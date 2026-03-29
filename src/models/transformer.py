from __future__ import annotations

import os
from itertools import product

import torch
import torch.nn.functional as F
from torch import nn

from src.models.lattice_repr import lattice_latent_to_gram, lattice_latent_to_y1


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def _softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(1e-8)
    return x + torch.log(-torch.expm1(-x))


class Mlp(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AdaLNBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.mlp = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.adaLN_mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)
        self.n_heads = n_heads

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        attn_head_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t_emb.dim() != 2 or t_emb.shape[1] != x.shape[-1]:
            raise RuntimeError(
                f"t_emb must be (B, {x.shape[-1]}), got {tuple(t_emb.shape)}"
            )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_mod(t_emb).chunk(6, dim=1)
        )
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_mask = None
        if attn_head_bias is not None:
            if attn_head_bias.dim() != 4:
                raise RuntimeError(
                    f"attn_head_bias must be (B, H, L, L), got {tuple(attn_head_bias.shape)}"
                )
            if attn_head_bias.shape[1] != self.n_heads:
                raise RuntimeError(
                    f"attn_head_bias has {attn_head_bias.shape[1]} heads, expected {self.n_heads}"
                )
            if attn_head_bias.shape[2] != x.shape[1] or attn_head_bias.shape[3] != x.shape[1]:
                raise RuntimeError(
                    "attn_head_bias sequence dims must match token length "
                    f"{x.shape[1]}, got {tuple(attn_head_bias.shape[2:])}"
                )
            head_bias = attn_head_bias.reshape(
                -1, attn_head_bias.shape[2], attn_head_bias.shape[3]
            ).to(dtype=h.dtype)
            attn_mask = head_bias if attn_mask is None else (attn_mask + head_bias)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=pad_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = x + gate_msa[:, None, :] * attn_out
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :] * self.mlp(h)
        return x


class GeometryEnhancementModule(nn.Module):
    def __init__(
        self,
        n_heads: int,
        pbc_radius: int = 1,
        use_distance_bias: bool = False,
        use_edge_bias: bool = False,
        edge_bias_n_freqs: int = 8,
        edge_bias_hidden_dim: int = 128,
        edge_bias_n_rbf: int = 16,
        edge_bias_rbf_max: float = 2.0,
        lattice_repr: str = "y1",
        dist_slope_init: float = -1.0,
        use_noise_gate: bool = True,
    ) -> None:
        super().__init__()
        self.use_distance_bias = use_distance_bias
        self.use_edge_bias = use_edge_bias
        self.use_noise_gate = use_noise_gate
        self.n_heads = n_heads
        self.lattice_repr = lattice_repr.lower()
        if self.lattice_repr not in {"y1", "ltri"}:
            raise ValueError(
                f"Unsupported lattice_repr: {lattice_repr}. Use 'y1' or 'ltri'."
            )
        if edge_bias_n_freqs <= 0:
            raise ValueError(f"edge_bias_n_freqs must be > 0, got {edge_bias_n_freqs}")
        if edge_bias_hidden_dim <= 0:
            raise ValueError(
                f"edge_bias_hidden_dim must be > 0, got {edge_bias_hidden_dim}"
            )
        if edge_bias_n_rbf <= 0:
            raise ValueError(f"edge_bias_n_rbf must be > 0, got {edge_bias_n_rbf}")
        if edge_bias_rbf_max <= 0.0:
            raise ValueError(f"edge_bias_rbf_max must be > 0, got {edge_bias_rbf_max}")
        if not (use_distance_bias or use_edge_bias):
            raise ValueError(
                "GeometryEnhancementModule requires use_distance_bias or use_edge_bias."
            )

        pbc_radius = int(pbc_radius)
        if pbc_radius not in {1, 2}:
            raise ValueError(f"pbc_radius must be 1 or 2, got {pbc_radius}")
        self.pbc_radius = pbc_radius

        offsets = torch.tensor(
            list(product(range(-pbc_radius, pbc_radius + 1), repeat=3)),
            dtype=torch.float32,
        )
        self.register_buffer(
            "_pbc_offsets",
            offsets.view(1, 1, 1, offsets.shape[0], 3),
            persistent=False,
        )

        if self.use_distance_bias:
            # GEM v2: monotone non-positive slope via -softplus(raw).
            slope_mag = torch.full(
                (1, n_heads, 1, 1),
                fill_value=abs(float(dist_slope_init)),
                dtype=torch.float32,
            ).clamp_min(1e-4)
            self.dist_slope_raw = nn.Parameter(
                _softplus_inverse(slope_mag) + 0.01 * torch.randn_like(slope_mag)
            )

        if self.use_edge_bias:
            self.register_buffer(
                "_edge_freqs",
                torch.arange(1, edge_bias_n_freqs + 1, dtype=torch.float32),
                persistent=False,
            )
            edge_fourier_dim = 6 * edge_bias_n_freqs
            self.register_buffer(
                "_edge_rbf_centers",
                torch.linspace(0.0, edge_bias_rbf_max, edge_bias_n_rbf, dtype=torch.float32),
                persistent=False,
            )
            if edge_bias_n_rbf == 1:
                rbf_gamma = 1.0
            else:
                delta = edge_bias_rbf_max / float(edge_bias_n_rbf - 1)
                rbf_gamma = 1.0 / max(delta * delta, 1e-6)
            self.register_buffer(
                "_edge_rbf_gamma", torch.tensor(rbf_gamma, dtype=torch.float32), persistent=False
            )
            edge_in_dim = edge_fourier_dim + edge_bias_n_rbf + 6  # Fourier(Δf_min), RBF(d_norm), y1
            self.edge_bias_mlp = nn.Sequential(
                nn.Linear(edge_in_dim, edge_bias_hidden_dim, bias=True),
                nn.SiLU(),
                nn.Linear(edge_bias_hidden_dim, n_heads, bias=True),
            )
            nn.init.zeros_(self.edge_bias_mlp[-1].weight)
            nn.init.zeros_(self.edge_bias_mlp[-1].bias)

        if self.use_noise_gate:
            gate_alpha_init = torch.ones((1, n_heads, 1, 1), dtype=torch.float32)
            self.noise_gate_alpha_raw = nn.Parameter(_softplus_inverse(gate_alpha_init))
            self.noise_gate_beta = nn.Parameter(torch.zeros((1, n_heads, 1, 1)))

    def _sanitize_lattice_latent(self, lattice: torch.Tensor) -> torch.Tensor:
        if lattice.dim() != 2 or lattice.shape[1] != 6:
            raise RuntimeError(f"lattice must be (B, 6), got {tuple(lattice.shape)}")
        lattice_safe = lattice.clone()
        if self.lattice_repr == "y1":
            lattice_safe[:, :3] = lattice_safe[:, :3].clamp(min=-5.0, max=5.0)
            lattice_safe[:, 3:] = lattice_safe[:, 3:].clamp(min=-0.9999, max=0.9999)
            return lattice_safe
        lattice_safe[:, 0] = lattice_safe[:, 0].clamp(min=-5.0, max=5.0)
        lattice_safe[:, 2] = lattice_safe[:, 2].clamp(min=-5.0, max=5.0)
        lattice_safe[:, 5] = lattice_safe[:, 5].clamp(min=-5.0, max=5.0)
        lattice_safe[:, 1] = lattice_safe[:, 1].clamp(min=-10.0, max=10.0)
        lattice_safe[:, 3] = lattice_safe[:, 3].clamp(min=-10.0, max=10.0)
        lattice_safe[:, 4] = lattice_safe[:, 4].clamp(min=-10.0, max=10.0)
        return lattice_safe

    def _lattice_y1_and_gram(self, lattice: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lattice_safe = self._sanitize_lattice_latent(lattice)
        lattice_y1 = lattice_latent_to_y1(lattice_safe, lattice_repr=self.lattice_repr)
        lattice_y1 = lattice_y1.clone()
        lattice_y1[:, :3] = lattice_y1[:, :3].clamp(min=-5.0, max=5.0)
        lattice_y1[:, 3:] = lattice_y1[:, 3:].clamp(min=-0.9999, max=0.9999)
        gram = lattice_latent_to_gram(lattice_safe, lattice_repr=self.lattice_repr)
        return (
            torch.nan_to_num(lattice_y1, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(gram, nan=0.0, posinf=0.0, neginf=0.0),
        )

    def _pairwise_min_image_geometry(
        self, coords: torch.Tensor, lattice: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if coords.dim() != 3:
            raise RuntimeError(f"coords must be (B, N, 3), got {tuple(coords.shape)}")
        b, _, c = coords.shape
        if c != 3:
            raise RuntimeError("coords last dimension must be 3 (fractional coordinates)")
        if lattice.shape[0] != b:
            raise RuntimeError(
                f"lattice batch {lattice.shape[0]} does not match coords batch {b}"
            )

        coords_f = coords.to(dtype=torch.float32)
        lattice_f = lattice.to(dtype=torch.float32)
        lattice_y1, gram = self._lattice_y1_and_gram(lattice_f)
        gram = gram.to(dtype=coords_f.dtype)

        delta = coords_f[:, :, None, :] - coords_f[:, None, :, :]
        offsets = self._pbc_offsets.to(device=coords_f.device, dtype=coords_f.dtype)
        delta_images = delta[:, :, :, None, :] + offsets
        dist2_images = torch.einsum(
            "bnmki,bij,bnmkj->bnmk", delta_images, gram, delta_images
        )
        min_idx = dist2_images.argmin(dim=-1, keepdim=True)
        min_dist2 = torch.gather(dist2_images, dim=-1, index=min_idx).squeeze(-1)
        min_dist2 = min_dist2.clamp_min(0.0)
        gather_idx = min_idx[..., None].expand(-1, -1, -1, 1, 3)
        min_delta = torch.gather(delta_images, dim=3, index=gather_idx).squeeze(3)
        min_dist = torch.sqrt(min_dist2)
        min_dist = torch.nan_to_num(min_dist, nan=0.0, posinf=0.0, neginf=0.0)
        min_dist2 = torch.nan_to_num(min_dist2, nan=0.0, posinf=0.0, neginf=0.0)

        lengths = torch.exp(lattice_y1[:, :3]).to(dtype=coords_f.dtype)
        cell_scale = lengths.mean(dim=-1).clamp_min(1e-6)
        min_dist_norm = min_dist / cell_scale[:, None, None]
        min_dist_norm = torch.nan_to_num(min_dist_norm, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "min_delta": min_delta,
            "min_dist_norm": min_dist_norm,
            "lattice_y1": lattice_y1,
        }

    @staticmethod
    def _mask_atom_pairwise_bias(
        pair_bias: torch.Tensor, atom_pad_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if atom_pad_mask is None:
            return pair_bias
        if pair_bias.dim() != 4:
            raise RuntimeError(
                f"pair_bias must have 4 dims (B, H, N, N), got {tuple(pair_bias.shape)}"
            )
        pair_bias = pair_bias.masked_fill(atom_pad_mask[:, None, :, None], 0.0)
        pair_bias = pair_bias.masked_fill(atom_pad_mask[:, None, None, :], 0.0)
        return pair_bias

    @staticmethod
    def _expand_atom_head_bias(
        bias_atoms: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        b, h, n, _ = bias_atoms.shape
        if seq_len == n:
            return bias_atoms
        if seq_len == n + 1:
            bias_full = torch.zeros(
                (b, h, seq_len, seq_len),
                device=bias_atoms.device,
                dtype=bias_atoms.dtype,
            )
            bias_full[:, :, :n, :n] = bias_atoms
            return bias_full
        raise RuntimeError(
            f"seq_len={seq_len} incompatible with coords length {n}; expected {n} or {n+1}."
        )

    def _compute_distance_bias_atoms(self, min_dist_norm: torch.Tensor) -> torch.Tensor:
        if not self.use_distance_bias:
            raise RuntimeError(
                "_compute_distance_bias_atoms called when use_distance_bias=False"
            )
        dist_slope = -F.softplus(self.dist_slope_raw)
        return dist_slope * min_dist_norm.unsqueeze(1)

    def _compute_edge_bias_atoms(
        self,
        min_delta_frac: torch.Tensor,
        min_dist_norm: torch.Tensor,
        lattice_y1: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_edge_bias:
            raise RuntimeError("_compute_edge_bias_atoms called when use_edge_bias=False")
        freqs = self._edge_freqs.to(
            device=min_delta_frac.device, dtype=min_delta_frac.dtype
        )
        args = 2.0 * torch.pi * min_delta_frac[..., None] * freqs.view(1, 1, 1, 1, -1)
        edge_fourier = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        b, n, m, _, _ = edge_fourier.shape
        edge_fourier = edge_fourier.reshape(b, n, m, -1)

        centers = self._edge_rbf_centers.to(
            device=min_dist_norm.device, dtype=min_dist_norm.dtype
        )
        gamma = self._edge_rbf_gamma.to(
            device=min_dist_norm.device, dtype=min_dist_norm.dtype
        )
        edge_rbf = torch.exp(
            -gamma * (min_dist_norm.unsqueeze(-1) - centers.view(1, 1, 1, -1)).pow(2)
        )
        lattice_feat = lattice_y1[:, None, None, :].to(dtype=min_dist_norm.dtype).expand(
            -1, n, m, -1
        )
        edge_feat = torch.cat([edge_fourier, edge_rbf, lattice_feat], dim=-1)
        edge_bias = self.edge_bias_mlp(edge_feat).permute(0, 3, 1, 2).contiguous()
        diag = torch.diagonal(edge_bias, offset=0, dim1=-2, dim2=-1)
        edge_bias = edge_bias - torch.diag_embed(diag)
        return edge_bias

    def _sigma_gate(
        self,
        t_sigma: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.use_noise_gate:
            return torch.ones((batch_size, self.n_heads, 1, 1), device=device, dtype=dtype)
        if t_sigma is None:
            raise RuntimeError("t_sigma must be provided when geometry bias is enabled.")
        sigma = t_sigma
        if sigma.dim() == 0:
            sigma = sigma.expand(batch_size)
        if sigma.dim() == 2 and sigma.shape[1] == 1:
            sigma = sigma[:, 0]
        if sigma.dim() != 1:
            raise RuntimeError(f"t_sigma must be (B,), got {tuple(sigma.shape)}")
        if sigma.shape[0] == 1 and batch_size > 1:
            sigma = sigma.expand(batch_size)
        if sigma.shape[0] != batch_size:
            raise RuntimeError(
                f"t_sigma batch {sigma.shape[0]} does not match expected batch {batch_size}"
            )
        sigma_safe = sigma.to(device=device, dtype=torch.float32).clamp_min(1e-8)
        sigma_feat = -torch.log(sigma_safe)
        alpha = F.softplus(self.noise_gate_alpha_raw).to(device=device, dtype=torch.float32)
        beta = self.noise_gate_beta.to(device=device, dtype=torch.float32)
        gate = torch.sigmoid(alpha * sigma_feat[:, None, None, None] + beta)
        return gate.to(dtype=dtype)

    def compute_geometry_features(
        self, coords: torch.Tensor, lattice: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self._pairwise_min_image_geometry(coords, lattice)

    def forward_from_features(
        self,
        geom_features: dict[str, torch.Tensor],
        coords_len: int,
        pad_mask: torch.Tensor | None = None,
        seq_len: int | None = None,
        t_sigma: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        if not (self.use_distance_bias or self.use_edge_bias):
            return None

        min_delta = geom_features["min_delta"]
        min_dist_norm = geom_features["min_dist_norm"]
        lattice_y1 = geom_features["lattice_y1"]

        geom_bias_atoms = None
        if self.use_distance_bias:
            geom_bias_atoms = self._compute_distance_bias_atoms(
                min_dist_norm=min_dist_norm,
            )
        if self.use_edge_bias:
            edge_atoms = self._compute_edge_bias_atoms(
                min_delta_frac=min_delta,
                min_dist_norm=min_dist_norm,
                lattice_y1=lattice_y1,
            )
            geom_bias_atoms = edge_atoms if geom_bias_atoms is None else (geom_bias_atoms + edge_atoms)
        if geom_bias_atoms is None:
            raise RuntimeError("Geometry bias branches produced no output.")

        atom_pad_mask = (
            pad_mask[:, :coords_len].bool() if pad_mask is not None else None
        )
        geom_bias_atoms = self._mask_atom_pairwise_bias(geom_bias_atoms, atom_pad_mask)
        gate = self._sigma_gate(
            t_sigma=t_sigma,
            batch_size=min_delta.shape[0],
            device=geom_bias_atoms.device,
            dtype=geom_bias_atoms.dtype,
        )
        geom_bias_atoms = gate * geom_bias_atoms
        target_seq_len = coords_len if seq_len is None else seq_len
        geom_bias = self._expand_atom_head_bias(geom_bias_atoms, seq_len=target_seq_len)
        if out_dtype is not None:
            geom_bias = geom_bias.to(dtype=out_dtype)
        return geom_bias

    def forward(
        self,
        coords: torch.Tensor,
        lattice: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        seq_len: int | None = None,
        t_sigma: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        geom_features = self.compute_geometry_features(coords=coords, lattice=lattice)
        return self.forward_from_features(
            geom_features=geom_features,
            coords_len=coords.shape[1],
            pad_mask=pad_mask,
            seq_len=seq_len,
            t_sigma=t_sigma,
            out_dtype=out_dtype,
        )


class TransformerTrunk(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        eps: float = 1e-6,
        use_distance_bias: bool = False,
        use_edge_bias: bool = False,
        edge_bias_n_freqs: int = 8,
        edge_bias_hidden_dim: int = 128,
        edge_bias_n_rbf: int = 16,
        edge_bias_rbf_max: float = 2.0,
        pbc_radius: int = 1,
        lattice_repr: str = "y1",
        dist_slope_init: float = -1.0,
        distance_bias_init: float | None = None,
        use_noise_gate: bool = True,
        gem_per_layer: bool = False,
    ) -> None:
        super().__init__()
        if distance_bias_init is not None:
            dist_slope_init = float(distance_bias_init)
        self.use_distance_bias = use_distance_bias
        self.use_edge_bias = use_edge_bias
        self.use_geom_bias = use_distance_bias or use_edge_bias
        self.gem_per_layer = gem_per_layer
        self.n_heads = n_heads
        self.lattice_repr = lattice_repr.lower()
        if self.lattice_repr not in {"y1", "ltri"}:
            raise ValueError(
                f"Unsupported lattice_repr: {lattice_repr}. Use 'y1' or 'ltri'."
            )
        self.blocks = nn.ModuleList(
            [
                AdaLNBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    eps=eps,
                )
                for _ in range(n_layers)
            ]
        )
        if self.use_geom_bias:
            if self.gem_per_layer:
                self.gems = nn.ModuleList(
                    [
                        GeometryEnhancementModule(
                            n_heads=n_heads,
                            pbc_radius=pbc_radius,
                            use_distance_bias=use_distance_bias,
                            use_edge_bias=use_edge_bias,
                            edge_bias_n_freqs=edge_bias_n_freqs,
                            edge_bias_hidden_dim=edge_bias_hidden_dim,
                            edge_bias_n_rbf=edge_bias_n_rbf,
                            edge_bias_rbf_max=edge_bias_rbf_max,
                            lattice_repr=lattice_repr,
                            dist_slope_init=dist_slope_init,
                            use_noise_gate=use_noise_gate,
                        )
                        for _ in range(n_layers)
                    ]
                )
                self.gem = None
            else:
                self.gem = GeometryEnhancementModule(
                    n_heads=n_heads,
                    pbc_radius=pbc_radius,
                    use_distance_bias=use_distance_bias,
                    use_edge_bias=use_edge_bias,
                    edge_bias_n_freqs=edge_bias_n_freqs,
                    edge_bias_hidden_dim=edge_bias_hidden_dim,
                    edge_bias_n_rbf=edge_bias_n_rbf,
                    edge_bias_rbf_max=edge_bias_rbf_max,
                    lattice_repr=lattice_repr,
                    dist_slope_init=dist_slope_init,
                    use_noise_gate=use_noise_gate,
                )
                self.gems = None
        else:
            self.gem = None
            self.gems = None
        self.norm_out = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        lattice: torch.Tensor | None = None,
        t_sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pad_mask is not None and pad_mask.dtype != torch.bool:
            pad_mask = pad_mask.bool()
        if pad_mask is not None and pad_mask.shape[:2] != x.shape[:2]:
            raise RuntimeError(
                f"pad_mask shape {tuple(pad_mask.shape)} does not match x {tuple(x.shape)}"
            )

        attn_head_bias = None
        geom_features = None
        if self.gem is not None or self.gems is not None:
            if coords is None:
                raise RuntimeError(
                    "coords must be provided when distance or edge bias is enabled"
                )
            if lattice is None:
                raise RuntimeError(
                    "lattice must be provided when distance or edge bias is enabled"
                )
            if coords.shape[0] != x.shape[0]:
                raise RuntimeError(
                    f"coords batch {coords.shape[0]} does not match x batch {x.shape[0]}"
                )
            if lattice.shape[0] != x.shape[0]:
                raise RuntimeError(
                    f"lattice batch {lattice.shape[0]} does not match x batch {x.shape[0]}"
                )
            if self.gems is not None:
                geom_features = self.gems[0].compute_geometry_features(
                    coords=coords,
                    lattice=lattice,
                )
            else:
                attn_head_bias = self.gem(
                    coords=coords,
                    lattice=lattice,
                    pad_mask=pad_mask,
                    seq_len=x.shape[1],
                    t_sigma=t_sigma,
                    out_dtype=x.dtype,
                )
        for layer_idx, block in enumerate(self.blocks):
            if pad_mask is not None:
                x = x.masked_fill(pad_mask[..., None], 0.0)
            if self.gems is not None:
                attn_head_bias = self.gems[layer_idx].forward_from_features(
                    geom_features=geom_features,
                    coords_len=coords.shape[1],
                    pad_mask=pad_mask,
                    seq_len=x.shape[1],
                    t_sigma=t_sigma,
                    out_dtype=x.dtype,
                )
            x = block(
                x,
                t_emb,
                pad_mask=pad_mask,
                attn_head_bias=attn_head_bias,
            )
        x = self.norm_out(x)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask[..., None], 0.0)
            if os.environ.get("DEBUG"):
                if pad_mask.any() and x[pad_mask].abs().max().item() != 0.0:
                    raise RuntimeError("Padded tokens not zero after trunk.")
        return x
