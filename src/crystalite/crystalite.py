from __future__ import annotations

import torch
from torch import nn

from src.models.embeddings import FourierCoordEmbedder, LatticeEmbedder, TimeEmbedder
from src.models.transformer import TransformerTrunk
from src.models.heads import CrystalHeads


def mod1(x: torch.Tensor) -> torch.Tensor:
    """Wrap values into [0, 1)."""
    return x - torch.floor(x)


class CrystaliteModel(nn.Module):
    """
    Transformer backbone for Crystalite EDM training on tokenized crystals.

    This keeps the original trunk/head architecture but swaps the token
    embedding to accept continuous (noisy) features:
      - type_feats: (B, N, type_dim) relaxed type features
      - frac_coords: (B, N, 3) fractional coords (wrapped to [0,1) internally)
      - lattice_feats: (B, 6)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        vz: int,
        type_dim: int | None = None,
        n_freqs: int = 32,
        coord_embed_mode: str = "rff",
        coord_rff_dim: int | None = None,
        coord_rff_sigma: float = 1.0,
        lattice_embed_mode: str = "rff",
        lattice_rff_dim: int = 256,
        lattice_rff_sigma: float = 5.0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_distance_bias: bool = False,
        use_edge_bias: bool = False,
        edge_bias_n_freqs: int = 8,
        edge_bias_hidden_dim: int = 128,
        edge_bias_n_rbf: int = 16,
        edge_bias_rbf_max: float = 2.0,
        pbc_radius: int = 1,
        lattice_repr: str = "y1",
        dist_slope_init: float = -1.0,
        use_noise_gate: bool = True,
        gem_per_layer: bool = False,
        coord_head_mode: str = "direct",
    ) -> None:
        super().__init__()
        self.type_dim = (vz + 1) if type_dim is None else int(type_dim)
        self.type_proj = nn.Sequential(
            nn.Linear(self.type_dim, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
        self.coord_embed = FourierCoordEmbedder(
            d_model=d_model,
            n_freqs=n_freqs,
            mode=coord_embed_mode,
            rff_dim=coord_rff_dim,
            rff_sigma=coord_rff_sigma,
        )
        self.lattice_embed = LatticeEmbedder(
            d_model=d_model,
            mode=lattice_embed_mode,
            rff_dim=lattice_rff_dim,
            rff_sigma=lattice_rff_sigma,
        )
        self.segment_embed = nn.Embedding(2, d_model)
        self.time = TimeEmbedder(d_model=d_model)
        self.trunk = TransformerTrunk(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            use_distance_bias=use_distance_bias,
            use_edge_bias=use_edge_bias,
            edge_bias_n_freqs=edge_bias_n_freqs,
            edge_bias_hidden_dim=edge_bias_hidden_dim,
            edge_bias_n_rbf=edge_bias_n_rbf,
            edge_bias_rbf_max=edge_bias_rbf_max,
            pbc_radius=pbc_radius,
            lattice_repr=lattice_repr,
            dist_slope_init=dist_slope_init,
            use_noise_gate=use_noise_gate,
            gem_per_layer=gem_per_layer,
        )
        self.heads = CrystalHeads(
            d_model=d_model,
            vz=vz,
            type_out_dim=self.type_dim,
            coord_head_mode=coord_head_mode,
        )

    def forward(
        self,
        type_feats: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice_feats: torch.Tensor,
        pad_mask: torch.Tensor,
        t_sigma: torch.Tensor,
        lattice_bias_feats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            type_feats: (B, N, type_dim) relaxed type features with padding zeroed.
            frac_coords: (B, N, 3) fractional coords (not necessarily wrapped).
            lattice_feats: (B, 6)
            pad_mask: (B, N) bool where True denotes padding.
            t_sigma: (B,) scalar noise embedding; shared for t_g/t_a.
            lattice_bias_feats: optional (B, 6) lattice features used only for
                geometry-aware attention bias. If None, lattice_feats is used.
        """
        frac_mod = mod1(frac_coords)
        h_type = self.type_proj(type_feats) + self.coord_embed(frac_mod) + self.segment_embed.weight[0]
        h_lat = self.lattice_embed(lattice_feats) + self.segment_embed.weight[1]
        h_lat = h_lat[:, None, :]

        x = torch.cat([h_type, h_lat], dim=1)
        pad_seq = torch.cat(
            [
                pad_mask.bool(),
                torch.zeros((pad_mask.shape[0], 1), device=pad_mask.device, dtype=torch.bool),
            ],
            dim=1,
        )
        t_emb = self.time(t_sigma, t_sigma)
        # In the EDM codepath, t_sigma is the noise embedding c_noise = 0.25 * log(sigma).
        # GEM's noise gate expects sigma (positive, on the same scale as the Karras schedule),
        # so convert back here while keeping the time embedding on c_noise.
        sigma_for_gem = torch.exp(4.0 * t_sigma.to(dtype=torch.float32))
        lattice_for_bias = lattice_feats if lattice_bias_feats is None else lattice_bias_feats
        h = self.trunk(
            x,
            t_emb,
            pad_mask=pad_seq,
            coords=frac_mod,
            lattice=lattice_for_bias,
            t_sigma=sigma_for_gem,
        )
        return self.heads(h, frac_coords=frac_mod, pad_mask=pad_mask.bool())


# Backwards-compatible alias during the rename from the old MP20-specific name.
MP20EDMModel = CrystaliteModel
