from __future__ import annotations

import torch
from torch import nn


class CrystalHeads(nn.Module):
    def __init__(
        self,
        d_model: int,
        vz: int,
        type_out_dim: int | None = None,
        coord_head_mode: str = "direct",
    ) -> None:
        super().__init__()
        # Predict element logits plus mask token (mask diffusion).
        if type_out_dim is None:
            type_out_dim = vz + 1
        mode = coord_head_mode.lower()
        if mode in {"non_relative", "non-relative"}:
            mode = "direct"
        if mode not in {"direct", "relative"}:
            raise ValueError(
                f"Unsupported coord_head_mode: {coord_head_mode}. Use 'direct' or 'relative'."
            )
        self.coord_head_mode = mode
        self.type_head = nn.Linear(d_model, type_out_dim, bias=True)
        if self.coord_head_mode == "direct":
            self.coord_head = nn.Linear(d_model, 3, bias=True)
        else:
            # Relative displacement head: attention-weighted minimum-image
            # fractional offsets, with learned feature similarity weights.
            self.coord_query = nn.Linear(d_model, d_model, bias=True)
            self.coord_key = nn.Linear(d_model, d_model, bias=True)
            self.coord_scale = float(d_model) ** -0.5
        self.lattice_head = nn.Linear(d_model, 6, bias=True)

    def _coord_from_relative(
        self,
        h_atoms: torch.Tensor,
        frac_coords: torch.Tensor,
        pad_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if frac_coords.dim() != 3 or frac_coords.shape[:2] != h_atoms.shape[:2]:
            raise RuntimeError(
                "frac_coords must be (B, N, 3) matching atom-token shape "
                f"{tuple(h_atoms.shape[:2])}, got {tuple(frac_coords.shape)}"
            )
        if frac_coords.shape[2] != 3:
            raise RuntimeError(f"frac_coords last dim must be 3, got {frac_coords.shape[2]}")

        q = self.coord_query(h_atoms)
        k = self.coord_key(h_atoms)
        scores = (
            torch.einsum("bid,bjd->bij", q, k).to(dtype=torch.float32) * self.coord_scale
        )

        key_pad_mask = None
        if pad_mask is not None:
            if pad_mask.shape != h_atoms.shape[:2]:
                raise RuntimeError(
                    f"pad_mask shape {tuple(pad_mask.shape)} does not match atoms "
                    f"{tuple(h_atoms.shape[:2])}"
                )
            key_pad_mask = pad_mask.bool()[:, None, :]
            scores = scores.masked_fill(key_pad_mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        if key_pad_mask is not None:
            attn = attn.masked_fill(key_pad_mask, 0.0)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Minimum-image displacement on the fractional torus.
        delta = frac_coords[:, :, None, :] - frac_coords[:, None, :, :]
        delta = delta - torch.round(delta)
        coord_vel = torch.einsum(
            "bij,bijc->bic", attn, delta.to(dtype=attn.dtype)
        ).to(dtype=h_atoms.dtype)
        if pad_mask is not None:
            coord_vel = coord_vel.masked_fill(pad_mask[..., None], 0.0)
        return coord_vel

    def forward(
        self,
        h: torch.Tensor,
        frac_coords: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # h: (B, S, d_model) with last token as lattice
        h_atoms = h[:, :-1, :]
        h_lat = h[:, -1, :]
        if self.coord_head_mode == "direct":
            coord_vel = self.coord_head(h_atoms)
        else:
            if frac_coords is None:
                raise RuntimeError(
                    "frac_coords must be provided when coord_head_mode='relative'."
                )
            coord_vel = self._coord_from_relative(h_atoms, frac_coords, pad_mask)
        return {
            "type_logits": self.type_head(h_atoms),
            "coord_vel": coord_vel,
            "lattice_vel": self.lattice_head(h_lat),
        }
