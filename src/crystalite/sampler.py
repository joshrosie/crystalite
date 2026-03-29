import math

import torch

from src.crystalite.edm_utils import karras_sigma_steps, denoise_edm
from src.crystalite.crystalite import CrystaliteModel


def wrap_frac(delta: torch.Tensor) -> torch.Tensor:
    # Map differences to [-0.5, 0.5] to respect periodicity.
    return delta - torch.round(delta)


def resolve_nonnegative_scalar(
    name: str, value: float | None, default: float = 0.0
) -> float:
    scalar = float(default) if value is None else float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {scalar}.")
    if scalar < 0.0:
        raise ValueError(f"{name} must be >= 0, got {scalar}.")
    return scalar


def resolve_aa_rho_pair(
    aa_rho_coords: float | None,
    aa_rho_lattice: float | None,
) -> tuple[float, float]:
    coords = resolve_nonnegative_scalar("aa_rho_coords", aa_rho_coords)
    lattice = resolve_nonnegative_scalar("aa_rho_lattice", aa_rho_lattice)
    return coords, lattice


def clamp_lattice_latent(lat: torch.Tensor, lattice_repr: str) -> torch.Tensor:
    rep = lattice_repr.lower()
    if rep == "y1":
        # log lengths ~ exp in [0.006, 148], cosines safe for arccos.
        lat[..., :3] = lat[..., :3].clamp(min=-5.0, max=5.0)
        lat[..., 3:] = lat[..., 3:].clamp(min=-0.999, max=0.999)
        return lat
    if rep == "ltri":
        # Diagonal log-scales and off-diagonal shear terms.
        lat[..., 0] = lat[..., 0].clamp(min=-5.0, max=5.0)
        lat[..., 2] = lat[..., 2].clamp(min=-5.0, max=5.0)
        lat[..., 5] = lat[..., 5].clamp(min=-5.0, max=5.0)
        lat[..., 1] = lat[..., 1].clamp(min=-10.0, max=10.0)
        lat[..., 3] = lat[..., 3].clamp(min=-10.0, max=10.0)
        lat[..., 4] = lat[..., 4].clamp(min=-10.0, max=10.0)
        return lat
    raise ValueError(f"Unsupported lattice representation: {lattice_repr}")


@torch.no_grad()
def edm_sampler(
    model: CrystaliteModel,
    pad_mask: torch.Tensor,
    type_dim: int,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    S_churn: float,
    S_min: float,
    S_max: float,
    S_noise: float,
    sigma_data_type: float,
    sigma_data_coord: float,
    sigma_data_lat: float,
    generator: torch.Generator | None = None,
    autocast_dtype: torch.dtype | None = None,
    fixed_atom_types: torch.Tensor | None = None,
    skip_type_scaling: bool = False,
    aa_frac_max_scale: float = 0.0,
    aa_rho_coords: float = 0.0,
    aa_rho_lattice: float = 0.0,
    aa_rho_types: float = 0.0,
    lattice_repr: str = "y1",
) -> dict[str, torch.Tensor]:
    device = pad_mask.device
    bsz, nmax = pad_mask.shape
    real_mask = ~pad_mask

    type_x = torch.randn((bsz, nmax, type_dim), device=device, generator=generator)
    frac_x = torch.randn(
        (bsz, nmax, 3), device=device, generator=generator
    )  # centered coords
    lat_x = torch.randn((bsz, 6), device=device, generator=generator)

    type_x = torch.where(real_mask[..., None], type_x, torch.zeros_like(type_x))
    frac_x = torch.where(real_mask[..., None], frac_x, torch.zeros_like(frac_x))

    t_steps = karras_sigma_steps(
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
    )
    aa_rho_coords_eff, aa_rho_lattice_eff = resolve_aa_rho_pair(
        aa_rho_coords=aa_rho_coords,
        aa_rho_lattice=aa_rho_lattice,
    )
    aa_rho_types_eff = resolve_nonnegative_scalar("aa_rho_types", aa_rho_types)
    aa_rho_by_target = {
        "types": float(aa_rho_types_eff),
        "coords": float(aa_rho_coords_eff),
        "lattice": float(aa_rho_lattice_eff),
    }
    aa_scale_steps: dict[str, torch.Tensor] = {}
    for aa_target, aa_rho_target in aa_rho_by_target.items():
        if aa_rho_target <= 0.0:
            continue
        aa_steps = karras_sigma_steps(
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=aa_rho_target,
            device=device,
        )
        # Build anti-annealing scales from the nonzero Karras grid only.
        # The appended terminal 0 is an integration endpoint, not part of the
        # schedule parameterization, and would force the final ratio to ~1.
        t_main = t_steps[:-1]
        aa_main = aa_steps[:-1]
        if t_main.numel() >= 2:
            delta_base = t_main[1:] - t_main[:-1]
            delta_aa = aa_main[1:] - aa_main[:-1]
            scale_steps = (
                delta_aa.abs() / delta_base.abs().clamp_min(1e-12)
            ).clamp_min(1.0)
            # One scale per integration interval in the main loop.
            scale_steps = torch.cat([scale_steps, scale_steps[-1:]], dim=0)
        else:
            scale_steps = torch.ones_like(t_main)

        # Keep legacy cap behavior on fractional-coordinate drift only.
        if aa_target == "coords" and aa_frac_max_scale > 0.0:
            scale_steps = scale_steps.clamp(max=aa_frac_max_scale)
        aa_scale_steps[aa_target] = scale_steps

    type_next = type_x * t_steps[0]
    frac_next = frac_x * t_steps[0]
    lat_next = lat_x * t_steps[0]

    # In CSP mode, fix atom types to ground truth (no noise)
    if fixed_atom_types is not None:
        type_next = fixed_atom_types.to(dtype=type_next.dtype)

    for i, (t_cur, t_next_val) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        gamma = (
            min(S_churn / num_steps, math.sqrt(2.0) - 1.0)
            if (t_cur >= S_min and t_cur <= S_max)
            else 0.0
        )
        t_hat = t_cur + gamma * t_cur
        noise_scale = (t_hat**2 - t_cur**2).sqrt()

        type_hat = type_next + noise_scale * S_noise * torch.randn(
            type_next.shape, device=device, dtype=type_next.dtype, generator=generator
        )
        frac_hat = frac_next + noise_scale * S_noise * torch.randn(
            frac_next.shape, device=device, dtype=frac_next.dtype, generator=generator
        )
        lat_hat = lat_next + noise_scale * S_noise * torch.randn(
            lat_next.shape, device=device, dtype=lat_next.dtype, generator=generator
        )

        # In CSP mode, fix atom types to ground truth (no noise)
        if fixed_atom_types is not None:
            type_hat = fixed_atom_types.to(dtype=type_hat.dtype)

        sigma_hat = torch.full((bsz,), float(t_hat), device=device)
        denoised = denoise_edm(
            model=model,
            type_noisy=type_hat,
            frac_noisy=frac_hat,
            lat_noisy=lat_hat,
            pad_mask=pad_mask,
            sigma=sigma_hat,
            sigma_data_type=sigma_data_type,
            sigma_data_coord=sigma_data_coord,
            sigma_data_lat=sigma_data_lat,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            autocast_dtype=autocast_dtype,
            skip_type_scaling=skip_type_scaling,
        )

        type_d = denoised["type"]
        frac_d = denoised["frac"]
        lat_d = denoised["lat"]

        type_dx = (type_hat - type_d) / t_hat
        # Fractional coords live on a torus; use minimum-image deltas for drift.
        frac_dx = wrap_frac(frac_hat - frac_d) / t_hat
        lat_dx = (lat_hat - lat_d) / t_hat
        scale_type_i = aa_scale_steps["types"][i] if "types" in aa_scale_steps else None
        scale_frac_i = (
            aa_scale_steps["coords"][i] if "coords" in aa_scale_steps else None
        )
        scale_lat_i = (
            aa_scale_steps["lattice"][i] if "lattice" in aa_scale_steps else None
        )
        if scale_type_i is not None:
            type_dx = scale_type_i * type_dx
        if scale_frac_i is not None:
            frac_dx = scale_frac_i * frac_dx
        if scale_lat_i is not None:
            lat_dx = scale_lat_i * lat_dx

        type_next = type_hat + (t_next_val - t_hat) * type_dx
        if fixed_atom_types is not None:
            type_next = fixed_atom_types.to(dtype=type_next.dtype)
        frac_next = frac_hat + (t_next_val - t_hat) * frac_dx
        lat_next = lat_hat + (t_next_val - t_hat) * lat_dx

        # Keep centered coords bounded to avoid drift/overflow.
        frac_next = frac_next - torch.round(frac_next)

        if i < num_steps - 1:
            sigma_next = torch.full((bsz,), float(t_next_val), device=device)
            denoised_next = denoise_edm(
                model=model,
                type_noisy=type_next,
                frac_noisy=frac_next,
                lat_noisy=lat_next,
                pad_mask=pad_mask,
                sigma=sigma_next,
                sigma_data_type=sigma_data_type,
                sigma_data_coord=sigma_data_coord,
                sigma_data_lat=sigma_data_lat,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                autocast_dtype=autocast_dtype,
                skip_type_scaling=skip_type_scaling,
            )
            type_d2 = denoised_next["type"]
            frac_d2 = denoised_next["frac"]
            lat_d2 = denoised_next["lat"]

            type_dx2 = (type_next - type_d2) / t_next_val
            frac_dx2 = wrap_frac(frac_next - frac_d2) / t_next_val
            lat_dx2 = (lat_next - lat_d2) / t_next_val
            if scale_type_i is not None:
                type_dx2 = scale_type_i * type_dx2
            if scale_frac_i is not None:
                frac_dx2 = scale_frac_i * frac_dx2
            if scale_lat_i is not None:
                lat_dx2 = scale_lat_i * lat_dx2

            type_next = type_hat + (t_next_val - t_hat) * (
                0.5 * type_dx + 0.5 * type_dx2
            )
            if fixed_atom_types is not None:
                type_next = fixed_atom_types.to(dtype=type_next.dtype)
            frac_next = frac_hat + (t_next_val - t_hat) * (
                0.5 * frac_dx + 0.5 * frac_dx2
            )
            lat_next = lat_hat + (t_next_val - t_hat) * (0.5 * lat_dx + 0.5 * lat_dx2)

            frac_next = frac_next - torch.round(frac_next)

    type_next = torch.where(
        real_mask[..., None], type_next, torch.zeros_like(type_next)
    )
    # Keep centered coords; wrapping applied at decode time by caller.
    frac_next = torch.where(
        real_mask[..., None], frac_next, torch.zeros_like(frac_next)
    )
    # Clamp to avoid NaNs/infs before visualization.
    lat_next = torch.nan_to_num(lat_next, nan=0.0, posinf=10.0, neginf=-10.0)
    type_next = torch.nan_to_num(type_next, nan=0.0, posinf=10.0, neginf=-10.0)
    frac_next = torch.nan_to_num(frac_next, nan=0.0, posinf=1.0, neginf=0.0)
    lat_next = clamp_lattice_latent(lat_next, lattice_repr=lattice_repr)
    # Keep type logits bounded to reduce overflow in softmax/argmax downstream.
    type_next = type_next.clamp(min=-50.0, max=50.0)

    return {"type": type_next, "frac": frac_next, "lat": lat_next}
