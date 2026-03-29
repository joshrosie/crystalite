from __future__ import annotations

import torch


def y1_to_gram(y1: torch.Tensor, cos_clip: float = 0.9999) -> torch.Tensor:
    """
    Convert Y1 lattice encoding to Gram matrix.

    Y1 convention:
      [log a, log b, log c, cos(alpha), cos(beta), cos(gamma)]
    Returns:
      Gram matrix G with shape (..., 3, 3), where d_cart^2 = d_frac^T G d_frac.
    """
    if y1.shape[-1] != 6:
        raise RuntimeError(f"Expected last dim 6 for Y1, got {tuple(y1.shape)}")
    log_a, log_b, log_c = y1[..., 0], y1[..., 1], y1[..., 2]
    cos_alpha = y1[..., 3].clamp(min=-cos_clip, max=cos_clip)
    cos_beta = y1[..., 4].clamp(min=-cos_clip, max=cos_clip)
    cos_gamma = y1[..., 5].clamp(min=-cos_clip, max=cos_clip)

    a = torch.exp(log_a)
    b = torch.exp(log_b)
    c = torch.exp(log_c)

    g00 = a * a
    g11 = b * b
    g22 = c * c
    g01 = a * b * cos_gamma
    g02 = a * c * cos_beta
    g12 = b * c * cos_alpha

    gram = torch.zeros((*y1.shape[:-1], 3, 3), device=y1.device, dtype=y1.dtype)
    gram[..., 0, 0] = g00
    gram[..., 1, 1] = g11
    gram[..., 2, 2] = g22
    gram[..., 0, 1] = g01
    gram[..., 1, 0] = g01
    gram[..., 0, 2] = g02
    gram[..., 2, 0] = g02
    gram[..., 1, 2] = g12
    gram[..., 2, 1] = g12
    return gram


def y1_to_loss_features(
    y1: torch.Tensor,
    num_atoms: torch.Tensor,
    log_len_min: float = -5.0,
    log_len_max: float = 5.0,
    cos_clip: float = 0.999,
) -> torch.Tensor:
    """
    Decode Y1 into loss-space features:
      - lengths normalized by N^(1/3)
      - angles in radians

    Clipping keeps the decode numerically stable when predictions wander
    outside the physically meaningful Y1 range early in training.
    """
    if y1.shape[-1] != 6:
        raise RuntimeError(f"Expected last dim 6 for Y1, got {tuple(y1.shape)}")
    if num_atoms.ndim != 1 or num_atoms.shape[0] != y1.shape[0]:
        raise RuntimeError(
            f"Expected num_atoms shape ({y1.shape[0]},), got {tuple(num_atoms.shape)}"
        )

    log_lengths = y1[..., :3].clamp(min=log_len_min, max=log_len_max)
    cos_angles = y1[..., 3:].clamp(min=-cos_clip, max=cos_clip)

    atom_scale = num_atoms.to(device=y1.device, dtype=y1.dtype).clamp_min(1.0)
    atom_scale = atom_scale.pow(1.0 / 3.0)[:, None]

    lengths = torch.exp(log_lengths) / atom_scale
    angles = torch.acos(cos_angles)
    return torch.cat([lengths, angles], dim=-1)


def ltri_params_to_lattice_matrix(params: torch.Tensor) -> torch.Tensor:
    """
    Map 6 unconstrained parameters to lower-triangular lattice matrix L:
      params = [p0, p1, p2, p3, p4, p5]
      L = [[exp(p0),    0,      0],
           [p1,      exp(p2),   0],
           [p3,         p4,   exp(p5)]]
    """
    if params.shape[-1] != 6:
        raise RuntimeError(f"Expected last dim 6 for Ltri params, got {tuple(params.shape)}")
    l = torch.zeros((*params.shape[:-1], 3, 3), device=params.device, dtype=params.dtype)
    l[..., 0, 0] = torch.exp(params[..., 0])
    l[..., 1, 0] = params[..., 1]
    l[..., 1, 1] = torch.exp(params[..., 2])
    l[..., 2, 0] = params[..., 3]
    l[..., 2, 1] = params[..., 4]
    l[..., 2, 2] = torch.exp(params[..., 5])
    return l


def ltri_params_to_gram(params: torch.Tensor) -> torch.Tensor:
    l = ltri_params_to_lattice_matrix(params)
    return l @ l.transpose(-1, -2)


def gram_to_y1(gram: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Convert Gram matrix G back to Y1 = [log lengths, cos angles].
    """
    if gram.shape[-2:] != (3, 3):
        raise RuntimeError(f"Expected trailing (3,3) Gram shape, got {tuple(gram.shape)}")

    g00 = gram[..., 0, 0].clamp_min(eps)
    g11 = gram[..., 1, 1].clamp_min(eps)
    g22 = gram[..., 2, 2].clamp_min(eps)

    a = torch.sqrt(g00)
    b = torch.sqrt(g11)
    c = torch.sqrt(g22)

    inv_ab = 1.0 / (a * b).clamp_min(eps)
    inv_ac = 1.0 / (a * c).clamp_min(eps)
    inv_bc = 1.0 / (b * c).clamp_min(eps)

    cos_gamma = (gram[..., 0, 1] * inv_ab).clamp(min=-1.0, max=1.0)
    cos_beta = (gram[..., 0, 2] * inv_ac).clamp(min=-1.0, max=1.0)
    cos_alpha = (gram[..., 1, 2] * inv_bc).clamp(min=-1.0, max=1.0)

    return torch.stack(
        [torch.log(a), torch.log(b), torch.log(c), cos_alpha, cos_beta, cos_gamma],
        dim=-1,
    )


def y1_to_ltri_params(y1: torch.Tensor) -> torch.Tensor:
    """
    Convert Y1 representation to lower-triangular latent params via Cholesky factor.
    """
    gram = y1_to_gram(y1)
    eye = torch.eye(3, device=gram.device, dtype=gram.dtype)

    chol = None
    for jitter in (0.0, 1e-8, 1e-6, 1e-4, 1e-3):
        gram_try = gram if jitter == 0.0 else gram + jitter * eye
        chol_try, info = torch.linalg.cholesky_ex(gram_try)
        if int(info.max().item()) == 0:
            chol = chol_try
            break
    if chol is None:
        evals, evecs = torch.linalg.eigh(gram)
        evals = evals.clamp_min(1e-6)
        gram_psd = evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)
        chol = torch.linalg.cholesky(gram_psd)

    d0 = chol[..., 0, 0].clamp_min(1e-12)
    d1 = chol[..., 1, 1].clamp_min(1e-12)
    d2 = chol[..., 2, 2].clamp_min(1e-12)
    return torch.stack(
        [
            torch.log(d0),
            chol[..., 1, 0],
            torch.log(d1),
            chol[..., 2, 0],
            chol[..., 2, 1],
            torch.log(d2),
        ],
        dim=-1,
    )


def lattice_latent_to_y1(latent: torch.Tensor, lattice_repr: str) -> torch.Tensor:
    rep = lattice_repr.lower()
    if rep == "y1":
        return latent
    if rep == "ltri":
        return gram_to_y1(ltri_params_to_gram(latent))
    raise ValueError(f"Unsupported lattice representation: {lattice_repr}")


def lattice_latent_to_gram(latent: torch.Tensor, lattice_repr: str) -> torch.Tensor:
    rep = lattice_repr.lower()
    if rep == "y1":
        return y1_to_gram(latent)
    if rep == "ltri":
        return ltri_params_to_gram(latent)
    raise ValueError(f"Unsupported lattice representation: {lattice_repr}")


def y1_to_lattice_latent(y1: torch.Tensor, lattice_repr: str) -> torch.Tensor:
    rep = lattice_repr.lower()
    if rep == "y1":
        return y1
    if rep == "ltri":
        return y1_to_ltri_params(y1)
    raise ValueError(f"Unsupported lattice representation: {lattice_repr}")
