"""Wasserstein distribution distance metrics for crystal structure evaluation."""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.stats import wasserstein_distance


# ---------------------------------------------------------------------------
# Crystal property helpers
# ---------------------------------------------------------------------------


def _sample_valid_crystals(crys_list: list, max_n: int = 1000, seed: int = 42) -> list:
    """Return up to max_n valid crystals sampled without replacement."""
    valid = [c for c in crys_list if getattr(c, "valid", False)]
    if not valid:
        return []
    if len(valid) <= max_n:
        return valid
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(valid), size=max_n, replace=False)
    return [valid[i] for i in idx]


def _mass_density(struct) -> float:
    return float(struct.density)


def _n_unique_elements(struct) -> int:
    return int(len({int(site.specie.Z) for site in struct.sites}))


def _atomic_density(struct) -> float | None:
    vol = float(struct.volume)
    if vol <= 0:
        return None
    return float(len(struct)) / vol


def _element_occurrence_support_weights(structs: list):
    counts = Counter()
    for s in structs:
        for site in s.sites:
            counts[int(site.specie.Z)] += 1
    if not counts:
        return None, None
    zs = np.fromiter(sorted(counts.keys()), dtype=float)
    ws = np.array([counts[int(z)] for z in zs], dtype=float)
    total = ws.sum()
    if total <= 0:
        return None, None
    ws = ws / total
    return zs, ws


# ---------------------------------------------------------------------------
# Distribution metrics
# ---------------------------------------------------------------------------


def _compute_wasserstein_metrics(
    pred_crys_list: list, ref_structs: list, max_samples: int = 1000, seed: int = 42
):
    """Compute four distribution distances between generated and reference sets."""
    if not pred_crys_list or not ref_structs:
        return {}

    pred_valid = _sample_valid_crystals(pred_crys_list, max_n=max_samples, seed=seed)
    if not pred_valid:
        return {}

    metrics: dict[str, float] = {}

    def _add_metric(name: str, gen_vals: list, ref_vals: list):
        gen = [v for v in gen_vals if v is not None and np.isfinite(v)]
        ref = [v for v in ref_vals if v is not None and np.isfinite(v)]
        if gen and ref:
            metrics[name] = float(wasserstein_distance(gen, ref))

    gen_mass = [
        _mass_density(c.structure)
        for c in pred_valid
        if getattr(c, "constructed", False)
    ]
    ref_mass = [_mass_density(s) for s in ref_structs]
    _add_metric("sample_dist/wdist_density_mass", gen_mass, ref_mass)

    gen_nary = [
        _n_unique_elements(c.structure)
        for c in pred_valid
        if getattr(c, "constructed", False)
    ]
    ref_nary = [_n_unique_elements(s) for s in ref_structs]
    _add_metric("sample_dist/wdist_nary", gen_nary, ref_nary)

    gen_atom_density = [
        _atomic_density(c.structure)
        for c in pred_valid
        if getattr(c, "constructed", False)
    ]
    ref_atom_density = [_atomic_density(s) for s in ref_structs]
    _add_metric("sample_dist/wdist_density_atomic", gen_atom_density, ref_atom_density)

    zG, wG = _element_occurrence_support_weights(
        [c.structure for c in pred_valid if getattr(c, "constructed", False)]
    )
    zT, wT = _element_occurrence_support_weights(ref_structs)
    if zG is not None and zT is not None:
        metrics["sample_dist/wdist_elem_occurrence"] = float(
            wasserstein_distance(zG, zT, u_weights=wG, v_weights=wT)
        )

    return metrics
