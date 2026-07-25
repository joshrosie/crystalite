"""Space-group match-rate evaluation for conditional generation.

The tightest "did conditioning work?" signal for space-group-conditioned
generation: convert generated crystals to pymatgen ``Structure`` objects, compute
each structure's space group symmetry-analytically (no DFT, no surrogate model),
and report the exact match-rate against the conditioning target. Because generated
structures usually have *slightly* broken symmetry, symmetry detection is
tolerance-sensitive, so match-rate is reported across a sweep of ``symprec``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from src.data.mp20_tokens import _to_numpy, decode_Y1

# Generated structures have imperfect symmetry, so evaluate across tolerances.
DEFAULT_SYMPREC = (0.01, 0.05, 0.1, 0.3)
DEFAULT_ANGLE_TOLERANCE = 5.0


def compute_space_group(
    structure: Structure,
    symprec: float = 0.1,
    angle_tolerance: float = DEFAULT_ANGLE_TOLERANCE,
) -> int | None:
    """Space-group number (1-230) of a structure, or None if analysis fails.

    ``SpacegroupAnalyzer`` can raise on degenerate/near-singular lattices that
    diffusion sometimes produces; those are treated as unanalysable (None).
    """
    try:
        sga = SpacegroupAnalyzer(
            structure, symprec=symprec, angle_tolerance=angle_tolerance
        )
        return int(sga.get_space_group_number())
    except Exception:
        return None


def compute_space_groups(
    structures: Sequence[Structure],
    symprec: float = 0.1,
    angle_tolerance: float = DEFAULT_ANGLE_TOLERANCE,
) -> list[int | None]:
    return [compute_space_group(s, symprec, angle_tolerance) for s in structures]


def structures_from_sample_batch(batch: dict) -> tuple[list[Structure], int]:
    """Build pymatgen Structures from a generated token batch.

    Mirrors :func:`src.data.mp20_tokens.tokens_batch_to_structures` but builds
    each structure independently and skips ones that fail to construct (e.g. a
    generated sample with a padding-position atom or a degenerate cell), so one
    bad sample does not abort the whole batch. Returns ``(structures, n_failed)``.
    """
    structures: list[Structure] = []
    n_failed = 0
    batch_size = _to_numpy(batch["A0"]).shape[0]
    for i in range(batch_size):
        try:
            pad = _to_numpy(batch["pad_mask"][i]).astype(bool)
            mask = ~pad
            atom_types = _to_numpy(batch["A0"][i])[mask]
            if atom_types.size == 0 or (atom_types <= 0).any():
                raise ValueError("empty or non-positive atomic numbers")
            species = [Element.from_Z(int(z)) for z in atom_types]
            frac_coords = _to_numpy(batch["F1"][i])[mask]
            lengths, angles = decode_Y1(batch["Y1"][i])
            lattice = Lattice.from_parameters(*(lengths.tolist() + angles.tolist()))
            structures.append(
                Structure(
                    lattice=lattice,
                    species=species,
                    coords=frac_coords,
                    coords_are_cartesian=False,
                )
            )
        except Exception:
            n_failed += 1
    return structures, n_failed


def space_group_match_rate(
    structures: Sequence[Structure],
    target: int | Sequence[int],
    symprec_list: Sequence[float] = DEFAULT_SYMPREC,
    angle_tolerance: float = DEFAULT_ANGLE_TOLERANCE,
) -> dict[float, dict]:
    """Exact space-group match-rate vs. the conditioning target, per symprec.

    ``target`` is a single SG number (all structures share it — the usual
    sampling setup) or a per-structure sequence. For each ``symprec`` returns:

    - ``match_rate``: matches / total (unanalysable structures count as misses —
      the honest end-to-end number).
    - ``match_rate_valid``: matches / analysable (isolates conditioning quality
      from structural validity).
    - ``n_match`` / ``n_valid`` / ``n_total`` and the raw ``computed`` SGs.
    """
    n = len(structures)
    if isinstance(target, (int, np.integer)):
        targets = [int(target)] * n
    else:
        targets = [int(t) for t in target]
        if len(targets) != n:
            raise ValueError(f"target length {len(targets)} != n structures {n}")

    results: dict[float, dict] = {}
    for symprec in symprec_list:
        computed = compute_space_groups(structures, symprec, angle_tolerance)
        pairs = [(c, t) for c, t in zip(computed, targets) if c is not None]
        n_match = sum(1 for c, t in pairs if c == t)
        n_valid = len(pairs)
        results[symprec] = {
            "symprec": symprec,
            "match_rate": (n_match / n) if n else 0.0,
            "match_rate_valid": (n_match / n_valid) if n_valid else 0.0,
            "n_match": n_match,
            "n_valid": n_valid,
            "n_total": n,
            "computed": computed,
        }
    return results


def sweep_guidance_match(
    generate_fn: Callable[[float], Sequence[Structure]],
    target: int,
    guidance_weights: Sequence[float],
    symprec_list: Sequence[float] = DEFAULT_SYMPREC,
    angle_tolerance: float = DEFAULT_ANGLE_TOLERANCE,
) -> list[dict]:
    """Sweep the CFG weight and report match-rate vs. w — the core CFG curve.

    ``generate_fn(w)`` must return a list of generated Structures produced with
    guidance weight ``w`` and the given target space group. Returns one summary
    row per weight (best-symprec match-rate plus the full per-symprec breakdown).
    """
    rows: list[dict] = []
    for w in guidance_weights:
        structures = list(generate_fn(w))
        per_symprec = space_group_match_rate(
            structures, target, symprec_list, angle_tolerance
        )
        best_symprec = max(
            per_symprec, key=lambda s: per_symprec[s]["match_rate"]
        )
        rows.append(
            {
                "guidance_scale": float(w),
                "target": int(target),
                "n_structures": len(structures),
                "best_match_rate": per_symprec[best_symprec]["match_rate"],
                "best_symprec": best_symprec,
                "per_symprec": per_symprec,
            }
        )
    return rows


def format_match_report(results: dict[float, dict], target: int) -> str:
    """Human-readable per-symprec table for a single generation run."""
    lines = [f"Space-group match-rate vs. target SG {target}:"]
    lines.append(f"  {'symprec':>8}  {'match%':>7}  {'valid%':>7}  {'n_valid':>8}  {'n_total':>8}")
    for symprec in sorted(results):
        r = results[symprec]
        lines.append(
            f"  {symprec:>8.3f}  {100 * r['match_rate']:>6.1f}%  "
            f"{100 * r['match_rate_valid']:>6.1f}%  {r['n_valid']:>8}  {r['n_total']:>8}"
        )
    return "\n".join(lines)
