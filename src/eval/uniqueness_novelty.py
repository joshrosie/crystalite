"""Uniqueness and novelty metrics for crystal structure generation evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Iterable

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from tqdm import tqdm

def _get_chemsys(structure: Structure) -> tuple[str, ...]:
    """Return sorted tuple of element symbols."""
    elements = [el.name for el in structure.composition.elements]
    return tuple(sorted(elements))


def _filter_by_nary(
    structures: Iterable[Structure],
    *,
    minimum_nary: int = 2,
    maximum_nary: int | None = None,
) -> list[tuple[int, Structure, tuple[str, ...]]]:
    """Filter structures by chemsys size (FlowMM minimum_nary/maximum_nary)."""
    kept: list[tuple[int, Structure, tuple[str, ...]]] = []
    for idx, struct in enumerate(structures):
        chemsys = _get_chemsys(struct)
        nary = len(chemsys)
        if nary < minimum_nary:
            continue
        if maximum_nary is not None and nary > maximum_nary:
            continue
        kept.append((idx, struct, chemsys))
    return kept


def _structures_match(
    struct: Structure, other: Structure, matcher: StructureMatcher
) -> bool:
    """Return True if matcher considers the two structures a match."""
    try:
        rms = matcher.get_rms_dist(struct, other)
    except Exception:
        return False
    return rms is not None


def _get_composition_hash(
    struct: Structure, matcher: StructureMatcher
) -> Hashable | None:
    """Get composition hash for fast bucketing."""
    if getattr(matcher, "allow_subset", False):
        return None

    comparator = getattr(matcher, "comparator", None) or getattr(
        matcher, "_comparator", None
    )
    if comparator is None or not hasattr(comparator, "get_hash"):
        return None

    ignored = (
        getattr(matcher, "ignored_species", None)
        or getattr(matcher, "_ignored_species", None)
        or ()
    )
    try:
        if ignored:
            struct_copy = struct.copy()
            struct_copy.remove_species(ignored)
            composition = struct_copy.composition
        else:
            composition = struct.composition

        comp_hash = comparator.get_hash(composition)
        hash(comp_hash)
        return comp_hash
    except Exception:
        return None


def _build_composition_index(
    structures: list[Structure], matcher: StructureMatcher
) -> tuple[dict[Hashable, list[Structure]], list[Structure]]:
    """Build composition-based index for fast novelty lookups."""
    index: dict[Hashable, list[Structure]] = defaultdict(list)
    fallback: list[Structure] = []

    for struct in structures:
        key = _get_composition_hash(struct, matcher)
        if key is None:
            fallback.append(struct)
        else:
            index[key].append(struct)

    return dict(index), fallback


def _is_finite_structure(struct: Structure) -> bool:
    """Best-effort filter for invalid structures."""
    try:
        mat = struct.lattice.matrix
        if not np.isfinite(mat).all():
            return False
        if not np.isfinite(struct.frac_coords).all():
            return False
        if struct.volume <= 1e-6:
            return False
    except Exception:
        return False
    return True


def compute_uniqueness_novelty(
    structures: list[Structure],
    train_structures: list[Structure],
    *,
    minimum_nary: int = 2,
    maximum_nary: int | None = None,
    matcher: StructureMatcher | None = None,
    use_fast_novelty: bool = True,
) -> dict:
    """Compute per-sample uniqueness/novelty/UN metrics in memory.

    Adapted from FlowMM:
    - flowmm/scripts_analysis/novelty.py
    - flowmm/src/flowmm/pymatgen_.py
    - flowmm/src/flowmm/pandas_.py

    Note: FlowMM's novelty.py passes ``minimum_nary=args.minimum_nary - 1`` (default 1)
    to ``filter_prerelaxed``, which keeps structures with ``len(chemsys) > minimum_nary``,
    i.e. 2+ elements. Our ``_filter_by_nary`` uses ``nary < minimum_nary`` (keeps
    ``nary >= minimum_nary``), so the equivalent default here is ``minimum_nary=2``.
    """
    matcher = matcher or StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    finite_structs = [s for s in structures if _is_finite_structure(s)]
    gen_items = _filter_by_nary(
        finite_structs,
        minimum_nary=minimum_nary,
        maximum_nary=maximum_nary,
    )
    if not gen_items:
        return {
            "unique_rate": 0.0,
            "novel_rate": 0.0,
            "un_rate": 0.0,
            "counts": {"unique": 0, "novel": 0, "unique_and_novel": 0, "total": 0},
            "is_unique": [],
            "is_novel": [],
            "is_un": [],
            "un_structs": [],
        }

    finite_train_structs = [s for s in train_structures if _is_finite_structure(s)]
    train_items = _filter_by_nary(
        finite_train_structs,
        minimum_nary=minimum_nary,
        maximum_nary=maximum_nary,
    )

    n = len(gen_items)
    uniq_adj: dict[int, list[int]] = defaultdict(list)
    if use_fast_novelty:
        comp_buckets: dict[Hashable, list[int]] = defaultdict(list)
        fallback_indices: list[int] = []
        for i, (_, struct, _) in enumerate(gen_items):
            key = _get_composition_hash(struct, matcher)
            if key is None:
                fallback_indices.append(i)
            else:
                comp_buckets[key].append(i)
        all_buckets = list(comp_buckets.values()) + (
            [fallback_indices] if fallback_indices else []
        )
        for bucket in tqdm(
            all_buckets, desc="    Uniqueness (FlowMM)", disable=len(all_buckets) < 10
        ):
            for a in range(len(bucket)):
                i = bucket[a]
                _, struct_i, _ = gen_items[i]
                for b in range(a + 1, len(bucket)):
                    j = bucket[b]
                    _, struct_j, _ = gen_items[j]
                    if _structures_match(struct_i, struct_j, matcher):
                        uniq_adj[i].append(j)
                        uniq_adj[j].append(i)
    else:
        for i in tqdm(range(n), desc="    Uniqueness (FlowMM)", disable=n < 100):
            _, struct_i, _ = gen_items[i]
            for j in range(i + 1, n):
                _, struct_j, _ = gen_items[j]
                if _structures_match(struct_i, struct_j, matcher):
                    uniq_adj[i].append(j)
                    uniq_adj[j].append(i)

    dupes: set[int] = set()
    for i in range(n):
        if i not in dupes:
            for j in uniq_adj.get(i, []):
                dupes.add(j)
    is_unique: dict[int, bool] = {gen_items[i][0]: i not in dupes for i in range(n)}

    is_novel: dict[int, bool] = {idx: False for idx, _, _ in gen_items}
    novel_candidates: set[int] = set()
    if train_items:
        gen_chemsys = {chemsys for _, _, chemsys in gen_items}
        train_chemsys = {chemsys for _, _, chemsys in train_items}
        intersection = gen_chemsys.intersection(train_chemsys)
        train_filtered = [
            struct for _, struct, chemsys in train_items if chemsys in intersection
        ]

        comp_index: dict[Hashable, list[Structure]] = {}
        comp_fallback: list[Structure] = []
        if use_fast_novelty and train_filtered:
            comp_index, comp_fallback = _build_composition_index(
                train_filtered, matcher
            )

        for idx, struct, chemsys in tqdm(
            gen_items, desc="    Novelty (FlowMM)", disable=len(gen_items) < 100
        ):
            if chemsys not in intersection:
                is_novel[idx] = True
                novel_candidates.add(idx)
                continue
            novel_candidates.add(idx)

            if use_fast_novelty and comp_index:
                key = _get_composition_hash(struct, matcher)
                if key is None:
                    candidates = train_filtered
                else:
                    candidates = comp_index.get(key, []) + comp_fallback
                    if not candidates:
                        is_novel[idx] = True
                        continue
            else:
                candidates = train_filtered

            match_found = False
            for other in candidates:
                if _structures_match(struct, other, matcher):
                    match_found = True
                    break
            is_novel[idx] = not match_found

    ordered_indices = [idx for idx, _, _ in gen_items]
    is_unique_list = [bool(is_unique[idx]) for idx in ordered_indices]
    is_novel_list = [bool(is_novel[idx]) for idx in ordered_indices]

    novel_dupes: set[int] = set()
    is_un_list: list[bool] = []
    for i in range(n):
        idx = gen_items[i][0]
        if not is_novel[idx] or i in novel_dupes:
            is_un_list.append(False)
        else:
            is_un_list.append(True)
            for j in uniq_adj.get(i, []):
                novel_dupes.add(j)

    unique_count = int(sum(is_unique_list))
    novel_count = int(sum(is_novel_list))
    un_count = int(sum(is_un_list))
    total = len(ordered_indices)
    novel_total = len(novel_candidates)

    un_structs = [
        struct for (idx, struct, _), is_un in zip(gen_items, is_un_list) if is_un
    ]

    return {
        "unique_rate": unique_count / total if total else 0.0,
        "novel_rate": novel_count / novel_total if novel_total else 0.0,
        "un_rate": un_count / novel_total if novel_total else 0.0,
        "counts": {
            "unique": unique_count,
            "novel": novel_count,
            "unique_and_novel": un_count,
            "total": total,
            "novel_total": novel_total,
        },
        "is_unique": is_unique_list,
        "is_novel": is_novel_list,
        "is_un": is_un_list,
        "un_structs": un_structs,
    }


