#!/usr/bin/env python3
"""Space-group guidance sweep — the CFG "did conditioning work?" payoff loop.

Loads a *conditional* (space-group LoRA) checkpoint once, then for each target
space group and each classifier-free-guidance weight ``w`` generates ``N``
crystals and measures the exact space-group match-rate (pymatgen
``SpacegroupAnalyzer``) against the target after MLIP relaxation. Reports
match-rate vs. ``w`` at a sweep of ``symprec`` tolerances, plus the **lift over
the unconditional base rate** (the fraction of *unconditioned* samples that land
in the target SG).

Protocol defaults follow the crystal-generation literature so the headline
number is directly comparable:
  * targets = the 10 most common MP-20 space groups (SymmCD "10 SGs")
  * headline symprec = 0.1 A (MatterGen / SymmCD standard)
  * guidance grid brackets MatterGen's gamma = 2 setting
MatterGen reference: ~20% target-SG match (~10% for high-symmetry groups).

Outputs to ``--output_dir``: ``sweep.csv`` (tidy, one row per target x w x
symprec), ``summary.json``, and ``match_rate_vs_w.png`` (headline symprec).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Hashable

import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm import tqdm

# Ensure repository root is on PYTHONPATH when run as a script (src/eval/<this>).
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.crystalite import mod1
from src.crystalite.sampler import (
    edm_sampler,
    clamp_lattice_latent as _clamp_lattice_latent,
)
from src.models.lattice_repr import lattice_latent_to_y1
from src.models.type_encoding import build_type_encoding
from src.data.mp20_tokens import MP20Tokens, VZ
from src.eval.spacegroup_match import (
    compute_space_groups,
    structures_from_sample_batch,
    DEFAULT_SYMPREC,
)
from src.eval.stability import load_phase_diagram
from src.eval.uniqueness_novelty import (
    _build_composition_index,
    _filter_by_nary,
    _get_composition_hash,
    _is_finite_structure,
    _structures_match,
)
from src.sample_crystalite_ckpt import (
    _load_checkpoint,
    _resolve_checkpoint_path,
    _build_model_from_ckpt,
    _apply_ema_state_dict,
    _cfg_value,
    _seed_everything,
    _sample_num_atoms,
    _prepare_dataset_context,
    _parse_allowed_elements,
)
from src.utils.sample_stats import (
    compute_e_above_hull_mp2020_like,
    compute_e_above_hull_uncorrected,
    make_chgnet_and_relaxer,
    make_nequip_batch_relaxer,
    make_nequip_relaxer,
)
from src.utils.dataset import dataset_to_structures

# The 10 most common space groups in MP-20 (SymmCD "10 SGs" evaluation set).
DEFAULT_TARGETS = [2, 12, 14, 62, 63, 139, 166, 194, 221, 225]
# Guidance grid: 0 = unconditional base rate; 2 = MatterGen's gamma; >2 = stronger.
DEFAULT_WEIGHTS = [0.0, 1.0, 2.0, 4.0, 8.0]
HEADLINE_SYMPREC = 0.1
MATTERGEN_REFERENCE = 0.20  # ~20% target-SG match; drawn on the plot for context.


def _resolve_sampling_params(model_args: dict) -> dict:
    """Pull EDM/sampling params from the checkpoint's model_args (same as the
    standalone sampler), so the base model's exact sampling recipe is inherited."""
    g = lambda k, d: _cfg_value(None, model_args, k, d)  # noqa: E731
    return {
        "lattice_repr": str(g("lattice_repr", "y1")),
        "sigma_min": float(g("sigma_min", 0.002)),
        "sigma_max": float(g("sigma_max", 80.0)),
        "rho": float(g("rho", 7.0)),
        "S_churn": float(g("S_churn", 20.0)),
        "S_min": float(g("S_min", 0.0)),
        "S_max": float(g("S_max", 999.0)),
        "S_noise": float(g("S_noise", 1.0)),
        "sigma_data_type": float(g("sigma_data_type", 1.0)),
        "sigma_data_coord": float(g("sigma_data_coord", 0.25)),
        "sigma_data_lattice": float(g("sigma_data_lattice", 1.0)),
        "aa_frac_max_scale": float(g("aa_frac_max_scale", 0.0)),
        "aa_rho_types": float(g("aa_rho_types", 0.0)),
        "aa_rho_coords": float(g("aa_rho_coords", 0.0)),
        "aa_rho_lattice": float(g("aa_rho_lattice", 0.0)),
    }


def _decode_to_batch(samples, pad_mask, type_encoding, allowed_mask, lattice_repr):
    """Decode raw sampler outputs into a stacked A0/F1/Y1/pad_mask batch dict.

    Mirrors the decode in src/sample_crystalite_ckpt.py so structures match what
    the standalone sampler would produce.
    """
    pm = pad_mask.to("cpu")
    real = ~pm
    a0 = type_encoding.decode_logits_to_A0(
        type_logits=samples["type"].detach().cpu(),
        pad_mask=pm,
        allowed_mask=allowed_mask,
    )
    a0 = torch.where(real, a0, torch.zeros_like(a0))
    f1 = mod1(samples["frac"].detach().cpu() + 0.5).clamp(0.0, 1.0)
    f1 = torch.where(real[..., None], f1, torch.zeros_like(f1))
    lat = _clamp_lattice_latent(samples["lat"].detach().cpu(), lattice_repr=lattice_repr)
    y1 = lattice_latent_to_y1(lat, lattice_repr=lattice_repr)
    y1 = _clamp_lattice_latent(y1, lattice_repr="y1")
    return {"A0": a0, "F1": f1, "Y1": y1, "pad_mask": pm}


def _generate_pool(
    *,
    model,
    n,
    nmax,
    target_sg,
    guidance_scale,
    strategy,
    count_probs,
    type_encoding,
    allowed_mask,
    sampling,
    num_steps,
    device,
    autocast_dtype,
    seed,
    chunk_size,
):
    """Generate ``n`` structures at a fixed (target SG, guidance weight)."""
    gen = torch.Generator(device=device).manual_seed(int(seed))
    arange = torch.arange(nmax, device=device)[None, :]
    batches = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        num_atoms = _sample_num_atoms(
            bsz=end - start,
            nmax=nmax,
            strategy=strategy,
            fixed_num_atoms=None,
            count_probs=count_probs,
            device=device,
            generator=gen,
        )
        pad = arange >= num_atoms[:, None]
        samples = edm_sampler(
            model=model,
            pad_mask=pad,
            type_dim=type_encoding.type_dim,
            num_steps=num_steps,
            sigma_min=sampling["sigma_min"],
            sigma_max=sampling["sigma_max"],
            rho=sampling["rho"],
            S_churn=sampling["S_churn"],
            S_min=sampling["S_min"],
            S_max=sampling["S_max"],
            S_noise=sampling["S_noise"],
            sigma_data_type=sampling["sigma_data_type"],
            sigma_data_coord=sampling["sigma_data_coord"],
            sigma_data_lat=sampling["sigma_data_lattice"],
            generator=gen,
            autocast_dtype=autocast_dtype,
            fixed_atom_types=None,
            skip_type_scaling=False,
            aa_frac_max_scale=sampling["aa_frac_max_scale"],
            aa_rho_types=sampling["aa_rho_types"],
            aa_rho_coords=sampling["aa_rho_coords"],
            aa_rho_lattice=sampling["aa_rho_lattice"],
            lattice_repr=sampling["lattice_repr"],
            target_spacegroup=int(target_sg),
            guidance_scale=float(guidance_scale),
        )
        batches.append(
            _decode_to_batch(
                samples, pad, type_encoding, allowed_mask, sampling["lattice_repr"]
            )
        )
    return {k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]}


def _match_stats(computed_by_symprec, target, denominator, symprecs):
    """Per-symprec match stats for one target from pre-computed SG lists."""
    out = {}
    for sp in symprecs:
        comp = computed_by_symprec[sp]
        valid = [c for c in comp if c is not None]
        n_match = sum(1 for c in valid if c == target)
        denom = int(denominator)
        out[sp] = {
            "match_rate": (n_match / denom) if denom else 0.0,
            "match_rate_valid": (n_match / len(valid)) if valid else 0.0,
            "n_match": n_match,
            "n_valid": len(valid),
            "n_total": denom,
        }
    return out


def _build_relaxer(args):
    mlip = str(args.relax_mlip).strip().lower()
    if mlip == "none":
        return None, {
            "mlip": "none",
            "steps": 0,
            "mode": "none",
        }

    if mlip == "chgnet":
        _, relaxer, device = make_chgnet_and_relaxer(args.relax_device)
        return relaxer, {
            "mlip": "chgnet",
            "device": device,
            "steps": int(args.relax_steps),
            "batch_size": int(args.relax_batch_size),
            "mode": "sequential",
        }

    if mlip == "nequip":
        mode = str(args.nequip_relax_mode).strip().lower()
        if mode == "batch":
            _, relaxer, device, model_path = make_nequip_batch_relaxer(
                compile_path=args.nequip_compile_path,
                stability_device=args.relax_device,
                optimizer_name=args.nequip_optimizer,
                cell_filter=args.nequip_cell_filter,
                max_force_abort=args.nequip_max_force_abort,
            )
        else:
            _, relaxer, device, model_path = make_nequip_relaxer(
                compile_path=args.nequip_compile_path,
                stability_device=args.relax_device,
                optimizer_name=args.nequip_optimizer,
                cell_filter=args.nequip_cell_filter,
                fmax=args.nequip_fmax,
                max_force_abort=args.nequip_max_force_abort,
            )
        return relaxer, {
            "mlip": "nequip",
            "device": device,
            "steps": int(args.relax_steps),
            "batch_size": int(args.relax_batch_size),
            "mode": mode,
            "compile_path": str(args.nequip_compile_path),
            "resolved_model": str(model_path),
            "optimizer": str(args.nequip_optimizer),
            "cell_filter": str(args.nequip_cell_filter),
            "fmax": float(args.nequip_fmax),
            "max_force_abort": float(args.nequip_max_force_abort),
        }

    raise ValueError(f"Unsupported relaxation backend: {args.relax_mlip!r}")


def _relax_structures(structures, relaxer, args, *, label: str):
    if relaxer is None:
        return list(structures), [None] * len(structures), 0

    all_structures = list(structures)
    if not all_structures:
        return [], [], 0

    steps = int(args.relax_steps)
    batch_size = max(1, int(args.relax_batch_size))
    relaxed = []
    energies = []
    n_failed = 0
    failure_examples: list[str] = []
    use_batched = hasattr(relaxer, "relax_many") and str(args.nequip_relax_mode) == "batch"

    def record_failure(exc: Exception):
        nonlocal n_failed
        n_failed += 1
        if len(failure_examples) < 3:
            failure_examples.append(f"{exc.__class__.__name__}: {exc}")

    def relax_one(struct):
        try:
            result = relaxer.relax(struct, steps=steps, verbose=False)
            relaxed.append(result["final_structure"])
            energies.append(float(result["trajectory"].energies[-1]))
        except Exception as exc:
            record_failure(exc)

    def relax_many(batch):
        if not batch:
            return
        try:
            results = relaxer.relax_many(batch, steps=steps, verbose=False)
            if len(results) != len(batch):
                raise RuntimeError(
                    "Relaxer returned a mismatched number of structures: "
                    f"{len(results)} for batch of {len(batch)}"
                )
            for result in results:
                relaxed.append(result["final_structure"])
                energies.append(float(result["trajectory"].energies[-1]))
        except Exception as exc:
            if len(batch) <= 1:
                record_failure(exc)
                return
            mid = len(batch) // 2
            relax_many(batch[:mid])
            relax_many(batch[mid:])

    iterator = range(0, len(all_structures), batch_size)
    for start in tqdm(iterator, desc=f"{label}/relax", dynamic_ncols=True):
        batch = all_structures[start : start + batch_size]
        if use_batched:
            relax_many(batch)
        else:
            for struct in batch:
                relax_one(struct)

    msg = (
        f"[sweep] {label}: relaxed {len(relaxed)}/{len(all_structures)} "
        f"structures; relax_failed={n_failed}"
    )
    if failure_examples:
        msg += f"; first_failure={failure_examples[0]}"
    print(msg)
    return relaxed, energies, n_failed


def _load_reference_structures(data_root: str, nmax: int, splits: list[str]) -> list:
    refs = []
    for split in splits:
        ds = MP20Tokens(
            root=data_root,
            augment_translate=False,
            split=split,
            nmax=nmax,
        )
        split_refs = dataset_to_structures(ds)
        refs.extend(split_refs)
        print(f"[sweep] loaded {len(split_refs)} novelty refs from split={split}")
    print(f"[sweep] total novelty refs={len(refs)}")
    return refs


def _compute_un_mask(
    structures,
    reference_structures,
    *,
    matcher: StructureMatcher,
    minimum_nary: int,
    maximum_nary: int | None,
) -> tuple[list[bool], dict[str, int]]:
    finite_positions = [
        i for i, struct in enumerate(structures) if _is_finite_structure(struct)
    ]
    finite_structures = [structures[i] for i in finite_positions]
    raw_gen_items = _filter_by_nary(
        finite_structures,
        minimum_nary=minimum_nary,
        maximum_nary=maximum_nary,
    )
    gen_items = [
        (finite_positions[pos], struct, chemsys)
        for pos, struct, chemsys in raw_gen_items
    ]
    mask = [False] * len(structures)
    if not gen_items:
        return mask, {
            "finite": len(finite_positions),
            "nary_kept": 0,
            "unique": 0,
            "novel": 0,
            "unique_and_novel": 0,
            "novel_total": 0,
        }

    finite_refs = [
        struct for struct in reference_structures if _is_finite_structure(struct)
    ]
    train_items = _filter_by_nary(
        finite_refs,
        minimum_nary=minimum_nary,
        maximum_nary=maximum_nary,
    )

    n = len(gen_items)
    uniq_adj: dict[int, list[int]] = defaultdict(list)
    comp_buckets: dict[Hashable, list[int]] = defaultdict(list)
    fallback_indices: list[int] = []
    for i, (_, struct, _) in enumerate(gen_items):
        key = _get_composition_hash(struct, matcher)
        if key is None:
            fallback_indices.append(i)
        else:
            comp_buckets[key].append(i)
    all_buckets = list(comp_buckets.values())
    if fallback_indices:
        all_buckets.append(fallback_indices)
    for bucket in all_buckets:
        for a in range(len(bucket)):
            i = bucket[a]
            _, struct_i, _ = gen_items[i]
            for b in range(a + 1, len(bucket)):
                j = bucket[b]
                _, struct_j, _ = gen_items[j]
                if _structures_match(struct_i, struct_j, matcher):
                    uniq_adj[i].append(j)
                    uniq_adj[j].append(i)

    dupes: set[int] = set()
    for i in range(n):
        if i not in dupes:
            for j in uniq_adj.get(i, []):
                dupes.add(j)
    is_unique = {i: i not in dupes for i in range(n)}

    is_novel = {i: False for i in range(n)}
    novel_candidates: set[int] = set()
    if train_items:
        gen_chemsys = {chemsys for _, _, chemsys in gen_items}
        train_chemsys = {chemsys for _, _, chemsys in train_items}
        intersection = gen_chemsys.intersection(train_chemsys)
        train_filtered = [
            struct for _, struct, chemsys in train_items if chemsys in intersection
        ]
        comp_index, comp_fallback = _build_composition_index(train_filtered, matcher)

        for i, (_, struct, chemsys) in enumerate(gen_items):
            if chemsys not in intersection:
                is_novel[i] = True
                novel_candidates.add(i)
                continue
            novel_candidates.add(i)
            key = _get_composition_hash(struct, matcher)
            if key is None:
                candidates = train_filtered
            else:
                candidates = comp_index.get(key, []) + comp_fallback
                if not candidates:
                    is_novel[i] = True
                    continue
            is_novel[i] = not any(
                _structures_match(struct, other, matcher) for other in candidates
            )

    novel_dupes: set[int] = set()
    is_un = {}
    for i, (original_idx, _, _) in enumerate(gen_items):
        if not is_novel[i] or i in novel_dupes:
            is_un[i] = False
        else:
            is_un[i] = True
            mask[original_idx] = True
            for j in uniq_adj.get(i, []):
                novel_dupes.add(j)

    return mask, {
        "finite": len(finite_positions),
        "nary_kept": n,
        "unique": int(sum(is_unique.values())),
        "novel": int(sum(is_novel.values())),
        "unique_and_novel": int(sum(is_un.values())),
        "novel_total": len(novel_candidates),
    }


def _compute_e_above_hull_values(
    structures,
    energies,
    *,
    ppd,
    ehull_method: str,
    mp2020_compat: Any | None,
) -> tuple[list[float | None], int]:
    values: list[float | None] = []
    failed = 0
    for i, (struct, energy) in enumerate(zip(structures, energies, strict=True)):
        if energy is None or not math.isfinite(float(energy)):
            values.append(None)
            failed += 1
            continue
        if ehull_method == "mp2020_like":
            value, reason = compute_e_above_hull_mp2020_like(
                ppd,
                struct,
                float(energy),
                mp2020_compat=mp2020_compat,
                entry_id=f"sg_sweep_{i}",
            )
        else:
            value, reason = compute_e_above_hull_uncorrected(
                ppd,
                struct,
                float(energy),
            )
        if reason is not None or value is None:
            values.append(None)
            failed += 1
        else:
            values.append(float(value))
    return values, failed


def _prepare_eval_structures(
    sample_batch,
    relaxer,
    args,
    *,
    reference_structures,
    ppd,
    mp2020_compat,
    matcher: StructureMatcher,
    label: str,
):
    raw_structs, n_decode_failed = structures_from_sample_batch(sample_batch)
    subset = str(args.eval_subset).strip().lower()
    if subset == "all":
        relaxed, energies, n_relax_failed = _relax_structures(
            raw_structs, relaxer, args, label=label
        )
        del energies
        return relaxed, {
            "n_decode_failed": n_decode_failed,
            "n_relax_failed": n_relax_failed,
            "n_failed": n_decode_failed + n_relax_failed,
        }, {
            "eval_subset": "all",
            "n_subset": int(args.num_samples),
            "n_un": 0,
            "n_metastable": 0,
            "n_msun": 0,
            "n_ehull_failed": 0,
        }

    if subset != "msun":
        raise ValueError(f"Unsupported eval subset: {args.eval_subset!r}")
    if ppd is None:
        raise ValueError("--eval_subset msun requires --thermo_ppd_mp.")

    un_mask, un_summary = _compute_un_mask(
        raw_structs,
        reference_structures,
        matcher=matcher,
        minimum_nary=int(args.msun_minimum_nary),
        maximum_nary=args.msun_maximum_nary,
    )
    un_structs = [
        struct for struct, keep in zip(raw_structs, un_mask, strict=True) if keep
    ]
    relaxed, energies, n_relax_failed = _relax_structures(
        un_structs, relaxer, args, label=f"{label}_UN"
    )
    e_values, n_ehull_failed = _compute_e_above_hull_values(
        relaxed,
        energies,
        ppd=ppd,
        ehull_method=str(args.thermo_ehull_method),
        mp2020_compat=mp2020_compat,
    )
    metastable_mask = [
        value is not None and value <= float(args.msun_max_e_above_hull)
        for value in e_values
    ]
    selected = [
        struct
        for struct, keep in zip(relaxed, metastable_mask, strict=True)
        if keep
    ]
    print(
        f"[sweep] {label}: MSUN subset {len(selected)}/{len(raw_structs)} "
        f"(UN={un_summary['unique_and_novel']}, "
        f"metastable={sum(metastable_mask)}, ehull_failed={n_ehull_failed})"
    )
    return selected, {
        "n_decode_failed": n_decode_failed,
        "n_relax_failed": n_relax_failed,
        "n_failed": n_decode_failed + n_relax_failed,
    }, {
        "eval_subset": "msun",
        "n_subset": len(selected),
        "n_un": int(un_summary["unique_and_novel"]),
        "n_metastable": int(sum(metastable_mask)),
        "n_msun": len(selected),
        "n_ehull_failed": n_ehull_failed,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Conditional (space-group) checkpoint to evaluate.")
    p.add_argument("--train_output_dir", type=str, default="")
    p.add_argument("--checkpoint_preference", type=str, default="best")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--sample_mode", type=str, default="ema", choices=["ema", "regular"])
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--num_samples", type=int, default=1000,
                   help="Structures generated per (target SG, guidance weight).")
    p.add_argument("--sample_num_steps", type=int, default=None)
    p.add_argument("--sample_chunk_size", type=int, default=256)
    p.add_argument("--sample_seed", type=int, default=1234)
    p.add_argument("--nmax", type=int, default=None)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--atom_count_strategy", type=str, default="empirical",
                   choices=["empirical", "max"])
    p.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS,
                   help="Target space-group numbers (1-230).")
    p.add_argument("--guidance_scales", type=float, nargs="+", default=DEFAULT_WEIGHTS)
    p.add_argument("--symprecs", type=float, nargs="+", default=list(DEFAULT_SYMPREC))
    p.add_argument("--headline_symprec", type=float, default=HEADLINE_SYMPREC)
    p.add_argument("--relax_mlip", type=str, default="nequip",
                   choices=["none", "chgnet", "nequip"],
                   help="Relax generated structures before SG analysis.")
    p.add_argument("--relax_steps", type=int, default=200)
    p.add_argument("--relax_batch_size", type=int, default=64)
    p.add_argument("--relax_device", type=str, default="cuda")
    p.add_argument("--nequip_compile_path", type=str,
                   default="data/mlip/nequip/*.nequip.pt2")
    p.add_argument("--nequip_relax_mode", type=str, default="batch",
                   choices=["sequential", "batch"])
    p.add_argument("--nequip_optimizer", type=str, default="FIRE",
                   choices=["FIRE", "LBFGS", "BFGS", "BFGSLineSearch", "LBFGSLineSearch"])
    p.add_argument("--nequip_cell_filter", type=str, default="frechet",
                   choices=["none", "frechet", "exp"])
    p.add_argument("--nequip_fmax", type=float, default=0.005)
    p.add_argument("--nequip_max_force_abort", type=float, default=1000000.0)
    p.add_argument("--eval_subset", type=str, default="msun",
                   choices=["all", "msun"],
                   help="Which generated structures count in the SG denominator.")
    p.add_argument("--msun_reference_splits", nargs="+", default=["train"])
    p.add_argument("--msun_minimum_nary", type=int, default=1)
    p.add_argument("--msun_maximum_nary", type=int, default=None)
    p.add_argument("--msun_max_e_above_hull", type=float, default=0.1)
    p.add_argument("--thermo_ppd_mp", type=str,
                   default="data/mp20/hull/2023-02-07-ppd-mp.pkl")
    p.add_argument("--thermo_ehull_method", type=str, default="mp2020_like",
                   choices=["uncorrected", "mp2020_like"])
    p.add_argument("--matcher_stol", type=float, default=0.5)
    p.add_argument("--matcher_ltol", type=float, default=0.3)
    p.add_argument("--matcher_angle_tol", type=float, default=10.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ckpt_path = _resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        train_output_dir=args.train_output_dir,
        preference=args.checkpoint_preference,
    )
    ckpt = _load_checkpoint(ckpt_path)

    use_cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")
    model, model_args = _build_model_from_ckpt(ckpt=ckpt, device=device)

    if getattr(model, "prop_encoder", None) is None:
        raise SystemExit(
            f"Checkpoint {ckpt_path} is UNCONDITIONAL (no prop_encoder). "
            "This sweep requires a checkpoint finetuned with --cond_prop spacegroup."
        )

    if args.sample_mode == "ema":
        ema_state = ckpt.get("ema_state_dict")
        if ema_state is not None:
            _apply_ema_state_dict(model, ema_state)
            print("[sweep] using EMA weights")
        else:
            print("[sweep] EMA weights not found; using regular weights")
    model.eval()

    nmax = int(_cfg_value(args.nmax, model_args, "nmax", 20))
    dataset_name = str(_cfg_value(args.dataset_name, model_args, "dataset_name", "mp20"))
    data_root = str(_cfg_value(args.data_root, model_args, "data_root", ""))
    num_steps = int(_cfg_value(args.sample_num_steps, model_args, "sample_num_steps", 100))
    sampling = _resolve_sampling_params(model_args)
    autocast_dtype = torch.bfloat16 if args.bf16 else None

    _seed_everything(args.sample_seed)
    type_encoding_name = str(
        ckpt.get("type_encoding", model_args.get("type_encoding", "atomic_number"))
    )
    type_encoding = build_type_encoding(type_encoding_name, vz=VZ)

    # Empirical atom-count distribution + element whitelist from the dataset,
    # matching how the base DNG model samples. Falls back to max-count if absent.
    count_probs = None
    allowed_mask = None
    strategy = args.atom_count_strategy
    has_dataset = bool(data_root) and Path(data_root).exists()
    if strategy == "empirical":
        if not has_dataset:
            print(f"[sweep] no dataset at {data_root!r}; falling back to max atom count")
            strategy = "max"
        else:
            count_probs, allowed_mask = _prepare_dataset_context(
                data_root=data_root,
                dataset_name=dataset_name,
                nmax=nmax,
                want_empirical_counts=True,
                want_allowed_mask=True,
            )
            print(f"[sweep] empirical atom counts + element mask from {data_root}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symprecs = sorted(args.symprecs)
    weights = sorted(args.guidance_scales)
    relaxer, relaxation_meta = _build_relaxer(args)
    print(f"[sweep] relaxation={json.dumps(relaxation_meta, sort_keys=True)}")
    if str(args.eval_subset).strip().lower() == "msun" and relaxer is None:
        raise ValueError("--eval_subset msun requires a relaxation backend.")
    matcher = StructureMatcher(
        stol=float(args.matcher_stol),
        ltol=float(args.matcher_ltol),
        angle_tol=float(args.matcher_angle_tol),
    )
    reference_structures = []
    ppd = None
    mp2020_compat = None
    if str(args.eval_subset).strip().lower() == "msun":
        reference_structures = _load_reference_structures(
            data_root=data_root,
            nmax=nmax,
            splits=list(args.msun_reference_splits),
        )
        ppd_path = Path(args.thermo_ppd_mp)
        if not ppd_path.exists():
            raise FileNotFoundError(f"MSUN phase diagram not found: {ppd_path}")
        ppd = load_phase_diagram(ppd_path)
        if str(args.thermo_ehull_method) == "mp2020_like":
            from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

            mp2020_compat = MaterialsProject2020Compatibility(check_potcar=False)
        print(
            "[sweep] eval_subset=msun "
            f"refs={len(reference_structures)} ppd={ppd_path} "
            f"ehull_method={args.thermo_ehull_method} "
            f"max_e_above_hull={args.msun_max_e_above_hull}"
        )
    else:
        print("[sweep] eval_subset=all")

    def gen(target_sg, w, seed):
        return _generate_pool(
            model=model, n=args.num_samples, nmax=nmax, target_sg=target_sg,
            guidance_scale=w, strategy=strategy, count_probs=count_probs,
            type_encoding=type_encoding, allowed_mask=allowed_mask, sampling=sampling,
            num_steps=num_steps, device=device, autocast_dtype=autocast_dtype,
            seed=seed, chunk_size=args.sample_chunk_size,
        )

    # Shared unconditional base pool (guidance 0 is target-independent): its SG
    # histogram gives the base rate for every target.
    print(f"[sweep] generating unconditional base pool (n={args.num_samples})")
    base_batch = gen(target_sg=0, w=0.0, seed=args.sample_seed)
    base_eval_structs, base_failures, base_subset = _prepare_eval_structures(
        base_batch,
        relaxer,
        args,
        reference_structures=reference_structures,
        ppd=ppd,
        mp2020_compat=mp2020_compat,
        matcher=matcher,
        label="base",
    )
    base_sgs = {sp: compute_space_groups(base_eval_structs, sp) for sp in symprecs}
    print(
        f"[sweep] base pool: {len(base_eval_structs)} eval structures, "
        f"{base_failures['n_failed']} failed, "
        f"{base_subset['eval_subset']} denominator={base_subset['n_subset']}"
    )

    rows = []
    for target in args.targets:
        base_stats = _match_stats(base_sgs, target, base_subset["n_subset"], symprecs)
        for w in weights:
            if w == 0.0:
                stats, failures, subset_meta = base_stats, base_failures, base_subset
            else:
                seed = args.sample_seed + 1000 * int(target) + int(round(w * 10))
                batch = gen(target_sg=target, w=w, seed=seed)
                eval_structs, failures, subset_meta = _prepare_eval_structures(
                    batch,
                    relaxer,
                    args,
                    reference_structures=reference_structures,
                    ppd=ppd,
                    mp2020_compat=mp2020_compat,
                    matcher=matcher,
                    label=f"SG{target}_w{w:g}",
                )
                sgs = {sp: compute_space_groups(eval_structs, sp) for sp in symprecs}
                stats = _match_stats(sgs, target, subset_meta["n_subset"], symprecs)
            for sp in symprecs:
                base_rate = base_stats[sp]["match_rate"]
                mr = stats[sp]["match_rate"]
                rows.append({
                    "target_sg": target,
                    "guidance_scale": w,
                    "symprec": sp,
                    "n_samples": args.num_samples,
                    "n_failed": failures["n_failed"],
                    "n_decode_failed": failures["n_decode_failed"],
                    "n_relax_failed": failures["n_relax_failed"],
                    "eval_subset": subset_meta["eval_subset"],
                    "n_subset": subset_meta["n_subset"],
                    "n_un": subset_meta["n_un"],
                    "n_metastable": subset_meta["n_metastable"],
                    "n_msun": subset_meta["n_msun"],
                    "n_ehull_failed": subset_meta["n_ehull_failed"],
                    "n_valid": stats[sp]["n_valid"],
                    "n_match": stats[sp]["n_match"],
                    "match_rate": round(mr, 5),
                    "match_rate_valid": round(stats[sp]["match_rate_valid"], 5),
                    "base_rate": round(base_rate, 5),
                    "lift": round(mr / base_rate, 4) if base_rate > 0 else float("inf"),
                })
            hl = stats[args.headline_symprec] if args.headline_symprec in stats else None
            if hl is not None:
                print(f"[sweep] SG {target:>3}  w={w:<4}  "
                      f"match@{args.headline_symprec}={100 * hl['match_rate']:.1f}%  "
                      f"(base {100 * base_stats[args.headline_symprec]['match_rate']:.1f}%)")

    csv_path = out_dir / "sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sweep] wrote {csv_path} ({len(rows)} rows)")

    summary = {
        "checkpoint": str(ckpt_path),
        "targets": args.targets,
        "guidance_scales": weights,
        "symprecs": symprecs,
        "headline_symprec": args.headline_symprec,
        "num_samples": args.num_samples,
        "relaxation": relaxation_meta,
        "eval_subset": args.eval_subset,
        "msun": {
            "reference_splits": list(args.msun_reference_splits),
            "minimum_nary": args.msun_minimum_nary,
            "maximum_nary": args.msun_maximum_nary,
            "max_e_above_hull": args.msun_max_e_above_hull,
            "thermo_ppd_mp": str(args.thermo_ppd_mp),
            "thermo_ehull_method": str(args.thermo_ehull_method),
            "matcher": {
                "stol": float(args.matcher_stol),
                "ltol": float(args.matcher_ltol),
                "angle_tol": float(args.matcher_angle_tol),
            },
        },
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _maybe_plot(rows, args.targets, weights, args.headline_symprec, out_dir, args.eval_subset)
    return 0


def _maybe_plot(rows, targets, weights, headline_symprec, out_dir, eval_subset):
    """Match-rate vs. guidance weight at the headline symprec, one line per target."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"[sweep] skipping plot ({exc})")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for target in targets:
        ys = [
            next(
                (r["match_rate"] for r in rows
                 if r["target_sg"] == target and r["guidance_scale"] == w
                 and r["symprec"] == headline_symprec),
                None,
            )
            for w in weights
        ]
        ax.plot(weights, [100 * y if y is not None else None for y in ys],
                marker="o", label=f"SG {target}")
    ax.axhline(100 * MATTERGEN_REFERENCE, ls="--", color="gray", lw=1,
               label="MatterGen ~20%")
    ax.set_xlabel("guidance weight w")
    subset_label = "MSUN" if str(eval_subset).lower() == "msun" else "all generated"
    ax.set_ylabel(f"{subset_label} SG match-rate (%) @ symprec {headline_symprec}")
    ax.set_title(f"Space-group conditioning ({subset_label}): match-rate vs CFG weight")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "match_rate_vs_w.png"
    fig.savefig(path, dpi=150)
    print(f"[sweep] wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
