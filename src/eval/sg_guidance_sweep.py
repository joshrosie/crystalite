#!/usr/bin/env python3
"""Space-group guidance sweep — the CFG "did conditioning work?" payoff loop.

Loads a *conditional* (space-group LoRA) checkpoint once, then for each target
space group and each classifier-free-guidance weight ``w`` generates ``N``
crystals and measures the exact space-group match-rate (pymatgen
``SpacegroupAnalyzer``) against the target. Reports match-rate vs. ``w`` at a
sweep of ``symprec`` tolerances, plus the **lift over the unconditional base
rate** (the fraction of *unconditioned* samples that land in the target SG).

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
import sys
from pathlib import Path

import torch

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
from src.data.mp20_tokens import VZ
from src.eval.spacegroup_match import (
    compute_space_groups,
    structures_from_sample_batch,
    DEFAULT_SYMPREC,
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


def _match_stats(computed_by_symprec, target, n, symprecs):
    """Per-symprec match stats for one target from pre-computed SG lists."""
    out = {}
    for sp in symprecs:
        comp = computed_by_symprec[sp]
        valid = [c for c in comp if c is not None]
        n_match = sum(1 for c in valid if c == target)
        out[sp] = {
            "match_rate": (n_match / n) if n else 0.0,
            "match_rate_valid": (n_match / len(valid)) if valid else 0.0,
            "n_match": n_match,
            "n_valid": len(valid),
            "n_total": n,
        }
    return out


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
    base_structs, base_failed = structures_from_sample_batch(base_batch)
    base_sgs = {sp: compute_space_groups(base_structs, sp) for sp in symprecs}
    print(f"[sweep] base pool: {len(base_structs)} valid, {base_failed} failed")

    rows = []
    for target in args.targets:
        base_stats = _match_stats(base_sgs, target, args.num_samples, symprecs)
        for w in weights:
            if w == 0.0:
                stats, n_failed = base_stats, base_failed
            else:
                seed = args.sample_seed + 1000 * int(target) + int(round(w * 10))
                batch = gen(target_sg=target, w=w, seed=seed)
                structs, n_failed = structures_from_sample_batch(batch)
                sgs = {sp: compute_space_groups(structs, sp) for sp in symprecs}
                stats = _match_stats(sgs, target, args.num_samples, symprecs)
            for sp in symprecs:
                base_rate = base_stats[sp]["match_rate"]
                mr = stats[sp]["match_rate"]
                rows.append({
                    "target_sg": target,
                    "guidance_scale": w,
                    "symprec": sp,
                    "n_samples": args.num_samples,
                    "n_failed": n_failed,
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
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _maybe_plot(rows, args.targets, weights, args.headline_symprec, out_dir)
    return 0


def _maybe_plot(rows, targets, weights, headline_symprec, out_dir):
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
    ax.set_ylabel(f"space-group match-rate (%) @ symprec {headline_symprec}")
    ax.set_title("Space-group conditioning: match-rate vs CFG weight")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "match_rate_vs_w.png"
    fig.savefig(path, dpi=150)
    print(f"[sweep] wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
