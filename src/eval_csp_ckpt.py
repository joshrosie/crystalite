"""Standalone CSP (crystal structure prediction) checkpoint evaluation.

Loads a CSP checkpoint, holds each target composition's atom types fixed, samples
coordinates + lattice, and reports StructureMatcher match-rate / RMSD against the
ground-truth structures of a dataset split. Supports a grid ablation over
coordinate/lattice anti-annealing (``--aa_rho_coords_values`` x
``--aa_rho_lattice_values``) and optional best-of-k metrics.

It reuses the same sampling + matching machinery the training loop uses for its
``csp_val_match_rate`` metric, so standalone numbers line up with train-time
numbers:

* model build + EMA:     ``_build_model_from_ckpt`` / ``_apply_ema_state_dict``
* CSP sampling:          ``_generate_csp_items_for_indices`` (fixed atom types)
* match-rate / RMSD:     ``RecEval`` / ``RecEvalBatch``

The checkpoint's ``model_args`` provide defaults for every sampler setting; CLI
flags override. This reads config the old-name-tolerant way (``_cfg_value``), so
both freshly-trained and cleaned checkpoints work.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.mp20_tokens import MP20Tokens, VZ
from src.eval.csp_eval import RecEval, RecEvalBatch
from src.eval.sample_runtime import (
    SamplingContext,
    _generate_csp_items_for_indices,
    _safe_crystal_from_tokens,
)
from src.eval_crystalite_ckpt import (
    _apply_ema_state_dict,
    _build_model_from_ckpt,
    _cfg_value,
    _load_checkpoint,
    _resolve_checkpoint_path,
    _seed_everything,
)
from src.models.type_encoding import build_type_encoding


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSP checkpoint evaluation (match-rate / RMSD).")

    # Checkpoint resolution.
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--train_output_dir", type=str, default="")
    p.add_argument(
        "--checkpoint_preference",
        type=str,
        default="auto",
        choices=["auto", "best", "final", "step_latest", "epoch_latest"],
    )
    p.add_argument("--best_meta_path", type=str, default="")

    # Dataset / targets.
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--num_samples", type=int, default=0, help="0 = full split.")
    p.add_argument(
        "--target_selection",
        type=str,
        default="sequential",
        choices=["sequential", "random"],
    )

    # Sampling.
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--nmax", type=int, default=None)
    p.add_argument("--sample_seed", type=int, default=None)
    p.add_argument(
        "--seed_mode",
        type=str,
        default="step_offset",
        choices=["step_offset", "fixed"],
        help="step_offset: base_seed = sample_seed + step (train-time behaviour).",
    )
    p.add_argument("--sample_num_steps", type=int, default=None)
    p.add_argument("--sample_chunk_size", type=int, default=256)
    p.add_argument("--sample_mode", type=str, default="ema", choices=["ema", "regular"])
    p.add_argument("--bf16", action="store_true")

    # Sampler schedule overrides.
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--s_churn", type=float, default=None)
    p.add_argument("--s_min", type=float, default=None)
    p.add_argument("--s_max", type=float, default=None)
    p.add_argument("--s_noise", type=float, default=None)

    # Anti-annealing grid (cross product of the two value lists).
    p.add_argument("--aa_rho_coords_values", type=float, nargs="+", default=None)
    p.add_argument("--aa_rho_lattice_values", type=float, nargs="+", default=None)
    p.add_argument("--aa_rho_types", type=float, default=None)
    p.add_argument("--aa_frac_max_scale", type=float, default=None)

    # Best-of-k.
    p.add_argument("--csp_precise_topk_list", type=int, nargs="+", default=None)
    p.add_argument("--csp_precise_topk_samples", type=int, default=0)

    # Output.
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--output_csv", type=str, default="")
    p.add_argument(
        "--save_pt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save sampled structures per grid cell (use --no-save_pt to disable).",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _build_sampler_args(
    cli: argparse.Namespace, model_args: dict[str, Any]
) -> SimpleNamespace:
    """Attribute bag consumed by _generate_csp_items_for_indices.

    aa_rho_coords / aa_rho_lattice are placeholders here; they are overwritten
    per grid cell in main().
    """
    return SimpleNamespace(
        sample_num_steps=int(_cfg_value(cli.sample_num_steps, model_args, "sample_num_steps", 400)),
        sigma_min=float(_cfg_value(None, model_args, "sigma_min", 0.002)),
        sigma_max=float(_cfg_value(None, model_args, "sigma_max", 80.0)),
        rho=float(_cfg_value(cli.rho, model_args, "rho", 7.0)),
        S_churn=float(_cfg_value(cli.s_churn, model_args, "S_churn", 30.0)),
        S_min=float(_cfg_value(cli.s_min, model_args, "S_min", 0.0)),
        S_max=float(_cfg_value(cli.s_max, model_args, "S_max", 999.0)),
        S_noise=float(_cfg_value(cli.s_noise, model_args, "S_noise", 1.003)),
        sigma_data_type=float(_cfg_value(None, model_args, "sigma_data_type", 1.0)),
        sigma_data_coord=float(_cfg_value(None, model_args, "sigma_data_coord", 0.3)),
        sigma_data_lattice=float(_cfg_value(None, model_args, "sigma_data_lattice", 0.3)),
        aa_frac_max_scale=float(_cfg_value(cli.aa_frac_max_scale, model_args, "aa_frac_max_scale", 0.0)),
        aa_rho_types=float(_cfg_value(cli.aa_rho_types, model_args, "aa_rho_types", 0.0)),
        aa_rho_coords=0.0,
        aa_rho_lattice=0.0,
        lattice_repr=str(_cfg_value(None, model_args, "lattice_repr", "ltri")),
        bf16=bool(cli.bf16 or _cfg_value(None, model_args, "bf16", False)),
    )


def _select_target_indices(
    n_total: int, n_request: int, mode: str, seed: int
) -> torch.Tensor:
    n = n_total if n_request <= 0 else min(n_request, n_total)
    if mode == "random":
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.randperm(n_total, generator=gen)[:n]
    return torch.arange(n, dtype=torch.long)


def _resolve_split(data_root: str, requested: str) -> str:
    raw = Path(data_root) / "raw"
    for candidate in (requested, "val", "test", "train", "all"):
        if (raw / f"{candidate}.csv").exists():
            if candidate != requested:
                print(f"[data] split '{requested}' CSV missing; using '{candidate}'.")
            return candidate
    print(f"[data] no split CSVs under {raw}; using '{requested}' as-is.")
    return requested


def main() -> None:
    cli = parse_args()

    ckpt_path = _resolve_checkpoint_path(
        checkpoint=cli.checkpoint,
        train_output_dir=cli.train_output_dir,
        preference=cli.checkpoint_preference,
    )
    ckpt = _load_checkpoint(ckpt_path)
    model_args = dict(ckpt.get("model_args", {}))
    if not bool(model_args.get("csp", False)):
        print("[warn] model_args.csp is not True; assuming a CSP checkpoint anyway.")

    use_cuda = torch.cuda.is_available() and str(cli.device).startswith("cuda")
    device = torch.device(cli.device if use_cuda else "cpu")
    if not use_cuda and str(cli.device).startswith("cuda"):
        print("[device] CUDA unavailable; falling back to CPU.")

    model, model_args = _build_model_from_ckpt(ckpt=ckpt, device=device)
    if cli.sample_mode == "ema":
        ema_state = ckpt.get("ema_state_dict", None)
        if ema_state is not None:
            _apply_ema_state_dict(model, ema_state)
            print("[ckpt] Using EMA weights.")
        else:
            print("[ckpt] EMA state missing; using regular weights.")
    else:
        print("[ckpt] Using regular weights.")

    step = int(ckpt.get("step", model_args.get("step", 0)))
    nmax = int(_cfg_value(cli.nmax, model_args, "nmax", 20))
    sample_seed = int(_cfg_value(cli.sample_seed, model_args, "sample_seed", 123))
    base_seed = sample_seed + step if cli.seed_mode == "step_offset" else sample_seed

    dataset_name = str(
        _cfg_value(
            cli.dataset_name,
            model_args,
            "metrics_dataset_name",
            _cfg_value(None, model_args, "dataset_name", "mp20"),
        )
    )
    data_root = str(_cfg_value(cli.data_root, model_args, "data_root", "data/mp20"))

    type_encoding_name = str(
        ckpt.get("type_encoding", model_args.get("type_encoding", "atomic_number"))
    )
    type_encoding = build_type_encoding(type_encoding_name, vz=VZ)

    split = _resolve_split(data_root, cli.split)
    ds = MP20Tokens(root=data_root, augment_translate=False, split=split, nmax=nmax)
    if len(ds) == 0:
        raise ValueError(f"Empty CSP dataset: root={data_root} split={split}")

    target_indices = _select_target_indices(
        len(ds), int(cli.num_samples), cli.target_selection, base_seed
    )
    total_count = int(target_indices.numel())

    args_ns = _build_sampler_args(cli, model_args)
    ctx = SamplingContext(
        args=args_ns, model=model, ema=None, device=device, nmax=nmax,
        type_encoding=type_encoding, count_probs=None, train_allowed_mask=None,
        train_element_dist=None, ref_stats=None, ref_structs=[],
        enable_evaluator_metrics=False, novelty_ref_structs=None, thermo_logger=None,
        thermo_reference_cache={}, sample_dir=Path("."), ase_view=None,
        wandb_enabled=False,
    )

    # Anti-annealing grid: default to the checkpoint's single value if no grid given.
    coords_grid = cli.aa_rho_coords_values or [
        float(_cfg_value(None, model_args, "aa_rho_coords", 0.0))
    ]
    lattice_grid = cli.aa_rho_lattice_values or [
        float(_cfg_value(None, model_args, "aa_rho_lattice", 0.0))
    ]
    topk_list = sorted(set(cli.csp_precise_topk_list or []))
    k_max = max(topk_list) if (topk_list and cli.csp_precise_topk_samples > 0) else 1

    chunk = max(1, int(cli.sample_chunk_size))
    _seed_everything(sample_seed)

    gt_all = [
        _safe_crystal_from_tokens(ds[int(target_indices[i].item())], i)
        for i in range(total_count)
    ]

    print(
        f"[csp] {dataset_name}/{split} n_eval={total_count}/{len(ds)} step={step} "
        f"base_seed={base_seed} steps={args_ns.sample_num_steps} "
        f"grid_coords={coords_grid} grid_lattice={lattice_grid} "
        f"topk={topk_list if k_max > 1 else 'off'}"
    )

    out_dir = Path(cli.output_dir) if cli.output_dir else (
        Path(cli.output_csv).parent if cli.output_csv else Path("outputs/csp_eval")
    )
    rows: list[dict[str, Any]] = []

    for aa_c in coords_grid:
        for aa_l in lattice_grid:
            args_ns.aa_rho_coords = float(aa_c)
            args_ns.aa_rho_lattice = float(aa_l)
            cell_seed = base_seed + int(round(aa_c * 1009)) + int(round(aa_l * 9176))

            # Candidate 0 over ALL targets (single-candidate match-rate).
            items0 = _generate_csp_items_for_indices(
                target_indices=target_indices, csp_source_ds=ds, ctx=ctx,
                seed=cell_seed, chunk_size=chunk,
            )
            pred0 = [_safe_crystal_from_tokens(it, i) for i, it in enumerate(items0)]
            rec = RecEval(pred0, gt_all)
            m = rec.get_metrics()
            match_rate = float(m["match_rate"])
            mean_rms = float(m["rms_dist"])
            matched_count = int(round(match_rate * total_count))

            row: dict[str, Any] = {
                "ckpt_path": str(ckpt_path),
                "ckpt_step": step,
                "dataset_name": dataset_name,
                "split": split,
                "num_samples": total_count,
                "total_count": total_count,
                "matched_count": matched_count,
                "target_selection": cli.target_selection,
                "seed_mode": cli.seed_mode,
                "base_seed": base_seed,
                "num_steps": args_ns.sample_num_steps,
                "rho": args_ns.rho,
                "S_churn": args_ns.S_churn,
                "S_min": args_ns.S_min,
                "S_max": args_ns.S_max,
                "S_noise": args_ns.S_noise,
                "aa_rho_coords": float(aa_c),
                "aa_rho_lattice": float(aa_l),
                "match_rate": match_rate,
                "rmse": mean_rms,
                "rms_dist": mean_rms,
            }
            print(
                f"  aa_coords={aa_c:g} aa_lattice={aa_l:g} -> "
                f"match_rate={match_rate:.4f} rms={mean_rms:.4f} "
                f"({matched_count}/{total_count})"
            )

            saved_candidate_sets = [items0]

            # Best-of-k over a subset of the targets.
            if k_max > 1:
                n_sub = min(int(cli.csp_precise_topk_samples), total_count)
                sub_idx = target_indices[:n_sub]
                gt_sub = gt_all[:n_sub]
                pred_batches = [[pred0[i] for i in range(n_sub)]]
                for cand in range(1, k_max):
                    items_c = _generate_csp_items_for_indices(
                        target_indices=sub_idx, csp_source_ds=ds, ctx=ctx,
                        seed=cell_seed + cand * 1_000_003, chunk_size=chunk,
                    )
                    pred_batches.append(
                        [_safe_crystal_from_tokens(it, i) for i, it in enumerate(items_c)]
                    )
                    saved_candidate_sets.append(items_c)
                rec_b = RecEvalBatch(pred_batches, gt_sub)
                for k in topk_list:
                    mk = rec_b.get_match_rate_and_rms_for_k(min(k, k_max))
                    row[f"top{k}_match_rate"] = float(mk["match_rate"])
                    row[f"top{k}_rms"] = float(mk["rms_dist"])
                    print(
                        f"    top{k}: match_rate={row[f'top{k}_match_rate']:.4f} "
                        f"rms={row[f'top{k}_rms']:.4f} (over {n_sub} targets)"
                    )

            rows.append(row)

            if cli.save_pt:
                out_dir.mkdir(parents=True, exist_ok=True)
                pt_path = out_dir / (
                    f"{dataset_name}_{split}_aa_c{aa_c:g}_l{aa_l:g}_samples.pt"
                )
                torch.save(
                    {
                        "candidate_sets": saved_candidate_sets,
                        "target_indices": target_indices,
                        "ckpt_path": str(ckpt_path),
                        "ckpt_step": step,
                        "aa_rho_coords": float(aa_c),
                        "aa_rho_lattice": float(aa_l),
                    },
                    pt_path,
                )
                print(f"    saved samples -> {pt_path}")

    best = max(rows, key=lambda r: (r["match_rate"], -r["rms_dist"]))
    print(
        f"\n[best] aa_coords={best['aa_rho_coords']:g} aa_lattice={best['aa_rho_lattice']:g} "
        f"match_rate={best['match_rate']:.4f} rms={best['rms_dist']:.4f}"
    )

    best_meta = None
    if cli.best_meta_path and Path(cli.best_meta_path).exists():
        try:
            best_meta = json.loads(Path(cli.best_meta_path).read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not read best_meta_path: {exc}")

    _write_outputs(
        rows=rows, best=best, best_meta=best_meta, topk_list=topk_list if k_max > 1 else [],
        csv_path=Path(cli.output_csv) if cli.output_csv else (out_dir / f"{dataset_name}_{split}_csp.csv"),
        overwrite=cli.overwrite,
        type_encoding_name=type_encoding_name,
    )


def _write_outputs(
    *,
    rows: list[dict[str, Any]],
    best: dict[str, Any],
    best_meta: dict[str, Any] | None,
    topk_list: list[int],
    csv_path: Path,
    overwrite: bool,
    type_encoding_name: str,
) -> None:
    base_cols = [
        "ckpt_path", "ckpt_step", "dataset_name", "split", "num_samples",
        "total_count", "matched_count", "target_selection", "seed_mode", "base_seed",
        "num_steps", "rho", "S_churn", "S_min", "S_max", "S_noise",
        "aa_rho_coords", "aa_rho_lattice", "match_rate", "rmse", "rms_dist",
    ]
    for k in topk_list:
        base_cols += [f"top{k}_match_rate", f"top{k}_rms"]

    if csv_path.exists() and not overwrite:
        print(f"[out] {csv_path} exists; pass --overwrite to replace. Skipping CSV.")
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=base_cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in base_cols})
        print(f"[out] wrote {csv_path}")

        summary = {
            "task": f"{best['dataset_name']} CSP {best['split']}",
            "type_encoding": type_encoding_name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "best": best,
            "grid_rows": rows,
        }
        if best_meta is not None:
            summary["best_meta"] = best_meta
        json_path = csv_path.with_suffix(".json")
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[out] wrote {json_path}")


if __name__ == "__main__":
    main()
