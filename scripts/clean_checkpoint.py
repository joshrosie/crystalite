"""Rewrite a checkpoint's ``model_args`` to the current codebase schema.

"Clean" means the saved ``model_args`` contains exactly the keys a checkpoint
trained in *this* codebase would contain — no removed/renamed legacy keys and no
stale deprecated values — while every architecture-, sampler-, dataset- and
anti-annealing-relevant value is preserved byte-for-byte. The model weights
(``model_state_dict`` / ``ema_state_dict``) and the top-level ``step`` /
``type_dim`` / ``ema_decay`` fields are never touched.

The canonical schema is derived authoritatively from the training config:

    args = build_parser().parse_args([]); validate_args(args); vars(args)

so it stays correct as the config evolves (it even includes derived fields such
as ``aa_rho_by_target`` that ``validate_args`` computes).

Transformations applied:
  * drop keys that are no longer part of the schema (e.g. ``attn_type``,
    ``simplicial_*``, ``gem_legacy_mode``, ``un_style``, ``precise_*``,
    ``test_*``, ``sample_every``, ``csp_eval_sources`` ...);
  * carry renamed values forward (``sample_every`` -> ``sample_frequency``,
    ``sample_metrics_count`` -> ``sample_count``);
  * reset deprecated-but-still-present flags to their clean defaults
    (``ema_use_for_sampling``, ``sample_compute_novelty``);
  * fill self-consistent values for newly-required keys (e.g. an absent
    ``metrics_dataset_name`` becomes ``dataset_name``);
  * normalise type-encoding aliases and recompute derived fields via
    ``validate_args``.

Usage:
    python scripts/clean_checkpoint.py --in dng.pt --out dng_clean.pt
    python scripts/clean_checkpoint.py --in dng.pt --inplace   # overwrite (backs up to .bak)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.training.config import build_parser, validate_args  # noqa: E402

# Old key -> current key for pure renames (value is carried over).
RENAMES = {
    "sample_every": "sample_frequency",
    "sample_metrics_count": "sample_count",
}

# Deprecated flags that survive in the parser but should not carry stale values.
DEPRECATED_RESET = {"ema_use_for_sampling", "sample_compute_novelty"}


def canonical_args():
    parser = build_parser()
    args = parser.parse_args([])
    return parser, args


def clean_model_args(old_ma: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (cleaned_model_args, report)."""
    parser, args = canonical_args()
    canonical = set(vars(args).keys())

    # 1) Overlay every still-valid value (skip deprecated flags -> clean default).
    for key in canonical:
        if key in old_ma and key not in DEPRECATED_RESET:
            setattr(args, key, old_ma[key])

    # 2) Carry renamed values forward.
    for old_key, new_key in RENAMES.items():
        if old_key in old_ma and new_key in canonical:
            setattr(args, new_key, old_ma[old_key])

    # 3) Self-consistent fills for metrics dataset pointers.
    if getattr(args, "metrics_dataset_name", None) in (None, ""):
        args.metrics_dataset_name = old_ma.get("dataset_name", args.dataset_name)
    if getattr(args, "metrics_data_root", None) in (None, ""):
        args.metrics_data_root = old_ma.get("data_root", args.data_root)

    # 4) Recompute derived fields + normalise aliases (aa_rho_by_target, etc.).
    validate_args(args, parser)

    cleaned = dict(vars(args))

    # Report against the post-validate key set so derived fields recomputed by
    # validate_args (e.g. aa_rho_by_target) are not mislabelled as dropped/added.
    canonical_final = set(cleaned)
    dropped = sorted(set(old_ma) - canonical_final)
    added = sorted(canonical_final - set(old_ma))
    renamed = sorted(
        f"{o}->{n}" for o, n in RENAMES.items() if o in old_ma and n in canonical
    )
    reset = sorted(
        k for k in DEPRECATED_RESET if k in old_ma and old_ma[k] != cleaned.get(k)
    )
    report = {
        "dropped_legacy_keys": dropped,
        "added_missing_keys": added,
        "renamed": renamed,
        "deprecated_reset": reset,
    }
    return cleaned, report


# Architecture + sampler + anti-annealing keys whose values MUST survive cleaning
# unchanged (used to assert we didn't alter anything inference-relevant).
_INVARIANT_KEYS = (
    "d_model", "n_heads", "n_layers", "coord_n_freqs", "coord_embed_mode",
    "coord_head_mode", "coord_rff_dim", "coord_rff_sigma", "lattice_embed_mode",
    "lattice_rff_dim", "lattice_rff_sigma", "lattice_repr", "dropout",
    "attn_dropout", "use_distance_bias", "use_edge_bias", "edge_bias_n_freqs",
    "edge_bias_hidden_dim", "edge_bias_n_rbf", "edge_bias_rbf_max", "pbc_radius",
    "dist_slope_init", "use_noise_gate", "gem_per_layer", "type_encoding",
    "sigma_min", "sigma_max", "rho", "S_churn", "S_min", "S_max", "S_noise",
    "sigma_data_type", "sigma_data_coord", "sigma_data_lattice", "sample_num_steps",
    "aa_rho_coords", "aa_rho_lattice", "aa_rho_types", "aa_frac_max_scale",
    "bf16", "csp", "dataset_name", "data_root", "nmax",
)


def verify_invariants(old_ma: dict[str, Any], cleaned: dict[str, Any]) -> list[str]:
    problems = []
    for key in _INVARIANT_KEYS:
        if key in old_ma:
            ov, nv = old_ma[key], cleaned.get(key, "<MISSING>")
            if ov != nv:
                problems.append(f"{key}: {ov!r} -> {nv!r}")
        elif key not in cleaned:
            problems.append(f"{key}: absent in both old and cleaned")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean a checkpoint's model_args to the current schema.")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", default=None)
    ap.add_argument("--inplace", action="store_true", help="Overwrite input (backs up to <in>.bak).")
    ap.add_argument("--no_load_check", action="store_true", help="Skip strict model-rebuild check.")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not args.inplace and not args.out_path:
        ap.error("Provide --out or --inplace.")
    out_path = in_path if args.inplace else Path(args.out_path)

    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    old_ma = dict(ckpt.get("model_args", {}))
    cleaned, report = clean_model_args(old_ma)

    print(f"[clean] {in_path}")
    print(f"  dropped legacy keys ({len(report['dropped_legacy_keys'])}): {report['dropped_legacy_keys']}")
    print(f"  added missing keys  ({len(report['added_missing_keys'])}): {report['added_missing_keys']}")
    print(f"  renamed:            {report['renamed']}")
    print(f"  deprecated reset:   {report['deprecated_reset']}")

    problems = verify_invariants(old_ma, cleaned)
    if problems:
        print("[ERROR] inference-relevant values changed; aborting:")
        for p in problems:
            print("   ", p)
        raise SystemExit(1)
    print("  invariant check:    OK (all architecture/sampler values preserved)")

    new_ckpt = dict(ckpt)
    new_ckpt["model_args"] = cleaned
    # Keep top-level type_encoding consistent with the (normalised) model_args one.
    if "type_encoding" in new_ckpt and isinstance(new_ckpt["type_encoding"], str):
        new_ckpt["type_encoding"] = cleaned.get("type_encoding", new_ckpt["type_encoding"])

    if not args.no_load_check:
        from src.eval_crystalite_ckpt import _build_model_from_ckpt

        _build_model_from_ckpt(ckpt=new_ckpt, device=torch.device("cpu"))
        print("  strict load check:  OK (_build_model_from_ckpt strict=True)")

    if args.inplace:
        backup = in_path.with_suffix(in_path.suffix + ".bak")
        if not backup.exists():
            backup.write_bytes(in_path.read_bytes())
            print(f"  backup written:     {backup}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, out_path)
    print(f"  wrote:              {out_path}")


if __name__ == "__main__":
    main()
