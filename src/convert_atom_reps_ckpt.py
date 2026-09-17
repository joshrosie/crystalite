from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

# Ensure repository root is on PYTHONPATH when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.crystalite import CrystaliteModel
from src.data.mp20_tokens import VZ
from src.models.type_encoding_state import (
    migrate_type_encoding, resolve_type_encoding, save_migrated_checkpoint,
    type_encoding_metadata,
)


SUPPORTED_CRYSTALITE_TYPE_ENCODINGS = {
    "atomic_number",
    "subatomic_tokenizer_raw",
    "subatomic_tokenizer_pca",
    "subatomic_tokenizer_pca_8",
    "subatomic_tokenizer_pca_16",
    "subatomic_tokenizer_pca_24",
}
SUPPORTED_LEGACY_PCA_DIMS = {"8", "16", "24"}
UNSUPPORTED_LEGACY_TYPE_ENCODINGS = {
    "table": "periodic_table_2d",
    "periodic_table_2d": "periodic_table_2d",
    "ec": "electron_config",
    "electron_config": "electron_config",
    "chemical_pca": "chem_pca",
    "chem_pca": "chem_pca",
}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected checkpoint dict at {path}, got {type(ckpt).__name__}.")
    return ckpt


def normalize_atom_reps_type_encoding_name(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    if not mode_norm:
        raise ValueError("Checkpoint type_encoding is empty.")

    if mode_norm in SUPPORTED_CRYSTALITE_TYPE_ENCODINGS:
        return mode_norm

    if mode_norm == "chem_raw_v2":
        return "subatomic_tokenizer_raw"

    if mode_norm == "chem_pca_v2":
        return "subatomic_tokenizer_pca_24"

    if mode_norm.startswith("chem_pca_v2_"):
        suffix = mode_norm.rsplit("_", 1)[-1]
        if suffix not in SUPPORTED_LEGACY_PCA_DIMS:
            raise ValueError(
                f"Unsupported atom-reps type encoding '{mode}'. Crystalite only supports "
                "chem_pca_v2 PCA dimensions 8, 16, and 24."
            )
        return f"subatomic_tokenizer_pca_{suffix}"

    unsupported_name = UNSUPPORTED_LEGACY_TYPE_ENCODINGS.get(mode_norm)
    if unsupported_name is not None:
        raise ValueError(
            f"Unsupported atom-reps type encoding '{unsupported_name}'. "
            "This converter only supports atomic_number, chem_raw_v2, and chem_pca_v2 variants "
            "that map to Crystalite's subatomic tokenizer encodings."
        )

    raise ValueError(
        f"Unknown or unsupported atom-reps type encoding '{mode}'. "
        "Expected atomic_number, chem_raw_v2, chem_pca_v2, or chem_pca_v2_<8|16|24>."
    )


def _validate_checkpoint_schema(ckpt: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint does not contain required key 'model_state_dict'.")
    if not isinstance(ckpt["model_state_dict"], Mapping):
        raise ValueError("Checkpoint key 'model_state_dict' must be a mapping.")

    if "ema_state_dict" in ckpt and not isinstance(ckpt["ema_state_dict"], Mapping):
        raise ValueError("Checkpoint key 'ema_state_dict' must be a mapping when present.")

    if "model_args" not in ckpt:
        raise ValueError("Checkpoint does not contain required key 'model_args'.")
    if not isinstance(ckpt["model_args"], Mapping):
        raise ValueError("Checkpoint key 'model_args' must be a mapping.")
    model_args = dict(ckpt["model_args"])

    top_type_encoding = ckpt.get("type_encoding")
    arg_type_encoding = model_args.get("type_encoding")
    if top_type_encoding is None and arg_type_encoding is None:
        raise ValueError(
            "Checkpoint must define type_encoding either at the top level or in model_args."
        )
    if top_type_encoding is not None and arg_type_encoding is not None:
        top_norm = normalize_atom_reps_type_encoding_name(top_type_encoding)
        arg_norm = normalize_atom_reps_type_encoding_name(arg_type_encoding)
        if top_norm != arg_norm:
            raise ValueError(
                "Checkpoint has conflicting type_encoding values: "
                f"top-level={top_type_encoding!r}, model_args={arg_type_encoding!r}."
            )

    source_type_encoding = top_type_encoding if top_type_encoding is not None else arg_type_encoding
    return model_args, str(source_type_encoding)


def _resolve_type_dim(
    *,
    ckpt: Mapping[str, Any],
    model_args: Mapping[str, Any],
    target_type_encoding: str,
) -> int:
    type_encoding = resolve_type_encoding(ckpt, vz=VZ)
    raw_type_dim = ckpt.get("type_dim", model_args.get("type_dim"))
    if raw_type_dim is None:
        return int(type_encoding.type_dim)

    type_dim = int(raw_type_dim)
    expected_type_dim = int(type_encoding.type_dim)
    if type_dim != expected_type_dim:
        raise ValueError(
            f"Checkpoint type_dim={type_dim} does not match {target_type_encoding} "
            f"type_dim={expected_type_dim}."
        )
    return type_dim


def convert_atom_reps_checkpoint(
    ckpt: Mapping[str, Any],
    *,
    source_path: Path | None = None,
    converted_at: str | None = None,
    type_encoding_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model_args, source_type_encoding = _validate_checkpoint_schema(ckpt)
    target_type_encoding = normalize_atom_reps_type_encoding_name(source_type_encoding)
    ckpt = migrate_type_encoding(ckpt, vz=VZ, verified_state=type_encoding_state)
    type_dim = _resolve_type_dim(
        ckpt=ckpt,
        model_args=model_args,
        target_type_encoding=target_type_encoding,
    )

    converted = dict(ckpt)
    converted_model_args = dict(model_args)
    converted_model_args["type_encoding"] = target_type_encoding
    converted_model_args["type_dim"] = type_dim

    converted["model_args"] = converted_model_args
    converted["type_encoding"] = target_type_encoding
    converted["type_dim"] = type_dim

    conversion_meta: dict[str, Any] = {
        "converter": "src.convert_atom_reps_ckpt",
        "source_format": "atom-reps",
        "target_format": "crystalite",
        "source_type_encoding": source_type_encoding,
        "target_type_encoding": target_type_encoding,
        "type_dim": type_dim,
        "converted_at": converted_at or _utc_timestamp(),
    }
    if source_path is not None:
        conversion_meta["source_path"] = str(source_path)
    if "conversion_meta" in ckpt:
        conversion_meta["previous_conversion_meta"] = ckpt["conversion_meta"]
    converted["conversion_meta"] = conversion_meta
    return converted


def build_crystalite_model_from_ckpt(
    ckpt: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> CrystaliteModel:
    model_args, source_type_encoding = _validate_checkpoint_schema(ckpt)
    target_type_encoding = normalize_atom_reps_type_encoding_name(source_type_encoding)
    type_dim = _resolve_type_dim(
        ckpt=ckpt,
        model_args=model_args,
        target_type_encoding=target_type_encoding,
    )

    if str(model_args.get("attn_type", "mha")).strip().lower() != "mha":
        raise ValueError(
            "Crystalite does not support this checkpoint because model_args.attn_type != 'mha'."
        )

    model = CrystaliteModel(
        d_model=int(model_args.get("d_model", 512)),
        n_heads=int(model_args.get("n_heads", 8)),
        n_layers=int(model_args.get("n_layers", 18)),
        vz=VZ,
        type_dim=type_dim,
        n_freqs=int(model_args.get("coord_n_freqs", model_args.get("n_freqs", 32))),
        coord_embed_mode=str(model_args.get("coord_embed_mode", "fourier")),
        coord_head_mode=str(model_args.get("coord_head_mode", "direct")),
        coord_rff_dim=model_args.get("coord_rff_dim", None),
        coord_rff_sigma=float(model_args.get("coord_rff_sigma", 1.0)),
        lattice_embed_mode=str(model_args.get("lattice_embed_mode", "mlp")),
        lattice_rff_dim=int(model_args.get("lattice_rff_dim", 256)),
        lattice_rff_sigma=float(model_args.get("lattice_rff_sigma", 5.0)),
        lattice_repr=str(model_args.get("lattice_repr", "y1")),
        dropout=float(model_args.get("dropout", 0.0)),
        attn_dropout=float(model_args.get("attn_dropout", 0.0)),
        use_distance_bias=bool(model_args.get("use_distance_bias", False)),
        use_edge_bias=bool(model_args.get("use_edge_bias", False)),
        edge_bias_n_freqs=int(model_args.get("edge_bias_n_freqs", 8)),
        edge_bias_hidden_dim=int(model_args.get("edge_bias_hidden_dim", 128)),
        edge_bias_n_rbf=int(model_args.get("edge_bias_n_rbf", 16)),
        edge_bias_rbf_max=float(model_args.get("edge_bias_rbf_max", 2.0)),
        pbc_radius=int(model_args.get("pbc_radius", 1)),
        dist_slope_init=float(model_args.get("dist_slope_init", -1.0)),
        use_noise_gate=bool(model_args.get("use_noise_gate", True)),
        gem_per_layer=bool(model_args.get("gem_per_layer", False)),
    ).to(torch.device(device))

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def verify_crystalite_load(
    ckpt: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> CrystaliteModel:
    return build_crystalite_model_from_ckpt(ckpt, device=device)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert atom-reps checkpoints to Crystalite checkpoint metadata."
    )
    parser.add_argument("--input", type=Path, required=True, help="atom-reps checkpoint path.")
    parser.add_argument("--output", type=Path, required=True, help="Converted checkpoint path.")
    parser.add_argument(
        "--type-encoding-state", type=Path,
        help="Verified recovery artifact bound to these model/EMA weights; unknown legacy PCA cannot be refitted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output checkpoint.",
    )
    parser.add_argument(
        "--verify-load",
        action="store_true",
        help="Strict-load the converted checkpoint into Crystalite before writing it.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used for --verify-load. Defaults to cpu.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.expanduser()
    output_path = args.output.expanduser()
    if (input_path.resolve() == output_path.resolve() or
            (output_path.exists() and input_path.samefile(output_path))):
        raise ValueError("Use a different output path; never overwrite the source checkpoint.")

    if not input_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    ckpt = load_checkpoint(input_path)
    state = (torch.load(args.type_encoding_state, map_location="cpu", weights_only=True)
             if args.type_encoding_state else None)
    converted = convert_atom_reps_checkpoint(ckpt, source_path=input_path, type_encoding_state=state)
    print(f"[encoding] {type_encoding_metadata(resolve_type_encoding(converted, vz=VZ))}")

    source_encoding = converted["conversion_meta"]["source_type_encoding"]
    target_encoding = converted["conversion_meta"]["target_type_encoding"]
    print(f"[convert] type_encoding: {source_encoding} -> {target_encoding}")
    print(f"[convert] type_dim: {converted['type_dim']}")

    if args.verify_load:
        verify_crystalite_load(converted, device=args.device)
        print("[convert] strict Crystalite load: ok")

    save_migrated_checkpoint(converted, source=input_path, destination=output_path, overwrite=args.overwrite)
    print(f"[convert] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
