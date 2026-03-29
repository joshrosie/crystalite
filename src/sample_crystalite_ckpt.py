from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

# Ensure repository root is on PYTHONPATH when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pymatgen.core.periodic_table import Element
from pymatgen.io.cif import CifWriter

from src.data.mp20_tokens import MP20Tokens, VZ, tokens_to_structure
from src.data.type_encoding import build_type_encoding
from src.crystalite import CrystaliteModel, mod1
from src.crystalite.sampler import clamp_lattice_latent as _clamp_lattice_latent, edm_sampler
from src.models.lattice_repr import lattice_latent_to_y1
from src.utils.dataset import compute_allowed_elements, ensure_dataset_splits
from src.utils.sample_stats import pmg_to_ase


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch versions do not support weights_only.
        return torch.load(path, map_location="cpu")


def _resolve_checkpoint_path(checkpoint: str, train_output_dir: str, preference: str) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    if not train_output_dir:
        raise ValueError("Provide either --checkpoint or --train_output_dir.")

    ckpt_dir = Path(train_output_dir) / "checkpoints"
    if preference == "auto":
        names = ["best.pt", "final.pt", "step_latest.pt", "epoch_latest.pt"]
    else:
        names = [f"{preference}.pt"]
    for name in names:
        path = ckpt_dir / name
        if path.exists():
            return path
    looked = ", ".join(str(ckpt_dir / n) for n in names)
    raise FileNotFoundError(f"Could not resolve checkpoint. Looked at: {looked}")


def _cfg_value(cli_val: Any, model_args: dict[str, Any], key: str, default: Any) -> Any:
    if isinstance(cli_val, str) and cli_val == "":
        cli_val = None
    if cli_val is not None:
        return cli_val
    if key in model_args:
        return model_args[key]
    return default


def _apply_ema_state_dict(model: torch.nn.Module, ema_state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in ema_state:
                src = ema_state[name].to(device=param.device, dtype=param.dtype)
                param.copy_(src)


def _build_model_from_ckpt(
    *,
    ckpt: dict[str, Any],
    device: torch.device,
) -> tuple[CrystaliteModel, dict[str, Any]]:
    model_args = dict(ckpt.get("model_args", {}))
    if str(model_args.get("attn_type", "mha")).strip().lower() != "mha":
        raise ValueError(
            "This codebase no longer supports simplicial attention checkpoints. "
            "The checkpoint was configured with attn_type != 'mha'."
        )
    type_dim = int(ckpt.get("type_dim", model_args.get("type_dim", VZ + 1)))

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
    ).to(device)

    model_state = ckpt.get("model_state_dict", None)
    if model_state is None:
        raise KeyError("Checkpoint does not contain model_state_dict.")
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model, model_args


def _build_count_distribution(dataset, nmax: int) -> torch.Tensor:
    counts = torch.zeros(nmax + 1, dtype=torch.float64)
    for i in range(len(dataset)):
        n = int(dataset[i]["num_atoms"])
        if 1 <= n <= nmax:
            counts[n] += 1
    if counts[1:].sum() == 0:
        raise ValueError("No valid num_atoms found for count distribution.")
    return counts[1:] / counts[1:].sum()


def _sample_num_atoms(
    *,
    bsz: int,
    nmax: int,
    strategy: str,
    fixed_num_atoms: int | None,
    count_probs: torch.Tensor | None,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if strategy == "max":
        return torch.full((bsz,), int(nmax), device=device, dtype=torch.long)

    if strategy == "fixed":
        if fixed_num_atoms is None:
            raise ValueError("--fixed_num_atoms is required when atom-count strategy is fixed.")
        if not (1 <= int(fixed_num_atoms) <= int(nmax)):
            raise ValueError(
                f"fixed_num_atoms must be in [1, nmax={nmax}], got {fixed_num_atoms}."
            )
        return torch.full((bsz,), int(fixed_num_atoms), device=device, dtype=torch.long)

    if count_probs is None:
        raise ValueError("Empirical atom-count sampling requires a reference dataset.")
    draw = torch.multinomial(
        count_probs.to(device=device, dtype=torch.float32),
        num_samples=bsz,
        replacement=True,
        generator=generator,
    )
    return draw + 1


def _parse_allowed_elements(raw_values: list[str] | None, vz: int) -> torch.Tensor | None:
    if not raw_values:
        return None

    allowed = torch.zeros(vz, dtype=torch.bool)
    parsed: list[int] = []
    for raw_value in raw_values:
        token = str(raw_value).strip()
        if not token:
            continue
        if token.isdigit():
            z = int(token)
        else:
            try:
                normalized = token if len(token) == 1 else token[0].upper() + token[1:].lower()
                z = int(Element(normalized).Z)
            except Exception as exc:
                raise ValueError(
                    f"Could not parse allowed element {raw_value!r} as an atomic number or symbol."
                ) from exc
        if not (1 <= z <= vz):
            raise ValueError(f"Allowed element {raw_value!r} resolved to Z={z}, outside [1, {vz}].")
        allowed[z - 1] = True
        parsed.append(z)

    if not parsed:
        return None
    return allowed


def _resolve_output_dir(output_dir: str, ckpt_path: Path) -> Path:
    if output_dir:
        return Path(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if ckpt_path.parent.name == "checkpoints" and ckpt_path.parent.parent.exists():
        base = ckpt_path.parent.parent / "offline_samples"
    else:
        base = ckpt_path.parent / "offline_samples"
    return base / f"{ckpt_path.stem}_{timestamp}"


def _prepare_dataset_context(
    *,
    data_root: str,
    dataset_name: str,
    nmax: int,
    want_empirical_counts: bool,
    want_allowed_mask: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not data_root:
        raise FileNotFoundError("A dataset root is required for empirical atom counts or dataset element masks.")

    data_root_path = Path(data_root)
    if not data_root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root_path}")

    has_split = ensure_dataset_splits(data_root_path, dataset_name)
    split = "train" if has_split else "all"
    dataset = MP20Tokens(
        root=str(data_root_path),
        augment_translate=False,
        split=split,
        nmax=nmax,
    )
    count_probs = _build_count_distribution(dataset, nmax=nmax) if want_empirical_counts else None
    allowed_mask = compute_allowed_elements(dataset) if want_allowed_mask else None
    return count_probs, allowed_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a Crystalite checkpoint and sample structures offline."
    )
    parser.add_argument("--train_output_dir", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--checkpoint_preference",
        type=str,
        default="auto",
        choices=["auto", "best", "final", "step_latest", "epoch_latest"],
    )
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument(
        "--sample_chunk_size",
        type=int,
        default=256,
        help="Chunk size for sampling. Set to 0 to disable chunking.",
    )
    parser.add_argument("--sample_seed", type=int, default=None)
    parser.add_argument("--sample_num_steps", type=int, default=None)
    parser.add_argument("--sample_mode", type=str, default="ema", choices=["ema", "regular"])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--nmax", type=int, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument(
        "--atom_count_strategy",
        type=str,
        default="auto",
        choices=["auto", "empirical", "fixed", "max"],
        help=(
            "Atom-count sampling mode. 'auto' uses the checkpoint's empirical setting "
            "when a dataset is available; otherwise it falls back to nmax."
        ),
    )
    parser.add_argument("--fixed_num_atoms", type=int, default=None)
    parser.add_argument(
        "--allowed_elements",
        nargs="*",
        default=None,
        help="Optional whitelist of atomic numbers or element symbols used when decoding atom types.",
    )
    parser.add_argument(
        "--save_pt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write raw token samples to samples.pt.",
    )
    parser.add_argument(
        "--save_extxyz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write exported structures to samples.xyz in extxyz format.",
    )
    parser.add_argument(
        "--save_cifs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write exported structures as individual CIF files.",
    )
    parser.add_argument(
        "--cif_limit",
        type=int,
        default=0,
        help="Maximum number of CIFs to write. 0 means write all exported structures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = _resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        train_output_dir=args.train_output_dir,
        preference=args.checkpoint_preference,
    )
    ckpt = _load_checkpoint(ckpt_path)
    model, model_args = _build_model_from_ckpt(ckpt=ckpt, device=torch.device("cpu"))

    nmax = int(_cfg_value(args.nmax, model_args, "nmax", 20))
    dataset_name = str(_cfg_value(args.dataset_name, model_args, "dataset_name", "mp20"))
    data_root = str(_cfg_value(args.data_root, model_args, "data_root", ""))
    sample_seed = int(_cfg_value(args.sample_seed, model_args, "sample_seed", 123))
    sample_num_steps = int(_cfg_value(args.sample_num_steps, model_args, "sample_num_steps", 100))
    lattice_repr = str(_cfg_value(None, model_args, "lattice_repr", "y1"))

    sigma_min = float(_cfg_value(None, model_args, "sigma_min", 0.002))
    sigma_max = float(_cfg_value(None, model_args, "sigma_max", 80.0))
    rho = float(_cfg_value(None, model_args, "rho", 7.0))
    S_churn = float(_cfg_value(None, model_args, "S_churn", 20.0))
    S_min = float(_cfg_value(None, model_args, "S_min", 0.0))
    S_max = float(_cfg_value(None, model_args, "S_max", 999.0))
    S_noise = float(_cfg_value(None, model_args, "S_noise", 1.0))
    sigma_data_type = float(_cfg_value(None, model_args, "sigma_data_type", 1.0))
    sigma_data_coord = float(_cfg_value(None, model_args, "sigma_data_coord", 0.25))
    sigma_data_lattice = float(_cfg_value(None, model_args, "sigma_data_lattice", 1.0))
    aa_frac_max_scale = float(_cfg_value(None, model_args, "aa_frac_max_scale", 0.0))
    aa_rho_types = float(_cfg_value(None, model_args, "aa_rho_types", 0.0))
    aa_rho_coords = float(_cfg_value(None, model_args, "aa_rho_coords", 0.0))
    aa_rho_lattice = float(_cfg_value(None, model_args, "aa_rho_lattice", 0.0))

    _seed_everything(sample_seed)
    use_cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")
    model = model.to(device)

    if args.sample_mode == "ema":
        ema_state = ckpt.get("ema_state_dict", None)
        if ema_state is not None:
            _apply_ema_state_dict(model, ema_state)
            print("[ckpt] Using EMA weights for sampling.")
        else:
            print("[ckpt] EMA state not found; falling back to regular model weights.")
    else:
        print("[ckpt] Using regular model weights for sampling.")

    type_encoding_name = str(ckpt.get("type_encoding", model_args.get("type_encoding", "atomic_number")))
    type_encoding = build_type_encoding(type_encoding_name, vz=VZ)

    allowed_mask = _parse_allowed_elements(args.allowed_elements, VZ)
    ckpt_atom_count_strategy = str(model_args.get("atom_count_strategy", "empirical")).lower()
    requested_atom_count_strategy = str(args.atom_count_strategy).lower()
    has_dataset = bool(data_root) and Path(data_root).exists()
    if requested_atom_count_strategy == "auto":
        if ckpt_atom_count_strategy == "empirical" and has_dataset:
            atom_count_strategy = "empirical"
        elif args.fixed_num_atoms is not None:
            atom_count_strategy = "fixed"
        else:
            atom_count_strategy = "max"
            if ckpt_atom_count_strategy == "empirical":
                print(
                    "[sample] Checkpoint prefers empirical atom counts, "
                    "but no local dataset root was available. Falling back to nmax."
                )
    else:
        atom_count_strategy = requested_atom_count_strategy

    count_probs = None
    if atom_count_strategy == "empirical" or (allowed_mask is None and has_dataset):
        try:
            want_allowed_mask = allowed_mask is None
            count_probs, dataset_allowed_mask = _prepare_dataset_context(
                data_root=data_root,
                dataset_name=dataset_name,
                nmax=nmax,
                want_empirical_counts=(atom_count_strategy == "empirical"),
                want_allowed_mask=want_allowed_mask,
            )
            if allowed_mask is None:
                allowed_mask = dataset_allowed_mask
            if atom_count_strategy == "empirical":
                print(f"[data] Using empirical atom counts from {data_root} ({dataset_name}).")
            if allowed_mask is dataset_allowed_mask and allowed_mask is not None:
                print("[data] Restricting decoded elements to the dataset train split support.")
        except Exception as exc:
            if atom_count_strategy == "empirical":
                raise
            print(f"[data] Could not load dataset context ({exc}). Continuing without dataset element mask.")

    if allowed_mask is not None and args.allowed_elements:
        kept = int(allowed_mask.sum().item())
        print(f"[decode] Restricting decoded elements to {kept} user-specified entries.")
    elif allowed_mask is None:
        print("[decode] No element whitelist applied while decoding atom types.")

    output_dir = _resolve_output_dir(args.output_dir, ckpt_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples = int(args.num_samples)
    chunk_size = max(1, int(args.sample_chunk_size or num_samples))
    generator = torch.Generator(device=device)
    generator.manual_seed(sample_seed)
    autocast_dtype = torch.bfloat16 if args.bf16 else None

    print(
        f"[sample] checkpoint={ckpt_path} step={int(ckpt.get('step', -1))} "
        f"device={device} samples={num_samples} chunk_size={chunk_size} "
        f"num_steps={sample_num_steps} atom_count={atom_count_strategy}"
    )

    sample_items: list[dict[str, torch.Tensor]] = []
    arange = torch.arange(nmax, device=device)[None, :]
    with torch.inference_mode():
        for start in tqdm(range(0, num_samples, chunk_size), desc="sampling", dynamic_ncols=True):
            end = min(start + chunk_size, num_samples)
            bsz = end - start

            num_atoms = _sample_num_atoms(
                bsz=bsz,
                nmax=nmax,
                strategy=atom_count_strategy,
                fixed_num_atoms=args.fixed_num_atoms,
                count_probs=count_probs,
                device=device,
                generator=generator,
            )
            pad_mask = arange >= num_atoms[:, None]
            real_mask = ~pad_mask

            samples = edm_sampler(
                model=model,
                pad_mask=pad_mask,
                type_dim=type_encoding.type_dim,
                num_steps=sample_num_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
                S_churn=S_churn,
                S_min=S_min,
                S_max=S_max,
                S_noise=S_noise,
                sigma_data_type=sigma_data_type,
                sigma_data_coord=sigma_data_coord,
                sigma_data_lat=sigma_data_lattice,
                generator=generator,
                autocast_dtype=autocast_dtype,
                fixed_atom_types=None,
                skip_type_scaling=False,
                aa_frac_max_scale=aa_frac_max_scale,
                aa_rho_types=aa_rho_types,
                aa_rho_coords=aa_rho_coords,
                aa_rho_lattice=aa_rho_lattice,
                lattice_repr=lattice_repr,
            )

            pad_mask_cpu = pad_mask.to("cpu")
            real_mask_cpu = ~pad_mask_cpu
            atom_idx = type_encoding.decode_logits_to_A0(
                type_logits=samples["type"].detach().cpu(),
                pad_mask=pad_mask_cpu,
                allowed_mask=allowed_mask,
            )
            atom_idx = torch.where(real_mask_cpu, atom_idx, torch.zeros_like(atom_idx))

            frac_coords = mod1(samples["frac"].detach().cpu() + 0.5).clamp(0.0, 1.0)
            frac_coords = torch.where(
                real_mask_cpu[..., None], frac_coords, torch.zeros_like(frac_coords)
            )

            lattice_latent = _clamp_lattice_latent(
                samples["lat"].detach().cpu(), lattice_repr=lattice_repr
            )
            lattice = lattice_latent_to_y1(lattice_latent, lattice_repr=lattice_repr)
            lattice = _clamp_lattice_latent(lattice, lattice_repr="y1")

            for i in range(bsz):
                sample_items.append(
                    {
                        "A0": atom_idx[i],
                        "F1": frac_coords[i],
                        "Y1": lattice[i],
                        "pad_mask": pad_mask_cpu[i],
                    }
                )

    print(f"[sample] Finished sampling. generated={len(sample_items)}")

    if args.save_pt:
        samples_pt = output_dir / "samples.pt"
        torch.save(sample_items, samples_pt)
        print(f"[save] Sample tensors: {samples_pt}")

    exported_structures = []
    exported_indices = []
    export_failures: Counter[str] = Counter()
    for sample_idx, item in enumerate(sample_items):
        try:
            exported_structures.append(tokens_to_structure(item))
            exported_indices.append(sample_idx)
        except Exception as exc:
            export_failures[f"{type(exc).__name__}: {exc}"] += 1

    if args.save_extxyz and exported_structures:
        from ase.io import write

        xyz_path = output_dir / "samples.xyz"
        ase_structs = [pmg_to_ase(struct) for struct in exported_structures]
        write(str(xyz_path), ase_structs, format="extxyz")
        print(f"[save] extxyz: {xyz_path}")

    cif_manifest: list[dict[str, Any]] = []
    if args.save_cifs and exported_structures:
        cif_dir = output_dir / "cifs"
        cif_dir.mkdir(parents=True, exist_ok=True)
        limit = (
            len(exported_structures)
            if int(args.cif_limit) <= 0
            else min(len(exported_structures), int(args.cif_limit))
        )
        for local_idx, (sample_idx, struct) in enumerate(zip(exported_indices, exported_structures)):
            if local_idx >= limit:
                break
            cif_name = f"sample_{sample_idx:05d}.cif"
            (cif_dir / cif_name).write_text(str(CifWriter(struct)))
            cif_manifest.append(
                {
                    "sample_idx": sample_idx,
                    "file": cif_name,
                    "formula": struct.composition.reduced_formula,
                    "num_sites": int(len(struct)),
                }
            )
        manifest_path = cif_dir / "manifest.json"
        manifest_path.write_text(json.dumps(cif_manifest, indent=2))
        print(f"[save] CIFs: {cif_dir} ({len(cif_manifest)} files)")

    summary = {
        "meta": {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_step": int(ckpt.get("step", -1)),
            "dataset_name": dataset_name,
            "data_root": data_root,
            "type_encoding": type_encoding.name,
            "lattice_repr": lattice_repr,
        },
        "sampling": {
            "device": str(device),
            "num_samples": num_samples,
            "sample_chunk_size": chunk_size,
            "sample_seed": sample_seed,
            "sample_num_steps": sample_num_steps,
            "sample_mode": args.sample_mode,
            "atom_count_strategy": atom_count_strategy,
            "fixed_num_atoms": args.fixed_num_atoms,
            "bf16": bool(args.bf16),
        },
        "outputs": {
            "save_pt": bool(args.save_pt),
            "save_extxyz": bool(args.save_extxyz),
            "save_cifs": bool(args.save_cifs),
        },
        "results": {
            "num_samples_generated": len(sample_items),
            "num_exported_structures": len(exported_structures),
            "num_export_failures": int(sum(export_failures.values())),
            "export_failure_counts": dict(export_failures),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[save] Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
