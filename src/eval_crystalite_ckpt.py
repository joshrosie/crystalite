from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
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

from src.data.mp20_tokens import MP20Tokens, VZ
from src.data.type_encoding import build_type_encoding
from src.crystalite import CrystaliteModel, mod1
from src.crystalite.sampler import clamp_lattice_latent as _clamp_lattice_latent, edm_sampler
from src.eval.dng_eval import (
    collect_constructed_structures,
    compute_evaluator_metrics,
    compute_novelty_metrics,
    compute_structure_stats_metrics,
    compute_sun_msun_from_thermo_rates,
    float_or_nan,
)
from src.eval.stability import _compute_thermo_metrics
from src.models.lattice_repr import lattice_latent_to_y1
from src.utils.dataset import (
    compute_allowed_elements as _compute_allowed_elements,
    dataset_to_structures as _dataset_to_structures,
)
from src.utils.stability_logger import StabilityLogger, _ThermoConfig


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
    if strategy == "fixed":
        if fixed_num_atoms is None:
            raise ValueError("--fixed_num_atoms is required when --atom_count_strategy=fixed.")
        if not (1 <= int(fixed_num_atoms) <= int(nmax)):
            raise ValueError(
                f"fixed_num_atoms must be in [1, nmax={nmax}], got {fixed_num_atoms}."
            )
        return torch.full((bsz,), int(fixed_num_atoms), device=device, dtype=torch.long)

    if count_probs is None:
        raise ValueError("Empirical atom-count sampling requires count_probs.")
    draw = torch.multinomial(
        count_probs.to(device=device, dtype=torch.float32),
        num_samples=bsz,
        replacement=True,
        generator=generator,
    )
    return draw + 1


def _build_count_distribution(dataset, nmax: int) -> torch.Tensor:
    counts = torch.zeros(nmax + 1, dtype=torch.float64)
    for i in range(len(dataset)):
        n = int(dataset[i]["num_atoms"])
        if 1 <= n <= nmax:
            counts[n] += 1
    if counts[1:].sum() == 0:
        raise ValueError("No valid num_atoms found for count distribution.")
    return counts[1:] / counts[1:].sum()


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


def _print_final_metrics(metrics: dict[str, float]) -> None:
    print("\n===== FINAL SAMPLE METRICS =====")
    for key in sorted(metrics):
        val = metrics[key]
        if not math.isfinite(val):
            print(f"{key}: nan")
        elif abs(val) >= 1e4 or (0 < abs(val) < 1e-4):
            print(f"{key}: {val:.6e}")
        else:
            print(f"{key}: {val:.6f}")


def _collect_sun_structures(
    un_structs: list,
    *,
    logger: "StabilityLogger",
    relax_steps: int,
    ehull_method: str,
) -> list[tuple[Any, float]]:
    """Relax UN structures; return (relaxed_struct, e_above_hull) for stable ones (ehull <= 0)."""
    from src.utils.sample_stats import (
        compute_e_above_hull_mp2020_like,
        compute_e_above_hull_uncorrected,
    )

    stable: list[tuple[Any, float]] = []
    for struct in tqdm(un_structs, desc="sun/thermo_relax", dynamic_ncols=True):
        try:
            relaxation = logger._thermo_relaxer.relax(struct, steps=relax_steps, verbose=False)
        except Exception:
            continue
        hull_struct = relaxation["final_structure"]
        e_total = float(relaxation["trajectory"].energies[-1])
        if ehull_method == "mp2020_like":
            e_val, fail = compute_e_above_hull_mp2020_like(
                logger._thermo_ppd,
                hull_struct,
                e_total,
                mp2020_compat=logger._thermo_mp2020_compat,
            )
        else:
            e_val, fail = compute_e_above_hull_uncorrected(
                logger._thermo_ppd, hull_struct, e_total
            )
        if fail is not None or e_val is None:
            continue
        if float(e_val) <= 0.0:
            stable.append((hull_struct, float(e_val)))
    return stable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a Crystalite checkpoint, sample structures, and compute "
            "diagnostics + generation metrics."
        )
    )
    parser.add_argument("--train_output_dir", type=str, default="outputs/DNG_nequip_alex_mp20_med_new_gem")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--checkpoint_preference",
        type=str,
        default="auto",
        choices=["auto", "best", "final", "step_latest", "epoch_latest"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--nmax", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--sample_chunk_size", type=int, default=256)
    parser.add_argument("--sample_seed", type=int, default=None)
    parser.add_argument("--sample_num_steps", type=int, default=None)
    parser.add_argument("--sample_mode", type=str, default="ema", choices=["ema", "regular"])
    parser.add_argument("--atom_count_strategy", type=str, default=None, choices=["empirical", "fixed"])
    parser.add_argument("--fixed_num_atoms", type=int, default=None)
    parser.add_argument("--sample_novelty_limit", type=int, default=None)
    parser.add_argument("--eval_jobs", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument(
        "--compute_novelty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute uniqueness/novelty/UN metrics.",
    )
    parser.add_argument(
        "--compute_wasserstein",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute Wasserstein distribution distances against reference set.",
    )
    parser.add_argument(
        "--compute_structure_stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute extra aggregate structure stats and invalid-reason counts.",
    )

    parser.add_argument("--report_dir", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--save_samples_pt", action="store_true")
    parser.add_argument(
        "--save_sun_samples",
        action="store_true",
        help=(
            "Save stable-unique-novel (SUN) structures as CIF files. "
            "Requires --compute_novelty and --thermo_count > 0."
        ),
    )

    parser.add_argument(
        "--thermo_count",
        type=int,
        default=0,
        help=(
            "Optional number of generated structures to run thermo relaxation on. "
            "0 disables thermo metrics."
        ),
    )
    parser.add_argument("--thermo_stability_batch", type=int, default=None)
    parser.add_argument("--thermo_relax_steps", type=int, default=None)
    parser.add_argument("--thermo_stability_device", type=str, default=None)
    parser.add_argument("--thermo_mlip", type=str, default=None, choices=["chgnet", "nequip"])
    parser.add_argument(
        "--thermo_ehull_method",
        type=str,
        default=None,
        choices=["uncorrected", "mp2020_like"],
    )
    parser.add_argument("--thermo_ppd_mp", type=str, default=None)
    parser.add_argument("--nequip_compile_path", type=str, default=None)
    parser.add_argument("--nequip_optimizer", type=str, default=None)
    parser.add_argument("--nequip_cell_filter", type=str, default=None)
    parser.add_argument("--nequip_fmax", type=float, default=None)
    parser.add_argument("--nequip_max_force_abort", type=float, default=None)
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

    dataset_name = str(
        args.dataset_name
        if args.dataset_name is not None
        else model_args.get(
            "metrics_dataset_name",
            _cfg_value(None, model_args, "dataset_name", "alex_mp20"),
        )
    )
    data_root = str(
        args.data_root
        if args.data_root is not None
        else model_args.get(
            "metrics_data_root",
            _cfg_value(None, model_args, "data_root", "data/alex_mp20"),
        )
    )
    nmax = int(_cfg_value(args.nmax, model_args, "nmax", 20))
    sample_seed = int(_cfg_value(args.sample_seed, model_args, "sample_seed", 123))
    sample_num_steps = int(_cfg_value(args.sample_num_steps, model_args, "sample_num_steps", 100))
    atom_count_strategy = str(
        _cfg_value(args.atom_count_strategy, model_args, "atom_count_strategy", "empirical")
    ).lower()
    sample_novelty_limit = int(
        _cfg_value(args.sample_novelty_limit, model_args, "sample_novelty_limit", 0)
    )
    legacy_sample_compute_novelty_requested = bool(
        model_args.get("sample_compute_novelty", False)
    )
    if legacy_sample_compute_novelty_requested:
        print(
            "[warn] Checkpoint config requested deprecated sample_compute_novelty; "
            "ADiT novelty is no longer computed in checkpoint eval."
        )
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

    train_split = "train" if (Path(data_root) / "raw" / "train.csv").exists() else "all"
    val_split = "val" if (Path(data_root) / "raw" / "val.csv").exists() else train_split
    ds_train_aug = MP20Tokens(
        root=data_root,
        augment_translate=True,
        split=train_split,
        nmax=nmax,
    )
    ds_train_ref = MP20Tokens(
        root=data_root,
        augment_translate=False,
        split=train_split,
        nmax=nmax,
    )
    ds_val_ref = MP20Tokens(
        root=data_root,
        augment_translate=False,
        split=val_split,
        nmax=nmax,
    )
    print(
        f"[data] dataset={dataset_name} train_split={train_split} val_split={val_split} "
        f"train={len(ds_train_aug)} val={len(ds_val_ref)} nmax={nmax}"
    )

    train_allowed_mask = _compute_allowed_elements(ds_train_aug)
    count_probs = None
    if atom_count_strategy == "empirical":
        count_probs = _build_count_distribution(ds_train_aug, nmax=nmax)

    novelty_ref_structs = _dataset_to_structures(ds_train_ref) if args.compute_novelty else []
    ref_structs = _dataset_to_structures(ds_val_ref)

    generator = torch.Generator(device=device)
    generator.manual_seed(sample_seed)
    autocast_dtype = torch.bfloat16 if args.bf16 else None

    num_samples = int(args.num_samples)
    chunk_size = max(1, int(args.sample_chunk_size))
    sample_items: list[dict[str, torch.Tensor]] = []
    print(
        f"[sample] Starting sampling: num_samples={num_samples}, chunk_size={chunk_size}, "
        f"num_steps={sample_num_steps}, seed={sample_seed}"
    )

    with torch.no_grad():
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
            arange = torch.arange(nmax, device=device)[None, :]
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
                allowed_mask=train_allowed_mask,
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

    metrics_out: dict[str, float] = {}
    metrics_out["summary/num_samples_requested"] = float(num_samples)
    metrics_out["summary/num_samples_generated"] = float(len(sample_items))

    evaluator_result = compute_evaluator_metrics(
        sample_items,
        limit=len(sample_items),
        ref_structs=ref_structs or [],
        sample_seed=sample_seed,
        include_diagnostics=True,
        include_wasserstein=bool(args.compute_wasserstein),
        wasserstein_max_samples=10000,
    )
    if evaluator_result.valid_rate is not None:
        metrics_out["valid_rate"] = float_or_nan(evaluator_result.valid_rate)
    if evaluator_result.comp_valid_rate is not None:
        metrics_out["comp_valid_rate"] = float_or_nan(evaluator_result.comp_valid_rate)
    if evaluator_result.struct_valid_rate is not None:
        metrics_out["struct_valid_rate"] = float_or_nan(
            evaluator_result.struct_valid_rate
        )
    for key, val in evaluator_result.diag_metrics.items():
        metrics_out[f"diagnostic/{key}"] = float_or_nan(val)
    if args.compute_wasserstein and evaluator_result.dist_metrics:
        print("[eval] Computing Wasserstein distribution metrics.")
        for key, val in evaluator_result.dist_metrics.items():
            metrics_out[key] = float_or_nan(val)

    novelty_metrics: dict[str, Any] = {}
    novelty_un_rate: float | None = None
    if args.compute_novelty:
        print("[eval] Computing novelty metrics.")
        novelty_result = compute_novelty_metrics(
            sample_items,
            novelty_ref_structs,
            limit=len(sample_items),
            minimum_nary=1,
        )
        novelty_metrics = novelty_result.novelty_metrics
        if novelty_result.unique_rate is not None:
            metrics_out["unique_rate"] = float_or_nan(novelty_result.unique_rate)
        if novelty_result.novel_rate is not None:
            metrics_out["novel_rate"] = float_or_nan(novelty_result.novel_rate)
        if novelty_result.un_rate is not None:
            novelty_un_rate = novelty_result.un_rate
            metrics_out["un_rate"] = float_or_nan(novelty_result.un_rate)

    if args.compute_structure_stats:
        print("[eval] Computing aggregate structure stats.")
        stats_result = compute_structure_stats_metrics(
            sample_items,
            total_count=len(sample_items),
            include_summary_stats=True,
        )
        for key, val in stats_result.metrics.items():
            metrics_out[f"structure_stats/{key}"] = float_or_nan(val)

    _sun_structs: list[tuple[Any, float]] = []
    if int(args.thermo_count) > 0:
        print(f"[eval] Running thermo metrics on up to {int(args.thermo_count)} structures.")
        thermo_count = min(int(args.thermo_count), len(sample_items))
        thermo_structs = collect_constructed_structures(
            sample_items,
            pred_crys_list=evaluator_result.pred_crys_list,
            count=thermo_count,
        )

        if thermo_structs:
            thermo_cfg = _ThermoConfig(
                batch_size=int(
                    _cfg_value(
                        args.thermo_stability_batch,
                        model_args,
                        "thermo_stability_batch",
                        32,
                    )
                ),
                relax_steps=int(
                    _cfg_value(args.thermo_relax_steps, model_args, "thermo_relax_steps", 200)
                ),
                ppd_path=str(
                    _cfg_value(
                        args.thermo_ppd_mp,
                        model_args,
                        "thermo_ppd_mp",
                        "data/mp20/hull/2023-02-07-ppd-mp.pkl",
                    )
                ),
                device=str(
                    _cfg_value(
                        args.thermo_stability_device,
                        model_args,
                        "thermo_stability_device",
                        "cuda",
                    )
                ),
                ehull_method=str(
                    _cfg_value(
                        args.thermo_ehull_method,
                        model_args,
                        "thermo_ehull_method",
                        "uncorrected",
                    )
                ),
                mlip=str(_cfg_value(args.thermo_mlip, model_args, "thermo_mlip", "chgnet")),
                nequip_compile_path=str(
                    _cfg_value(
                        args.nequip_compile_path,
                        model_args,
                        "nequip_compile_path",
                        "",
                    )
                ),
                nequip_optimizer=str(
                    _cfg_value(args.nequip_optimizer, model_args, "nequip_optimizer", "FIRE")
                ),
                nequip_cell_filter=str(
                    _cfg_value(args.nequip_cell_filter, model_args, "nequip_cell_filter", "none")
                ),
                nequip_fmax=float(_cfg_value(args.nequip_fmax, model_args, "nequip_fmax", 0.01)),
                nequip_max_force_abort=float(
                    _cfg_value(
                        args.nequip_max_force_abort,
                        model_args,
                        "nequip_max_force_abort",
                        1e6,
                    )
                ),
            )
            logger = StabilityLogger(gamma_cfg=None, thermo_cfg=thermo_cfg)
            thermo_metrics = _compute_thermo_metrics(
                logger,
                thermo_structs,
                tag="eval",
                step=0,
                enabled=True,
                show_progress=True,
            )
            for key, val in thermo_metrics.items():
                metrics_out[key] = float_or_nan(val)

            sun_summary = compute_sun_msun_from_thermo_rates(
                un_rate=novelty_un_rate,
                thermo_metrics=thermo_metrics,
                thermo_tag="eval",
            )
            for key, val in sun_summary.items():
                metrics_out[key] = float_or_nan(val)
            if args.save_sun_samples and novelty_metrics.get("un_structs"):
                un_candidates = novelty_metrics["un_structs"]
                print(
                    f"[eval] Relaxing {len(un_candidates)} UN structures to find SUN samples."
                )
                _sun_structs = _collect_sun_structures(
                    un_candidates,
                    logger=logger,
                    relax_steps=thermo_cfg.relax_steps,
                    ehull_method=logger._thermo_ehull_method,
                )
                metrics_out["sun_count"] = float(len(_sun_structs))
                print(f"[eval] Found {len(_sun_structs)} SUN structures.")
        else:
            print("[eval] No constructed structures available for thermo metrics.")

    run_name = str(args.run_name).strip()
    if not run_name:
        run_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_report_dir = Path(args.report_dir) if args.report_dir else (Path(args.train_output_dir) / "eval_reports")
    report_dir = base_report_dir / run_name
    report_dir.mkdir(parents=True, exist_ok=True)

    if _sun_structs:
        sun_dir = report_dir / "sun_samples"
        sun_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for i, (struct, e_above) in enumerate(_sun_structs):
            cif_name = f"sun_{i:04d}.cif"
            struct.to(filename=str(sun_dir / cif_name), fmt="cif")
            manifest.append({
                "file": cif_name,
                "formula": struct.formula,
                "e_above_hull": e_above,
            })
        (sun_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[save] {len(_sun_structs)} SUN CIFs → {sun_dir}")

    report = {
        "meta": {
            "run_name": run_name,
            "checkpoint_path": str(ckpt_path),
            "checkpoint_step": int(ckpt.get("step", -1)),
            "dataset_name": dataset_name,
            "data_root": data_root,
            "nmax": nmax,
            "sample_mode": args.sample_mode,
            "num_samples": num_samples,
            "sample_chunk_size": chunk_size,
            "sample_num_steps": sample_num_steps,
            "sample_seed": sample_seed,
            "legacy_sample_compute_novelty_requested": legacy_sample_compute_novelty_requested,
            "compute_novelty_metrics": bool(args.compute_novelty),
            "compute_wasserstein": bool(args.compute_wasserstein),
            "thermo_count": int(args.thermo_count),
        },
        "metrics": {k: float(v) for k, v in metrics_out.items()},
    }
    metrics_json = report_dir / "metrics.json"
    metrics_json.write_text(json.dumps(report, indent=2))

    if args.save_samples_pt:
        samples_pt = report_dir / "samples.pt"
        torch.save(sample_items, samples_pt)
        print(f"[save] Sample tensors: {samples_pt}")

    print(f"[save] Metrics JSON: {metrics_json}")
    _print_final_metrics(report["metrics"])


if __name__ == "__main__":
    main()
