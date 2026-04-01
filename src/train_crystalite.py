from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on PYTHONPATH when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.utils.dataset import (
    compute_allowed_elements,
    compute_dataset_element_distribution,
    dataset_to_structures,
    ensure_dataset_splits,
    sample_input_stats,
)
from src.utils.ema import EMA

from src.data.mp20_tokens import (
    MP20Tokens,
    collate_mp20_tokens,
    NMAX as DEFAULT_NMAX,
    VZ,
)

from src.eval.stability import _compute_thermo_metrics
from src.eval.wasserstein import _compute_wasserstein_metrics
from src.models.type_encoding import build_type_encoding
from src.models.lattice_repr import (
    lattice_latent_to_y1,
    y1_to_lattice_latent,
)

from src.utils.sample_stats import collect_structure_stats
from src.utils.stability_logger import StabilityLogger, _ThermoConfig
from src.utils.wandb_utils import init_wandb, log_images, log_metrics
from src.utils.constants import DATASET_NMAX_DEFAULTS, _DIAGNOSTIC_SECTION_KEYS
from src.utils.seeding import seed_everything, seed_dataloader_worker
from src.utils.checkpoint import (
    BestCkptState,
    _build_best_candidate,
    build_val_fallback_candidate,
    maybe_update_best_ckpt,
    resolve_post_training_eval_ckpt,
    select_primary_candidate_from_sampling,
    BEST_CKPT_SELECTOR_CHOICES,
)

from src.crystalite.sampler import (
    clamp_lattice_latent as _clamp_lattice_latent,
    edm_sampler,
    wrap_frac,
)
from src.crystalite.edm_utils import (
    sample_sigma,
    denoise_edm,
    compute_edm_loss,
)
from src.crystalite import CrystaliteModel, mod1
from src.eval.sample_runtime import (
    SamplingContext,
    SamplingRequest,
    generate_sampling_batch,
    evaluate_dng_sampling_batch,
    evaluate_csp_sampling_batch,
    save_sampling_artifacts,
    maybe_use_ema,
    _build_sampling_runs,
)

_build_best_candidate = _build_best_candidate
_BestCkptState = BestCkptState
_build_val_fallback_candidate = build_val_fallback_candidate
_maybe_update_best_ckpt = maybe_update_best_ckpt
_select_primary_candidate_from_sampling = select_primary_candidate_from_sampling
_resolve_post_training_eval_ckpt = resolve_post_training_eval_ckpt
_compute_allowed_elements = compute_allowed_elements
_dataset_to_structures = dataset_to_structures


def _build_count_distribution(dataset, nmax: int) -> torch.Tensor:
    counts = torch.zeros(nmax + 1, dtype=torch.float64)
    for i in range(len(dataset)):
        n = int(dataset[i]["num_atoms"])
        if 1 <= n <= nmax:
            counts[n] += 1
    if counts[1:].sum() == 0:
        raise ValueError("No valid num_atoms found for count distribution.")
    probs = counts[1:] / counts[1:].sum()
    return probs

def main() -> None:
    from src.training.config import build_parser, validate_args
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    nmax = args.nmax

    seed_everything(args.seed, deterministic=args.deterministic)
    print(f"[seed] seed={args.seed} deterministic={bool(args.deterministic)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metrics_data_root = (
        str(args.metrics_data_root)
        if args.metrics_data_root is not None
        else str(args.data_root)
    )
    metrics_dataset_name = (
        str(args.metrics_dataset_name)
        if args.metrics_dataset_name is not None
        else str(args.dataset_name)
    )
    args.metrics_data_root = metrics_data_root
    args.metrics_dataset_name = metrics_dataset_name

    has_split = ensure_dataset_splits(args.data_root, args.dataset_name)

    ds = MP20Tokens(
        root=args.data_root,
        augment_translate=True,
        split="train" if has_split else "all",
        nmax=nmax,
    )
    val_ds = MP20Tokens(
        root=args.data_root,
        augment_translate=False,
        split="val" if has_split else "all",
        nmax=nmax,
    )
    ref_ds = MP20Tokens(
        root=args.data_root,
        augment_translate=False,
        split="train" if has_split else "all",
        nmax=nmax,
    )
    train_element_dist = compute_dataset_element_distribution(ds)
    train_allowed_mask = compute_allowed_elements(ds)

    print(
        f"Dataset split sizes ({args.dataset_name}, nmax={nmax}):",
        f"train={len(ds)}",
        f"val={len(val_ds)}",
    )
    same_metrics_source = (
        metrics_dataset_name == str(args.dataset_name)
        and Path(metrics_data_root).resolve() == Path(args.data_root).resolve()
    )
    if not args.csp:
        if same_metrics_source:
            print(
                "[eval] Metric references will reuse the training dataset "
                f"({metrics_dataset_name} @ {metrics_data_root})."
            )
        else:
            print(
                "[eval] Metric references will use a separate dataset "
                f"({metrics_dataset_name} @ {metrics_data_root})."
            )

    train_loader_gen = torch.Generator()
    train_loader_gen.manual_seed(int(args.seed))

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_mp20_tokens,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        generator=train_loader_gen,
        worker_init_fn=seed_dataloader_worker,
    )

    def _infinite_loader(dl):
        while True:
            yield from dl

    steps_per_epoch = max(1, len(loader))
    data_iter = _infinite_loader(loader)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_mp20_tokens,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    type_encoding = build_type_encoding(args.type_encoding, vz=VZ)

    model = CrystaliteModel(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        vz=VZ,
        type_dim=type_encoding.type_dim,
        n_freqs=args.coord_n_freqs,
        coord_embed_mode=args.coord_embed_mode,
        coord_head_mode=args.coord_head_mode,
        coord_rff_dim=args.coord_rff_dim,
        coord_rff_sigma=args.coord_rff_sigma,
        lattice_embed_mode=args.lattice_embed_mode,
        lattice_rff_dim=args.lattice_rff_dim,
        lattice_rff_sigma=args.lattice_rff_sigma,
        lattice_repr=args.lattice_repr,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        use_distance_bias=args.use_distance_bias,
        use_edge_bias=args.use_edge_bias,
        edge_bias_n_freqs=args.edge_bias_n_freqs,
        edge_bias_hidden_dim=args.edge_bias_hidden_dim,
        edge_bias_n_rbf=args.edge_bias_n_rbf,
        edge_bias_rbf_max=args.edge_bias_rbf_max,
        pbc_radius=args.pbc_radius,
        dist_slope_init=args.dist_slope_init,
        use_noise_gate=args.use_noise_gate,
        gem_per_layer=args.gem_per_layer,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {num_params}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    warmup_steps = max(0, args.lr_warmup_steps)
    max_steps = max(1, args.max_steps)
    if warmup_steps >= max_steps:
        warmup_steps = max_steps - 1

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if step >= max_steps:
            return 0.0
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    ema = None
    if args.ema_decay > 0.0:
        ema = EMA(model, decay=args.ema_decay)
    bf16_dtype = torch.bfloat16 if args.bf16 else None

    run = init_wandb(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
        enabled=(not args.no_wandb),
    )
    log_metrics(
        {
            "dataset_splits/train": len(ds),
            "dataset_splits/val": len(val_ds),
            "dataset_splits/num_params": num_params,
            "dataset/name": args.dataset_name,
            "dataset/nmax": nmax,
            "dataset/metrics_reference_name": metrics_dataset_name,
            "dataset/metrics_reference_same_as_train": float(same_metrics_source),
        },
        step=0,
        enabled=(not args.no_wandb),
    )

    weights = {
        "A": 0.0 if args.csp else float(args.loss_weights[0]),
        "F": float(args.loss_weights[1]),
        "Y": float(args.loss_weights[2]),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build a checkpoint dict (includes EMA when available) so saves stay consistent.
    def build_ckpt(step_value: int):
        ckpt = {
            "model_state_dict": model.state_dict(),
            "model_args": vars(args),
            "step": step_value,
            "type_encoding": type_encoding.name,
            "type_dim": type_encoding.type_dim,
        }
        if ema is not None:
            ckpt["ema_state_dict"] = ema.state_dict()
            ckpt["ema_decay"] = args.ema_decay
        return ckpt

    ase_view = None

    enable_dng_metrics = args.sample_count > 0
    novelty_ref_structs = None
    ref_stats = None
    ref_structs = []
    metrics_ref_ds = None
    metrics_val_ds = None
    if not args.csp:
        if same_metrics_source:
            metrics_ref_ds = ref_ds
            metrics_val_ds = val_ds if len(val_ds) > 0 else None
        else:
            metrics_has_split = ensure_dataset_splits(
                metrics_data_root, metrics_dataset_name
            )
            metrics_ref_ds = MP20Tokens(
                root=metrics_data_root,
                augment_translate=False,
                split="train" if metrics_has_split else "all",
                nmax=nmax,
            )
            if metrics_has_split:
                maybe_metrics_val = MP20Tokens(
                    root=metrics_data_root,
                    augment_translate=False,
                    split="val",
                    nmax=nmax,
                )
                if len(maybe_metrics_val) > 0:
                    metrics_val_ds = maybe_metrics_val

        if metrics_ref_ds is None or len(metrics_ref_ds) == 0:
            raise RuntimeError(
                "Metric reference train split is empty under " f"{metrics_data_root}."
            )

        if enable_dng_metrics:
            novelty_ref_structs = dataset_to_structures(metrics_ref_ds)

        if (
            args.sample_frequency > 0
            and metrics_ref_ds is not None
            and len(metrics_ref_ds) > 0
        ):
            ref_items = (
                metrics_ref_ds.items
                if hasattr(metrics_ref_ds, "items")
                else [metrics_ref_ds[i] for i in range(len(metrics_ref_ds))]
            )
            ref_stats = collect_structure_stats(ref_items)

        ref_struct_source = (
            metrics_val_ds
            if (metrics_val_ds is not None and len(metrics_val_ds) > 0)
            else metrics_ref_ds
        )
        if ref_struct_source is not None and len(ref_struct_source) > 0:
            ref_structs = dataset_to_structures(ref_struct_source)
            ref_path = getattr(ref_struct_source, "raw_csv", None)
            ref_split = getattr(ref_struct_source, "split", "unknown")
            if ref_path:
                print(
                    f"[eval] Reference structures will use split='{ref_split}' at {ref_path}"
                )
            else:
                print(
                    f"[eval] Reference structures will use split='{ref_split}' (path unavailable)"
                )
        else:
            print(
                "[eval] No reference structures available; reference-based metrics will be skipped."
            )
    else:
        print("[eval] CSP mode: skipping de novo evaluator/reference setup.")

    thermo_logger = None
    if args.thermo_stability_check:
        if args.thermo_ppd_mp is None or not args.thermo_ppd_mp.exists():
            raise FileNotFoundError(
                "Thermo stability requires --thermo_ppd_mp pointing to a valid PPD pickle."
            )
        thermo_cfg = _ThermoConfig(
            batch_size=max(1, int(args.thermo_stability_batch)),
            relax_steps=int(args.thermo_relax_steps),
            ppd_path=str(args.thermo_ppd_mp),
            device=str(args.thermo_stability_device),
            ehull_method=str(args.thermo_ehull_method),
            mlip=str(args.thermo_mlip),
            nequip_compile_path=str(args.nequip_compile_path),
            nequip_relax_mode=str(args.nequip_relax_mode),
            nequip_optimizer=str(args.nequip_optimizer),
            nequip_cell_filter=str(args.nequip_cell_filter),
            nequip_fmax=float(args.nequip_fmax),
            nequip_max_force_abort=float(args.nequip_max_force_abort),
        )
        thermo_logger = StabilityLogger(gamma_cfg=None, thermo_cfg=thermo_cfg)
    thermo_reference_cache: dict[tuple[str, int, str], dict[str, float]] = {}

    best_ckpt_state = BestCkptState()

    count_probs = None
    if args.atom_count_strategy == "empirical":
        count_probs = _build_count_distribution(ds, nmax=nmax)

    # Optional: report input feature stats on a random subset.
    if args.stat_samples > 0:
        stats = sample_input_stats(
            ds, sample_size=args.stat_samples, type_encoding=type_encoding
        )
        if stats:
            # Print to stdout for quick inspection.
            print("Input stats (sampled):")
            for k, v in stats.items():
                print(f"  {k}: {v.detach().cpu().numpy()}")
            # Flatten and log to wandb if enabled.
            log_payload = {}
            for k, v in stats.items():
                if v.dim() == 1:
                    # Log the full vector as one entry (avoid per-element spam).
                    log_payload[f"data_stats/{k}"] = v.detach().cpu().numpy().tolist()
                else:
                    log_payload[f"data_stats/{k}"] = float(v)
            log_metrics(log_payload, step=0, enabled=(not args.no_wandb))

    sampling_ctx = SamplingContext(
        args=args,
        model=model,
        ema=ema,
        device=device,
        nmax=nmax,
        type_encoding=type_encoding,
        count_probs=count_probs,
        train_allowed_mask=train_allowed_mask,
        train_element_dist=train_element_dist,
        ref_stats=ref_stats,
        ref_structs=ref_structs,
        enable_evaluator_metrics=(not args.csp) and enable_dng_metrics,
        novelty_ref_structs=novelty_ref_structs,
        thermo_logger=thermo_logger,
        thermo_reference_cache=thermo_reference_cache,
        sample_dir=sample_dir,
        ase_view=ase_view,
        wandb_enabled=(not args.no_wandb),
    )

    model.train()
    progress = tqdm(range(1, args.max_steps + 1), desc="train", dynamic_ncols=True)
    for step in progress:
        batch = next(data_iter)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        sigma = sample_sigma(
            bsz=batch["A0"].shape[0],
            device=device,
            P_mean=args.edm_P_mean,
            P_std=args.edm_P_std,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
        )
        pad_mask = batch["pad_mask"].bool()
        real_mask = ~pad_mask

        type_clean = type_encoding.encode_from_A0(batch["A0"], pad_mask)
        frac_clean = batch["F1"]
        frac_clean_c = frac_clean - 0.5  # centered coordinates for Euclidean EDM
        lat_clean = y1_to_lattice_latent(batch["Y1"], args.lattice_repr)

        g_type = torch.randn_like(type_clean)
        g_frac = torch.randn_like(frac_clean_c)
        g_lat = torch.randn_like(lat_clean)

        if args.csp:
            type_noisy = type_clean
        else:
            type_noisy = type_clean + sigma[:, None, None] * g_type
        frac_noisy = (
            frac_clean_c + sigma[:, None, None] * g_frac
        )  # unwrapped Euclidean diffusion
        frac_noisy = torch.where(
            real_mask[..., None], frac_noisy, torch.zeros_like(frac_noisy)
        )
        lat_noisy = lat_clean + sigma[:, None] * g_lat

        denoised = denoise_edm(
            model=model,
            type_noisy=type_noisy,
            frac_noisy=frac_noisy,
            lat_noisy=lat_noisy,
            pad_mask=pad_mask,
            sigma=sigma,
            sigma_data_type=args.sigma_data_type,
            sigma_data_coord=args.sigma_data_coord,
            sigma_data_lat=args.sigma_data_lattice,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            autocast_dtype=bf16_dtype,
            skip_type_scaling=args.csp,
        )

        losses = compute_edm_loss(
            denoised=denoised,
            clean={"type": type_clean, "frac_c": frac_clean_c, "lat": lat_clean},
            frac_noisy=frac_noisy,
            sigma=sigma,
            pad_mask=pad_mask,
            sigma_data_type=args.sigma_data_type,
            sigma_data_coord=args.sigma_data_coord,
            sigma_data_lat=args.sigma_data_lattice,
            loss_weights=weights,
            coord_loss_mode=args.coord_loss_mode,
            lattice_repr=args.lattice_repr,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss_total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        if step % args.log_every == 0:
            metrics = {
                "loss/total": float(losses["loss_total"].item()),
                "loss/type": float(losses["loss_a"].item() * weights["A"]),
                "loss/coord": float(losses["loss_f"].item() * weights["F"]),
                "loss/lattice": float(losses["loss_y"].item() * weights["Y"]),
                "lr": optimizer.param_groups[0]["lr"],
                "sigma/mean": float(sigma.mean().item()),
                "sigma/std": float(sigma.std().item()),
            }
            if real_mask.any():
                type_pred = type_encoding.decode_logits_to_A0(
                    type_logits=denoised["type"], pad_mask=pad_mask
                )
                target_zero = batch["A0"]
                correct = (type_pred == target_zero) & real_mask
                metrics["stats/type_acc"] = float(
                    correct.float().sum().item() / real_mask.sum().item()
                )
                frac_delta = wrap_frac(denoised["frac"] - frac_clean_c)
                mask_exp = real_mask[..., None].float()
                metrics["stats/coord_l2"] = float(
                    ((frac_delta**2) * mask_exp).sum().item()
                    / mask_exp.sum().clamp_min(1.0).item()
                )
            else:
                metrics["stats/type_acc"] = 0.0
                metrics["stats/coord_l2"] = 0.0
            metrics["stats/lattice_l2"] = float(
                ((denoised["lat"] - lat_clean) ** 2).mean().item()
            )
            log_metrics(metrics, step=step, enabled=(not args.no_wandb))
            progress.set_postfix(loss=f"{metrics['loss/total']:.4f}")

        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            ckpt_path = (
                ckpt_dir / "step_latest.pt"
                if args.ckpt_latest_only
                else ckpt_dir / f"step_{step:07d}.pt"
            )
            torch.save(build_ckpt(step), ckpt_path)

        if args.val_every > 0 and step % args.val_every == 0 and has_split:
            model.eval()
            val_losses = {
                "loss_total": 0.0,
                "loss_a": 0.0,
                "loss_f": 0.0,
                "loss_y": 0.0,
            }
            num = 0
            with torch.no_grad():
                for v_idx, v_batch in enumerate(val_loader):
                    if v_idx >= args.val_batches:
                        break
                    v_batch = {
                        k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in v_batch.items()
                    }
                    sigma_v = sample_sigma(
                        bsz=v_batch["A0"].shape[0],
                        device=device,
                        P_mean=args.edm_P_mean,
                        P_std=args.edm_P_std,
                        sigma_min=args.sigma_min,
                        sigma_max=args.sigma_max,
                    )
                    pad_v = v_batch["pad_mask"].bool()
                    real_v = ~pad_v
                    type_clean_v = type_encoding.encode_from_A0(v_batch["A0"], pad_v)
                    frac_clean_v = v_batch["F1"]
                    frac_clean_v_c = frac_clean_v - 0.5
                    lat_clean_v = y1_to_lattice_latent(v_batch["Y1"], args.lattice_repr)
                    if args.csp:
                        type_noisy_v = type_clean_v
                    else:
                        type_noisy_v = type_clean_v + sigma_v[
                            :, None, None
                        ] * torch.randn_like(type_clean_v)
                    frac_noisy_v = frac_clean_v_c + sigma_v[
                        :, None, None
                    ] * torch.randn_like(frac_clean_v_c)
                    frac_noisy_v = torch.where(
                        real_v[..., None], frac_noisy_v, torch.zeros_like(frac_noisy_v)
                    )
                    lat_noisy_v = lat_clean_v + sigma_v[:, None] * torch.randn_like(
                        lat_clean_v
                    )

                    denoised_v = denoise_edm(
                        model=model,
                        type_noisy=type_noisy_v,
                        frac_noisy=frac_noisy_v,
                        lat_noisy=lat_noisy_v,
                        pad_mask=pad_v,
                        sigma=sigma_v,
                        sigma_data_type=args.sigma_data_type,
                        sigma_data_coord=args.sigma_data_coord,
                        sigma_data_lat=args.sigma_data_lattice,
                        sigma_min=args.sigma_min,
                        sigma_max=args.sigma_max,
                        autocast_dtype=bf16_dtype,
                        skip_type_scaling=args.csp,
                    )
                    v_losses = compute_edm_loss(
                        denoised=denoised_v,
                        clean={
                            "type": type_clean_v,
                            "frac_c": frac_clean_v_c,
                            "lat": lat_clean_v,
                        },
                        frac_noisy=frac_noisy_v,
                        sigma=sigma_v,
                        pad_mask=pad_v,
                        sigma_data_type=args.sigma_data_type,
                        sigma_data_coord=args.sigma_data_coord,
                        sigma_data_lat=args.sigma_data_lattice,
                        loss_weights=weights,
                        coord_loss_mode=args.coord_loss_mode,
                        lattice_repr=args.lattice_repr,
                    )
                    for k in val_losses:
                        val_losses[k] += float(v_losses[k].item())
                    num += 1
            if num > 0:
                for k in val_losses:
                    val_losses[k] /= num
            log_metrics(
                {
                    "val/loss_total": val_losses["loss_total"],
                    "val/loss_type": val_losses["loss_a"],
                    "val/loss_coord": val_losses["loss_f"],
                    "val/loss_lattice": val_losses["loss_y"],
                },
                step=step,
                enabled=(not args.no_wandb),
            )
            progress.set_postfix(val_loss=f"{val_losses['loss_total']:.4f}")

            # Best-checkpoint fallback: track best val loss before sampling metrics are available.
            _fb = build_val_fallback_candidate(
                step=step,
                epoch=math.ceil(step / steps_per_epoch),
                mode="csp" if args.csp else "dng",
                val_loss=val_losses["loss_total"],
            )
            if _fb is not None:
                maybe_update_best_ckpt(
                    state=best_ckpt_state,
                    candidate=_fb,
                    maximize=False,
                    ckpt_dir=ckpt_dir,
                    build_ckpt_fn=build_ckpt,
                    enabled=args.best_ckpt,
                )

            model.train()

        do_sample = args.sample_frequency > 0 and (step % args.sample_frequency == 0)
        if do_sample:
            base_seed = args.sample_seed + step
            was_training = model.training

            # Accumulators for best-checkpoint metric collection.
            _dng_payloads: dict[str, dict[str, float]] = {}
            _csp_payloads: dict[str, list[dict[str, Any]]] = {}

            def _run_one_sampling(
                tag: str,
                use_ema: bool,
                metrics_count: int,
                csp_source_ds=None,
                csp_source_label: str = "val",
            ) -> None:
                request = SamplingRequest(
                    tag=tag,
                    step=step,
                    base_seed=base_seed,
                    use_ema=use_ema,
                    metrics_count=metrics_count,
                    csp_source_ds=csp_source_ds,
                    csp_source_label=csp_source_label,
                )
                with maybe_use_ema(model, ema, use_ema):
                    batch = generate_sampling_batch(request, sampling_ctx)
                if args.csp:
                    outcome = evaluate_csp_sampling_batch(batch, request, sampling_ctx)
                else:
                    outcome = evaluate_dng_sampling_batch(batch, request, sampling_ctx)
                save_sampling_artifacts(batch, request, sampling_ctx)
                if outcome.dng_payload:
                    _dng_payloads.setdefault(tag, {}).update(outcome.dng_payload)
                for payload in outcome.csp_payloads:
                    _csp_payloads.setdefault(tag, []).append(payload)

            runs, ema_missing = _build_sampling_runs(
                do_sample=do_sample,
                sample_mode=args.sample_mode,
                ema_use_for_sampling=args.ema_use_for_sampling,
                ema_available=(ema is not None),
                sample_count=args.sample_count,
            )

            if ema_missing:
                print(
                    "[sample] EMA requested via --sample_mode but unavailable; using regular weights instead."
                )

            for tag, use_ema, mcount in runs:
                if args.csp:
                    _run_one_sampling(tag, use_ema, mcount, csp_source_ds=val_ds, csp_source_label="val")
                else:
                    _run_one_sampling(tag, use_ema, mcount)

            # Best-checkpoint: select primary candidate from sampling metrics.
            if args.best_ckpt and (_dng_payloads or _csp_payloads):
                _primary = select_primary_candidate_from_sampling(
                    is_csp=args.csp,
                    step=step,
                    epoch=math.ceil(step / steps_per_epoch),
                    dng_payloads=_dng_payloads if not args.csp else None,
                    csp_payloads=_csp_payloads if args.csp else None,
                    best_ckpt_selector=args.best_ckpt_selector,
                )
                if _primary is not None:
                    maybe_update_best_ckpt(
                        state=best_ckpt_state,
                        candidate=_primary,
                        maximize=True,
                        ckpt_dir=ckpt_dir,
                        build_ckpt_fn=build_ckpt,
                        enabled=True,
                    )

            if was_training:
                model.train()

    # Reuse the epoch_latest file for the final save to avoid writing twice.
    final_epoch_path = ckpt_dir / "epoch_latest.pt"
    ckpt = build_ckpt(args.max_steps)
    ckpt["epoch"] = math.ceil(args.max_steps / steps_per_epoch)
    torch.save(ckpt, final_epoch_path)

    # Provide a compatibility link at checkpoints/final.pt without duplicating data.
    final_path = ckpt_dir / "final.pt"
    try:
        if final_path.exists() or final_path.is_symlink():
            final_path.unlink()
        os.link(final_epoch_path, final_path)
    except OSError:
        # Fallback to a symlink; if that fails, we still have epoch_latest.pt.
        try:
            if final_path.exists() or final_path.is_symlink():
                final_path.unlink()
            os.symlink(final_epoch_path.name, final_path)
        except OSError:
            pass

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
