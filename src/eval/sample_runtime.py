from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from src.data.mp20_tokens import tokens_to_eval_dict
from src.crystalite import mod1
from src.crystalite.sampler import clamp_lattice_latent, edm_sampler
from src.eval.crystal import Crystal
from src.eval.csp_eval import RecEval, RecEvalBatch
from src.eval.diagnostics import _log_element_histogram
from src.eval.stability import _compute_thermo_metrics
from src.eval.dng_eval import (
    build_reference_thermo_comparison_metrics,
    collect_constructed_structures,
    compute_evaluator_metrics,
    compute_novelty_metrics,
    compute_structure_stats_metrics,
    compute_sun_metrics,
)
from src.models.lattice_repr import lattice_latent_to_y1
from src.training.config import _compute_topk_target_count, _normalize_topk_list
from src.utils.sample_stats import plot_sample_vs_ref_stats
from src.utils.wandb_utils import log_images, log_metrics


# ---------------------------------------------------------------------------
# Small helpers (sampling-only utilities)
# ---------------------------------------------------------------------------


def _pad_mask_from_counts(num_atoms: torch.Tensor, nmax: int) -> torch.Tensor:
    arange = torch.arange(nmax, device=num_atoms.device)[None, :]
    return arange >= num_atoms[:, None]


def _save_sample_images(
    out_dir: Path,
    samples: list[dict],
    max_images: int,
    ase_view: Any | None,
) -> list:
    if ase_view is None or max_images <= 0:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for i, item in enumerate(samples[:max_images]):
        try:
            tensors = [item["A0"], item["F1"], item["Y1"]]
            if any(torch.isnan(t).any() or torch.isinf(t).any() for t in tensors):
                continue
            from src.data.mp20_tokens import tokens_to_structure as _tts

            struct = _tts(item)
            if not np.isfinite(struct.frac_coords).all():
                continue
            if not np.isfinite(struct.lattice.matrix).all():
                continue
        except Exception:
            continue
        try:
            from src.utils.ase_notebook.backend.svg import svg_to_pil

            svg = ase_view.make_svg(struct, center_in_uc=True)
            img = svg_to_pil(svg)
            img.save(out_dir / f"sample_{i:03d}.png")
            images.append(img)
        except Exception:
            continue
    return images


def _safe_crystal_from_tokens(item: dict[str, Any], sample_idx: int) -> Crystal:
    try:
        return Crystal(tokens_to_eval_dict(item, sample_idx=sample_idx))
    except Exception:
        dummy = {
            "frac_coords": np.zeros((1, 3), dtype=np.float32),
            "atom_types": np.zeros((1,), dtype=np.int64),
            "lengths": np.ones(3, dtype=np.float32),
            "angles": np.ones(3, dtype=np.float32) * 90.0,
            "sample_idx": sample_idx,
        }
        return Crystal(dummy)


def _build_sampling_runs(
    do_sample: bool,
    sample_mode: str,
    ema_use_for_sampling: bool,
    ema_available: bool,
    sample_count: int,
) -> tuple[list[tuple[str, bool, int]], bool]:
    """Build the list of (tag, use_ema, metrics_count) sampling runs.

    Returns (runs, ema_missing).
    """
    want_regular = sample_mode in {"regular", "both"}
    want_ema = sample_mode in {"ema", "both"} or ema_use_for_sampling
    ema_missing = want_ema and not ema_available

    runs: list[tuple[str, bool, int]] = []
    if do_sample:
        if want_regular:
            runs.append(("sample", False, sample_count))
        if want_ema and ema_available:
            runs.append(("sample_ema", True, sample_count))
        elif ema_missing and not want_regular:
            runs.append(("sample", False, sample_count))

    return runs, ema_missing


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SamplingRequest:
    tag: str
    step: int
    base_seed: int
    use_ema: bool
    metrics_count: int
    csp_source_ds: Any | None = None
    csp_source_label: str = "val"

    @property
    def is_csp(self) -> bool:
        return self.csp_source_ds is not None


@dataclass
class SamplingContext:
    args: Any
    model: torch.nn.Module
    ema: Any | None
    device: torch.device
    nmax: int
    type_encoding: Any
    count_probs: torch.Tensor | None
    train_allowed_mask: torch.Tensor | None
    train_element_dist: torch.Tensor | None
    ref_stats: dict[str, Any] | None
    ref_structs: list[Any]
    enable_evaluator_metrics: bool
    novelty_ref_structs: list[Any] | None
    thermo_logger: Any | None
    thermo_reference_cache: dict[tuple[str, int, str], dict[str, float]]
    sample_dir: Path
    ase_view: Any
    wandb_enabled: bool


@dataclass
class SamplingBatch:
    sample_items: list[dict[str, Any]]
    atom_idx: torch.Tensor
    pad_mask_cpu: torch.Tensor
    n_samples_requested: int
    n_samples_generated: int
    csp_indices: torch.Tensor | None = None


@dataclass
class SamplingOutcome:
    dng_payload: dict[str, float] = field(default_factory=dict)
    csp_payloads: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# EMA context manager
# ---------------------------------------------------------------------------


@contextmanager
def maybe_use_ema(
    model: torch.nn.Module,
    ema: Any | None,
    use_ema: bool,
) -> Iterator[None]:
    backup = None
    try:
        if use_ema and ema is not None:
            backup = ema.apply(model)
        yield
    finally:
        if backup is not None:
            ema.restore(model, backup)


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def generate_sampling_batch(
    request: SamplingRequest,
    ctx: SamplingContext,
) -> SamplingBatch:
    args = ctx.args
    device = ctx.device

    requested_samples = max(int(args.sample_vis_count), int(request.metrics_count))
    if (
        request.is_csp
        and args.csp_precise_topk_list
        and int(args.csp_precise_topk_samples) > 0
    ):
        requested_samples = max(requested_samples, int(args.csp_precise_topk_samples))

    n_samples = requested_samples
    num_atoms = None
    if str(args.atom_count_strategy) == "empirical" and ctx.count_probs is not None:
        probs = ctx.count_probs.to(device=device, dtype=torch.float32)
        count_gen = torch.Generator(device=device).manual_seed(int(request.base_seed))
        num_atoms = (
            torch.multinomial(
                probs,
                n_samples,
                replacement=True,
                generator=count_gen,
            )
            + 1
        )
    if num_atoms is None:
        num_atoms = torch.full((n_samples,), ctx.nmax, device=device, dtype=torch.long)

    generator = torch.Generator(device=device).manual_seed(int(request.base_seed))
    chunk_size = max(1, int(args.sample_chunk_size or n_samples))

    sample_items: list[dict[str, Any]] = []
    atom_idx_chunks: list[torch.Tensor] = []
    pad_mask_chunks: list[torch.Tensor] = []

    csp_indices: torch.Tensor | None = None
    if request.is_csp:
        csp_source_ds = request.csp_source_ds
        if len(csp_source_ds) == 0:
            raise ValueError("CSP source dataset empty, cannot sample for CSP.")
        n_samples = min(n_samples, len(csp_source_ds))
        cpu_gen = torch.Generator(device="cpu").manual_seed(int(request.base_seed))
        csp_indices = torch.randperm(len(csp_source_ds), generator=cpu_gen)[:n_samples]

    with torch.no_grad():
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)

            fixed_atom_types = None
            if request.is_csp:
                batch_indices = csp_indices[start:end].tolist()
                batch_items = [request.csp_source_ds[i] for i in batch_indices]
                a0_chunk = torch.stack([item["A0"] for item in batch_items]).to(device)
                pad_mask_chunk = (
                    torch.stack([item["pad_mask"] for item in batch_items])
                    .to(device)
                    .bool()
                )
                fixed_atom_types = ctx.type_encoding.encode_from_A0(
                    a0=a0_chunk,
                    pad_mask=pad_mask_chunk,
                )
            else:
                pad_mask_chunk = _pad_mask_from_counts(
                    num_atoms[start:end],
                    nmax=ctx.nmax,
                ).to(device)

            samples = edm_sampler(
                model=ctx.model,
                pad_mask=pad_mask_chunk,
                type_dim=ctx.type_encoding.type_dim,
                num_steps=args.sample_num_steps,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                rho=args.rho,
                S_churn=args.S_churn,
                S_min=args.S_min,
                S_max=args.S_max,
                S_noise=args.S_noise,
                sigma_data_type=args.sigma_data_type,
                sigma_data_coord=args.sigma_data_coord,
                sigma_data_lat=args.sigma_data_lattice,
                generator=generator,
                autocast_dtype=ctx.args.bf16 and torch.bfloat16 or None,
                fixed_atom_types=fixed_atom_types,
                skip_type_scaling=request.is_csp,
                aa_frac_max_scale=args.aa_frac_max_scale,
                aa_rho_types=args.aa_rho_types,
                aa_rho_coords=args.aa_rho_coords,
                aa_rho_lattice=args.aa_rho_lattice,
                lattice_repr=args.lattice_repr,
            )

            pad_mask_cpu_chunk = pad_mask_chunk.to("cpu")
            real_mask_chunk = ~pad_mask_cpu_chunk

            if request.is_csp:
                batch_indices = csp_indices[start:end].tolist()
                atom_idx = torch.stack(
                    [request.csp_source_ds[i]["A0"] for i in batch_indices]
                ).to("cpu")
            else:
                atom_idx = ctx.type_encoding.decode_logits_to_A0(
                    type_logits=samples["type"].detach().cpu(),
                    pad_mask=pad_mask_cpu_chunk,
                    allowed_mask=ctx.train_allowed_mask,
                )

            atom_idx = torch.where(
                real_mask_chunk, atom_idx, torch.zeros_like(atom_idx)
            )
            frac_coords = mod1(samples["frac"].detach().cpu() + 0.5).clamp(0.0, 1.0)
            frac_coords = torch.where(
                real_mask_chunk[..., None],
                frac_coords,
                torch.zeros_like(frac_coords),
            )

            lattice_latent = clamp_lattice_latent(
                samples["lat"].detach().cpu(),
                args.lattice_repr,
            )
            lattice = lattice_latent_to_y1(
                lattice_latent,
                lattice_repr=args.lattice_repr,
            )
            lattice = clamp_lattice_latent(lattice, lattice_repr="y1")

            for i in range(atom_idx.shape[0]):
                sample_items.append(
                    {
                        "A0": atom_idx[i],
                        "F1": frac_coords[i],
                        "Y1": lattice[i],
                        "pad_mask": pad_mask_cpu_chunk[i],
                    }
                )
            atom_idx_chunks.append(atom_idx)
            pad_mask_chunks.append(pad_mask_cpu_chunk)

    pad_mask_cpu = (
        torch.cat(pad_mask_chunks, dim=0)
        if pad_mask_chunks
        else torch.empty((0, ctx.nmax), dtype=torch.bool)
    )
    atom_idx = (
        torch.cat(atom_idx_chunks, dim=0)
        if atom_idx_chunks
        else torch.empty((0, ctx.nmax), dtype=torch.long)
    )

    return SamplingBatch(
        sample_items=sample_items,
        atom_idx=atom_idx,
        pad_mask_cpu=pad_mask_cpu,
        n_samples_requested=requested_samples,
        n_samples_generated=len(sample_items),
        csp_indices=csp_indices,
    )


def _generate_csp_items_for_indices(
    target_indices: torch.Tensor,
    csp_source_ds: Any,
    ctx: SamplingContext,
    seed: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    """Generate one new set of CSP samples for specific indices (used for top-k)."""
    args = ctx.args
    device = ctx.device
    items: list[dict[str, Any]] = []
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        for start in range(0, len(target_indices), chunk_size):
            end = min(start + chunk_size, len(target_indices))
            batch_indices = target_indices[start:end].tolist()
            batch_items = [csp_source_ds[i] for i in batch_indices]
            a0_chunk = torch.stack([item["A0"] for item in batch_items]).to(device)
            pad_mask_chunk = (
                torch.stack([item["pad_mask"] for item in batch_items])
                .to(device)
                .bool()
            )
            fixed_atom_types = ctx.type_encoding.encode_from_A0(
                a0=a0_chunk, pad_mask=pad_mask_chunk
            )
            samples = edm_sampler(
                model=ctx.model,
                pad_mask=pad_mask_chunk,
                type_dim=ctx.type_encoding.type_dim,
                num_steps=args.sample_num_steps,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                rho=args.rho,
                S_churn=args.S_churn,
                S_min=args.S_min,
                S_max=args.S_max,
                S_noise=args.S_noise,
                sigma_data_type=args.sigma_data_type,
                sigma_data_coord=args.sigma_data_coord,
                sigma_data_lat=args.sigma_data_lattice,
                generator=generator,
                autocast_dtype=args.bf16 and torch.bfloat16 or None,
                fixed_atom_types=fixed_atom_types,
                skip_type_scaling=True,
                aa_frac_max_scale=args.aa_frac_max_scale,
                aa_rho_types=args.aa_rho_types,
                aa_rho_coords=args.aa_rho_coords,
                aa_rho_lattice=args.aa_rho_lattice,
                lattice_repr=args.lattice_repr,
            )
            pad_mask_cpu = pad_mask_chunk.to("cpu")
            real_mask = ~pad_mask_cpu
            atom_idx = torch.stack([csp_source_ds[i]["A0"] for i in batch_indices]).to(
                "cpu"
            )
            atom_idx = torch.where(real_mask, atom_idx, torch.zeros_like(atom_idx))
            frac_coords = mod1(samples["frac"].detach().cpu() + 0.5).clamp(0.0, 1.0)
            frac_coords = torch.where(
                real_mask[..., None], frac_coords, torch.zeros_like(frac_coords)
            )
            lattice_latent = clamp_lattice_latent(
                samples["lat"].detach().cpu(), args.lattice_repr
            )
            lattice = lattice_latent_to_y1(
                lattice_latent, lattice_repr=args.lattice_repr
            )
            lattice = clamp_lattice_latent(lattice, lattice_repr="y1")
            for i in range(atom_idx.shape[0]):
                items.append(
                    {
                        "A0": atom_idx[i],
                        "F1": frac_coords[i],
                        "Y1": lattice[i],
                        "pad_mask": pad_mask_cpu[i],
                    }
                )
    return items


# ---------------------------------------------------------------------------
# DNG evaluation
# ---------------------------------------------------------------------------


def evaluate_dng_sampling_batch(
    batch: SamplingBatch,
    request: SamplingRequest,
    ctx: SamplingContext,
) -> SamplingOutcome:
    args = ctx.args
    step = request.step
    tag = request.tag
    base_seed = request.base_seed
    metrics_count = request.metrics_count
    n_samples = batch.n_samples_generated
    sample_items = batch.sample_items

    outcome = SamplingOutcome()

    # Element histogram vs training distribution
    _log_element_histogram(
        tag=tag,
        atom_types=batch.atom_idx,
        pad_mask=batch.pad_mask_cpu,
        step=step,
        enabled=ctx.wandb_enabled,
        train_distribution=ctx.train_element_dist,
    )

    # Structure statistics vs reference
    if ctx.ref_stats is not None and sample_items:
        stats_result = compute_structure_stats_metrics(
            sample_items,
            total_count=n_samples,
        )
        plot_sample_vs_ref_stats(
            tag=tag,
            sample_stats=stats_result.sample_stats or {},
            ref_stats=ctx.ref_stats,
            step=step,
            enabled=ctx.wandb_enabled,
        )

    # Evaluator-backed validity + diagnostics + Wasserstein
    evaluator_result = compute_evaluator_metrics(
        sample_items,
        enabled=ctx.enable_evaluator_metrics,
        limit=metrics_count,
        ref_structs=ctx.ref_structs or [],
        sample_seed=args.sample_seed,
        include_wasserstein=True,
        wasserstein_max_samples=max(int(metrics_count), 1),
    )
    pred_crys_list = evaluator_result.pred_crys_list
    if evaluator_result.eval_count > 0:
        log_payload: dict[str, float] = {
            f"{tag}/valid_rate": float(evaluator_result.valid_rate or 0.0),
            f"{tag}/comp_valid_rate": float(evaluator_result.comp_valid_rate or 0.0),
            f"{tag}/struct_valid_rate": float(
                evaluator_result.struct_valid_rate or 0.0
            ),
        }
        dist_payload: dict[str, float] = {}
        if evaluator_result.dist_metrics:
            if "sample_dist/wdist_nary" in evaluator_result.dist_metrics:
                dist_payload[f"{tag}/wdist_nary"] = evaluator_result.dist_metrics[
                    "sample_dist/wdist_nary"
                ]
            if "sample_dist/wdist_density_mass" in evaluator_result.dist_metrics:
                dist_payload[f"{tag}/wdist_density_mass"] = evaluator_result.dist_metrics[
                    "sample_dist/wdist_density_mass"
                ]

        log_metrics(
            {**log_payload, **dist_payload}, step=step, enabled=ctx.wandb_enabled
        )

    # Canonical novelty/uniqueness + SUN/MSUN.
    novelty_result = compute_novelty_metrics(
        sample_items,
        ctx.novelty_ref_structs,
        limit=metrics_count,
        minimum_nary=1,
    )
    if novelty_result.novelty_metrics:
        novelty_metrics = novelty_result.novelty_metrics
        novelty_payload: dict[str, float] = {
            f"{tag}/unique_rate": float(novelty_result.unique_rate or 0.0),
            f"{tag}/novel_rate": float(novelty_result.novel_rate or 0.0),
            f"{tag}/un_rate": float(novelty_result.un_rate or 0.0),
        }
        log_metrics(novelty_payload, step=step, enabled=ctx.wandb_enabled)

        if ctx.thermo_logger is not None and step >= args.no_thermo_before_steps:
            sun_target = args.sun_k
            if sun_target > 0:
                sun_result = compute_sun_metrics(
                    novelty_metrics,
                    thermo_logger=ctx.thermo_logger,
                    tag=tag,
                    step=step,
                    enabled=ctx.wandb_enabled,
                    base_seed=base_seed,
                    sun_target=int(sun_target),
                    show_progress=True,
                )
                if sun_result.thermo_metrics:
                    log_metrics(
                        sun_result.thermo_metrics,
                        step=step,
                        enabled=ctx.wandb_enabled,
                    )
                if sun_result.summary_metrics:
                    log_metrics(
                        sun_result.summary_metrics,
                        step=step,
                        enabled=ctx.wandb_enabled,
                    )
                outcome.dng_payload.update(sun_result.dng_payload)

    # Standalone thermo (when sun_k <= 0) + reference comparison
    if (
        ctx.thermo_logger is not None
        and step >= args.no_thermo_before_steps
        and args.sun_k <= 0
    ):
        thermo_count = (
            args.thermo_stability_count
            if args.thermo_stability_count > 0
            else metrics_count
        )
        thermo_count = min(int(thermo_count), n_samples)
        if thermo_count > 0:
            thermo_structs = collect_constructed_structures(
                sample_items,
                pred_crys_list=pred_crys_list,
                count=thermo_count,
            )
            thermo_metrics = _compute_thermo_metrics(
                ctx.thermo_logger,
                thermo_structs,
                tag=tag,
                step=step,
                enabled=ctx.wandb_enabled,
                show_progress=True,
            )
            if thermo_metrics:
                log_metrics(thermo_metrics, step=step, enabled=ctx.wandb_enabled)

                if ctx.ref_structs:
                    ref_count = min(int(thermo_count), len(ctx.ref_structs))
                    if ref_count > 0:
                        backend = (
                            str(getattr(ctx.thermo_logger, "thermo_backend", "thermo"))
                            .strip()
                            .lower()
                        )
                        ref_tag = f"{tag}/reference"
                        cache_key = (ref_tag, ref_count, backend)
                        ref_metrics = ctx.thermo_reference_cache.get(cache_key)
                        if ref_metrics is None:
                            ref_struct_subset = [
                                s for s in ctx.ref_structs[:ref_count] if s is not None
                            ]
                            ref_metrics = _compute_thermo_metrics(
                                ctx.thermo_logger,
                                ref_struct_subset,
                                tag=ref_tag,
                                step=step,
                                enabled=ctx.wandb_enabled,
                                show_progress=False,
                            )
                            ctx.thermo_reference_cache[cache_key] = ref_metrics

                        compare_payload = build_reference_thermo_comparison_metrics(
                            tag=tag,
                            ref_tag=ref_tag,
                            backend=backend,
                            thermo_metrics=thermo_metrics,
                            ref_metrics=ref_metrics,
                        )
                        if compare_payload:
                            log_metrics(
                                compare_payload, step=step, enabled=ctx.wandb_enabled
                            )

    return outcome


# ---------------------------------------------------------------------------
# CSP evaluation
# ---------------------------------------------------------------------------


def evaluate_csp_sampling_batch(
    batch: SamplingBatch,
    request: SamplingRequest,
    ctx: SamplingContext,
) -> SamplingOutcome:
    args = ctx.args
    step = request.step
    tag = request.tag
    base_seed = request.base_seed
    sample_items = batch.sample_items
    csp_indices = batch.csp_indices
    csp_source_ds = request.csp_source_ds
    csp_source_label = request.csp_source_label

    outcome = SamplingOutcome()

    if not sample_items or csp_indices is None:
        return outcome

    # Build pred/gt Crystal lists and compute RecEval
    pred_crys = []
    gt_crys = []
    for i, item in enumerate(sample_items):
        gt_item = csp_source_ds[csp_indices[i].item()]
        pred_crys.append(_safe_crystal_from_tokens(item, i))
        gt_crys.append(_safe_crystal_from_tokens(gt_item, i))

    rec_eval = RecEval(pred_crys, gt_crys)
    rec_metrics = rec_eval.get_metrics()
    match_rate = float(rec_metrics["match_rate"])
    mean_rms = float(rec_metrics["rms_dist"])
    log_metrics(
        {
            f"{tag}/csp_{csp_source_label}_match_rate": match_rate,
            f"{tag}/csp_{csp_source_label}_mean_rms": mean_rms,
        },
        step=step,
        enabled=ctx.wandb_enabled,
    )
    outcome.csp_payloads.append(
        {
            "csp_source_label": csp_source_label,
            "match_rate": match_rate,
            "mean_rms": mean_rms,
        }
    )

    # Top-k evaluation: generate additional candidates per target
    if (
        args.csp_precise_topk_list
        and args.csp_precise_topk_samples > 0
    ):
        topk_list = _normalize_topk_list(args.csp_precise_topk_list)
        k_max = topk_list[-1]
        topk_target_count = _compute_topk_target_count(
            len(sample_items), args.csp_precise_topk_samples
        )
        if topk_target_count <= 0:
            print(
                f"[csp-topk] Skipping {tag}/{csp_source_label}: "
                f"--csp_precise_topk_samples={args.csp_precise_topk_samples} "
                "left zero available targets."
            )
        else:
            if topk_target_count < args.csp_precise_topk_samples:
                print(
                    f"[csp-topk] Truncating {tag}/{csp_source_label} targets "
                    f"from requested {args.csp_precise_topk_samples} to "
                    f"{topk_target_count} due to available samples."
                )
            target_indices = csp_indices[:topk_target_count]
            gt_topk = [
                _safe_crystal_from_tokens(csp_source_ds[target_indices[i].item()], i)
                for i in range(topk_target_count)
            ]
            pred_topk_batches = [
                [
                    _safe_crystal_from_tokens(sample_items[i], i)
                    for i in range(topk_target_count)
                ]
            ]
            chunk_size = max(1, int(args.sample_chunk_size or topk_target_count))
            with maybe_use_ema(ctx.model, ctx.ema, request.use_ema):
                tag_seed = sum(ord(c) for c in tag)
                label_seed = sum(ord(c) for c in csp_source_label)
                for candidate_idx in range(1, k_max):
                    candidate_seed = (
                        int(base_seed)
                        + tag_seed * 1009
                        + label_seed * 9176
                        + candidate_idx * 1000003
                    )
                    candidate_items = _generate_csp_items_for_indices(
                        target_indices=target_indices,
                        csp_source_ds=csp_source_ds,
                        ctx=ctx,
                        seed=candidate_seed,
                        chunk_size=chunk_size,
                    )
                    pred_topk_batches.append(
                        [
                            _safe_crystal_from_tokens(item, i)
                            for i, item in enumerate(candidate_items)
                        ]
                    )

            rec_topk = RecEvalBatch(pred_topk_batches, gt_topk)
            topk_payload: dict[str, float] = {}
            for k in topk_list:
                topk_metrics = rec_topk.get_match_rate_and_rms_for_k(k)
                topk_payload[f"{tag}/csp_{csp_source_label}_top{k}_match_rate"] = float(
                    topk_metrics["match_rate"]
                )
                topk_payload[f"{tag}/csp_{csp_source_label}_top{k}_mean_rms"] = float(
                    topk_metrics["rms_dist"]
                )
            log_metrics(topk_payload, step=step, enabled=ctx.wandb_enabled)

    return outcome


# ---------------------------------------------------------------------------
# Artifact saving + entry point
# ---------------------------------------------------------------------------


def save_sampling_artifacts(
    batch: SamplingBatch,
    request: SamplingRequest,
    ctx: SamplingContext,
) -> None:
    csp_suffix = f"_{request.csp_source_label}" if request.is_csp else ""
    sample_step_dir = (
        ctx.sample_dir / f"{request.tag}{csp_suffix}_step_{request.step:07d}"
    )
    images = _save_sample_images(
        sample_step_dir,
        batch.sample_items,
        ctx.args.sample_vis_count,
        ctx.ase_view,
    )
    image_tag = (
        f"{request.tag}/images{csp_suffix}"
        if request.is_csp
        else f"{request.tag}/images"
    )
    log_images(
        image_tag,
        images,
        step=request.step,
        enabled=ctx.wandb_enabled,
    )


def run_sampling_request(
    request: SamplingRequest,
    ctx: SamplingContext,
) -> SamplingBatch:
    with maybe_use_ema(ctx.model, ctx.ema, request.use_ema):
        return generate_sampling_batch(request, ctx)
