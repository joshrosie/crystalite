"""Shared helpers for de novo generation (DNG) evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.data.mp20_tokens import tokens_to_eval_dict, tokens_to_structure as _tokens_to_structure
from src.eval.crystal import array_dict_to_crystal
from src.eval.diagnostics import _compute_sample_diagnostics
from src.eval.stability import _compute_thermo_metrics
from src.eval.uniqueness_novelty import compute_uniqueness_novelty
from src.eval.wasserstein import _compute_wasserstein_metrics
from src.utils.sample_stats import collect_structure_stats


@dataclass
class StructureStatsResult:
    metrics: dict[str, float] = field(default_factory=dict)
    sample_stats: dict[str, np.ndarray] | None = None
    invalid_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluatorMetricsResult:
    valid_rate: float | None = None
    comp_valid_rate: float | None = None
    struct_valid_rate: float | None = None
    diag_metrics: dict[str, float] = field(default_factory=dict)
    dist_metrics: dict[str, float] = field(default_factory=dict)
    pred_crys_list: list[Any] = field(default_factory=list)
    eval_count: int = 0


@dataclass
class NoveltyMetricsResult:
    unique_rate: float | None = None
    novel_rate: float | None = None
    un_rate: float | None = None
    novelty_metrics: dict[str, Any] = field(default_factory=dict)
    novelty_structs: list[Any] = field(default_factory=list)
    eval_count: int = 0


@dataclass
class SunMetricsResult:
    thermo_metrics: dict[str, float] = field(default_factory=dict)
    summary_metrics: dict[str, float] = field(default_factory=dict)
    dng_payload: dict[str, float] = field(default_factory=dict)


def float_or_nan(x: Any) -> float:
    try:
        out = float(x)
    except Exception:
        return float("nan")
    if not math.isfinite(out):
        return float("nan")
    return out


def _metric_key(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


def _add_np_stats(metrics: dict[str, float], name: str, arr: np.ndarray) -> None:
    if arr.size == 0:
        return
    vals = arr.reshape(-1)
    metrics[f"{name}_mean"] = float_or_nan(vals.mean())
    metrics[f"{name}_min"] = float_or_nan(vals.min())
    metrics[f"{name}_max"] = float_or_nan(vals.max())


def _is_finite_structure(struct) -> bool:
    """Best-effort filter for invalid structures (avoid matcher failures)."""
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


def compute_structure_stats_metrics(
    sample_items: list[dict[str, Any]],
    *,
    total_count: int | None = None,
    include_summary_stats: bool = False,
) -> StructureStatsResult:
    invalid_summary: dict[str, Any] = {}
    sample_stats = collect_structure_stats(
        sample_items,
        invalid_summary=invalid_summary,
    )

    metrics: dict[str, float] = {}
    denom = max(1, int(total_count if total_count is not None else len(sample_items)))
    invalid_total = int(invalid_summary.get("invalid_total", 0))
    metrics["invalid_total"] = float(invalid_total)
    metrics["invalid_rate"] = float(invalid_total) / float(denom)
    for reason, count in invalid_summary.get("invalid_reasons", {}).items():
        metrics[f"invalid_reason/{reason}"] = float(count)

    if include_summary_stats:
        _add_np_stats(metrics, "length", sample_stats.get("lengths", np.zeros((0, 3))))
        _add_np_stats(metrics, "angle", sample_stats.get("angles", np.zeros((0, 3))))
        _add_np_stats(
            metrics,
            "volume",
            sample_stats.get("volumes", np.zeros((0,), dtype=np.float32)),
        )
        _add_np_stats(
            metrics,
            "volume_per_atom",
            sample_stats.get("volumes_per_atom", np.zeros((0,), dtype=np.float32)),
        )
        _add_np_stats(
            metrics,
            "min_dist",
            sample_stats.get("min_dists", np.zeros((0,), dtype=np.float32)),
        )
        _add_np_stats(
            metrics,
            "num_atoms",
            sample_stats.get("num_atoms", np.zeros((0,), dtype=np.float32)),
        )

    return StructureStatsResult(
        metrics=metrics,
        sample_stats=sample_stats,
        invalid_summary=invalid_summary,
    )


def compute_evaluator_metrics(
    sample_items: list[dict[str, Any]],
    *,
    enabled: bool = True,
    limit: int,
    ref_structs: list[Any] | None,
    sample_seed: int,
    include_diagnostics: bool = True,
    include_wasserstein: bool = True,
    wasserstein_max_samples: int | None = None,
) -> EvaluatorMetricsResult:
    result = EvaluatorMetricsResult()
    if not enabled or limit <= 0 or not sample_items:
        return result

    eval_count = min(int(limit), len(sample_items))
    result.eval_count = eval_count
    crystals = [
        array_dict_to_crystal(tokens_to_eval_dict(sample_items[i], sample_idx=i))
        for i in range(eval_count)
    ]
    result.pred_crys_list = crystals
    result.valid_rate = sum(c.valid for c in crystals) / eval_count
    result.comp_valid_rate = sum(c.comp_valid for c in crystals) / eval_count
    result.struct_valid_rate = sum(c.struct_valid for c in crystals) / eval_count

    if include_diagnostics:
        result.diag_metrics = _compute_sample_diagnostics(result.pred_crys_list)

    if include_wasserstein and ref_structs:
        max_samples = (
            eval_count
            if wasserstein_max_samples is None
            else min(int(wasserstein_max_samples), eval_count)
        )
        result.dist_metrics = _compute_wasserstein_metrics(
            result.pred_crys_list,
            ref_structs,
            max_samples=min(max(max_samples, 1), eval_count),
            seed=sample_seed,
        )

    return result


def compute_novelty_metrics(
    sample_items: list[dict[str, Any]],
    novelty_ref_structs: list[Any] | None,
    *,
    limit: int,
    minimum_nary: int = 1,
) -> NoveltyMetricsResult:
    result = NoveltyMetricsResult()
    if limit <= 0 or not sample_items:
        return result

    result.eval_count = min(int(limit), len(sample_items))
    for item in sample_items[: result.eval_count]:
        try:
            result.novelty_structs.append(_tokens_to_structure(item))
        except Exception:
            continue

    if not result.novelty_structs or not novelty_ref_structs:
        return result

    result.novelty_metrics = compute_uniqueness_novelty(
        result.novelty_structs,
        novelty_ref_structs,
        minimum_nary=minimum_nary,
    )
    result.unique_rate = float(result.novelty_metrics["unique_rate"])
    result.novel_rate = float(result.novelty_metrics["novel_rate"])
    result.un_rate = float(result.novelty_metrics["un_rate"])
    return result


def collect_constructed_structures(
    sample_items: list[dict[str, Any]],
    *,
    pred_crys_list: list[Any] | None = None,
    count: int,
) -> list[Any]:
    if count <= 0:
        return []

    thermo_structs: list[Any] = []
    if pred_crys_list:
        for crys in pred_crys_list:
            if getattr(crys, "constructed", False):
                thermo_structs.append(crys.structure)
            if len(thermo_structs) >= count:
                return thermo_structs
        return thermo_structs[:count]

    for item in sample_items[:count]:
        try:
            thermo_structs.append(_tokens_to_structure(item))
        except Exception:
            continue
    return thermo_structs[:count]


def compute_sun_metrics(
    novelty_metrics: dict[str, Any],
    *,
    thermo_logger: Any,
    tag: str,
    step: int,
    enabled: bool,
    base_seed: int,
    sun_target: int,
    show_progress: bool = True,
) -> SunMetricsResult:
    result = SunMetricsResult()
    if thermo_logger is None or sun_target <= 0:
        return result

    un_rate = float(novelty_metrics.get("un_rate", 0.0))
    sun_key = _metric_key(tag, "SUN")
    msun_key = _metric_key(tag, "MSUN")
    sun_structs = list(novelty_metrics.get("un_structs", []))
    sun_tag = _metric_key(tag, "sun")

    if not sun_structs:
        if un_rate >= 0.0:
            result.summary_metrics = {sun_key: 0.0, msun_key: 0.0}
            result.dng_payload["MSUN"] = 0.0
        return result

    sun_k = min(int(sun_target), len(sun_structs))
    if 0 < sun_k < len(sun_structs):
        tag_seed = sum(ord(c) for c in tag)
        rng = np.random.default_rng(base_seed + tag_seed)
        idx = rng.choice(len(sun_structs), size=sun_k, replace=False)
        sun_structs = [sun_structs[i] for i in idx]

    result.thermo_metrics = _compute_thermo_metrics(
        thermo_logger,
        sun_structs,
        tag=sun_tag,
        step=step,
        enabled=enabled,
        show_progress=show_progress,
    )

    stable_rate = result.thermo_metrics.get(f"{sun_tag}/thermo_stable_rate", 0.0)
    metastable_rate = result.thermo_metrics.get(
        f"{sun_tag}/thermo_metastable_rate", 0.0
    )
    if un_rate >= 0.0:
        msun_val = float(metastable_rate) * float(un_rate)
        result.summary_metrics = {
            sun_key: float(stable_rate) * float(un_rate),
            msun_key: msun_val,
        }
        result.dng_payload["MSUN"] = msun_val

    return result


def compute_sun_msun_from_thermo_rates(
    *,
    un_rate: float | None,
    thermo_metrics: dict[str, float],
    thermo_tag: str,
    metric_prefix: str = "",
) -> dict[str, float]:
    if un_rate is None:
        return {}

    metrics: dict[str, float] = {}
    stable_key = f"{thermo_tag}/thermo_stable_rate"
    metastable_key = f"{thermo_tag}/thermo_metastable_rate"
    if stable_key in thermo_metrics:
        metrics[_metric_key(metric_prefix, "SUN")] = float(un_rate) * float(
            thermo_metrics[stable_key]
        )
    if metastable_key in thermo_metrics:
        metrics[_metric_key(metric_prefix, "MSUN")] = float(un_rate) * float(
            thermo_metrics[metastable_key]
        )
    return metrics


def build_reference_thermo_comparison_metrics(
    *,
    tag: str,
    ref_tag: str,
    backend: str,
    thermo_metrics: dict[str, float],
    ref_metrics: dict[str, float],
) -> dict[str, float]:
    compare_payload: dict[str, float] = {}

    gen_mean_key = f"{tag}/thermo_e_above_hull_mean"
    ref_mean_key = f"{ref_tag}/thermo_e_above_hull_mean"
    if gen_mean_key in thermo_metrics and ref_mean_key in ref_metrics:
        gen_mean = float(thermo_metrics[gen_mean_key])
        ref_mean = float(ref_metrics[ref_mean_key])
        compare_payload[f"{tag}/e_above_hull/generated_mean"] = gen_mean
        compare_payload[f"{tag}/e_above_hull/reference_mean"] = ref_mean
        compare_payload[f"{tag}/e_above_hull/generated_minus_reference"] = (
            gen_mean - ref_mean
        )

    gen_stable_key = f"{tag}/{backend}_stability/stability"
    ref_stable_key = f"{ref_tag}/{backend}_stability/stability"
    if gen_stable_key in thermo_metrics and ref_stable_key in ref_metrics:
        gen_stable = float(thermo_metrics[gen_stable_key])
        ref_stable = float(ref_metrics[ref_stable_key])
        compare_payload[f"{tag}/{backend}_stability/reference_stability"] = ref_stable
        compare_payload[
            f"{tag}/{backend}_stability/stability_minus_reference"
        ] = (gen_stable - ref_stable)

    gen_meta_key = f"{tag}/{backend}_stability/metastability"
    ref_meta_key = f"{ref_tag}/{backend}_stability/metastability"
    if gen_meta_key in thermo_metrics and ref_meta_key in ref_metrics:
        gen_meta = float(thermo_metrics[gen_meta_key])
        ref_meta = float(ref_metrics[ref_meta_key])
        compare_payload[
            f"{tag}/{backend}_stability/reference_metastability"
        ] = ref_meta
        compare_payload[
            f"{tag}/{backend}_stability/metastability_minus_reference"
        ] = (gen_meta - ref_meta)

    return compare_payload
