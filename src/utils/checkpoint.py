from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Optional
import torch


@dataclass
class _BestCandidate:
    step: int
    epoch: int
    mode: str  # "dng" or "csp"
    selection_source: str  # "primary" or "fallback"
    selector_name: str
    selector_value: float
    raw_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class BestCkptState:
    best_primary: Optional[_BestCandidate] = None
    best_fallback: Optional[_BestCandidate] = None


BEST_CKPT_SELECTOR_CHOICES = (
    "auto",
    "legacy",
    "dng_precise_msun",
    "dng_sample_msun",
    "csp_val_ratio",
    "csp_precise_val_ratio",
    "csp_sample_val_ratio",
)


def _best_ckpt_dng_order(best_ckpt_selector: str) -> list[tuple[str, str]]:
    auto = [
        ("precise_ema", "MSUN"),
        ("precise", "MSUN"),
        ("sample_ema", "MSUN"),
        ("sample", "MSUN"),
    ]
    legacy = [
        ("sample_ema", "MSUN"),
        ("sample", "MSUN"),
        ("precise_ema", "MSUN"),
        ("precise", "MSUN"),
    ]
    by_selector: dict[str, list[tuple[str, str]]] = {
        "auto": auto,
        "legacy": legacy,
        "dng_precise_msun": [("precise_ema", "MSUN"), ("precise", "MSUN")],
        "dng_sample_msun": [("sample_ema", "MSUN"), ("sample", "MSUN")],
    }
    return by_selector.get(best_ckpt_selector, auto)


def _best_ckpt_csp_tag_order(best_ckpt_selector: str) -> list[str]:
    auto = ["sample_ema", "sample", "precise_ema", "precise"]
    by_selector = {
        "auto": auto,
        "legacy": auto,
        "csp_val_ratio": auto,
        "csp_precise_val_ratio": ["precise_ema", "precise"],
        "csp_sample_val_ratio": ["sample_ema", "sample"],
    }
    return by_selector.get(best_ckpt_selector, auto)


def _build_best_candidate(
    step: int,
    epoch: int,
    mode: str,
    selection_source: str,
    selector_name: str,
    selector_value: float,
    raw_metrics: dict[str, float],
) -> Optional[_BestCandidate]:
    if not math.isfinite(selector_value):
        return None
    return _BestCandidate(
        step=step,
        epoch=epoch,
        mode=mode,
        selection_source=selection_source,
        selector_name=selector_name,
        selector_value=selector_value,
        raw_metrics=dict(raw_metrics),
    )


def build_val_fallback_candidate(
    step: int, epoch: int, mode: str, val_loss: float
) -> Optional[_BestCandidate]:
    if not math.isfinite(val_loss):
        return None
    return _BestCandidate(
        step=step,
        epoch=epoch,
        mode=mode,
        selection_source="fallback",
        selector_name="val_loss",
        selector_value=val_loss,
        raw_metrics={"val_loss": val_loss},
    )


def select_primary_candidate_from_sampling(
    is_csp: bool,
    step: int,
    epoch: int,
    dng_payloads: Optional[dict[str, dict[str, float]]] = None,
    csp_payloads: Optional[dict[str, list[dict[str, Any]]]] = None,
    best_ckpt_selector: str = "auto",
) -> Optional[_BestCandidate]:
    if not is_csp:
        # DNG: choose the first finite candidate from a selector-specific order.
        dng_payloads = dng_payloads or {}
        metric_order = ["MSUN"]
        for tag, metric in _best_ckpt_dng_order(best_ckpt_selector):
            payload = dng_payloads.get(tag)
            if payload is None:
                continue
            val = payload.get(metric)
            if val is None or not math.isfinite(val):
                continue
            raw = {
                f"{tag}/{m}": payload[m]
                for m in metric_order
                if payload.get(m) is not None
            }
            return _build_best_candidate(
                step=step,
                epoch=epoch,
                mode="dng",
                selection_source="primary",
                selector_name=f"{tag}/{metric}",
                selector_value=val,
                raw_metrics=raw,
            )
        return None
    else:
        # CSP: use val match_rate / mean_rms as selector.
        csp_payloads = csp_payloads or {}
        for tag in _best_ckpt_csp_tag_order(best_ckpt_selector):
            entries = csp_payloads.get(tag)
            if entries is None:
                continue
            val_entry = None
            for entry in entries:
                if entry.get("csp_source_label") == "val":
                    val_entry = entry
                    break
            if val_entry is None:
                continue
            match_rate = val_entry.get("match_rate")
            mean_rms = val_entry.get("mean_rms")
            if match_rate is None or mean_rms is None:
                continue
            if not math.isfinite(match_rate) or not math.isfinite(mean_rms):
                continue
            if mean_rms == 0.0:
                continue
            ratio = match_rate / mean_rms
            if not math.isfinite(ratio):
                continue
            raw = {
                f"{tag}/csp_val_match_rate": match_rate,
                f"{tag}/csp_val_mean_rms": mean_rms,
                f"{tag}/csp_val_ratio": ratio,
            }
            return _build_best_candidate(
                step=step,
                epoch=epoch,
                mode="csp",
                selection_source="primary",
                selector_name=f"{tag}/csp_val_ratio",
                selector_value=ratio,
                raw_metrics=raw,
            )
        return None


def maybe_update_best_ckpt(
    state: BestCkptState,
    candidate: _BestCandidate,
    maximize: bool,
    ckpt_dir: Path,
    build_ckpt_fn,
    enabled: bool,
) -> bool:
    """Update best checkpoint state and save to disk when improved.

    Returns True if the on-disk best.pt was updated.
    """
    if not enabled:
        return False

    is_primary = candidate.selection_source != "fallback"
    tol = 1e-12

    if is_primary:
        prev = state.best_primary
        if prev is not None:
            if maximize:
                improved = candidate.selector_value > prev.selector_value + tol
            else:
                improved = candidate.selector_value < prev.selector_value - tol
        else:
            improved = True
        if improved:
            state.best_primary = candidate
            # Save to disk
            ckpt = build_ckpt_fn(candidate.step)
            torch.save(ckpt, ckpt_dir / "best.pt")
            meta = {
                "step": candidate.step,
                "epoch": candidate.epoch,
                "mode": candidate.mode,
                "selection_source": candidate.selection_source,
                "selector_name": candidate.selector_name,
                "selector_value": candidate.selector_value,
                "raw_metrics": candidate.raw_metrics,
                "updated_at_step": candidate.step,
            }
            (ckpt_dir / "best_meta.json").write_text(json.dumps(meta, indent=2))
            return True
        return False
    else:
        # Fallback: always track the best fallback value
        prev_fb = state.best_fallback
        if prev_fb is None or (
            (not maximize and candidate.selector_value < prev_fb.selector_value - tol)
            or (maximize and candidate.selector_value > prev_fb.selector_value + tol)
        ):
            state.best_fallback = candidate

        # Only write to disk if no primary exists yet
        if state.best_primary is None:
            ckpt = build_ckpt_fn(candidate.step)
            torch.save(ckpt, ckpt_dir / "best.pt")
            meta = {
                "step": candidate.step,
                "epoch": candidate.epoch,
                "mode": candidate.mode,
                "selection_source": candidate.selection_source,
                "selector_name": candidate.selector_name,
                "selector_value": candidate.selector_value,
                "raw_metrics": candidate.raw_metrics,
                "updated_at_step": candidate.step,
            }
            (ckpt_dir / "best_meta.json").write_text(json.dumps(meta, indent=2))
            return True
        return False


def resolve_post_training_eval_ckpt(ckpt_dir: Path) -> Path | None:
    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        return best_path
    final_path = ckpt_dir / "final.pt"
    if final_path.exists():
        return final_path
    return None
