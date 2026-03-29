"""Stability evaluation: CHGNet relaxation metrics and thermodynamic stability metrics."""

# Copyright (c) Meta Platforms, Inc. and affiliates. (original chgnet_ portions)

from __future__ import annotations

import dataclasses
import pickle
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from chgnet.model import StructOptimizer
from chgnet.model.dynamics import TrajectoryObserver
from chgnet.model.model import CHGNet
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from torch import Tensor
from tqdm import tqdm

if TYPE_CHECKING:
    from src.utils.stability_logger import StabilityLogger


# ---------------------------------------------------------------------------
# Phase diagram utilities
# ---------------------------------------------------------------------------


def load_phase_diagram(ppd_path: Path | str):
    with open(ppd_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CHGNet relaxation pair
# ---------------------------------------------------------------------------


@dataclass
class UnrelaxedRelaxedStructurePair:
    """Collector of endpoints of relaxations. Values are in ase units."""

    structure_dicts: tuple[dict, dict]
    energies: tuple[float, float]
    n_steps_to_relax: int
    stol: float = 0.5
    angle_tol: int = 10
    ltol: float = 0.3

    def __post_init__(self):
        self.matcher = StructureMatcher(
            stol=self.stol,
            angle_tol=self.angle_tol,
            ltol=self.ltol,
        )

    @cached_property
    def structures(self) -> tuple[Structure, Structure]:
        return tuple(Structure.from_dict(sd) for sd in self.structure_dicts)

    @cached_property
    def atoms(self) -> tuple[Atoms, Atoms]:
        return tuple(
            AseAtomsAdaptor.get_atoms(structure) for structure in self.structures
        )

    @cached_property
    def match(self) -> bool:
        return self.rms_dist is not None

    @cached_property
    def rms_dist(self) -> float | None:
        out = self.matcher.get_rms_dist(self.structures[0], self.structures[1])
        if out is None:
            return None
        if isinstance(out, tuple):
            return out[0]
        raise ValueError(f"Unexpected output from StructureMatcher.get_rms_dist: {out}")

    @classmethod
    def from_chgnet(
        cls,
        initial_structure: Structure,
        prediction: dict[str, Tensor],
        relaxation: dict[str, Structure | TrajectoryObserver],
    ):
        initial_structure.add_site_property("magmom", prediction["m"])
        final_structure = relaxation["final_structure"]
        trajectory = relaxation["trajectory"]
        return cls(
            structure_dicts=(initial_structure.as_dict(), final_structure.as_dict()),
            energies=(
                prediction["e"] * initial_structure.num_sites,
                trajectory.energies[-1],
            ),
            n_steps_to_relax=len(trajectory.energies),
        )


def compute_relaxation_metrics(
    initial_structure: Structure,
    chgnet_model: CHGNet,
    relaxer: StructOptimizer,
    steps: int = 200,
) -> dict:
    """Take an initial structure, relax it with CHGNet, and return CSP metrics."""
    try:
        prediction = chgnet_model.predict_structure(initial_structure)
        relaxation = relaxer.relax(initial_structure, steps=steps, verbose=False)
        pair = UnrelaxedRelaxedStructurePair.from_chgnet(
            initial_structure, prediction, relaxation
        )
        return {
            "csp_match": pair.match,
            "csp_rms_dist": pair.rms_dist,
            "csp_steps_to_relax": pair.n_steps_to_relax,
            "csp_e_gen": pair.energies[0],
            "csp_e_relax": pair.energies[1],
            "csp_success": True,
        }
    except Exception as e:
        return {
            "csp_match": False,
            "csp_rms_dist": None,
            "csp_steps_to_relax": None,
            "csp_e_gen": None,
            "csp_e_relax": None,
            "csp_success": False,
            "csp_error": str(e),
        }


# ---------------------------------------------------------------------------
# Thermodynamic metrics helpers
# ---------------------------------------------------------------------------


def _aggregate_thermo_records(records: list[dict], tag: str) -> dict[str, float]:
    t_checked = t_success = t_failed = t_divergence = 0.0
    t_stable = t_meta = t_e_sum = 0.0
    fail_reason_totals: dict[str, float] = {}
    for rec in records:
        for k, v in rec.items():
            if not k.endswith("/thermo_checked"):
                continue
            prefix = k[: -len("/thermo_checked")]
            success = rec.get(f"{prefix}/thermo_success", 0.0)
            failed = rec.get(
                f"{prefix}/thermo_failed", max(0.0, float(v) - float(success))
            )
            divergence = rec.get(f"{prefix}/thermo_divergence", 0.0)
            stable_count = rec.get(
                f"{prefix}/thermo_stable_count",
                float(rec.get(f"{prefix}/thermo_stable_rate", 0.0)) * float(v),
            )
            meta_count = rec.get(
                f"{prefix}/thermo_metastable_count",
                float(rec.get(f"{prefix}/thermo_metastable_rate", 0.0)) * float(v),
            )
            e_mean = rec.get(f"{prefix}/thermo_e_above_hull_mean", None)
            t_checked += float(v)
            t_success += float(success)
            t_failed += float(failed)
            t_divergence += float(divergence)
            t_stable += float(stable_count)
            t_meta += float(meta_count)
            if e_mean is not None:
                t_e_sum += float(e_mean) * float(success)
            fail_prefix = f"{prefix}/thermo_fail/"
            for fail_key, fail_value in rec.items():
                if fail_key.startswith(fail_prefix):
                    suffix = fail_key[len(prefix) + 1 :]
                    fail_reason_totals[suffix] = fail_reason_totals.get(
                        suffix, 0.0
                    ) + float(fail_value)

    out: dict[str, float] = {}
    if t_checked > 0:
        out[f"{tag}/thermo_checked_total"] = float(t_checked)
        out[f"{tag}/thermo_success_total"] = float(t_success)
        out[f"{tag}/thermo_failed_total"] = float(t_failed)
        out[f"{tag}/thermo_divergence_total"] = float(t_divergence)
        out[f"{tag}/thermo_success_rate"] = float(t_success) / float(t_checked)
        out[f"{tag}/thermo_failure_rate"] = float(t_failed) / float(t_checked)
        out[f"{tag}/thermo_stable_rate"] = float(t_stable) / float(t_checked)
        out[f"{tag}/thermo_metastable_rate"] = float(t_meta) / float(t_checked)
    if t_success > 0:
        out[f"{tag}/thermo_e_above_hull_mean"] = float(t_e_sum) / float(t_success)
    for fail_reason, fail_total in fail_reason_totals.items():
        out[f"{tag}/{fail_reason}"] = float(fail_total)
    return out


def _compute_thermo_metrics(
    logger: StabilityLogger | None,
    structures: list,
    *,
    tag: str,
    step: int,
    enabled: bool,
    show_progress: bool = False,
) -> dict[str, float]:
    if logger is None or not enabled or not structures:
        return {}
    cfg = logger.thermo_cfg
    if cfg is None:
        return {}
    records: list[dict] = []

    def log_fn(payload, step=None, enabled=True):
        records.append({k: float(v) for k, v in payload.items()})

    orig_batch = cfg.batch_size
    iterator = range(0, len(structures), orig_batch)
    if show_progress:
        iterator = tqdm(iterator, desc=f"{tag}/thermo_relax", dynamic_ncols=True)
    for start in iterator:
        end = min(len(structures), start + orig_batch)
        cfg.batch_size = end - start
        logger.update(
            structures[start:end], tag=tag, step=step, log_fn=log_fn, enabled=True
        )
    cfg.batch_size = orig_batch

    metrics = _aggregate_thermo_records(records, tag)
    if not metrics:
        metrics = {
            f"{tag}/thermo_checked_total": float(len(structures)),
            f"{tag}/thermo_success_total": 0.0,
            f"{tag}/thermo_failed_total": float(len(structures)),
            f"{tag}/thermo_divergence_total": 0.0,
            f"{tag}/thermo_success_rate": 0.0,
            f"{tag}/thermo_failure_rate": 1.0,
            f"{tag}/thermo_stable_rate": 0.0,
            f"{tag}/thermo_metastable_rate": 0.0,
        }

    backend = str(getattr(logger, "thermo_backend", "thermo")).strip().lower()
    checked_total = float(metrics.get(f"{tag}/thermo_checked_total", 0.0))
    success_total = float(metrics.get(f"{tag}/thermo_success_total", 0.0))
    failed_total = float(
        metrics.get(
            f"{tag}/thermo_failed_total",
            max(0.0, checked_total - success_total),
        )
    )
    divergence_total = float(metrics.get(f"{tag}/thermo_divergence_total", 0.0))
    success_rate = (
        float(metrics.get(f"{tag}/thermo_success_rate", 0.0))
        if checked_total > 0
        else 0.0
    )
    failure_rate = (
        float(metrics.get(f"{tag}/thermo_failure_rate", 0.0))
        if checked_total > 0
        else 0.0
    )
    stable_rate = float(metrics.get(f"{tag}/thermo_stable_rate", 0.0))
    metastable_rate = float(metrics.get(f"{tag}/thermo_metastable_rate", 0.0))

    metrics[f"{tag}/{backend}_stability/checked_total"] = checked_total
    metrics[f"{tag}/{backend}_stability/success_total"] = success_total
    metrics[f"{tag}/{backend}_stability/failed_total"] = failed_total
    metrics[f"{tag}/{backend}_stability/divergence_total"] = divergence_total
    metrics[f"{tag}/{backend}_stability/success_rate"] = success_rate
    metrics[f"{tag}/{backend}_stability/failure_rate"] = failure_rate
    metrics[f"{tag}/{backend}_stability/stability"] = stable_rate
    metrics[f"{tag}/{backend}_stability/metastability"] = metastable_rate

    e_key = f"{tag}/thermo_e_above_hull_mean"
    if e_key in metrics:
        metrics[f"{tag}/{backend}_stability/e_above_hull_mean"] = float(metrics[e_key])
    return metrics
