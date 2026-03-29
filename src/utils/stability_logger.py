from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from pathlib import Path

from src.eval.stability import load_phase_diagram
from src.utils.sample_stats import (
    compute_e_above_hull_uncorrected,
    compute_e_above_hull_mp2020_like,
    gamma_phonon_stability,
    make_nequip_batch_relaxer,
    make_chgnet_and_relaxer,
    make_chgnet_ase_calculator,
    make_nequip_relaxer,
    pmg_to_ase,
)


@dataclass
class _GammaConfig:
    batch_size: int
    relax_steps: int
    supercell: tuple[int, int, int]
    delta: float
    neg_tol: float
    acoustic_tol: float
    device: str | None


@dataclass
class _ThermoConfig:
    batch_size: int
    relax_steps: int
    ppd_path: str
    device: str
    ehull_method: str = "uncorrected"
    mlip: str = "chgnet"
    nequip_compile_path: str | None = None
    nequip_relax_mode: str = "sequential"
    nequip_optimizer: str = "FIRE"
    nequip_cell_filter: str = "none"
    nequip_fmax: float = 0.01
    nequip_max_force_abort: float = 1e6


class StabilityLogger:
    def __init__(
        self,
        *,
        gamma_cfg: _GammaConfig | None,
        thermo_cfg: _ThermoConfig | None,
        buffer_mult: int = 5,
    ) -> None:
        self.gamma_cfg = gamma_cfg
        self.thermo_cfg = thermo_cfg
        self.buffer_mult = max(1, int(buffer_mult))

        self.gamma_buffers: dict[str, list] = defaultdict(list)
        self.thermo_buffers: dict[str, list] = defaultdict(list)

        self._gamma_ready = False
        self._thermo_ready = False

        self._gamma_chgnet = None
        self._gamma_relaxer = None
        self._gamma_calc = None

        self._thermo_relaxer = None
        self._thermo_batch_relaxer = None
        self._thermo_ppd = None
        self._thermo_backend = "chgnet"
        self._thermo_ehull_method = "uncorrected"
        self._thermo_nequip_relax_mode = "sequential"
        self._thermo_mp2020_compat = None

        if self.gamma_cfg is not None:
            chgnet, relaxer, dev = make_chgnet_and_relaxer(self.gamma_cfg.device)
            self._gamma_chgnet = chgnet
            self._gamma_relaxer = relaxer
            self._gamma_calc = make_chgnet_ase_calculator(chgnet, dev)
            self._gamma_ready = True

        if self.thermo_cfg is not None:
            backend = str(self.thermo_cfg.mlip).strip().lower()
            if backend == "chgnet":
                if (
                    self.gamma_cfg is not None
                    and self.gamma_cfg.device == self.thermo_cfg.device
                ):
                    self._thermo_relaxer = self._gamma_relaxer
                else:
                    _, relaxer, _ = make_chgnet_and_relaxer(self.thermo_cfg.device)
                    self._thermo_relaxer = relaxer
            elif backend == "nequip":
                if not self.thermo_cfg.nequip_compile_path:
                    raise ValueError(
                        "NequIP thermo backend requires --nequip_compile_path."
                    )
                relax_mode = str(
                    getattr(self.thermo_cfg, "nequip_relax_mode", "sequential")
                ).strip().lower()
                if relax_mode not in {"sequential", "batch"}:
                    raise ValueError(
                        "Unsupported NequIP relax mode. "
                        "Choose from: sequential, batch."
                    )
                _, relaxer, _, _ = make_nequip_relaxer(
                    compile_path=self.thermo_cfg.nequip_compile_path,
                    stability_device=self.thermo_cfg.device,
                    optimizer_name=self.thermo_cfg.nequip_optimizer,
                    cell_filter=self.thermo_cfg.nequip_cell_filter,
                    fmax=self.thermo_cfg.nequip_fmax,
                    max_force_abort=self.thermo_cfg.nequip_max_force_abort,
                )
                self._thermo_relaxer = relaxer
                self._thermo_nequip_relax_mode = relax_mode
                if relax_mode == "batch":
                    _, batch_relaxer, _, _ = make_nequip_batch_relaxer(
                        compile_path=self.thermo_cfg.nequip_compile_path,
                        stability_device=self.thermo_cfg.device,
                        optimizer_name=self.thermo_cfg.nequip_optimizer,
                        cell_filter=self.thermo_cfg.nequip_cell_filter,
                        max_force_abort=self.thermo_cfg.nequip_max_force_abort,
                    )
                    self._thermo_batch_relaxer = batch_relaxer
            else:
                raise ValueError(f"Unsupported thermo MLIP backend: {backend!r}")

            self._thermo_backend = backend
            ehull_method = str(self.thermo_cfg.ehull_method).strip().lower()
            if ehull_method not in {"uncorrected", "mp2020_like"}:
                raise ValueError(
                    f"Unsupported thermo e_hull method: {ehull_method!r}. "
                    "Choose from: uncorrected, mp2020_like."
                )
            self._thermo_ehull_method = ehull_method
            if ehull_method == "mp2020_like":
                from pymatgen.entries.compatibility import (
                    MaterialsProject2020Compatibility,
                )

                self._thermo_mp2020_compat = MaterialsProject2020Compatibility(
                    check_potcar=False
                )
            self._thermo_ppd = load_phase_diagram(self.thermo_cfg.ppd_path)
            self._thermo_ready = True

    @classmethod
    def from_args(cls, args) -> "StabilityLogger | None":
        gamma_cfg = None
        thermo_cfg = None

        if getattr(args, "gamma_phonon_check", False):
            gamma_device = args.gamma_stability_device or "cpu"
            gamma_cfg = _GammaConfig(
                batch_size=max(1, int(args.gamma_phonon_batch)),
                relax_steps=int(args.gamma_relax_steps),
                supercell=tuple(args.gamma_phonon_supercell),
                delta=float(args.gamma_phonon_delta),
                neg_tol=float(args.gamma_phonon_neg_tol),
                acoustic_tol=float(args.gamma_phonon_acoustic_tol),
                device=gamma_device,
            )

        if getattr(args, "thermo_stability_check", False):
            if args.thermo_ppd_mp is None or not Path(args.thermo_ppd_mp).exists():
                raise FileNotFoundError(
                    "Thermo stability requires --thermo_ppd_mp pointing to a valid PPD pickle."
                )
            thermo_device = args.thermo_stability_device or "cpu"
            thermo_mlip = str(getattr(args, "thermo_mlip", "chgnet")).strip().lower()
            nequip_compile_path = getattr(args, "nequip_compile_path", None)
            thermo_cfg = _ThermoConfig(
                batch_size=max(1, int(args.thermo_stability_batch)),
                relax_steps=int(args.thermo_relax_steps),
                ppd_path=str(args.thermo_ppd_mp),
                device=thermo_device,
                ehull_method=str(getattr(args, "thermo_ehull_method", "uncorrected")),
                mlip=thermo_mlip,
                nequip_compile_path=(
                    str(nequip_compile_path) if nequip_compile_path else None
                ),
                nequip_relax_mode=str(
                    getattr(args, "nequip_relax_mode", "sequential")
                ),
                nequip_optimizer=str(getattr(args, "nequip_optimizer", "FIRE")),
                nequip_cell_filter=str(getattr(args, "nequip_cell_filter", "none")),
                nequip_fmax=float(getattr(args, "nequip_fmax", 0.01)),
                nequip_max_force_abort=float(
                    getattr(args, "nequip_max_force_abort", 1e6)
                ),
            )

        if gamma_cfg is None and thermo_cfg is None:
            return None

        return cls(gamma_cfg=gamma_cfg, thermo_cfg=thermo_cfg)

    def update(
        self,
        structures: Iterable,
        *,
        tag: str,
        step: int,
        log_fn: Callable[..., None],
        enabled: bool,
    ) -> None:
        if not enabled:
            return
        structs = list(structures)
        if not structs:
            return
        if self._gamma_ready:
            self._update_gamma(structs, tag=tag, step=step, log_fn=log_fn, enabled=enabled)
        if self._thermo_ready:
            self._update_thermo(structs, tag=tag, step=step, log_fn=log_fn, enabled=enabled)

    @property
    def thermo_backend(self) -> str:
        return str(self._thermo_backend)

    def _trim_buffer(self, buf: list, batch_size: int) -> list:
        buffer_max = batch_size * self.buffer_mult
        if len(buf) > buffer_max:
            return buf[-buffer_max:]
        return buf

    @staticmethod
    def _thermo_exception_reason(exc: Exception) -> tuple[str, bool]:
        message = str(exc).strip().lower()
        if "force divergence" in message:
            return "relax_divergence", True
        if "non-finite" in message:
            return "relax_non_finite", True
        return f"exception:{type(exc).__name__}", False

    @staticmethod
    def _is_structure_level_batch_exception(exc: Exception) -> bool:
        message = str(exc).strip().lower()
        return ("force divergence" in message) or ("non-finite" in message)

    @staticmethod
    def _extract_batched_failure_indices(exc: Exception, batch_size: int) -> list[int]:
        message = str(exc)
        match = re.search(r"systems\s+(\[[^\]]*\])", message)
        if match is None:
            return []
        try:
            parsed = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            return []
        if not isinstance(parsed, list):
            return []
        seen: set[int] = set()
        indices: list[int] = []
        for item in parsed:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= batch_size or idx in seen:
                continue
            seen.add(idx)
            indices.append(idx)
        return indices

    def _update_gamma(
        self,
        structures: list,
        *,
        tag: str,
        step: int,
        log_fn: Callable[..., None],
        enabled: bool,
    ) -> None:
        cfg = self.gamma_cfg
        if cfg is None or not self._gamma_ready:
            return
        buf = self.gamma_buffers[tag]
        buf.extend(structures)
        self.gamma_buffers[tag] = self._trim_buffer(buf, cfg.batch_size)
        if len(self.gamma_buffers[tag]) < cfg.batch_size:
            return
        batch = self.gamma_buffers[tag][: cfg.batch_size]
        self.gamma_buffers[tag] = self.gamma_buffers[tag][cfg.batch_size :]

        checked = 0
        ok = 0
        minfreqs = []
        fail_reasons: dict[str, int] = {}

        for struct in batch:
            try:
                relaxation = self._gamma_relaxer.relax(
                    struct,
                    steps=cfg.relax_steps,
                    verbose=False,
                )
                struct_relaxed = relaxation["final_structure"]
            except Exception:
                fail_reasons["relax_exception"] = fail_reasons.get("relax_exception", 0) + 1
                continue

            try:
                atoms = pmg_to_ase(struct_relaxed)
                atoms.pbc = True
                atoms.calc = self._gamma_calc
            except Exception:
                fail_reasons["ase_convert_exception"] = (
                    fail_reasons.get("ase_convert_exception", 0) + 1
                )
                continue

            checked += 1
            res = gamma_phonon_stability(
                atoms,
                supercell=cfg.supercell,
                delta=cfg.delta,
                neg_tol=cfg.neg_tol,
                acoustic_tol=cfg.acoustic_tol,
            )
            if res["ok"]:
                ok += 1
                minfreqs.append(res["min_gamma_freq"])
            else:
                fail_reasons[res["fail_reason"]] = fail_reasons.get(res["fail_reason"], 0) + 1

        payload = {
            f"{tag}/gamma_phonon_checked": float(checked),
            f"{tag}/gamma_phonon_ok_rate": float(ok) / float(max(1, checked)),
        }
        if minfreqs:
            payload[f"{tag}/gamma_phonon_min_freq_mean"] = float(sum(minfreqs) / len(minfreqs))
        for reason, cnt in fail_reasons.items():
            payload[f"{tag}/gamma_phonon_fail/{reason}"] = float(cnt)

        log_fn(payload, step=step, enabled=enabled)

    def _update_thermo(
        self,
        structures: list,
        *,
        tag: str,
        step: int,
        log_fn: Callable[..., None],
        enabled: bool,
    ) -> None:
        cfg = self.thermo_cfg
        if cfg is None or not self._thermo_ready:
            return
        buf = self.thermo_buffers[tag]
        buf.extend(structures)
        self.thermo_buffers[tag] = self._trim_buffer(buf, cfg.batch_size)
        if len(self.thermo_buffers[tag]) < cfg.batch_size:
            return
        batch = self.thermo_buffers[tag][: cfg.batch_size]
        self.thermo_buffers[tag] = self.thermo_buffers[tag][cfg.batch_size :]

        e_above = []
        stable_count = 0
        metastable_count = 0
        success_count = 0
        failed_count = 0
        divergence_count = 0
        fail_reasons: dict[str, int] = {}
        ehull_method = str(self._thermo_ehull_method).strip().lower()

        def record_failure(
            reason: str,
            *,
            count: int = 1,
            is_divergence: bool = False,
        ) -> None:
            nonlocal failed_count, divergence_count
            if count <= 0:
                return
            failed_count += int(count)
            fail_reasons[reason] = fail_reasons.get(reason, 0) + int(count)
            if is_divergence:
                divergence_count += int(count)

        def record_relaxation(relaxation) -> None:
            nonlocal success_count, stable_count, metastable_count
            e_relax_total = relaxation["trajectory"].energies[-1]
            hull_struct = relaxation["final_structure"]
            if ehull_method == "mp2020_like":
                e_above_val, fail_reason = compute_e_above_hull_mp2020_like(
                    self._thermo_ppd,
                    hull_struct,
                    e_relax_total,
                    mp2020_compat=self._thermo_mp2020_compat,
                )
            else:
                e_above_val, fail_reason = compute_e_above_hull_uncorrected(
                    self._thermo_ppd,
                    hull_struct,
                    e_relax_total,
                )
            if fail_reason is not None:
                record_failure(fail_reason)
                return
            if e_above_val is not None:
                e_above_val = float(e_above_val)
                e_above.append(e_above_val)
                success_count += 1
                stable_count += int(e_above_val <= 0.0)
                metastable_count += int(e_above_val <= 0.1)

        def relax_sequential(structures: list) -> None:
            for struct in structures:
                try:
                    relaxation = self._thermo_relaxer.relax(
                        struct,
                        steps=cfg.relax_steps,
                        verbose=False,
                    )
                    record_relaxation(relaxation)
                except Exception as exc:
                    reason, is_divergence = self._thermo_exception_reason(exc)
                    record_failure(reason, is_divergence=is_divergence)

        def relax_batched(structures: list) -> None:
            if not structures:
                return
            try:
                relaxations = self._thermo_batch_relaxer.relax_many(
                    structures,
                    steps=cfg.relax_steps,
                    verbose=False,
                )
                if len(relaxations) != len(structures):
                    raise RuntimeError(
                        "Batched NequIP returned a mismatched number of relaxations: "
                        f"{len(relaxations)} for batch of {len(structures)}"
                    )
                for relaxation in relaxations:
                    record_relaxation(relaxation)
            except Exception as exc:
                if not self._is_structure_level_batch_exception(exc):
                    raise
                reason, is_divergence = self._thermo_exception_reason(exc)
                offender_indices = self._extract_batched_failure_indices(
                    exc,
                    len(structures),
                )
                if offender_indices:
                    offender_set = set(offender_indices)
                    record_failure(
                        reason,
                        count=len(offender_set),
                        is_divergence=is_divergence,
                    )
                    survivors = [
                        struct
                        for idx, struct in enumerate(structures)
                        if idx not in offender_set
                    ]
                    if survivors:
                        relax_batched(survivors)
                    return
                if len(structures) <= 1:
                    record_failure(reason, is_divergence=is_divergence)
                    return
                midpoint = len(structures) // 2
                relax_batched(structures[:midpoint])
                relax_batched(structures[midpoint:])

        if (
            self._thermo_backend == "nequip"
            and self._thermo_nequip_relax_mode == "batch"
            and self._thermo_batch_relaxer is not None
        ):
            relax_batched(batch)
        else:
            relax_sequential(batch)

        payload: dict[str, float] = {
            f"{tag}/thermo_checked": float(len(batch)),
            f"{tag}/thermo_success": float(success_count),
            f"{tag}/thermo_failed": float(failed_count),
            f"{tag}/thermo_divergence": float(divergence_count),
            f"{tag}/thermo_stable_count": float(stable_count),
            f"{tag}/thermo_metastable_count": float(metastable_count),
        }
        payload[f"{tag}/thermo_ehull_is_mp2020_like"] = (
            1.0 if ehull_method == "mp2020_like" else 0.0
        )
        payload[f"{tag}/thermo_ehull_is_uncorrected"] = (
            1.0 if ehull_method == "uncorrected" else 0.0
        )
        if batch:
            payload[f"{tag}/thermo_stable_rate"] = float(stable_count) / float(len(batch))
            payload[f"{tag}/thermo_metastable_rate"] = float(metastable_count) / float(
                len(batch)
            )
        if e_above:
            payload[f"{tag}/thermo_e_above_hull_mean"] = float(np.mean(e_above))
        for reason, cnt in fail_reasons.items():
            payload[f"{tag}/thermo_fail/{reason}"] = float(cnt)
        log_fn(payload, step=step, enabled=enabled)
