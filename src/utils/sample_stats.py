from __future__ import annotations

import json
import warnings
from glob import glob
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch

from src.data.mp20_tokens import decode_Y1, tokens_to_structure, VZ
from src.eval.crystal import chemical_symbols


def volume_from_lengths_angles(lengths: np.ndarray, angles_deg: np.ndarray) -> float:
    a, b, c = lengths.tolist()
    alpha, beta, gamma = np.deg2rad(angles_deg.tolist())
    cos_a = float(np.cos(alpha))
    cos_b = float(np.cos(beta))
    cos_c = float(np.cos(gamma))
    term = 1.0 + 2.0 * cos_a * cos_b * cos_c - cos_a**2 - cos_b**2 - cos_c**2
    term = max(term, 0.0)
    return float(a * b * c * np.sqrt(term))


def collect_structure_stats(
    items: list[dict[str, Any]],
    *,
    invalid_log_path: str | Path | None = None,
    invalid_summary: dict[str, Any] | None = None,
    max_invalid_logs: int = 25,
) -> dict[str, np.ndarray]:
    elem_list: list[int] = []
    lengths_list: list[np.ndarray] = []
    angles_list: list[np.ndarray] = []
    volumes_list: list[float] = []
    vol_per_atom_list: list[float] = []
    min_dists: list[float] = []
    num_atoms_list: list[int] = []

    invalid_entries: list[dict[str, Any]] = []
    invalid_count = 0
    reason_counts: dict[str, int] = {}

    def log_invalid(
        idx: int,
        reason: str,
        *,
        num_atoms: int | None = None,
        lengths: np.ndarray | None = None,
        angles: np.ndarray | None = None,
    ) -> None:
        nonlocal invalid_count
        invalid_count += 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if invalid_log_path is None or len(invalid_entries) >= max_invalid_logs:
            return
        entry: dict[str, Any] = {"index": idx, "reason": reason}
        if num_atoms is not None:
            entry["num_atoms"] = num_atoms
        if lengths is not None:
            entry["lengths"] = [float(x) for x in lengths.tolist()]
        if angles is not None:
            entry["angles"] = [float(x) for x in angles.tolist()]
        invalid_entries.append(entry)

    for idx, item in enumerate(items):
        a0 = item["A0"]
        pad_mask = item["pad_mask"]
        if torch.is_tensor(a0):
            a0 = a0.detach().cpu().numpy()
        if torch.is_tensor(pad_mask):
            pad_mask = pad_mask.detach().cpu().numpy()
        mask = ~pad_mask.astype(bool)
        num_atoms = int(mask.sum())

        lengths, angles = decode_Y1(item["Y1"])
        if not np.all(np.isfinite(lengths)) or not np.all(np.isfinite(angles)):
            log_invalid(
                idx,
                "non_finite_lattice",
                num_atoms=num_atoms,
                lengths=lengths,
                angles=angles,
            )
            continue
        if (lengths <= 0).any():
            log_invalid(
                idx,
                "non_positive_length",
                num_atoms=num_atoms,
                lengths=lengths,
                angles=angles,
            )
            continue
        if (angles <= 0).any() or (angles >= 180.0).any():
            log_invalid(
                idx,
                "invalid_angle",
                num_atoms=num_atoms,
                lengths=lengths,
                angles=angles,
            )
            continue
        volume = volume_from_lengths_angles(lengths, angles)
        if not np.isfinite(volume) or volume <= 0.0:
            log_invalid(
                idx,
                "non_positive_volume",
                num_atoms=num_atoms,
                lengths=lengths,
                angles=angles,
            )
            continue

        try:
            struct = tokens_to_structure(item)
        except Exception as exc:
            log_invalid(
                idx,
                f"tokens_to_structure_error: {exc.__class__.__name__}",
                num_atoms=num_atoms,
                lengths=lengths,
                angles=angles,
            )
            continue
        if struct is not None and len(struct) > 1:
            try:
                dist = struct.distance_matrix
                tri = dist[np.triu_indices(dist.shape[0], k=1)]
                if tri.size:
                    min_dists.append(float(tri.min()))
            except Exception as exc:
                log_invalid(
                    idx,
                    f"distance_matrix_error: {exc.__class__.__name__}",
                    num_atoms=num_atoms,
                    lengths=lengths,
                    angles=angles,
                )
                continue

        num_atoms_list.append(num_atoms)
        elements = a0[mask]
        elements = elements[elements > 0]
        elem_list.extend([int(z) for z in elements.tolist()])
        lengths_list.append(lengths)
        angles_list.append(angles)
        volumes_list.append(volume)
        if num_atoms > 0:
            vol_per_atom_list.append(volume / float(num_atoms))

    if invalid_log_path is not None and invalid_count > 0:
        path = Path(invalid_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "invalid_count": invalid_count,
            "logged_count": len(invalid_entries),
            "max_invalid_logs": max_invalid_logs,
            "invalid_samples": invalid_entries,
        }
        path.write_text(json.dumps(payload, indent=2))
    if invalid_summary is not None:
        invalid_summary["invalid_total"] = invalid_count
        invalid_summary["invalid_reasons"] = dict(reason_counts)

    stats = {
        "elements": np.array(elem_list, dtype=np.int64),
        "lengths": np.vstack(lengths_list) if lengths_list else np.zeros((0, 3)),
        "angles": np.vstack(angles_list) if angles_list else np.zeros((0, 3)),
        "volumes": np.array(volumes_list, dtype=np.float32),
        "volumes_per_atom": np.array(vol_per_atom_list, dtype=np.float32),
        "min_dists": np.array(min_dists, dtype=np.float32),
        "num_atoms": np.array(num_atoms_list, dtype=np.int64),
    }
    return stats


def plot_sample_vs_ref_stats(
    *,
    tag: str,
    sample_stats: dict[str, np.ndarray],
    ref_stats: dict[str, np.ndarray],
    step: int,
    enabled: bool,
) -> None:
    if not enabled:
        return
    import matplotlib.pyplot as plt
    import wandb

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), dpi=120)
    axes = axes.reshape(-1)

    # Elements bar chart (by atomic number).
    ref_counts = np.bincount(ref_stats["elements"], minlength=VZ + 1)[1 : VZ + 1]
    samp_counts = np.bincount(sample_stats["elements"], minlength=VZ + 1)[1 : VZ + 1]
    labels = chemical_symbols[1 : VZ + 1]
    ref_vals = ref_counts.astype(np.float32)
    samp_vals = samp_counts.astype(np.float32)
    if ref_vals.sum() > 0:
        ref_vals /= ref_vals.sum()
    if samp_vals.sum() > 0:
        samp_vals /= samp_vals.sum()
    x = np.arange(len(labels))
    width = 0.4
    ax = axes[0]
    ax.bar(x - width / 2, ref_vals, width, label="ref", alpha=0.7)
    ax.bar(x + width / 2, samp_vals, width, label="sample", alpha=0.7)
    ax.set_title("Element distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=7)
    ax.legend()

    # Lengths histogram.
    ax = axes[1]
    ref_len = ref_stats["lengths"].reshape(-1)
    samp_len = sample_stats["lengths"].reshape(-1)
    if ref_len.size and samp_len.size:
        bins = np.linspace(
            min(ref_len.min(), samp_len.min()),
            max(ref_len.max(), samp_len.max()),
            40,
        )
        ax.hist(ref_len, bins=bins, density=True, alpha=0.5, label="ref")
        ax.hist(samp_len, bins=bins, density=True, alpha=0.5, label="sample")
    ax.set_title("Lattice lengths")
    ax.legend()

    # Angles histogram.
    ax = axes[2]
    ref_ang = ref_stats["angles"].reshape(-1)
    samp_ang = sample_stats["angles"].reshape(-1)
    if ref_ang.size and samp_ang.size:
        bins = np.linspace(0.0, 180.0, 40)
        ax.hist(ref_ang, bins=bins, density=True, alpha=0.5, label="ref")
        ax.hist(samp_ang, bins=bins, density=True, alpha=0.5, label="sample")
    ax.set_title("Lattice angles (deg)")
    ax.legend()

    # Volume per atom histogram.
    ax = axes[3]
    ref_vol = ref_stats["volumes_per_atom"]
    samp_vol = sample_stats["volumes_per_atom"]
    if ref_vol.size and samp_vol.size:
        vmin = float(min(ref_vol.min(), samp_vol.min()))
        vmax = float(max(ref_vol.max(), samp_vol.max()))
        if vmin == vmax:
            vmin -= 1e-3
            vmax += 1e-3
        bins = np.linspace(vmin, vmax, 40)
        ax.hist(ref_vol, bins=bins, density=True, alpha=0.5, label="ref")
        ax.hist(samp_vol, bins=bins, density=True, alpha=0.5, label="sample")
    ax.set_title("Volume per atom")
    ax.legend()

    # Min distance histogram.
    ax = axes[4]
    ref_md = ref_stats["min_dists"]
    samp_md = sample_stats["min_dists"]
    if ref_md.size and samp_md.size:
        bins = np.linspace(
            min(ref_md.min(), samp_md.min()),
            max(ref_md.max(), samp_md.max()),
            40,
        )
        ax.hist(ref_md, bins=bins, density=True, alpha=0.5, label="ref")
        ax.hist(samp_md, bins=bins, density=True, alpha=0.5, label="sample")
    ax.set_title("Min pairwise distance")
    ax.legend()

    # Atoms per structure histogram.
    ax = axes[5]
    ref_n = ref_stats["num_atoms"]
    samp_n = sample_stats["num_atoms"]
    if ref_n.size and samp_n.size:
        nmax = int(max(ref_n.max(), samp_n.max()))
        edges = np.arange(0.5, nmax + 1.5, 1.0)
        ax.hist(ref_n, bins=edges, density=True, alpha=0.5, label="ref")
        ax.hist(samp_n, bins=edges, density=True, alpha=0.5, label="sample")
    ax.set_title("Atoms per structure")
    ax.legend()

    fig.suptitle(f"{tag}: sample vs train", y=1.02)
    fig.tight_layout()
    wandb.log({f"{tag}/sample_vs_ref": wandb.Image(fig)}, step=step)
    plt.close(fig)


def make_chgnet_and_relaxer(stability_device: str):
    from chgnet.model import CHGNet, StructOptimizer

    if stability_device != "cpu" and not torch.cuda.is_available():
        stability_device = "cpu"
    chgnet = CHGNet.load(use_device=stability_device)
    chgnet.eval()
    relaxer = StructOptimizer(model=chgnet, use_device=stability_device)
    return chgnet, relaxer, stability_device


def make_chgnet_ase_calculator(chgnet, stability_device: str):
    # Try common constructor signatures across CHGNet versions.
    from chgnet.model.dynamics import CHGNetCalculator

    for kwargs in (
        {"model": chgnet, "use_device": stability_device},
        {"model": chgnet, "device": stability_device},
        {"use_device": stability_device},
        {"device": stability_device},
        {},
    ):
        try:
            return CHGNetCalculator(**kwargs)
        except TypeError:
            continue
    return CHGNetCalculator()


def resolve_nequip_compiled_model(compile_path: str | Path) -> str:
    path = Path(compile_path)
    if path.is_file():
        return str(path.resolve())

    candidates = set(glob(str(path)))
    if not path.is_absolute():
        repo_path = Path(__file__).resolve().parents[2] / path
        if repo_path.is_file():
            return str(repo_path.resolve())
        candidates.update(glob(str(repo_path)))

    resolved = sorted(str(Path(p).resolve()) for p in candidates if Path(p).is_file())
    if len(resolved) == 1:
        return resolved[0]
    if len(resolved) > 1:
        msg = ", ".join(resolved[:5])
        raise FileNotFoundError(
            f"Matched {len(resolved)} NequIP models for {compile_path!r} "
            f"(first 5: {msg}). Please provide a specific path."
        )
    raise FileNotFoundError(
        f"Could not resolve NequIP compiled model from {compile_path!r}."
    )


def _resolve_nequip_device(stability_device: str) -> str:
    if stability_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if stability_device != "cpu" and not torch.cuda.is_available():
        return "cpu"
    return stability_device


def make_nequip_ase_calculator(compiled_model: str, stability_device: str):
    try:  # required only for some compiled model backends
        import cuequivariance_torch  # noqa: F401
        import openequivariance  # noqa: F401
    except ImportError:
        pass

    from nequip.ase import NequIPCalculator

    stability_device = _resolve_nequip_device(stability_device)

    prev_default_dtype = torch.get_default_dtype()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Trying to use model type names")
            calc = NequIPCalculator.from_compiled_model(
                compile_path=compiled_model,
                device=stability_device,
            )
    finally:
        # NequIP compiled model loading mutates torch global default dtype.
        torch.set_default_dtype(prev_default_dtype)
    return calc, stability_device


class _ContiguousRowVectorCellStateProxy:
    """Proxy a torch-sim state while materializing row-vector cells contiguously."""

    def __init__(self, state) -> None:
        object.__setattr__(self, "_state", state)

    def __getattr__(self, name: str):
        if name == "row_vector_cell":
            return self._state.row_vector_cell.contiguous()
        return getattr(self._state, name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self._state, name, value)


def patch_nequip_torchsim_calc_contiguous_row_vector_cell(calculator):
    """Wrap NequIP TorchSim forward so model cell input is contiguous.

    TorchSim exposes ``row_vector_cell`` as a transpose view of ``state.cell``.
    Some downstream Warp-backed paths reject the resulting non-contiguous
    ``(batch, 3, 3)`` tensor when stress/cell-filter code is active.
    """

    if getattr(calculator, "_atomreps_contiguous_row_vector_cell_patch", False):
        return calculator

    original_forward = calculator.forward

    def _patched_forward(self, state):
        proxy_state = _ContiguousRowVectorCellStateProxy(state)
        return original_forward(proxy_state)

    calculator.forward = MethodType(_patched_forward, calculator)
    calculator._atomreps_contiguous_row_vector_cell_patch = True
    return calculator


def make_nequip_batch_calculator(compiled_model: str, stability_device: str):
    try:  # required only for some compiled model backends
        import cuequivariance_torch  # noqa: F401
        import openequivariance  # noqa: F401
    except ImportError:
        pass

    try:
        from nequip.integrations.torchsim import NequIPTorchSimCalc
    except Exception as exc:
        raise RuntimeError(
            "Could not import NequIP TorchSim integration. Install a NequIP build "
            "that includes `nequip.integrations.torchsim` and install `torch_sim`."
        ) from exc

    stability_device = _resolve_nequip_device(stability_device)

    prev_default_dtype = torch.get_default_dtype()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Trying to use model type names")
            calc = NequIPTorchSimCalc.from_compiled_model(
                compile_path=compiled_model,
                device=stability_device,
            )
            calc = patch_nequip_torchsim_calc_contiguous_row_vector_cell(calc)
    finally:
        # NequIP compiled model loading mutates torch global default dtype.
        torch.set_default_dtype(prev_default_dtype)
    return calc, stability_device


def torch_system_max_forces(forces: torch.Tensor, system_idx: torch.Tensor) -> list[float]:
    norms = torch.linalg.vector_norm(forces, dim=1)
    n_systems = int(system_idx.max().item()) + 1 if norms.numel() else 0
    values: list[float] = []
    for sys_idx in range(n_systems):
        mask = system_idx == sys_idx
        if not torch.any(mask):
            values.append(float("nan"))
            continue
        values.append(float(norms[mask].max().detach().cpu().item()))
    return values


class _EnergyTrajectory:
    def __init__(self, energies: list[float]) -> None:
        self.energies = energies


class NequIPRelaxer:
    def __init__(
        self,
        *,
        calculator,
        optimizer_cls,
        filter_cls,
        fmax: float,
        max_force_abort: float,
    ) -> None:
        self.calculator = calculator
        self.optimizer_cls = optimizer_cls
        self.filter_cls = filter_cls
        self.fmax = float(fmax)
        self.max_force_abort = float(max_force_abort)

    def relax(self, struct, steps: int = 200, verbose: bool = False) -> dict[str, Any]:
        del verbose
        atoms = pmg_to_ase(struct)
        atoms.pbc = True
        atoms.calc = self.calculator
        target = atoms if self.filter_cls is None else self.filter_cls(atoms)

        nsteps = 0
        if steps > 0:
            with self.optimizer_cls(target, logfile=None) as optimizer:
                for _ in optimizer.irun(fmax=self.fmax, steps=steps):
                    forces = target.get_forces()
                    if forces.size:
                        max_force = float(np.linalg.norm(forces, axis=1).max())
                        if np.isfinite(self.max_force_abort) and max_force > self.max_force_abort:
                            raise RuntimeError(
                                "Force divergence detected: "
                                f"max force {max_force:.3e} > {self.max_force_abort:.3e}"
                            )
                nsteps = int(getattr(optimizer, "nsteps", steps))

        final_atoms = target.atoms if hasattr(target, "atoms") else target
        final_structure = ase_to_pmg(final_atoms)
        total_energy = float(final_atoms.get_potential_energy())
        return {
            "final_structure": final_structure,
            "trajectory": _EnergyTrajectory([total_energy]),
            "nsteps": nsteps,
        }


class NequIPBatchRelaxer:
    def __init__(
        self,
        *,
        calculator,
        cell_filter: str,
        max_force_abort: float,
    ) -> None:
        self.calculator = calculator
        self.cell_filter = str(cell_filter).lower()
        self.max_force_abort = float(max_force_abort)

    def _resolve_cell_filter(self, ts):
        if self.cell_filter == "none":
            return None
        if self.cell_filter == "frechet":
            return ts.CellFilter.frechet
        raise ValueError(
            "Batched NequIP only supports cell_filter='none' or 'frechet'. "
            f"Got: {self.cell_filter!r}"
        )

    def relax_many(
        self,
        structs,
        steps: int = 200,
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        del verbose
        structures = list(structs)
        if not structures:
            return []

        try:
            import torch_sim as ts
        except Exception as exc:
            raise RuntimeError(
                "Could not import `torch_sim`. Install `torch-sim-atomistic`."
            ) from exc

        state = ts.io.structures_to_state(
            structures,
            device=self.calculator.device,
            dtype=self.calculator.dtype,
        )
        cell_filter_arg = self._resolve_cell_filter(ts)

        if steps > 0:
            optim_state = ts.fire_init(
                state=state,
                model=self.calculator,
                cell_filter=cell_filter_arg,
                fire_flavor="ase_fire",
            )
            for _ in range(int(steps)):
                optim_state = ts.fire_step(
                    state=optim_state,
                    model=self.calculator,
                    fire_flavor="ase_fire",
                )
                if not torch.isfinite(optim_state.energy).all():
                    raise RuntimeError("Non-finite batch energy detected during relaxation")
                if not torch.isfinite(optim_state.forces).all():
                    raise RuntimeError("Non-finite batch forces detected during relaxation")
                if np.isfinite(self.max_force_abort):
                    max_forces = torch_system_max_forces(
                        optim_state.forces,
                        optim_state.system_idx,
                    )
                    offenders = [
                        idx for idx, force in enumerate(max_forces) if force > self.max_force_abort
                    ]
                    if offenders:
                        raise RuntimeError(
                            "Force divergence detected in batched relaxation for "
                            f"systems {offenders}; max force threshold={self.max_force_abort:.3e}"
                        )
            final_structures = ts.io.state_to_structures(optim_state)
            energies = optim_state.energy.detach().cpu().numpy()
            nsteps = int(steps)
        else:
            model_output = self.calculator(state)
            if not torch.isfinite(model_output["energy"]).all():
                raise RuntimeError("Non-finite batch energy detected in single-point mode")
            if not torch.isfinite(model_output["forces"]).all():
                raise RuntimeError("Non-finite batch forces detected in single-point mode")
            final_structures = list(structures)
            energies = model_output["energy"].detach().cpu().numpy()
            nsteps = 0

        return [
            {
                "final_structure": final_structure,
                "trajectory": _EnergyTrajectory([float(total_energy)]),
                "nsteps": nsteps,
            }
            for final_structure, total_energy in zip(final_structures, energies, strict=True)
        ]

    def relax(self, struct, steps: int = 200, verbose: bool = False) -> dict[str, Any]:
        return self.relax_many([struct], steps=steps, verbose=verbose)[0]


def make_nequip_relaxer(
    *,
    compile_path: str | Path,
    stability_device: str,
    optimizer_name: str = "FIRE",
    cell_filter: str = "none",
    fmax: float = 0.01,
    max_force_abort: float = 1e6,
):
    import ase.optimize
    from ase.filters import ExpCellFilter, FrechetCellFilter

    optimizer_map = {
        "FIRE": ase.optimize.FIRE,
        "LBFGS": ase.optimize.LBFGS,
        "BFGS": ase.optimize.BFGS,
        "BFGSLineSearch": ase.optimize.BFGSLineSearch,
        "LBFGSLineSearch": ase.optimize.LBFGSLineSearch,
    }
    if optimizer_name not in optimizer_map:
        allowed = ", ".join(sorted(optimizer_map))
        raise ValueError(f"Unsupported NequIP optimizer {optimizer_name!r}. Choose from: {allowed}")

    filter_map = {
        "none": None,
        "frechet": FrechetCellFilter,
        "exp": ExpCellFilter,
    }
    filter_key = str(cell_filter).lower()
    if filter_key not in filter_map:
        allowed = ", ".join(sorted(filter_map))
        raise ValueError(f"Unsupported NequIP cell filter {cell_filter!r}. Choose from: {allowed}")

    model_path = resolve_nequip_compiled_model(compile_path)
    calc, resolved_device = make_nequip_ase_calculator(
        compiled_model=model_path,
        stability_device=stability_device,
    )
    relaxer = NequIPRelaxer(
        calculator=calc,
        optimizer_cls=optimizer_map[optimizer_name],
        filter_cls=filter_map[filter_key],
        fmax=fmax,
        max_force_abort=max_force_abort,
    )
    return calc, relaxer, resolved_device, model_path


def make_nequip_batch_relaxer(
    *,
    compile_path: str | Path,
    stability_device: str,
    optimizer_name: str = "FIRE",
    cell_filter: str = "none",
    max_force_abort: float = 1e6,
):
    if str(optimizer_name) != "FIRE":
        raise ValueError(
            "Batched NequIP currently supports only optimizer_name='FIRE'. "
            f"Got: {optimizer_name!r}"
        )

    filter_key = str(cell_filter).lower()
    if filter_key not in {"none", "frechet"}:
        raise ValueError(
            "Batched NequIP currently supports only cell_filter='none' or 'frechet'. "
            f"Got: {cell_filter!r}"
        )

    model_path = resolve_nequip_compiled_model(compile_path)
    calc, resolved_device = make_nequip_batch_calculator(
        compiled_model=model_path,
        stability_device=stability_device,
    )
    relaxer = NequIPBatchRelaxer(
        calculator=calc,
        cell_filter=filter_key,
        max_force_abort=max_force_abort,
    )
    return calc, relaxer, resolved_device, model_path


def pmg_to_ase(struct):
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_atoms(struct)


def ase_to_pmg(atoms):
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_structure(atoms)


def gamma_phonon_stability(
    atoms,
    supercell=(2, 2, 2),
    delta=0.02,
    neg_tol=1e-3,
    acoustic_tol=1e-2,
    run_tag="phonon",
):
    import numpy as np
    import tempfile
    from pathlib import Path
    from ase.phonons import Phonons

    out = {
        "ok": False,
        "fail_reason": "",
        "fail_detail": "",
        "min_gamma_freq": float("nan"),
        "min_gamma_freq_all": float("nan"),
        "num_neg_optical": 0,
    }

    try:
        _ = atoms.get_forces()
    except Exception as e:
        out["fail_reason"] = "forces_exception"
        out["fail_detail"] = f"{type(e).__name__}: {str(e)[:120]}"
        return out

    try:
        with tempfile.TemporaryDirectory(prefix=f"{run_tag}_") as d:
            phonon_prefix = str(Path(d) / "ph")
            ph = Phonons(
                atoms, atoms.calc, supercell=supercell, delta=delta, name=phonon_prefix
            )
            try:
                ph.run()
                ph.read(acoustic=True)

                q = np.array([[0.0, 0.0, 0.0]])
                freqs = np.asarray(ph.band_structure(q)[0])
                freqs = np.real_if_close(freqs, tol=1000).astype(float)

                if freqs.size:
                    if not np.all(np.isfinite(freqs)):
                        out["fail_reason"] = "nan_freq"
                        out["fail_detail"] = "non-finite frequencies"
                        return out
                    out["min_gamma_freq_all"] = float(freqs.min())

                    # Drop 3 acoustic modes (near-zero); safer than unconditional drop
                    idx = np.argsort(np.abs(freqs))
                    drop = [k for k in idx if abs(freqs[k]) < acoustic_tol][
                        : min(3, freqs.size)
                    ]
                    if len(drop) < min(3, freqs.size):
                        drop = idx[: min(3, freqs.size)].tolist()

                    keep = np.ones_like(freqs, dtype=bool)
                    keep[drop] = False
                    optical = freqs[keep]
                else:
                    optical = freqs

                if optical.size:
                    out["min_gamma_freq"] = float(optical.min())
                    out["num_neg_optical"] = int((optical < -neg_tol).sum())
                else:
                    out["min_gamma_freq"] = (
                        float(freqs.min()) if freqs.size else float("nan")
                    )
                    out["num_neg_optical"] = (
                        int((freqs < -neg_tol).sum()) if freqs.size else 0
                    )

                if out["num_neg_optical"] > 0:
                    out["fail_reason"] = "imaginary_gamma"
                    return out

                out["ok"] = True
                return out
            finally:
                try:
                    ph.clean()
                except Exception:
                    pass

    except Exception as e:
        out["fail_reason"] = f"phonon_exception:{type(e).__name__}"
        out["fail_detail"] = f"{type(e).__name__}: {str(e)[:120]}"
        return out


def compute_e_above_hull_uncorrected(
    ppd, structure, e_total: float
) -> tuple[float | None, str | None]:
    """Uncorrected parity: e_above_hull = (E_total / num_sites) - e_hull_per_atom."""
    num_sites = structure.num_sites
    if num_sites <= 0:
        return None, "invalid_num_sites"
    try:
        e_hull = ppd.get_hull_energy_per_atom(structure.composition)
    except Exception as exc:
        return None, f"hull_exception:{type(exc).__name__}"
    if not np.isfinite(e_hull) or not np.isfinite(e_total):
        return None, "nan_eabove"
    e_above = (e_total / num_sites) - e_hull
    if not np.isfinite(e_above):
        return None, "nan_eabove"
    return float(e_above), None


def _build_mp2020_parameters(composition: Any, mp2020_compat: Any) -> dict[str, Any]:
    """Build synthetic metadata expected by MP2020 compatibility corrections."""
    sorted_elements = sorted(
        (el for el in composition.elements if composition[el] > 0),
        key=lambda el: el.X,
    )
    most_electroneg = sorted_elements[-1].symbol if sorted_elements else None
    u_settings = mp2020_compat.u_settings.get(most_electroneg, {})
    hubbards = {
        el.symbol: float(u_settings.get(el.symbol, 0.0))
        for el in composition.elements
        if float(u_settings.get(el.symbol, 0.0)) != 0.0
    }
    run_type = "GGA+U" if hubbards else "GGA"
    return {"run_type": run_type, "hubbards": hubbards, "software": "vasp"}


def compute_e_above_hull_mp2020_like(
    ppd,
    structure,
    e_total: float,
    *,
    mp2020_compat: Any | None = None,
    entry_id: str | None = None,
) -> tuple[float | None, str | None]:
    """MP2020-like parity: apply MP2020 correction to an ML entry, then compute hull distance."""
    if not np.isfinite(e_total):
        return None, "nan_eabove"

    try:
        from pymatgen.entries.computed_entries import ComputedStructureEntry
    except Exception as exc:
        return None, f"mp2020_import_exception:{type(exc).__name__}"

    if mp2020_compat is None:
        try:
            from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
        except Exception as exc:
            return None, f"mp2020_import_exception:{type(exc).__name__}"
        mp2020_compat = MaterialsProject2020Compatibility(check_potcar=False)

    try:
        mp2020_params = _build_mp2020_parameters(
            composition=structure.composition,
            mp2020_compat=mp2020_compat,
        )
        raw_entry = ComputedStructureEntry(
            composition=structure.composition,
            energy=float(e_total),
            structure=structure,
            entry_id=entry_id,
            parameters=mp2020_params,
        )
        corrected = mp2020_compat.process_entry(raw_entry.copy(), on_error="raise")
        if corrected is None:
            return None, "mp2020_removed"
        e_above = float(ppd.get_e_above_hull(corrected, allow_negative=True))
    except Exception as exc:
        return None, f"mp2020_exception:{type(exc).__name__}"

    if not np.isfinite(e_above):
        return None, "nan_eabove"
    return float(e_above), None
