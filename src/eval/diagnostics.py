"""Sample quality diagnostics and element histogram logging."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from src.data.mp20_tokens import VZ
from src.eval.crystal import chemical_symbols
from src.utils.constants import _DIAGNOSTIC_SECTION_KEYS


def _compute_sample_diagnostics(
    pred_crys_list: list,
    min_dist_cutoff: float = 0.5,
    volume_cutoff: float = 0.1,
) -> dict[str, float]:
    import smact

    total = len(pred_crys_list)
    if total == 0:
        return {}

    constructed = 0
    min_dists = []
    volumes = []
    lengths = []
    angles = []
    reason_counts: dict[str, int] = {}
    missing_ox = 0
    comp_invalid = 0
    comp_invalid_missing = 0
    num_atoms_list = []
    num_elems_list = []
    invalid_atoms = 0
    total_atoms = 0

    for crys in pred_crys_list:
        atom_types = getattr(crys, "atom_types", None)
        if atom_types is not None:
            atoms = [int(z) for z in atom_types]
            num_atoms_list.append(len(atoms))
            num_elems_list.append(len(set(atoms)))
            for z in atoms:
                total_atoms += 1
                if z <= 0 or z >= len(chemical_symbols):
                    invalid_atoms += 1

        if getattr(crys, "constructed", False):
            constructed += 1
            struct = crys.structure
            volumes.append(float(struct.volume))
            lengths.append(np.asarray(crys.lengths, dtype=np.float32))
            angles.append(np.asarray(crys.angles, dtype=np.float32))
            if len(struct) > 1:
                dist = struct.distance_matrix
                dist = dist + np.eye(dist.shape[0]) * 1e6
                min_dists.append(float(dist.min()))

        reason = getattr(crys, "invalid_reason", None)
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        missing = False
        if atom_types is not None:
            try:
                elem_symbols = []
                for z in atom_types:
                    z_int = int(z)
                    if z_int <= 0 or z_int >= len(chemical_symbols):
                        elem_symbols = []
                        break
                    elem_symbols.append(chemical_symbols[z_int])
                if elem_symbols:
                    space = smact.element_dictionary(tuple(set(elem_symbols)))
                    smact_elems = [e[1] for e in space.items()]
                    missing = any(len(e.oxidation_states) == 0 for e in smact_elems)
            except Exception:
                missing = False
        if missing:
            missing_ox += 1
        if not crys.comp_valid:
            comp_invalid += 1
            if missing:
                comp_invalid_missing += 1

    metrics: dict[str, float] = {
        "constructed_rate": float(constructed) / float(total),
        "comp_missing_ox_frac": float(missing_ox) / float(total),
    }

    if min_dists:
        min_d = np.array(min_dists, dtype=np.float32)
        metrics["struct_min_dist_mean"] = float(min_d.mean())
        metrics["struct_min_dist_min"] = float(min_d.min())
        metrics["struct_min_dist_frac_lt_cutoff"] = float(
            (min_d < min_dist_cutoff).mean()
        )

    if volumes:
        vol = np.array(volumes, dtype=np.float32)
        metrics["struct_volume_mean"] = float(vol.mean())
        metrics["struct_volume_min"] = float(vol.min())
        metrics["struct_volume_frac_lt_cutoff"] = float((vol < volume_cutoff).mean())
    if lengths:
        len_vals = np.vstack(lengths).reshape(-1)
        metrics["lattice_length_mean"] = float(len_vals.mean())
        metrics["lattice_length_min"] = float(len_vals.min())
        metrics["lattice_length_max"] = float(len_vals.max())
        metrics["lattice_length_frac_lt_1"] = float((len_vals < 1.0).mean())
    if angles:
        ang_vals = np.vstack(angles).reshape(-1)
        metrics["lattice_angle_mean"] = float(ang_vals.mean())
        metrics["lattice_angle_min"] = float(ang_vals.min())
        metrics["lattice_angle_max"] = float(ang_vals.max())
        metrics["lattice_angle_frac_outside_50_130"] = float(
            ((ang_vals < 50.0) | (ang_vals > 130.0)).mean()
        )

    if num_atoms_list:
        atoms_arr = np.array(num_atoms_list, dtype=np.float32)
        metrics["num_atoms_mean"] = float(atoms_arr.mean())
        metrics["num_atoms_min"] = float(atoms_arr.min())
        metrics["num_atoms_max"] = float(atoms_arr.max())
    if num_elems_list:
        elems_arr = np.array(num_elems_list, dtype=np.float32)
        metrics["num_elems_mean"] = float(elems_arr.mean())
        metrics["num_elems_min"] = float(elems_arr.min())
        metrics["num_elems_max"] = float(elems_arr.max())
        metrics["num_elems_frac_single"] = float((elems_arr == 1).mean())

    if total_atoms > 0:
        metrics["invalid_atom_frac"] = float(invalid_atoms) / float(total_atoms)

    if comp_invalid > 0:
        metrics["comp_invalid_missing_ox_frac"] = float(comp_invalid_missing) / float(
            comp_invalid
        )
        metrics["comp_invalid_no_ox_frac"] = float(
            comp_invalid - comp_invalid_missing
        ) / float(comp_invalid)

    for reason, count in reason_counts.items():
        metrics[f"invalid_reason/{reason}"] = float(count) / float(total)

    return metrics


def _add_diagnostic_section_metrics(
    log_payload: dict[str, float],
    *,
    tag: str,
    diag: Mapping[str, float],
) -> None:
    for key in _DIAGNOSTIC_SECTION_KEYS:
        val = diag.get(key, None)
        if val is None:
            continue
        log_payload[f"{tag}/diagnostic/{key}"] = float(val)


def _log_element_histogram(
    tag: str,
    atom_types: torch.Tensor,
    pad_mask: torch.Tensor,
    step: int,
    enabled: bool,
    train_distribution: torch.Tensor | None = None,
) -> None:
    if not enabled:
        return
    values = atom_types.detach().cpu()
    mask = (~pad_mask).detach().cpu()
    vals = values[mask].reshape(-1)
    vals = vals[vals > 0]
    if vals.numel() == 0:
        return
    counts = torch.bincount(vals, minlength=VZ + 1)[1 : VZ + 1]
    total = int(counts.sum().item())
    if total == 0:
        return
    import matplotlib.pyplot as plt
    import wandb

    elems = chemical_symbols[1 : VZ + 1]
    gen_frac = counts.to(dtype=torch.float64) / float(total)

    width = 0.4
    positions = torch.arange(len(elems), dtype=torch.float64)
    pos_np = positions.numpy()
    fig_width = max(24.0, 0.35 * len(elems))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    if train_distribution is not None:
        ax.bar(
            pos_np - width / 2.0,
            train_distribution.detach().cpu().numpy(),
            width=width,
            label="train",
            color="#4F81BD",
        )
    ax.bar(
        pos_np + width / 2.0,
        gen_frac.detach().cpu().numpy(),
        width=width,
        label="generated",
        color="#C0504D",
    )
    ax.set_xticks(pos_np)
    ax.set_xticklabels(elems, rotation=90, ha="center")
    ax.set_ylabel("fraction")
    ax.set_title("Element distribution (train vs generated)")
    ax.legend()
    fig.tight_layout()
    wandb.log({f"{tag}/element_histogram": wandb.Image(fig)}, step=step)
    plt.close(fig)
