#!/usr/bin/env python3
"""Build reviewer-facing case studies for generated crystal candidates.

The script is intentionally usable in two environments:

1. Locally, with only a CIF directory and its manifest. This reports candidate
   selection, symmetry, oxidation plausibility, and MLIP metadata already stored
   in the manifest.
2. On the server, with MP20 reference data and a phase-diagram pickle. This adds
   nearest-reference searches and decomposition products.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mp20_tokens import MP20Tokens, tokens_to_structure  # noqa: E402
from src.eval.crystal import smact_validity  # noqa: E402


@dataclass
class ReferenceRecord:
    ref_idx: int
    split: str
    mp_id: str
    structure: Structure
    reduced_formula: str
    chemsys: tuple[str, ...]
    volume_per_atom: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select low-e_above_hull generated CIFs and produce detailed "
            "case-study tables for rebuttal/manuscript use."
        )
    )
    parser.add_argument("--cif_dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--max_e_above_hull",
        type=float,
        default=0.1,
        help="Prefer candidates at or below this eV/atom threshold.",
    )
    parser.add_argument(
        "--min_e_above_hull",
        type=float,
        default=0.0,
        help=(
            "Exclude candidates below this eV/atom value. Use the default of "
            "0.0 for reviewer-facing examples so strongly negative MLIP hull "
            "artifacts are not selected."
        ),
    )
    parser.add_argument(
        "--candidate_pool_size",
        type=int,
        default=200,
        help="Number of low-hull manifest rows to inspect before final filtering.",
    )
    parser.add_argument(
        "--require_smact_valid",
        action="store_true",
        help="Only keep candidates passing the repo's SMACT-style plausibility check.",
    )
    parser.add_argument(
        "--reference_data_root",
        type=Path,
        default=None,
        help="Optional MP20 token dataset root for nearest-reference analysis.",
    )
    parser.add_argument(
        "--reference_splits",
        nargs="+",
        default=["train"],
        help="MP20 splits to search when --reference_data_root is provided.",
    )
    parser.add_argument("--nmax", type=int, default=20)
    parser.add_argument(
        "--phase_diagram",
        type=Path,
        default=None,
        help="Optional PhaseDiagram/PatchedPhaseDiagram pickle for decomposition products.",
    )
    parser.add_argument("--stol", type=float, default=0.5)
    parser.add_argument("--ltol", type=float, default=0.3)
    parser.add_argument("--angle_tol", type=float, default=10.0)
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument(
        "--max_reference_candidates",
        type=int,
        default=5000,
        help="Maximum same-formula/chemsys references scored per candidate.",
    )
    parser.add_argument(
        "--reference_limit",
        type=int,
        default=0,
        help="Debug limit for loaded reference structures. 0 means no limit.",
    )
    parser.add_argument("--rdf_cutoff", type=float, default=8.0)
    parser.add_argument("--rdf_bins", type=int, default=80)
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def load_manifest(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Manifest is empty: {path}")
    data = json.loads(text)
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for key in ("manifest", "samples", "structures", "records"):
            if isinstance(data.get(key), list):
                rows = data[key]
                break
        else:
            raise ValueError(
                f"Manifest dict must contain one of manifest/samples/structures/records: {path}"
            )
    else:
        raise TypeError(f"Unsupported manifest type {type(data).__name__}: {path}")
    return [dict(row) for row in rows]


def row_success(row: dict[str, Any]) -> bool:
    if "success" not in row:
        return True
    return bool(row["success"])


def select_candidates(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    min_e_above_hull: float,
    max_e_above_hull: float,
) -> list[dict[str, Any]]:
    usable = [
        row
        for row in rows
        if row_success(row)
        and (e_hull := finite_float(row.get("e_above_hull"))) is not None
        and min_e_above_hull <= e_hull <= max_e_above_hull
    ]
    usable.sort(key=lambda r: finite_float(r.get("e_above_hull")) or float("inf"))
    return usable[:top_k]


def load_structure_for_row(cif_dir: Path, row: dict[str, Any]) -> Structure:
    file_name = row.get("file")
    if not file_name:
        sample_idx = int(row["sample_idx"])
        file_name = f"sample_{sample_idx:05d}.cif"
    path = cif_dir / str(file_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing CIF for sample {row.get('sample_idx')}: {path}")
    return Structure.from_file(str(path))


def chemsys(structure: Structure) -> tuple[str, ...]:
    return tuple(sorted(el.symbol for el in structure.composition.elements))


def symmetry_summary(structure: Structure, symprec: float) -> dict[str, Any]:
    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
        return {
            "spacegroup_symbol": analyzer.get_space_group_symbol(),
            "spacegroup_number": analyzer.get_space_group_number(),
            "crystal_system": analyzer.get_crystal_system(),
        }
    except Exception as exc:
        return {
            "spacegroup_symbol": None,
            "spacegroup_number": None,
            "crystal_system": None,
            "symmetry_error": f"{type(exc).__name__}: {str(exc)[:160]}",
        }


def oxidation_summary(structure: Structure) -> dict[str, Any]:
    comp = structure.composition
    el_amt = comp.get_el_amt_dict()
    atomic_numbers: list[int] = []
    counts: list[int] = []
    for symbol, amount in el_amt.items():
        atomic_numbers.append(Element(symbol).Z)
        counts.append(max(1, int(round(float(amount)))))

    try:
        smact_ok = bool(
            smact_validity(
                atomic_numbers,
                counts,
                use_pauling_test=True,
                include_alloys=True,
                allow_missing_ox_states=False,
            )
        )
    except Exception:
        smact_ok = None

    oxi_guess = None
    try:
        guesses = comp.oxi_state_guesses()
        if guesses:
            oxi_guess = {
                str(key): float(val)
                for key, val in list(guesses[0].items())
            }
    except Exception:
        oxi_guess = None

    return {
        "smact_valid": smact_ok,
        "oxidation_guess": oxi_guess,
        "oxidation_guess_json": json.dumps(oxi_guess, sort_keys=True)
        if oxi_guess is not None
        else "",
    }


def rdf_fingerprint(structure: Structure, cutoff: float, bins: int) -> np.ndarray:
    dists: list[float] = []
    try:
        for neighs in structure.get_all_neighbors(r=cutoff):
            for nn in neighs:
                dist = float(nn.nn_distance)
                if dist > 1e-6:
                    dists.append(dist)
    except Exception:
        return np.zeros(bins, dtype=np.float64)

    hist, _ = np.histogram(dists, bins=bins, range=(0.0, cutoff))
    fp = hist.astype(np.float64)
    total = fp.sum()
    if total > 0:
        fp /= total
    return fp


def lattice_feature(structure: Structure) -> np.ndarray:
    lengths = np.array(structure.lattice.abc, dtype=np.float64)
    angles = np.array(structure.lattice.angles, dtype=np.float64)
    volume_per_atom = structure.volume / max(1, structure.num_sites)
    return np.concatenate(
        [
            np.log(np.clip(lengths, 1e-8, None)),
            angles / 180.0,
            np.array([math.log(max(volume_per_atom, 1e-8))], dtype=np.float64),
        ]
    )


def load_reference_records(
    root: Path | None,
    *,
    splits: list[str],
    nmax: int,
    limit: int,
) -> list[ReferenceRecord]:
    if root is None:
        return []
    if not root.exists():
        raise FileNotFoundError(f"Reference data root does not exist: {root}")

    records: list[ReferenceRecord] = []
    seen: set[tuple[str, int]] = set()
    for split in splits:
        ds = MP20Tokens(
            root=str(root),
            augment_translate=False,
            split=split,
            nmax=nmax,
        )
        for i in range(len(ds)):
            item = ds[i]
            try:
                struct = tokens_to_structure(item)
            except Exception:
                continue
            key = (split, i)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                ReferenceRecord(
                    ref_idx=i,
                    split=split,
                    mp_id=str(item.get("mp_id") or f"{split}:{i}"),
                    structure=struct,
                    reduced_formula=struct.composition.reduced_formula,
                    chemsys=chemsys(struct),
                    volume_per_atom=struct.volume / max(1, struct.num_sites),
                )
            )
            if limit > 0 and len(records) >= limit:
                return records
    return records


def build_reference_index(
    refs: list[ReferenceRecord],
) -> tuple[dict[str, list[ReferenceRecord]], dict[tuple[str, ...], list[ReferenceRecord]]]:
    by_formula: dict[str, list[ReferenceRecord]] = {}
    by_chemsys: dict[tuple[str, ...], list[ReferenceRecord]] = {}
    for rec in refs:
        by_formula.setdefault(rec.reduced_formula, []).append(rec)
        by_chemsys.setdefault(rec.chemsys, []).append(rec)
    return by_formula, by_chemsys


def prefilter_reference_pool(
    structure: Structure,
    pool: list[ReferenceRecord],
    max_reference_candidates: int,
) -> list[ReferenceRecord]:
    if len(pool) <= max_reference_candidates:
        return pool
    cand_feat = lattice_feature(structure)

    def score(rec: ReferenceRecord) -> float:
        ref_feat = lattice_feature(rec.structure)
        return float(np.linalg.norm(cand_feat - ref_feat))

    return sorted(pool, key=score)[:max_reference_candidates]


def nearest_reference_summary(
    structure: Structure,
    refs: list[ReferenceRecord],
    by_formula: dict[str, list[ReferenceRecord]],
    by_chemsys: dict[tuple[str, ...], list[ReferenceRecord]],
    *,
    matcher: StructureMatcher,
    max_reference_candidates: int,
    rdf_cutoff: float,
    rdf_bins: int,
) -> dict[str, Any]:
    if not refs:
        return {}

    formula = structure.composition.reduced_formula
    system = chemsys(structure)
    pool_kind = "formula"
    pool = by_formula.get(formula, [])
    if not pool:
        pool_kind = "chemsys"
        pool = by_chemsys.get(system, [])
    if not pool:
        pool_kind = "all"
        pool = refs
    pool_size_unfiltered = len(pool)
    pool = prefilter_reference_pool(
        structure,
        pool,
        max_reference_candidates=max_reference_candidates,
    )

    best_sm: tuple[float, float | None, ReferenceRecord] | None = None
    best_lattice: tuple[float, ReferenceRecord] | None = None
    best_rdf: tuple[float, ReferenceRecord] | None = None
    cand_lattice = lattice_feature(structure)
    cand_rdf = rdf_fingerprint(structure, cutoff=rdf_cutoff, bins=rdf_bins)

    for rec in pool:
        lat_dist = float(np.linalg.norm(cand_lattice - lattice_feature(rec.structure)))
        if best_lattice is None or lat_dist < best_lattice[0]:
            best_lattice = (lat_dist, rec)

        ref_rdf = rdf_fingerprint(rec.structure, cutoff=rdf_cutoff, bins=rdf_bins)
        rdf_dist = float(np.linalg.norm(cand_rdf - ref_rdf))
        if best_rdf is None or rdf_dist < best_rdf[0]:
            best_rdf = (rdf_dist, rec)

        try:
            rms = matcher.get_rms_dist(structure, rec.structure)
        except Exception:
            rms = None
        if rms is None:
            continue
        rms_val = float(rms[0])
        max_val = float(rms[1]) if isinstance(rms, tuple) and len(rms) > 1 else None
        if best_sm is None or rms_val < best_sm[0]:
            best_sm = (rms_val, max_val, rec)

    out: dict[str, Any] = {
        "reference_pool": pool_kind,
        "reference_pool_size": pool_size_unfiltered,
        "reference_scored": len(pool),
    }
    if best_sm is not None:
        _, _, rec = best_sm
        out.update(
            {
                "sm_match_ref_id": rec.mp_id,
                "sm_match_ref_split": rec.split,
                "sm_match_ref_idx": rec.ref_idx,
                "sm_match_rms": best_sm[0],
                "sm_match_max_dist": best_sm[1],
            }
        )
    if best_lattice is not None:
        _, rec = best_lattice
        out.update(
            {
                "lattice_nearest_ref_id": rec.mp_id,
                "lattice_nearest_ref_split": rec.split,
                "lattice_nearest_distance": best_lattice[0],
            }
        )
    if best_rdf is not None:
        _, rec = best_rdf
        out.update(
            {
                "rdf_nearest_ref_id": rec.mp_id,
                "rdf_nearest_ref_split": rec.split,
                "rdf_nearest_distance": best_rdf[0],
            }
        )
    return out


def load_phase_diagram(path: Path | None) -> Any | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Phase diagram path does not exist: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        return pickle.load(handle)


def decomposition_summary(
    phase_diagram: Any | None,
    structure: Structure,
    *,
    e_total: float | None,
    entry_id: str,
) -> dict[str, Any]:
    if phase_diagram is None or e_total is None:
        if phase_diagram is None:
            return {}
    out: dict[str, Any] = {}
    try:
        decomp = phase_diagram.get_decomposition(structure.composition)
        products = []
        for product_entry, amount in sorted(
            decomp.items(),
            key=lambda item: item[0].composition.reduced_formula,
        ):
            products.append(
                {
                    "formula": product_entry.composition.reduced_formula,
                    "entry_id": str(getattr(product_entry, "entry_id", "")),
                    "amount": float(amount),
                }
            )
        out["decomposition_products_json"] = json.dumps(products, sort_keys=True)
    except Exception as exc:
        out["decomposition_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

    if e_total is None:
        return out

    try:
        entry = ComputedStructureEntry(
            composition=structure.composition,
            energy=float(e_total),
            structure=structure,
            entry_id=entry_id,
        )
        decomp, e_above = phase_diagram.get_decomp_and_e_above_hull(
            entry,
            allow_negative=True,
            on_error="raise",
        )
    except Exception as exc:
        out["phase_diagram_e_above_hull_error"] = (
            f"{type(exc).__name__}: {str(exc)[:200]}"
        )
        return out

    out["phase_diagram_e_above_hull"] = float(e_above)
    return out


def structure_basic_summary(structure: Structure) -> dict[str, Any]:
    lengths = structure.lattice.abc
    angles = structure.lattice.angles
    return {
        "formula": structure.composition.formula,
        "reduced_formula": structure.composition.reduced_formula,
        "num_sites": structure.num_sites,
        "volume": structure.volume,
        "volume_per_atom": structure.volume / max(1, structure.num_sites),
        "a": lengths[0],
        "b": lengths[1],
        "c": lengths[2],
        "alpha": angles[0],
        "beta": angles[1],
        "gamma": angles[2],
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}g}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Crystalite Candidate Case Studies",
        "",
        "These candidates are selected from generated CIFs by low nonnegative "
        "manifest `e_above_hull`. Values are MLIP/phase-diagram screening "
        "quantities, not DFT-confirmed discovery claims.",
        "",
        "| sample | formula | e_hull | SG | SMACT | SM match | RDF nearest | CIF |",
        "|---:|---|---:|---|---|---|---|---|",
    ]
    for row in rows:
        sg = ""
        if row.get("spacegroup_symbol"):
            sg = f"{row.get('spacegroup_symbol')} ({row.get('spacegroup_number')})"
        sm = row.get("sm_match_ref_id") or ""
        if row.get("sm_match_rms") is not None:
            sm = f"{sm}, RMS={fmt(row.get('sm_match_rms'))}"
        rdf = row.get("rdf_nearest_ref_id") or ""
        if row.get("rdf_nearest_distance") is not None:
            rdf = f"{rdf}, d={fmt(row.get('rdf_nearest_distance'))}"
        lines.append(
            "| {sample_idx} | {formula} | {ehull} | {sg} | {smact} | {sm} | {rdf} | {cif} |".format(
                sample_idx=row.get("sample_idx", ""),
                formula=row.get("reduced_formula") or row.get("formula", ""),
                ehull=fmt(row.get("e_above_hull")),
                sg=sg,
                smact=row.get("smact_valid", ""),
                sm=sm,
                rdf=rdf,
                cif=row.get("copied_cif", row.get("cif_path", "")),
            )
        )

    lines.extend(["", "## Details", ""])
    for row in rows:
        lines.extend(
            [
                f"### sample_{int(row['sample_idx']):05d}",
                "",
                f"- Formula: {row.get('formula')} ({row.get('reduced_formula')})",
                f"- Sites: {row.get('num_sites')}",
                f"- Manifest e_above_hull: {fmt(row.get('e_above_hull'))} eV/atom",
                f"- Symmetry: {row.get('spacegroup_symbol')} "
                f"({row.get('spacegroup_number')}), {row.get('crystal_system')}",
                f"- SMACT validity: {row.get('smact_valid')}",
                f"- Oxidation-state guess: {row.get('oxidation_guess_json') or 'none'}",
            ]
        )
        if row.get("e_form") is not None:
            lines.append(f"- Manifest formation energy: {fmt(row.get('e_form'))} eV/atom")
        if row.get("reference_pool"):
            lines.append(
                f"- Reference search: pool={row.get('reference_pool')}, "
                f"pool_size={row.get('reference_pool_size')}, "
                f"scored={row.get('reference_scored')}"
            )
        if row.get("sm_match_ref_id"):
            lines.append(
                f"- StructureMatcher equivalent/tolerance match: "
                f"{row.get('sm_match_ref_id')} (split={row.get('sm_match_ref_split')}, "
                f"RMS={fmt(row.get('sm_match_rms'))}, "
                f"max_dist={fmt(row.get('sm_match_max_dist'))})"
            )
        elif row.get("reference_pool"):
            lines.append("- StructureMatcher equivalent/tolerance match: none found")
        if row.get("lattice_nearest_ref_id"):
            lines.append(
                f"- Lattice-feature nearest reference: {row.get('lattice_nearest_ref_id')} "
                f"(split={row.get('lattice_nearest_ref_split')}, "
                f"distance={fmt(row.get('lattice_nearest_distance'))})"
            )
        if row.get("rdf_nearest_ref_id"):
            lines.append(
                f"- RDF nearest reference: {row.get('rdf_nearest_ref_id')} "
                f"(split={row.get('rdf_nearest_ref_split')}, "
                f"distance={fmt(row.get('rdf_nearest_distance'))})"
            )
        if row.get("decomposition_products_json"):
            lines.append(
                f"- Decomposition products: `{row.get('decomposition_products_json')}`"
            )
        if row.get("decomposition_error"):
            lines.append(f"- Decomposition error: {row.get('decomposition_error')}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_cif_dir = args.out_dir / "selected_cifs"
    selected_cif_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest(args.manifest)
    candidates = select_candidates(
        manifest_rows,
        top_k=max(int(args.top_k), int(args.candidate_pool_size)),
        min_e_above_hull=float(args.min_e_above_hull),
        max_e_above_hull=float(args.max_e_above_hull),
    )
    if not candidates:
        raise SystemExit("No successful candidates found within the e_above_hull window.")

    refs = load_reference_records(
        args.reference_data_root,
        splits=list(args.reference_splits),
        nmax=int(args.nmax),
        limit=int(args.reference_limit),
    )
    by_formula, by_chemsys = build_reference_index(refs)
    matcher = StructureMatcher(
        stol=float(args.stol),
        ltol=float(args.ltol),
        angle_tol=float(args.angle_tol),
    )
    phase_diagram = load_phase_diagram(args.phase_diagram)

    output_rows: list[dict[str, Any]] = []
    for row in candidates:
        sample_idx = int(row.get("sample_idx", len(output_rows)))
        structure = load_structure_for_row(args.cif_dir, row)
        file_name = row.get("file") or f"sample_{sample_idx:05d}.cif"
        cif_path = args.cif_dir / str(file_name)
        oxidation = oxidation_summary(structure)
        if bool(args.require_smact_valid) and oxidation.get("smact_valid") is not True:
            continue

        result: dict[str, Any] = {
            "sample_idx": sample_idx,
            "manifest_file": str(args.manifest),
            "cif_path": str(cif_path),
            "success": row_success(row),
            "e_above_hull": finite_float(row.get("e_above_hull")),
            "e_form": finite_float(row.get("e_form")),
            "e_total": finite_float(row.get("e_total")),
            "nsteps": row.get("nsteps"),
            "manifest_formula": row.get("relaxed_formula") or row.get("formula"),
        }
        result.update(structure_basic_summary(structure))
        result.update(symmetry_summary(structure, symprec=float(args.symprec)))
        result.update(oxidation)
        result.update(
            nearest_reference_summary(
                structure,
                refs,
                by_formula,
                by_chemsys,
                matcher=matcher,
                max_reference_candidates=int(args.max_reference_candidates),
                rdf_cutoff=float(args.rdf_cutoff),
                rdf_bins=int(args.rdf_bins),
            )
        )
        result.update(
            decomposition_summary(
                phase_diagram,
                structure,
                e_total=finite_float(row.get("e_total")),
                entry_id=f"generated-{sample_idx}",
            )
        )
        copied = selected_cif_dir / f"sample_{sample_idx:05d}.cif"
        shutil.copy2(cif_path, copied)
        result["copied_cif"] = str(copied)
        output_rows.append(result)
        if len(output_rows) >= int(args.top_k):
            break

    if not output_rows:
        raise SystemExit(
            "No candidates survived final filters. Try relaxing --min_e_above_hull "
            "or omitting --require_smact_valid."
        )

    csv_path = args.out_dir / "case_studies.csv"
    json_path = args.out_dir / "case_studies.json"
    md_path = args.out_dir / "case_studies.md"

    fieldnames = sorted({key for row in output_rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    json_path.write_text(json.dumps(output_rows, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(output_rows, md_path)

    print(f"[case-studies] wrote {csv_path}")
    print(f"[case-studies] wrote {json_path}")
    print(f"[case-studies] wrote {md_path}")
    print(f"[case-studies] copied CIFs to {selected_cif_dir}")
    if not refs:
        print("[case-studies] reference search skipped; pass --reference_data_root on the server.")
    if phase_diagram is None:
        print("[case-studies] decomposition skipped; pass --phase_diagram on the server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
