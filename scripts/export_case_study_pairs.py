#!/usr/bin/env python3
"""Export generated case-study CIFs with their nearest MP20 reference analogues.

The input is a case-study output directory (or its ``case_studies.csv``). For
each row, the script copies the generated CIF and reconstructs the requested
MP20 reference structure from the token dataset, then writes Avogadro-friendly
paired folders plus a zip archive.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mp20_tokens import MP20Tokens, tokens_to_structure  # noqa: E402


REFERENCE_FIELDS = {
    "rdf": ("rdf_nearest_ref_id", "rdf_nearest_ref_split"),
    "lattice": ("lattice_nearest_ref_id", "lattice_nearest_ref_split"),
    "sm": ("sm_match_ref_id", "sm_match_ref_split"),
}


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def resolve_csv(path: Path) -> tuple[Path, Path]:
    if path.is_dir():
        out_dir = path
        csv_path = path / "case_studies.csv"
    else:
        csv_path = path
        out_dir = path.parent
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return csv_path, out_dir


def resolve_generated_cif(row: dict[str, str], out_dir: Path) -> Path:
    for key in ("copied_cif", "cif_path"):
        value = row.get(key)
        if not value:
            continue
        path = Path(value)
        candidates = [path]
        if not path.is_absolute():
            candidates.append(out_dir / path)
            candidates.append(REPO_ROOT / path)
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No generated CIF found for sample {row.get('sample_idx')}")


def load_needed_references(
    root: Path,
    needed: dict[str, set[str]],
    *,
    nmax: int,
) -> dict[tuple[str, str], object]:
    found: dict[tuple[str, str], object] = {}
    for split, mp_ids in sorted(needed.items()):
        if not mp_ids:
            continue
        remaining = set(mp_ids)
        ds = MP20Tokens(
            root=str(root),
            augment_translate=False,
            split=split,
            nmax=nmax,
        )
        for i in range(len(ds)):
            item = ds[i]
            mp_id = str(item.get("mp_id") or "")
            if mp_id not in remaining:
                continue
            found[(split, mp_id)] = tokens_to_structure(item)
            remaining.remove(mp_id)
            if not remaining:
                break
        if remaining:
            print(
                f"[case-pairs] warning: missing references in split={split}: "
                f"{sorted(remaining)}",
                file=sys.stderr,
            )
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=Path,
        help="Case-study output directory or case_studies.csv.",
    )
    parser.add_argument(
        "--reference_data_root",
        type=Path,
        required=True,
        help="MP20 token dataset root, e.g. $HOME/crysfinity/data/mp20.",
    )
    parser.add_argument(
        "--reference_kind",
        choices=sorted(REFERENCE_FIELDS),
        default="rdf",
        help="Which recorded MP20 analogue to export.",
    )
    parser.add_argument("--nmax", type=int, default=20)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <case-study-dir>/avogadro_pairs_<kind>.",
    )
    args = parser.parse_args()

    csv_path, case_dir = resolve_csv(args.path)
    output_dir = args.out_dir or (case_dir / f"avogadro_pairs_{args.reference_kind}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    id_field, split_field = REFERENCE_FIELDS[args.reference_kind]
    needed: dict[str, set[str]] = {}
    for row in rows:
        ref_id = row.get(id_field, "").strip()
        ref_split = row.get(split_field, "").strip()
        if ref_id and ref_split:
            needed.setdefault(ref_split, set()).add(ref_id)

    references = load_needed_references(
        args.reference_data_root,
        needed,
        nmax=int(args.nmax),
    )

    manifest_rows: list[dict[str, str]] = []
    for row in rows:
        sample_idx = str(row.get("sample_idx") or "unknown")
        formula = safe_name(row.get("reduced_formula") or row.get("formula") or "")
        ref_id = row.get(id_field, "").strip()
        ref_split = row.get(split_field, "").strip()
        if not ref_id or not ref_split:
            print(
                f"[case-pairs] warning: sample {sample_idx} has no "
                f"{args.reference_kind} reference; skipping",
                file=sys.stderr,
            )
            continue
        ref_struct = references.get((ref_split, ref_id))
        if ref_struct is None:
            continue

        pair_dir = output_dir / f"sample_{int(sample_idx):05d}_{formula}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        generated_src = resolve_generated_cif(row, case_dir)
        generated_dst = pair_dir / f"generated_sample_{int(sample_idx):05d}_{formula}.cif"
        reference_dst = pair_dir / f"mp20_{args.reference_kind}_{ref_id}_{ref_split}.cif"
        notes_dst = pair_dir / "README.txt"

        shutil.copy2(generated_src, generated_dst)
        ref_struct.to(filename=str(reference_dst), fmt="cif")
        notes_dst.write_text(
            "\n".join(
                [
                    f"sample_idx: {sample_idx}",
                    f"formula: {row.get('reduced_formula')}",
                    f"e_above_hull: {row.get('e_above_hull')}",
                    f"spacegroup: {row.get('spacegroup_symbol')} ({row.get('spacegroup_number')})",
                    f"reference_kind: {args.reference_kind}",
                    f"reference_id: {ref_id}",
                    f"reference_split: {ref_split}",
                    f"rdf_nearest_ref_id: {row.get('rdf_nearest_ref_id')}",
                    f"rdf_nearest_distance: {row.get('rdf_nearest_distance')}",
                    f"lattice_nearest_ref_id: {row.get('lattice_nearest_ref_id')}",
                    f"lattice_nearest_distance: {row.get('lattice_nearest_distance')}",
                    f"sm_match_ref_id: {row.get('sm_match_ref_id')}",
                    f"oxidation_guess_json: {row.get('oxidation_guess_json')}",
                    f"decomposition_products_json: {row.get('decomposition_products_json')}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "sample_idx": sample_idx,
                "formula": row.get("reduced_formula", ""),
                "generated_cif": str(generated_dst),
                "reference_cif": str(reference_dst),
                "reference_kind": args.reference_kind,
                "reference_id": ref_id,
                "reference_split": ref_split,
            }
        )

    manifest_path = output_dir / "pair_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "sample_idx",
            "formula",
            "generated_cif",
            "reference_cif",
            "reference_kind",
            "reference_id",
            "reference_split",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    zip_path = output_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir.parent))

    print(f"[case-pairs] wrote {len(manifest_rows)} pairs to {output_dir}")
    print(f"[case-pairs] wrote manifest {manifest_path}")
    print(f"[case-pairs] wrote zip {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
