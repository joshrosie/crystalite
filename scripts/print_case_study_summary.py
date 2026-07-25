#!/usr/bin/env python3
"""Print a compact, terminal-friendly summary of case-study CSV output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_COLUMNS = [
    "sample_idx",
    "reduced_formula",
    "e_above_hull",
    "e_form",
    "spacegroup_symbol",
    "spacegroup_number",
    "smact_valid",
    "oxidation_guess_json",
    "is_unique",
    "is_novel",
    "is_un",
    "is_msun",
    "reference_pool",
    "sm_match_ref_id",
    "sm_match_ref_split",
    "rdf_nearest_ref_id",
    "rdf_nearest_ref_split",
    "rdf_nearest_distance",
    "decomposition_products_json",
]


def resolve_csv(path: Path) -> Path:
    if path.is_dir():
        path = path / "case_studies.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=Path,
        help="Path to case_studies.csv or an output directory containing it.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=DEFAULT_COLUMNS,
        help="Columns to print. Defaults to reviewer-relevant fields.",
    )
    args = parser.parse_args()

    csv_path = resolve_csv(args.path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    print(f"[case-summary] {csv_path}")
    print(f"[case-summary] rows={len(rows)}")
    for row in rows:
        print()
        print("=" * 100)
        for col in args.columns:
            print(f"{col}: {row.get(col, '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
