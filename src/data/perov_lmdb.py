from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _record_to_cif(record: dict) -> str:
    cell = _to_numpy(record["cell"]).astype(np.float64)
    pos = _to_numpy(record["pos"]).astype(np.float64)
    atomic_numbers = _to_numpy(record["atomic_numbers"]).astype(np.int64).tolist()

    structure = Structure(
        lattice=Lattice(cell),
        species=atomic_numbers,
        coords=pos,
        coords_are_cartesian=True,
        to_unit_cell=True,
    )
    return str(CifWriter(structure))


def convert_lmdb_split_to_csv(
    input_lmdb: Path,
    output_csv: Path,
    *,
    material_prefix: str = "perov",
) -> int:
    try:
        import lmdb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LMDB support is required for perov_5 conversion. "
            "Install with: uv pip install lmdb"
        ) from exc

    env = lmdb.open(
        str(input_lmdb),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
    )
    rows = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with env.begin() as txn, output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["material_id", "cif"])
        writer.writeheader()
        for _, value in txn.cursor():
            record = pickle.loads(value)
            raw_id = int(record["ids"])
            writer.writerow(
                {
                    "material_id": f"{material_prefix}_{raw_id}",
                    "cif": _record_to_cif(record),
                }
            )
            rows += 1
    env.close()
    return rows


def ensure_csv_splits_from_lmdb(
    data_root: str | Path,
    *,
    splits: Iterable[str] = ("train", "val", "test"),
    material_prefix: str = "perov",
) -> dict[str, int]:
    """Create missing raw/{split}.csv from raw/{split}.lmdb when possible.

    Returns a map {split: rows_written} for splits that were newly converted.
    """
    root = Path(data_root)
    raw_dir = root / "raw"
    converted: dict[str, int] = {}
    for split in splits:
        csv_path = raw_dir / f"{split}.csv"
        if csv_path.exists():
            continue
        lmdb_path = raw_dir / f"{split}.lmdb"
        if not lmdb_path.exists():
            continue
        converted[split] = convert_lmdb_split_to_csv(
            lmdb_path, csv_path, material_prefix=material_prefix
        )
    return converted
