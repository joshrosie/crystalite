#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "jbungle/crystalite-datasets"
VALID_DATASETS = {"mp20", "alex_mp20", "mpts_52"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download selected datasets from a Hugging Face dataset repo."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(VALID_DATASETS) + ["all"],
        help="Which dataset folders to download (e.g. --datasets mp20 mpts_52).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./data"),
        help="Local output directory.",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch/tag/commit to download.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HF token for private repos.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print available dataset names and exit.",
    )
    return parser.parse_args()


def choose_datasets_interactively() -> list[str]:
    print("Available datasets:")
    for i, name in enumerate(sorted(VALID_DATASETS), start=1):
        print(f"  {i}. {name}")
    print("  4. all")

    raw = input(
        "\nEnter one or more choices separated by commas "
        "(example: 1,3 or mp20,mpts_52 or all): "
    ).strip()

    if not raw:
        raise ValueError("No selection provided.")

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    mapped: list[str] = []

    index_map = {
        "1": "alex_mp20",
        "2": "mp20",
        "3": "mpts_52",
        "4": "all",
    }

    for p in parts:
        p = index_map.get(p, p)
        if p == "all":
            return sorted(VALID_DATASETS)
        if p not in VALID_DATASETS:
            raise ValueError(f"Invalid selection: {p}")
        mapped.append(p)

    # deduplicate while preserving order
    seen = set()
    result = []
    for x in mapped:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def resolve_datasets(arg_datasets: list[str] | None) -> list[str]:
    if arg_datasets:
        if "all" in arg_datasets:
            return sorted(VALID_DATASETS)
        return list(dict.fromkeys(arg_datasets))

    return choose_datasets_interactively()


def download_dataset(
    repo_id: str,
    dataset_name: str,
    out_dir: Path,
    revision: str | None = None,
    token: str | None = None,
) -> str:
    allow_patterns = [f"{dataset_name}/**"]

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=str(out_dir),   # <- not out_dir / dataset_name
        revision=revision,
        token=token,
    )
    return path


def main() -> int:
    args = parse_args()

    if args.list_only:
        print("Available datasets:")
        for name in sorted(VALID_DATASETS):
            print(f"  - {name}")
        return 0

    try:
        selected = resolve_datasets(args.datasets)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Repo: {args.repo_id}")
    print(f"Output dir: {args.out.resolve()}")
    print(f"Selected: {', '.join(selected)}\n")

    args.out.mkdir(parents=True, exist_ok=True)

    for name in selected:
        print(f"Downloading {name}...")
        try:
            local_path = download_dataset(
                repo_id=args.repo_id,
                dataset_name=name,
                out_dir=args.out,
                revision=args.revision,
                token=args.token,
            )
            print(f"  done -> {local_path}\n")
        except Exception as e:
            print(f"  failed for {name}: {e}\n", file=sys.stderr)

    print("Finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())