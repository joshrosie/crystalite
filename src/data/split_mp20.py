from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _ensure_raw_all(root: Path) -> Path:
    """Guarantee raw/all.csv exists by downloading from HF if needed."""
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_csv = raw_dir / "all.csv"
    if all_csv.exists():
        return all_csv
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to download MP20 raw data.") from exc

    hf_hub_download(
        repo_id="chaitjo/MP20_ADiT",
        filename="raw/all.csv",
        repo_type="dataset",
        local_dir=str(root),
    )
    if not all_csv.exists():
        raise FileNotFoundError(f"Failed to download {all_csv}")
    return all_csv


def _split_dataframe(
    df: pd.DataFrame,
    *,
    strategy: str,
    train_frac: float,
    seed: int,
    train_len: int,
    val_len: int,
):
    """Return train/val/test DataFrames according to the chosen strategy."""
    if strategy == "random":
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_train = int(len(df) * train_frac)
        n_val = len(df) - n_train
        n_test = 0
    else:  # contiguous
        n_train = min(train_len, len(df))
        n_val = min(val_len, max(len(df) - n_train, 0))
        n_test = max(len(df) - n_train - n_val, 0)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]
    return train_df, val_df, test_df


def auto_split_if_missing(
    data_root: str,
    strategy: str = "contiguous",
    train_frac: float = 0.9,
    seed: int = 42,
    train_len: int = 27138,
    val_len: int = 9046,
) -> bool:
    """Create train/val/test CSVs if they do not already exist.

    Returns True if new split files were written, False if splits already existed.
    """
    root = Path(data_root)
    raw_dir = root / "raw"
    train_path = raw_dir / "train.csv"
    val_path = raw_dir / "val.csv"
    test_path = raw_dir / "test.csv"

    if train_path.exists() and val_path.exists():
        return False

    all_csv = _ensure_raw_all(root)
    df = pd.read_csv(all_csv)
    train_df, val_df, test_df = _split_dataframe(
        df,
        strategy=strategy,
        train_frac=train_frac,
        seed=seed,
        train_len=train_len,
        val_len=val_len,
    )

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    if len(test_df) > 0:
        test_df.to_csv(test_path, index=False)

    print(
        f"[auto_split_if_missing] wrote splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/mp20")
    parser.add_argument(
        "--strategy",
        type=str,
        default="contiguous",
        choices=["contiguous", "random"],
        help="contiguous: slice train/val/test without shuffling (matches upstream repo); random: shuffle then split.",
    )
    parser.add_argument("--train_frac", type=float, default=0.9, help="Used when strategy=random")
    parser.add_argument("--seed", type=int, default=42, help="Used when strategy=random")
    parser.add_argument("--train_len", type=int, default=27138, help="Used when strategy=contiguous")
    parser.add_argument("--val_len", type=int, default=9046, help="Used when strategy=contiguous")
    args = parser.parse_args()

    root = Path(args.data_root)
    all_csv = _ensure_raw_all(root)
    df = pd.read_csv(all_csv)
    train_df, val_df, test_df = _split_dataframe(
        df,
        strategy=args.strategy,
        train_frac=args.train_frac,
        seed=args.seed,
        train_len=args.train_len,
        val_len=args.val_len,
    )

    raw_dir = root / "raw"
    train_path = raw_dir / "train.csv"
    val_path = raw_dir / "val.csv"
    test_path = raw_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    if len(test_df) > 0:
        test_df.to_csv(test_path, index=False)

    print(f"Wrote {len(train_df)} rows to {train_path}")
    print(f"Wrote {len(val_df)} rows to {val_path}")
    if len(test_df) > 0:
        print(f"Wrote {len(test_df)} rows to {test_path}")


if __name__ == "__main__":
    main()
