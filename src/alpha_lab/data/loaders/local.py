"""Local file loaders — for manual pulls and IBKR exports.

Uses a glob pattern so you can drop CSV or parquet snapshots into
``data/raw/<source>/`` and load them without custom parsing per dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_parquet_dir(path: str | Path, pattern: str = "*.parquet") -> pd.DataFrame:
    """Concatenate every matching parquet file under *path*.

    Each file is read and appended; a ``__source__`` column with the file stem
    is added so rows can be traced back to their origin.
    """
    root = Path(path)
    files = sorted(root.glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = []
    for fp in files:
        df = pd.read_parquet(fp)
        df["__source__"] = fp.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_csv_dir(path: str | Path, pattern: str = "*.csv", **read_csv_kwargs) -> pd.DataFrame:
    """Same idea, but for CSVs. Extra kwargs are passed to ``pd.read_csv``."""
    root = Path(path)
    files = sorted(root.glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = []
    for fp in files:
        df = pd.read_csv(fp, **read_csv_kwargs)
        df["__source__"] = fp.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
