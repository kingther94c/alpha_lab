"""Parquet / CSV read-write-cache helpers.

Keeps cache code out of notebooks. ``cached_parquet`` is the common
build-or-load pattern: if the file already exists, load it; else build it,
write it, and return it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from alpha_lab.utils.paths import INTERIM_DIR, ensure_dir


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a parquet file into a DataFrame."""
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write *df* to parquet, creating parent dirs as needed. Returns the path."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p)
    return p


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Thin wrapper around pandas.read_csv."""
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    """Write *df* to csv (no index by default), creating parent dirs as needed."""
    p = Path(path)
    ensure_dir(p.parent)
    kwargs.setdefault("index", False)
    df.to_csv(p, **kwargs)
    return p


def cached_parquet(
    key: str,
    builder: Callable[[], pd.DataFrame],
    cache_dir: str | Path = INTERIM_DIR,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Build-or-load cache.

    Parameters
    ----------
    key : filename (without extension) used under *cache_dir*.
    builder : zero-arg callable that returns a DataFrame when the cache miss.
    cache_dir : directory to store the parquet file in.
    refresh : force rebuild even if cache exists.
    """
    path = Path(cache_dir) / f"{key}.parquet"
    if path.exists() and not refresh:
        return read_parquet(path)
    df = builder()
    write_parquet(df, path)
    return df
