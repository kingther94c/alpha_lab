"""Project path helpers.

Walks up from this file to find the repo root (marker: ``pyproject.toml``) so
notebooks launched from anywhere under the repo still resolve paths correctly.

An optional ``ALPHA_LAB_DATA_DIR`` env var can redirect ``DATA_DIR`` to an
absolute path outside the repo (useful for large private datasets).
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Walk up from *start* until a directory containing pyproject.toml is found."""
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: two levels up from this file's package dir.
    return start.parents[2]


PROJECT_ROOT: Path = _find_project_root(Path(__file__).resolve())

_data_override = os.environ.get("ALPHA_LAB_DATA_DIR")
DATA_DIR: Path = Path(_data_override).expanduser().resolve() if _data_override else PROJECT_ROOT / "data"

RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
FEATURES_DIR: Path = DATA_DIR / "features"
RESULTS_DIR: Path = DATA_DIR / "results"
PRIVATE_DIR: Path = DATA_DIR / "private"

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create *path* (and parents) if missing, return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
