"""YAML config loading.

Usage:
    from alpha_lab.utils.config import load_config
    cfg = load_config("default")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from alpha_lab.utils.paths import CONFIGS_DIR


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(name: str = "default") -> dict[str, Any]:
    """Load ``configs/<name>.yaml``."""
    return load_yaml(CONFIGS_DIR / f"{name}.yaml")
