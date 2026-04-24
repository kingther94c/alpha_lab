"""Minimal logging helper. Use in place of print() in reusable code."""

from __future__ import annotations

import logging

_DEFAULT_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "alpha_lab", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger. Idempotent — safe to call in notebooks."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FMT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
