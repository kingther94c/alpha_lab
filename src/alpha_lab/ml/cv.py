"""Time-series cross-validation splitters.

Produces :class:`Split` objects (``.train, .val, .embargo`` as
``DatetimeIndex``) without touching the underlying data. Callers slice
their features / labels with ``df.loc[split.train]``.

Splitter taxonomy
-----------------
- :class:`WalkForwardSplit` — chained expanding/rolling train→val windows.
  Standard walk-forward analysis.
- :class:`PurgedKFold` — Lopez de Prado purged-embargo K-fold for
  overlapping multi-horizon labels. Training rows whose label window leaks
  into the validation block are purged; an embargo follows the validation
  block.
- :class:`BlockBootstrap` — stationary or circular block bootstrap for
  uncertainty estimation on time-series metrics.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    """One train / val pair plus its fold id and (if purged) embargo block."""

    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    fold_id: int
    embargo: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))


class WalkForwardSplit:
    """Walk-forward (expanding or rolling) train→val splits.

    Parameters
    ----------
    train_size : length of the train window
    val_size : length of the validation window
    step : how far to advance the train-end pointer between folds
    embargo : gap inserted between the train end and the val start
        (useful when label horizons could cross the boundary)
    mode : ``"expanding"`` (train always starts at the global start) or
        ``"rolling"`` (train window slides; constant size)
    """

    def __init__(
        self,
        *,
        train_size: pd.Timedelta | str,
        val_size: pd.Timedelta | str,
        step: pd.Timedelta | str,
        embargo: pd.Timedelta | str = pd.Timedelta(0),
        mode: Literal["expanding", "rolling"] = "expanding",
    ):
        self.train_size = pd.Timedelta(train_size)
        self.val_size = pd.Timedelta(val_size)
        self.step = pd.Timedelta(step)
        self.embargo = pd.Timedelta(embargo)
        if mode not in ("expanding", "rolling"):
            raise ValueError(f"mode must be 'expanding' or 'rolling', got {mode!r}")
        self.mode = mode

    def split(self, index: pd.DatetimeIndex) -> Iterator[Split]:
        if len(index) == 0:
            return
        idx_sorted = index.sort_values()
        start = idx_sorted.min()
        end = idx_sorted.max()
        train_end = start + self.train_size
        fold_id = 0
        while True:
            val_start = train_end + self.embargo
            val_end = val_start + self.val_size
            if val_end > end:
                break
            train_start = start if self.mode == "expanding" else train_end - self.train_size
            train = idx_sorted[(idx_sorted >= train_start) & (idx_sorted < train_end)]
            val = idx_sorted[(idx_sorted >= val_start) & (idx_sorted < val_end)]
            if len(train) == 0 or len(val) == 0:
                break
            yield Split(train=train, val=val, fold_id=fold_id)
            train_end = train_end + self.step
            fold_id += 1


class PurgedKFold:
    """Purged + embargoed K-fold for overlapping labels.

    For each validation fold ``[val_start, val_end]``, training rows whose
    label window ``(t, t + label_horizon]`` overlaps the fold are purged.
    An additional ``embargo`` period after the validation block is also
    excluded from training.

    Parameters
    ----------
    n_splits : number of folds (>= 2)
    label_horizon : the maximum label horizon used by any feature/label
        downstream. Required — researchers must state this explicitly so
        the purge window is sized correctly.
    embargo : extra exclusion period after the val block (default 0)
    """

    def __init__(
        self,
        *,
        n_splits: int,
        label_horizon: pd.Timedelta | str,
        embargo: pd.Timedelta | str = pd.Timedelta(0),
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = int(n_splits)
        self.label_horizon = pd.Timedelta(label_horizon)
        self.embargo = pd.Timedelta(embargo)

    def split(self, index: pd.DatetimeIndex) -> Iterator[Split]:
        if len(index) < self.n_splits:
            return
        idx_sorted = index.sort_values()
        n = len(idx_sorted)
        fold_size = n // self.n_splits
        bounds = [
            (i * fold_size, (i + 1) * fold_size if i < self.n_splits - 1 else n)
            for i in range(self.n_splits)
        ]
        for fid, (s, e) in enumerate(bounds):
            val = idx_sorted[s:e]
            val_start = val.min()
            val_end = val.max()
            purge_lo = val_start - self.label_horizon
            purge_hi = val_end + self.embargo
            # Exclude any row in [purge_lo, purge_hi] from training
            mask = ~((idx_sorted >= purge_lo) & (idx_sorted <= purge_hi))
            train = idx_sorted[mask]
            embargo_idx = idx_sorted[(idx_sorted > val_end) & (idx_sorted <= purge_hi)]
            yield Split(train=train, val=val, fold_id=fid, embargo=embargo_idx)


class BlockBootstrap:
    """Stationary or circular block bootstrap (Politis–Romano).

    Yields ``n_resamples`` ``DatetimeIndex`` objects of the same length as
    the input, each constructed by concatenating blocks of consecutive
    (modulo n) positions. Use for confidence intervals on Sharpe, CAGR, etc.

    Parameters
    ----------
    block_size : either a ``pd.Timedelta`` (converted to integer bars via
        the median consecutive-diff) or an integer number of bars.
    n_resamples : number of resamples to yield (default 1000).
    mode : ``"stationary"`` (block length ~ Geometric(1/block_size); average
        length = block_size) or ``"circular"`` (fixed block length).
    seed : RNG seed for reproducibility.
    """

    def __init__(
        self,
        *,
        block_size: pd.Timedelta | int,
        n_resamples: int = 1000,
        mode: Literal["stationary", "circular"] = "stationary",
        seed: int | None = None,
    ):
        self.block_size = block_size
        self.n_resamples = int(n_resamples)
        if mode not in ("stationary", "circular"):
            raise ValueError(f"mode must be 'stationary' or 'circular', got {mode!r}")
        self.mode = mode
        self.seed = seed

    def _bsize_bars(self, index: pd.DatetimeIndex) -> int:
        if isinstance(self.block_size, pd.Timedelta):
            if len(index) < 2:
                return 1
            diffs = index[1:] - index[:-1]
            bar = pd.Series(diffs).value_counts().idxmax()
            if bar <= pd.Timedelta(0):
                return 1
            return max(1, int(self.block_size / bar))
        return max(1, int(self.block_size))

    def resample(self, index: pd.DatetimeIndex) -> Iterator[pd.DatetimeIndex]:
        rng = np.random.default_rng(self.seed)
        idx_sorted = index.sort_values()
        n = len(idx_sorted)
        if n == 0:
            return
        bsize = self._bsize_bars(idx_sorted)
        p = 1.0 / bsize  # geometric block-length parameter
        for _ in range(self.n_resamples):
            positions: list[int] = []
            while len(positions) < n:
                start = int(rng.integers(0, n))
                length = int(rng.geometric(p)) if self.mode == "stationary" else bsize
                for k in range(length):
                    if len(positions) >= n:
                        break
                    positions.append((start + k) % n)
            yield idx_sorted[positions]
