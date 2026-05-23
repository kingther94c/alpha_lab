"""Tests for src/alpha_lab/ml/cv.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.ml.cv import BlockBootstrap, PurgedKFold, Split, WalkForwardSplit


# --- WalkForwardSplit ------------------------------------------------------

def test_walk_forward_expanding_chain():
    idx = pd.date_range("2024-01-01", "2024-06-30", freq="D", tz="UTC")
    wf = WalkForwardSplit(
        train_size="60D", val_size="30D", step="30D", mode="expanding",
    )
    splits = list(wf.split(idx))
    assert len(splits) >= 2
    # Trains all start at the same point (expanding)
    starts = {s.train.min() for s in splits}
    assert len(starts) == 1
    # Trains grow monotonically
    train_lengths = [len(s.train) for s in splits]
    assert train_lengths == sorted(train_lengths)
    # Each train precedes its val
    for s in splits:
        assert s.train.max() < s.val.min()


def test_walk_forward_rolling_chain():
    idx = pd.date_range("2024-01-01", "2024-06-30", freq="D", tz="UTC")
    wf = WalkForwardSplit(
        train_size="60D", val_size="30D", step="30D", mode="rolling",
    )
    splits = list(wf.split(idx))
    assert len(splits) >= 2
    # Rolling: train lengths roughly constant (give or take 1 day for edge effects)
    train_lengths = [len(s.train) for s in splits]
    assert max(train_lengths) - min(train_lengths) <= 1


def test_walk_forward_with_embargo():
    idx = pd.date_range("2024-01-01", "2024-06-30", freq="D", tz="UTC")
    wf = WalkForwardSplit(
        train_size="30D", val_size="30D", step="30D", embargo="7D",
    )
    splits = list(wf.split(idx))
    assert len(splits) >= 1
    for s in splits:
        gap_days = (s.val.min() - s.train.max()).days
        assert gap_days >= 7


def test_walk_forward_invalid_mode():
    with pytest.raises(ValueError):
        WalkForwardSplit(train_size="10D", val_size="5D", step="5D", mode="weird")


# --- PurgedKFold -----------------------------------------------------------

def test_purged_kfold_no_train_val_overlap():
    idx = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    pkf = PurgedKFold(n_splits=5, label_horizon="3D")
    splits = list(pkf.split(idx))
    assert len(splits) == 5
    for s in splits:
        train_set = set(s.train)
        val_set = set(s.val)
        assert len(train_set & val_set) == 0


def test_purged_kfold_purges_label_horizon():
    """Training rows within ``label_horizon`` before the val window must be
    excluded (their labels would peek into val)."""
    idx = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    pkf = PurgedKFold(n_splits=5, label_horizon="5D")
    splits = list(pkf.split(idx))
    # Pick the middle fold
    mid = splits[2]
    val_start = mid.val.min()
    # No training row should fall in [val_start - 5D, val_start)
    purge_window_start = val_start - pd.Timedelta("5D")
    in_purge = [t for t in mid.train if purge_window_start <= t < val_start]
    assert in_purge == []


def test_purged_kfold_embargo_excluded_from_train():
    idx = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    pkf = PurgedKFold(n_splits=5, label_horizon="0D", embargo="7D")
    splits = list(pkf.split(idx))
    for s in splits:
        val_end = s.val.max()
        # Rows in (val_end, val_end + 7D] must NOT be in train
        embargo_window_end = val_end + pd.Timedelta("7D")
        in_embargo = [t for t in s.train if val_end < t <= embargo_window_end]
        assert in_embargo == []


def test_purged_kfold_n_splits_validation():
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1, label_horizon="1D")


# --- BlockBootstrap --------------------------------------------------------

def test_block_bootstrap_reproducible_with_seed():
    idx = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    bb1 = BlockBootstrap(block_size=5, n_resamples=3, seed=42)
    bb2 = BlockBootstrap(block_size=5, n_resamples=3, seed=42)
    r1 = [r.tolist() for r in bb1.resample(idx)]
    r2 = [r.tolist() for r in bb2.resample(idx)]
    assert r1 == r2


def test_block_bootstrap_different_seeds_differ():
    idx = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    bb1 = BlockBootstrap(block_size=5, n_resamples=2, seed=1)
    bb2 = BlockBootstrap(block_size=5, n_resamples=2, seed=2)
    r1 = [r.tolist() for r in bb1.resample(idx)]
    r2 = [r.tolist() for r in bb2.resample(idx)]
    assert r1 != r2


def test_block_bootstrap_length_equals_input():
    idx = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
    bb = BlockBootstrap(block_size=4, n_resamples=5, seed=0)
    for sample in bb.resample(idx):
        assert len(sample) == 20


def test_block_bootstrap_timedelta_block_size():
    idx = pd.date_range("2024-01-01", periods=30, freq="1h", tz="UTC")
    # 6-hour blocks on 1-hour bars = 6 bars per block
    bb = BlockBootstrap(block_size=pd.Timedelta("6h"), n_resamples=2, seed=0, mode="circular")
    samples = list(bb.resample(idx))
    assert len(samples) == 2
    assert all(len(s) == 30 for s in samples)


def test_block_bootstrap_circular_mode_fixed_blocks():
    idx = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    bb = BlockBootstrap(block_size=5, n_resamples=1, seed=0, mode="circular")
    sample = next(bb.resample(idx))
    assert len(sample) == 30
