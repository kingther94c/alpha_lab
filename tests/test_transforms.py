"""Tests for src/alpha_lab/features/transforms.py — focused on Standardizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.features.transforms import Standardizer


@pytest.fixture
def train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    train_idx = pd.date_range("2024-01-01", periods=200, freq="D", tz="UTC")
    val_idx = pd.date_range("2024-07-19", periods=100, freq="D", tz="UTC")
    train = pd.DataFrame(rng.normal(5.0, 2.0, (200, 2)), index=train_idx, columns=["A", "B"])
    # Val has a DIFFERENT mean/std on purpose — Standardizer must use train's params
    val = pd.DataFrame(rng.normal(100.0, 50.0, (100, 2)), index=val_idx, columns=["A", "B"])
    return train, val


def test_per_column_fit_then_train_is_centered(train_val):
    train, _ = train_val
    sc = Standardizer(mode="per_column")
    z_train = sc.fit_transform(train)
    assert abs(z_train["A"].mean()) < 1e-9
    assert abs(z_train["A"].std() - 1.0) < 0.1   # exactly 1.0 modulo Bessel adjustments


def test_per_column_val_uses_train_params(train_val):
    train, val = train_val
    sc = Standardizer(mode="per_column").fit(train)
    z_val = sc.transform(val)
    # Val has mean ~100; train's mean was ~5; transformed val should be VERY far from 0
    assert z_val["A"].mean() > 20  # nowhere near 0 — proves train params were used
    # And val's standardization is NOT 1
    assert abs(z_val["A"].std() - 1.0) > 1.0


def test_pooled_mode():
    train = pd.DataFrame({"A": [1.0, 2, 3], "B": [4.0, 5, 6]})
    sc = Standardizer(mode="pooled").fit(train)
    # Pooled mean = (1+2+3+4+5+6)/6 = 3.5
    assert isinstance(sc._mean, float)
    assert sc._mean == pytest.approx(3.5)


def test_transform_before_fit_raises():
    sc = Standardizer()
    with pytest.raises(RuntimeError):
        sc.transform(pd.DataFrame({"A": [1.0]}))


def test_winsorize_bounds_applied_on_train(train_val):
    train, val = train_val
    # Add a few outliers
    train_with_outliers = train.copy()
    train_with_outliers.iloc[0, 0] = 1e9
    train_with_outliers.iloc[-1, 1] = -1e9
    sc = Standardizer(mode="per_column", winsorize_bounds=(0.05, 0.95)).fit(train_with_outliers)
    # Bounds stored
    assert sc._winsor_lo is not None
    assert sc._winsor_hi is not None
    # Transform also winsorizes (uses train bounds)
    z = sc.transform(train_with_outliers)
    assert not np.isinf(z.to_numpy()).any()


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        Standardizer(mode="weird")  # type: ignore[arg-type]
