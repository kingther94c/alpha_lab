import numpy as np
import pandas as pd
import pytest

from alpha_lab.portfolio.vol_target import (
    VolTargetInfeasible,
    match_benchmark_vol_weights,
    rolling_match_benchmark_vol_weights,
)


def _cov(vols: list[float]) -> pd.DataFrame:
    labels = ["A", "B"]
    return pd.DataFrame(np.diag(np.square(vols)), index=labels, columns=labels)


def test_match_benchmark_vol_weights_preserves_simplex_and_band():
    raw = pd.Series({"A": 0.5, "B": 0.5})

    result = match_benchmark_vol_weights(
        raw,
        _cov([0.20, 0.10]),
        _cov([0.20, 0.10]),
        benchmark_vol_short=0.15,
        benchmark_vol_long=0.15,
    )

    assert result.weights.sum() == pytest.approx(1.0)
    assert (result.weights >= 0.0).all()
    assert 0.9 - 1e-8 <= result.vol_ratio <= 1.1 + 1e-8


def test_match_benchmark_vol_weights_keeps_feasible_raw_target():
    raw = pd.Series({"A": 1.0, "B": 0.0})

    result = match_benchmark_vol_weights(
        raw,
        _cov([0.15, 0.10]),
        _cov([0.15, 0.10]),
        benchmark_vol_short=0.15,
        benchmark_vol_long=0.15,
    )

    pd.testing.assert_series_equal(result.weights, raw)
    assert result.vol_ratio == pytest.approx(1.0)


def test_match_benchmark_vol_weights_applies_turnover_penalty_to_feasible_raw_target():
    raw = pd.Series({"A": 1.0, "B": 0.0})
    previous = pd.Series({"A": 0.0, "B": 1.0})

    result = match_benchmark_vol_weights(
        raw,
        _cov([0.15, 0.15]),
        _cov([0.15, 0.15]),
        benchmark_vol_short=0.15,
        benchmark_vol_long=0.15,
        previous_weights=previous,
        turnover_penalty=10.0,
    )

    assert result.weights["A"] < 0.5
    assert 0.9 - 1e-8 <= result.vol_ratio <= 1.1 + 1e-8


def test_match_benchmark_vol_weights_raises_when_all_assets_too_quiet():
    raw = pd.Series({"A": 0.5, "B": 0.5})

    with pytest.raises(VolTargetInfeasible, match="no feasible"):
        match_benchmark_vol_weights(
            raw,
            _cov([0.08, 0.10]),
            _cov([0.08, 0.10]),
            benchmark_vol_short=0.15,
            benchmark_vol_long=0.15,
        )


def test_match_benchmark_vol_weights_honors_position_cap():
    raw = pd.Series({"A": 1.0, "B": 0.0})

    result = match_benchmark_vol_weights(
        raw,
        _cov([0.20, 0.12]),
        _cov([0.20, 0.12]),
        benchmark_vol_short=0.15,
        benchmark_vol_long=0.15,
        max_weight=0.7,
    )

    assert result.weights.max() <= 0.7 + 1e-8
    assert result.weights.sum() == pytest.approx(1.0)
    assert 0.9 - 1e-8 <= result.vol_ratio <= 1.1 + 1e-8


def test_rolling_vol_match_does_not_use_future_returns():
    idx = pd.bdate_range("2023-01-02", periods=100)
    returns = pd.DataFrame(
        {
            "A": np.tile([0.01, -0.008], 50),
            "B": np.tile([0.004, -0.003], 50),
        },
        index=idx,
    )
    benchmark = pd.Series(np.tile([0.007, -0.006], 50), index=idx)
    targets = pd.DataFrame(
        {"A": [0.7, 0.7], "B": [0.3, 0.3]},
        index=[idx[70], idx[80]],
    )
    changed = returns.copy()
    changed.loc[idx[90]:, "A"] = 0.20

    original = rolling_match_benchmark_vol_weights(targets, returns, benchmark)
    revised = rolling_match_benchmark_vol_weights(targets, changed, benchmark)

    pd.testing.assert_frame_equal(original.weights, revised.weights)
    assert original.diagnostics["vol_ratio"].between(0.9 - 1e-8, 1.1 + 1e-8).all()
