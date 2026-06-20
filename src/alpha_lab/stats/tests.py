"""Significance tests for strategy returns.

Guards against the two ways a backtest lies (research-plan §7): **data-snooping**
(trying many configs and reporting the best) and **overlapping-return inflation**
(naive t-stats on autocorrelated returns). Provides:

- :func:`bootstrap_sharpe_ci` — block-bootstrap CI for the annualized Sharpe
  (preserves autocorrelation from overlapping/held positions).
- :func:`deflated_sharpe_ratio` — Bailey & López de Prado DSR: probability the best
  Sharpe found across N trials is truly > 0, after deflating for selection.
- :func:`newey_west_tstat` — HAC t-stat of the mean return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

_EULER = 0.5772156649015329


def annualized_sharpe(returns, periods: int = 252) -> float:
    """Annualized Sharpe of a return stream (0 mean assumption on rf already netted)."""
    r = pd.Series(returns).dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(periods)) if sd and len(r) > 1 else float("nan")


def bootstrap_sharpe_ci(
    returns, *, periods: int = 252, n_boot: int = 2000, alpha: float = 0.05,
    block: int = 21, seed: int = 0,
) -> dict:
    """Block-bootstrap confidence interval for the annualized Sharpe.

    Resamples contiguous blocks (default ~1 month) to preserve autocorrelation, so the
    CI isn't falsely tight for overlapping signals. Returns the point Sharpe, the
    ``[alpha/2, 1-alpha/2]`` CI, and ``p_gt_0`` (bootstrap probability Sharpe > 0).
    """
    r = pd.Series(returns).dropna().to_numpy()
    n = len(r)
    if n < block * 2:
        return {"sharpe": annualized_sharpe(r, periods), "lo": float("nan"),
                "hi": float("nan"), "p_gt_0": float("nan"), "n_boot": 0}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    sh = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        samp = np.concatenate([r[s:s + block] for s in starts])[:n]
        sd = samp.std()
        sh[i] = samp.mean() / sd * np.sqrt(periods) if sd > 0 else np.nan
    lo, hi = np.nanpercentile(sh, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": annualized_sharpe(r, periods), "lo": float(lo), "hi": float(hi),
            "p_gt_0": float(np.nanmean(sh > 0)), "n_boot": n_boot}


def deflated_sharpe_ratio(
    observed_sharpe: float, *, n_obs: int, trial_sharpes=None, n_trials: int | None = None,
    periods: int = 252, skew: float = 0.0, kurt: float = 3.0,
) -> dict:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Probability the *best* annualized Sharpe across the search is genuinely > 0 after
    deflating for ``n_trials`` of selection and the return distribution's skew/kurtosis.
    DSR > 0.95 ≈ significant at 5% after multiple testing.

    Pass ``trial_sharpes`` (the annualized Sharpes of every config tried) to estimate
    both the trial count and their variance; otherwise pass ``n_trials`` and the
    per-trial Sharpe variance is approximated as 1 (the null benchmark variance).
    """
    sr = observed_sharpe / np.sqrt(periods)  # per-period
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        tv = np.asarray([s / np.sqrt(periods) for s in trial_sharpes], dtype=float)
        n_trials = len(tv)
        var_trials = float(np.nanvar(tv, ddof=1))
    else:
        n_trials = n_trials or 1
        var_trials = 1.0 / n_obs
    n_trials = max(int(n_trials), 2)
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr0 = np.sqrt(max(var_trials, 1e-12)) * ((1 - _EULER) * z1 + _EULER * z2)
    num = (sr - sr0) * np.sqrt(max(n_obs - 1, 1))
    den = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2, 1e-12))
    return {"dsr": float(sps.norm.cdf(num / den)), "expected_max_sharpe_ann": float(sr0 * np.sqrt(periods)),
            "n_trials": n_trials}


def newey_west_tstat(returns, *, lags: int | None = None) -> float:
    """t-stat of the mean return with Newey-West (HAC) standard errors.

    Use for overlapping-return / autocorrelated strategies where the naive
    ``mean/std`` t-stat is too generous. ``lags=None`` uses the ``4(n/100)^(2/9)`` rule.
    """
    r = pd.Series(returns).dropna().to_numpy()
    n = len(r)
    if n < 3:
        return float("nan")
    lags = int(np.floor(4 * (n / 100) ** (2 / 9))) if lags is None else lags
    e = r - r.mean()
    var = np.mean(e * e)
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1)
        var += 2 * w * np.mean(e[lag:] * e[:-lag])
    se = np.sqrt(var / n)
    return float(r.mean() / se) if se > 0 else float("nan")
