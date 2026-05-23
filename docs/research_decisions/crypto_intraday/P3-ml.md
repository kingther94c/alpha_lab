# P3 — ML attempt (BTC/ETH perp 1h, Q1+Q2 2024)

**Slug**: `crypto_intraday/P3-ml`
**Date**: 2026-05-23
**Researcher**: Kelvin Chen
**Status**: reject (does not beat `funding_momentum` baseline OOS after costs)
**Notebook**: [30_ml.ipynb](../../../notebooks/90_crypto_intraday/30_ml.ipynb)
**Baseline**: P2's `funding_momentum` at perp_stress (net +0.4082, Sharpe 1.5564)

## Research question

Train logistic regression, ridge regression, and LightGBM on the leak-safe feature catalog (19 features per symbol — multi-window returns, realized vol, volume z-score, MA / breakout / ATR / RSI / MACD / Bollinger / Donchian, time-of-day, funding-z/cum/level). Use walk-forward expanding CV with `Standardizer` fit on train only. Convert OOS predictions to weights, backtest with stress costs.

**Promotion rule (from goal)**: "ML must beat simple baselines OOS after costs — else reject/downgrade." The relevant baseline is `funding_momentum`.

## Setup

- 1h sampling, perp, Q1+Q2 2024 (same window as P2).
- Walk-forward expanding: train_size=60D, val_size=30D, step=30D → 3 folds completed (≈ 90D train + 90D val coverage).
- Logistic / LightGBM: binary direction target (forward 1h return > 0). Threshold 0.55 for class 1, 0.45 for class -1 (else flat).
- Ridge: continuous forward 1h return target; predicted value sign → weight.
- Per-symbol models (one per BTC, ETH); predictions stacked into a 2-col weight matrix and backtested jointly.
- `Standardizer(mode='per_column')` fit on train only per fold.

## Results

### Net total return — Q1+Q2 2024

| model     | zero    | perp_base | perp_stress |
|-----------|---------|-----------|-------------|
| logistic  | -0.0181 | -0.2005   | -0.2636     |
| ridge     | -0.0175 | -0.1931   | -0.2494     |
| lightgbm  |  0.0920 | -0.2661   | -0.3833     |

### Annualized Sharpe

| model     | zero    | perp_base | perp_stress |
|-----------|---------|-----------|-------------|
| logistic  |  0.0332 | -1.3173   | -1.8574     |
| ridge     |  0.0727 | -1.0576   | -1.4713     |
| lightgbm  |  0.7498 | -1.9710   | -3.1601     |

### Turnover (perp_base)

| model     | turnover_sum |
|-----------|--------------|
| logistic  |  304.5       |
| ridge     |  294.5       |
| lightgbm  |  618.0       |

### Comparison vs `funding_momentum` baseline (perp_stress)

| model     | net (vs 0.4082)  | Sharpe (vs 1.5564) | beats? |
|-----------|------------------|--------------------|--------|
| logistic  | -0.2636          | -1.8574            | NO     |
| ridge     | -0.2494          | -1.4713            | NO     |
| lightgbm  | -0.3833          | -3.1601            | NO     |

## Observations

1. **LightGBM has the strongest gross signal** (zero-cost Sharpe 0.75) — about 10× the linear models' gross signal. This is the expected pattern: tree models capture non-linearities that linear features miss.

2. **All three models are crushed by costs**. LightGBM's higher gross is paid for with 2× turnover; logistic / ridge under-trade because their predictions are nearly random.

3. **None of the models beats the baseline.** This is the honest result. The framework's promotion rule fires correctly.

4. **The base rate is ~51%** — markets are noisy at 1h on this window; an ML classifier needs to find non-trivial structure.

5. **Why ML doesn't help here**:
   - The strongest source of edge in this window (Q1+Q2 2024) is the **funding-rate sign**, which is a slow-moving, low-turnover signal. ML models trained on a feature panel that includes funding-z/cum/level should pick this up — and they don't, because the models are predicting BAR-LEVEL direction rather than the slow regime signal.
   - To "find" funding_momentum, an ML model would need to learn that the level/sign of funding_cum is the dominant predictor, then output low-noise sign-only predictions. Logistic regression at the 0.55 threshold is too noisy to do that.
   - Tree models (LightGBM) over-fit to short-horizon noise.

6. **No purged-k-fold tested here** — P3 used standard walk-forward expanding. The `PurgedKFold` machinery is in `ml/cv.py` and ready; using it requires defining `label_horizon` (here, 1h = 1 bar = trivial purge). Since the label horizon is 1 bar (= the rebalance period), purging is essentially a no-op. For longer-horizon label experiments (e.g., 24h forward returns), `PurgedKFold` would matter — that's a P4-deferred extension.

## Decision

**Reject ML** for this engagement scope. The framework's explicit rule fires:
> ML must beat simple baselines OOS after costs — else reject/downgrade.

None of logistic, ridge, or LightGBM matches `funding_momentum`'s net +40.8% / Sharpe 1.56 at stress costs over Q1+Q2 2024.

This is a **negative result for ML on this specific configuration**, NOT a final verdict on ML for crypto intraday research broadly. Caveats for the final report:

- Sample window is short (6 months) — ML benefits from more training data.
- 2-symbol cross-section is degenerate — ML thrives on broader universes.
- Label horizon (1h) may be too short; longer horizons + purged CV may reveal slower-moving structure.
- Feature engineering is canonical-only — meta-features (funding sign persistence, regime indicators) might help.
- No hyperparameter tuning was performed.

These caveats are all P4-deferred. The headline finding is that ML does NOT clear the high bar set by funding_momentum in this window.

## Failure modes

- **Multiple-testing across 3 model families** — minor, n=3.
- **Feature set is leak-safe by construction** but `funding_z_grid` / `funding_cum_grid` use `reindex(method='ffill')` from an 8h cadence onto 1h. The pad is monotonically lagging, but for the bar that contains a funding event the model "sees" the new rate — that's the correct semantic (the rate is published at the event time and known thereafter).
- **No combinatorial purged CV** (too expensive at this scope; documented as a P2-bucket future enhancement).
- **No formal Sharpe SE / bootstrap on ML returns** — P4-deferred.

## Next steps (P4 — final report)

1. Assemble the PM-style markdown+HTML report with the 11 required sections.
2. Headline verdict: likely `monitor` (funding signals work in this window but are regime-dependent and need cross-period validation).
3. Document the rejected ML attempt as a negative finding.
4. Audit log: PM holdout never accessed across 30+ events.

## Appendix

- Notebook: [30_ml.ipynb](../../../notebooks/90_crypto_intraday/30_ml.ipynb).
- PM holdout audit at notebook end: 30 events, 0 raises.
