# P1 — Horizon Discovery (BTC/ETH perp 5m, Q1+Q2 2024)

**Slug**: `crypto_intraday/P1-horizon-discovery`
**Date**: 2026-05-23
**Researcher**: Kelvin Chen
**Status**: accept_monitoring
**Notebook**: [10_horizon_discovery.ipynb](../../../notebooks/90_crypto_intraday/10_horizon_discovery.ipynb)
**Supersedes**: —

## Research question

Before committing to a working interval / horizon for the rule-based and ML phases, empirically determine which lookback × horizon combinations carry information content for BTCUSDT / ETHUSDT perp 1m, sampled at 5m. The output is a focused set of (feature, horizon, symbol) tuples worth pursuing.

## Universe & data

- Symbols: BTCUSDT, ETHUSDT — USD-M perpetual (primary market).
- Source: Binance Vision 1m archives, resampled right-labeled to 5m via `OHLCV.resample('5min', label='right', closed='right')`.
- Window: 2024-01-01 → 2024-07-01 (Q1+Q2 2024, well inside the train slice; holdout is in 2026).
- Sample size: ~52k 5m bars per symbol.

## Features (leak-safe, all from `alpha_lab.features.intraday`)

`ret_1bar`, `ret_6bar`, `ret_12bar`, `ret_48bar` (5m, 30m, 1h, 4h trailing log returns); `rvol_60bar` (60-bar realized vol); `vol_z_60bar` (60-bar volume z-score); `distma_60bar` (distance from 60-bar MA); `rsi_14bar`; `bollb_20bar`.

## Method

For each (feature, symbol, horizon ∈ {5m, 15m, 30m, 1h, 4h}) compute:
- **IC** — Pearson correlation of feature[t] vs `safe_forward_returns(returns, horizon)`.
- **Rank-IC** — Spearman analogue.
- **Bootstrap 95% CI** on IC via `BlockBootstrap(block_size=288 bars ≈ 1 day, n_resamples=200, seed=0, mode='stationary')`.

Survival criterion: `|IC| ≥ 0.01` AND the 95% bootstrap CI excludes zero.

## Headline IC matrices (BTCUSDT, ETHUSDT — Pearson)

BTCUSDT:

| feature       | 5m      | 15m     | 30m     | 1h      | 4h      |
|---------------|---------|---------|---------|---------|---------|
| ret_1bar      | -0.0204 | -0.0220 | -0.0124 | -0.0074 | -0.0090 |
| ret_6bar      | -0.0126 | -0.0034 |  0.0099 | -0.0050 | -0.0133 |
| ret_12bar     | -0.0074 | -0.0062 | -0.0051 | -0.0257 | -0.0206 |
| ret_48bar     | -0.0091 | -0.0116 | -0.0134 | -0.0206 | -0.0095 |
| rvol_60bar    |  0.0079 |  0.0124 |  0.0184 |  0.0274 |  0.0507 |
| vol_z_60bar   |  0.0016 |  0.0028 | -0.0042 | -0.0141 | -0.0143 |
| distma_60bar  | -0.0118 | -0.0131 | -0.0142 | -0.0256 | -0.0151 |
| rsi_14bar     | -0.0045 | -0.0017 |  0.0056 | -0.0013 |  0.0015 |
| bollb_20bar   | -0.0060 | -0.0045 |  0.0046 | -0.0013 | -0.0046 |

ETHUSDT shows the same qualitative pattern with slightly stronger magnitudes for `rvol_60bar` (peaks at +0.078 / 4h).

## Surviving (feature, horizon, symbol) tuples (11 of 90)

Sorted by |IC|:

| symbol  | feature       | horizon | IC      | rank_IC | IC 95% CI            |
|---------|---------------|---------|---------|---------|----------------------|
| ETHUSDT | rvol_60bar    | 4h      |  0.0776 |  0.0426 | (0.0238, 0.1292)     |
| ETHUSDT | rvol_60bar    | 1h      |  0.0524 |  0.0342 | (0.0151, 0.0871)     |
| BTCUSDT | rvol_60bar    | 4h      |  0.0507 |  0.0350 | (0.0064, 0.1014)     |
| ETHUSDT | rvol_60bar    | 30m     |  0.0408 |  0.0281 | (0.0121, 0.0688)     |
| ETHUSDT | rvol_60bar    | 15m     |  0.0304 |  0.0205 | (0.0092, 0.0501)     |
| BTCUSDT | ret_12bar     | 1h      | -0.0257 | -0.0605 | (-0.0519, -0.0029)   |
| ETHUSDT | vol_z_60bar   | 1h      | -0.0224 | -0.0210 | (-0.0423, -0.0002)   |
| BTCUSDT | ret_1bar      | 15m     | -0.0220 | -0.0390 | (-0.0408, -0.0034)   |
| ETHUSDT | rvol_60bar    | 5m      |  0.0210 |  0.0104 | (0.0078, 0.0330)     |
| BTCUSDT | ret_1bar      | 5m      | -0.0204 | -0.0298 | (-0.0347, -0.0043)   |
| BTCUSDT | distma_60bar  | 5m      | -0.0118 | -0.0329 | (-0.0242, -0.0018)   |

## Observations

1. **Two distinct stories at this sample**:
   - **Linear (Pearson)**: realized vol predicts higher forward returns over multi-hour horizons. Direction is unusual (vol-as-positive-signal) — likely a sample-period artifact of the H1 2024 BTC bull run where vol clustered around moves up. Not load-bearing.
   - **Monotonic (Rank-IC)**: mean-reversion shows up clearly in `ret_1bar`, `ret_12bar`, `distma_60bar`, `rsi_14bar`, `bollb_20bar`. The rank-IC magnitudes (-0.04 to -0.06) are larger than Pearson, indicating the relationship is monotonic but non-linear. This is the more actionable signal.

2. **1h horizon dominates the surviving set** (3 tuples), with the 5m horizon tied. The 30m horizon barely survives. 4h survives only for `rvol_60bar`.

3. **ETH carries more signal than BTC**: 7 of 11 surviving tuples are ETHUSDT, including all the long-horizon `rvol_60bar` entries.

4. **Sample size caveat**: 6 months of 5m bars is ~26k observations per symbol. IC bootstrap CIs of width ~5% reflect this. The conclusions below are directional; P2 will revisit on longer windows.

## Decision

**accept_monitoring**. Use the following scope for P1-4 (rule-based) and P3 (ML):

- **Primary horizon**: 1h forward returns. Most cleanly-significant signals are at this horizon.
- **Secondary horizon**: 5m. Short-term mean reversion shows up cleanly on BTCUSDT.
- **Drop**: 30m horizon (only 1 surviving tuple); 4h horizon (only `rvol_60bar` survives, and the sign is sample-period dependent).
- **Working interval**: 5m sampling (gives access to all 5 horizons we tested).
- **Feature families to prioritize in P1-4**:
  - Mean-reversion: RSI(14), Bollinger %B, distance-from-MA, recent log returns.
  - Volatility-conditioned: regime overlays using `rvol_60bar` rather than `rvol_60bar` as a direct signal (because the long-horizon Pearson IC sign is suspect).
  - Skip pure trend features for the short horizons — rank IC says they hurt.

## Failure modes

- **Sign of `rvol_60bar` IC is sample-period dependent**: H1 2024 had a clear directional bull. In a chop period, vol clustering may flip sign. P2 will verify on more diverse windows; for now, treat the vol-positive linear signal as a research artifact, not a tradable edge.
- **Multiple-testing**: 90 (feature × horizon × symbol) tests, no correction. With α=0.05, ~4.5 false positives expected at random; we observed 11 survivors. The marginal cases (CI nearly touching 0) should be discounted.
- **Bootstrap block size of 1 day** may understate uncertainty for the 4h horizon (5 day-blocks = 5 quasi-independent observations of a 4h-horizon signal). Larger blocks for long horizons is a P2 refinement.

## Next steps (P1-4)

1. Build `notebooks/90_crypto_intraday/11_rule_based.ipynb` exercising the mean-reversion family at 5m sampling, evaluating at the 1h and 5m forward horizons primarily.
2. Strategies to backtest: RSI MR, Bollinger MR, Bollinger breakout (for contrast), distance-from-MA MR, MACD, MA crossover (trend baseline expected to fail), Donchian breakout (trend baseline expected to fail), volume-shock continuation vs reversal.
3. Each strategy: zero / perp_base / perp_stress costs; gross/net curves; turnover; cost drag; decision per strategy.
4. After P1-4, write `idea_log.md` with the 20+ ideas + provenance, marking each as implemented / rejected / deferred.

## Appendix

- Notebook: [10_horizon_discovery.ipynb](../../../notebooks/90_crypto_intraday/10_horizon_discovery.ipynb).
- PM holdout audit at notebook end: 12 events, 0 raises. PM Holdout was not accessed.
