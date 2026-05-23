# P1 — Rule-Based Strategies (BTC/ETH perp 5m, Q1+Q2 2024)

**Slug**: `crypto_intraday/P1-rule-based`
**Date**: 2026-05-23
**Researcher**: Kelvin Chen
**Status**: reject_all_with_findings
**Notebook**: [11_rule_based.ipynb](../../../notebooks/90_crypto_intraday/11_rule_based.ipynb)
**Builds on**: [P1-horizon-discovery](./P1-horizon-discovery.md)

## Research question

Among canonical rule-based strategy families — mean-reversion (RSI, Bollinger %B, distance-from-MA, short-term reversal), breakout (Bollinger, Donchian), trend (MA crossover, MACD), and volume-shock — do any survive realistic costs (`perp_base`, `perp_stress`) on Binance BTCUSDT / ETHUSDT perp at 5m sampling over Q1+Q2 2024?

## Setup

- 5m sampling, Q1+Q2 2024, perp.
- Per-symbol slippage from `configs/crypto_intraday.yaml` (now supported by `run_backtest`).
- Funding included in `perp_base` / `perp_stress`.
- Per-symbol weight capped at 0.5 (gross ≤ 1.0).

## Headline results

### Zero-cost annualized Sharpe (gross)

| strategy                  | Sharpe (zero) |
|---------------------------|---------------|
| volume_shock_continuation | **1.11**      |
| macd_trend                | 0.69          |
| bollinger_mr              | 0.63          |
| rsi_mr                    | 0.57          |
| donchian_breakout         | 0.47          |
| distma_reversion          | 0.31          |
| ma_crossover              | 0.04          |
| short_term_reversal       | -0.43         |
| bollinger_breakout        | -1.19         |

### Net total return, all three cost scenarios

| strategy                  | zero    | perp_base | perp_stress |
|---------------------------|---------|-----------|-------------|
| rsi_mr                    | 0.0732  | -0.9233   | -0.9760     |
| bollinger_mr              | 0.0861  | -0.9732   | -0.9947     |
| bollinger_breakout        | -0.1749 | -0.9234   | -0.9731     |
| distma_reversion          | 0.0200  | -0.8606   | -0.9419     |
| short_term_reversal       | -0.1281 | -0.9937   | -0.9993     |
| ma_crossover              | -0.0565 | -0.6247   | -0.7489     |
| macd_trend                | 0.1186  | -0.9196   | -0.9748     |
| donchian_breakout         | 0.0333  | -0.8201   | -0.9165     |
| volume_shock_continuation | 0.1168  | -0.6996   | -0.8314     |

### Attribution under perp_base (6-month cumulative)

| strategy                  | gross_total | cost_drag | funding_drag | net_total | turnover_sum |
|---------------------------|-------------|-----------|--------------|-----------|--------------|
| rsi_mr                    |  0.0732     | 2.6394    | -0.0011      | -0.9233   | 4,223        |
| bollinger_mr              |  0.0861     | 3.7010    |  0.0018      | -0.9732   | 5,923        |
| bollinger_breakout        | -0.1749     | 2.3790    | -0.0027      | -0.9234   | 3,807        |
| distma_reversion          |  0.0200     | 1.9931    | -0.0031      | -0.8606   | 3,190        |
| short_term_reversal       | -0.1281     | 4.9277    |  0.0034      | -0.9937   | 7,886        |
| ma_crossover              | -0.0565     | 0.9145    |  0.0071      | -0.6247   | 1,464        |
| macd_trend                |  0.1186     | 2.6397    | -0.0078      | -0.9196   | 4,226        |
| donchian_breakout         |  0.0333     | 1.7482    | -0.0004      | -0.8201   | 2,799        |
| volume_shock_continuation |  0.1168     | 1.3135    | -0.0003      | -0.6996   | 2,102        |

## Observations

1. **Every strategy is cost-killed at 5m.** Cost drag ranges from 0.91× capital (the slowest, ma_crossover) to 4.93× capital (short_term_reversal). At 5m with smoothly-varying weights, turnover compounds destructively.

2. **Funding drag is immaterial** in this window — < 1% of capital in every case. Funding ≠ the binding constraint at 5m; commission + slippage on turnover is.

3. **The interesting gross signals are**:
   - `volume_shock_continuation` (Sharpe 1.11 zero-cost, lowest turnover of the alive set at 2,102). The signal is event-driven (volume z-score > 2 gate), which naturally controls turnover.
   - `macd_trend` (Sharpe 0.69 zero-cost). Surprising — the horizon-discovery rank-IC suggested trend should not pay at this horizon. The discrepancy is likely because MACD picks up regime persistence in the H1 2024 bull run rather than pure short-term trend. Sample-period sensitive.
   - `bollinger_mr` and `rsi_mr` (Sharpe 0.63 / 0.57). Confirm the rank-IC mean-reversion story but lose massively to costs at this frequency.

4. **Strategies that fail even gross**:
   - `short_term_reversal` (gross Sharpe -0.43, gross net -12.8%). The "reversal" sign was wrong at this horizon at 5m for this period. Inconsistent with the rank-IC story for `ret_1bar` and `ret_12bar` in P1-horizon-discovery — likely because rank-IC's negative is driven by tail-events while the strategy used a continuous z-scored signal.
   - `bollinger_breakout` (gross Sharpe -1.19). Breakout chasing at 5m is anti-edge here — consistent with rank-IC's strong negative for `bollb_20bar`.

5. **Asymmetry between Pearson and Spearman signals**: features with strong rank-IC didn't necessarily translate to working strategies with continuous weight mappings. Threshold-based signals would convert rank monotonicity to performance more directly. Logical next step.

## Decision per strategy

| strategy                  | decision           | reason                                                      |
|---------------------------|--------------------|-------------------------------------------------------------|
| rsi_mr                    | reject_cost_killed | gross positive, lost to costs at 5m                         |
| bollinger_mr              | reject_cost_killed | gross positive, lost to costs                               |
| bollinger_breakout        | reject             | negative gross — anti-edge at this horizon                  |
| distma_reversion          | reject_cost_killed | gross positive (small), lost to costs                       |
| short_term_reversal       | reject             | negative gross at 5m continuous spec                        |
| ma_crossover              | reject             | negative gross + cost drag                                  |
| macd_trend                | reject_cost_killed | gross positive (sample-period dependent), lost to costs     |
| donchian_breakout         | reject_cost_killed | gross positive (small), lost to costs                       |
| volume_shock_continuation | reject_cost_killed | gross strong (Sharpe 1.11), still lost to costs; **revisit** |

## Headline finding (honest)

At naive 5m specifications with continuous, smoothly-varying weights, NONE of the canonical rule-based families survives realistic Binance perp costs over Q1+Q2 2024. This is the expected outcome at this frequency without turnover control. The single bright spot is `volume_shock_continuation`, an event-driven family whose turnover is naturally bounded — it still loses, but by less, and the gross-signal quality (Sharpe 1.11) is high enough to justify iteration with turnover control.

## Failure modes

- **Multiple testing across 9 strategies** without correction. With α=0.05 we'd expect ~0.5 false positives at random; the gross signals we observed at Sharpe 0.5+ are too large to be pure noise but are concentrated on a single market regime (H1 2024 bull).
- **Continuous z-scored signals over-trade.** The same gross alpha at threshold-only entry could net positive after costs (testable directly in P2).
- **Per-symbol slip averages cross-sectional differences** — for a strategy that flips one symbol while holding the other steady, BTC slip (1bp base) vs ETH slip (1.5bp base) matters. Now correctly handled via the dict-slip kwarg; the impact is second-order at this scope.
- **No survivorship issue** (only 2 symbols, both extant throughout).

## Next steps (P2)

1. Re-run `volume_shock_continuation` and `bollinger_mr` / `rsi_mr` with:
   - threshold-based gating (only take signal when |z| > 1.5) → reduces turnover by ~10×
   - hourly rebalance (`rebalance="1h"`) → reduces turnover by 12×
   - holding-period floor (no signal change within N bars after entry)
2. Funding-rate features (`funding_zscore`, `funding_cumulative` — already in `features/intraday.py`).
3. Cross-asset signals: BTC→ETH lead-lag, spread z-score, rolling beta residual.
4. Volatility-regime overlays — long-only when vol < median, etc.
5. Write `idea_log.md` cataloguing what's been tried and what's deferred.

## Appendix

- Notebook: [11_rule_based.ipynb](../../../notebooks/90_crypto_intraday/11_rule_based.ipynb).
- PM holdout audit at notebook end: 15 events, 0 raises. PM Holdout was not accessed.
- `volume_shock_continuation` is flagged in `idea_log.md` as `needs_revision`, not `reject` — it's the most promising candidate for a turnover-controlled re-run in P2.
