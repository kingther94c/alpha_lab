# P2 — Perp-specific + Cross-asset + Vol-regime (BTC/ETH perp 1h, Q1+Q2 2024)

**Slug**: `crypto_intraday/P2-strategies`
**Date**: 2026-05-23
**Researcher**: Kelvin Chen
**Status**: accept_monitoring (funding-based) — see per-strategy table
**Notebook**: [20_p2_strategies.ipynb](../../../notebooks/90_crypto_intraday/20_p2_strategies.ipynb)
**Supersedes**: P1-rule-based (extends with perp-specific + cross-asset families)

## Research questions

1. Can the best P1 gross signal (`volume_shock_continuation`, Sharpe 1.11 zero-cost) survive realistic costs once we resample to 1h (cuts turnover ~12× vs 5m)?
2. Does the perp-specific information (funding rate) carry tradable signal that close-only doesn't?
3. Do cross-asset signals (BTC ↔ ETH spread / beta residual / lead-lag) survive costs?
4. Does a vol-regime overlay (only trade when realized vol is moderate) improve the best survivor?

## Setup

- 1h sampling (resampled from 1m), perp, Q1+Q2 2024.
- Per-symbol slippage from config; funding included in `perp_base` / `perp_stress`.
- `bars_per_year = 8760` for annualization.

## Headline results

### Net total return — Q1+Q2 2024, 1h sampling

| strategy                       | zero    | perp_base | perp_stress |
|--------------------------------|---------|-----------|-------------|
| volume_shock_continuation_v2   | -0.0109 | -0.1478   | -0.2024     |
| bollinger_mr_threshold         | -0.0077 | -0.2179   | -0.2975     |
| rsi_mr_threshold               | -0.0506 | -0.1209   | -0.1525     |
| **funding_contrarian**         | 0.2174  | **0.1741**| **0.1501**  |
| **funding_momentum**           | 0.5224  | **0.4084**| **0.4082**  |
| spread_z_pair_trade            | -0.0936 | -0.2418   | -0.2994     |
| btc_leads_eth                  | -0.0536 | -0.3290   | -0.4415     |
| beta_residual_mr               | 0.1334  | -0.2066   | -0.3446     |

### Annualized Sharpe

| strategy                       | zero    | perp_base | perp_stress |
|--------------------------------|---------|-----------|-------------|
| volume_shock_continuation_v2   | -0.0312 | -1.6825   | -2.4134     |
| bollinger_mr_threshold         |  0.0939 | -1.5362   | -2.2699     |
| rsi_mr_threshold               | -0.3754 | -1.0910   | -1.4311     |
| **funding_contrarian**         |  1.5132 | **1.2603**| **1.1168**  |
| **funding_momentum**           |  1.8503 | **1.5569**| **1.5564**  |
| spread_z_pair_trade            | -1.3569 | -3.9431   | -5.0834     |
| btc_leads_eth                  | -0.5362 | -4.4290   | -6.4995     |
| beta_residual_mr               |  1.6590 | -2.8363   | -5.2395     |

### Attribution under `perp_base`

| strategy                       | gross_total | cost_drag | funding_drag | turnover_sum |
|--------------------------------|-------------|-----------|--------------|--------------|
| volume_shock_continuation_v2   | -0.0109     | 0.1509    | -0.0018      | 241.5        |
| bollinger_mr_threshold         | -0.0077     | 0.2422    | -0.0043      | 387.0        |
| rsi_mr_threshold               | -0.0506     | 0.0831    | -0.0062      | 133.0        |
| funding_contrarian             |  0.2174     | 0.0463    | -0.0101      |  74.0        |
| **funding_momentum**           |  0.5224     | 0.0003    | +0.0775      |   **0.5**    |
| spread_z_pair_trade            | -0.0936     | 0.1795    | -0.0010      | 287.2        |
| btc_leads_eth                  | -0.0536     | 0.3407    | +0.0032      | 524.2        |
| beta_residual_mr               |  0.1334     | 0.3550    | +0.0015      | 546.2        |

Note: `funding_drag > 0` means net funding paid (cost). `funding_momentum` paid +7.75% of capital in funding over the period — that's the price of staying long during a positive-funding regime — but still cleared +40.8% net.

### Decisions per strategy

| strategy                       | decision           | reason                                                                   |
|--------------------------------|--------------------|--------------------------------------------------------------------------|
| volume_shock_continuation_v2   | reject             | gross went negative at 1h; the P1 zero-cost edge was 5m-specific         |
| bollinger_mr_threshold         | reject             | barely-negative gross even at zero costs                                 |
| rsi_mr_threshold               | reject             | negative gross + cost drag                                                |
| **funding_contrarian**         | **accept_monitoring** | net +15% / Sharpe 1.12 at stress; low turnover (74 over 6mo)            |
| **funding_momentum**           | **accept_monitoring** | net +40.8% / Sharpe 1.56 at stress; near-zero turnover                  |
| spread_z_pair_trade            | reject             | wrong sign — log(BTC/ETH) didn't mean-revert at 1h in this period       |
| btc_leads_eth                  | reject             | high cost drag, weak gross                                                |
| beta_residual_mr               | reject_cost_killed | gross +13% but lost everything to costs at 1h on noisy residual          |

### Vol-regime overlay on best survivor (`funding_momentum`)

| scenario     | ungated net | ungated Sharpe | gated net | gated Sharpe |
|--------------|-------------|----------------|-----------|--------------|
| zero         | +0.5224     | 1.85           | +0.1512   | 0.97         |
| perp_base    | +0.4084     | 1.56           | +0.0632   | 0.52         |
| perp_stress  | +0.4082     | 1.56           | +0.0474   | 0.44         |

**Low-vol gate hurts `funding_momentum`**. Mechanism: H1 2024 BTC bull happened during high-vol regimes, and the funding-positive signal coincided with up-vol days. Gating out high-vol periods removed most of the alpha. Confirms that funding_momentum is in part a "trend-during-bull" proxy rather than a robust regime-independent signal.

## Observations

1. **Funding-based signals are the only ones that survive realistic costs in this window**. Both `funding_momentum` and `funding_contrarian` clear the `perp_stress` bar.

2. **The signal source is the 8h funding cadence**, which is forward-filled to the 1h grid leak-safely. Turnover is naturally ~zero because the funding sign flips slowly (multi-day persistence).

3. **`funding_momentum` is partly a bull-market trend proxy**:
   - Q1+Q2 2024 was a BTC bull (~$42k → ~$60k).
   - Positive funding → longs paying → confirms longs are crowded → strategy stays long.
   - In a chop / bear regime the signal will likely flip and the bull-capture mechanism breaks.
   - Vol-regime overlay confirms: gating out high-vol periods (which coincided with the bull moves) cuts net stress return from +40.8% → +4.7%.

4. **Cross-asset signals all failed**:
   - `spread_z_pair_trade` had the **wrong sign** — log(BTC/ETH) did not mean-revert at 1h in this period. (BTC outperformed ETH steadily, no reversion.)
   - `btc_leads_eth` over-traded with no edge.
   - `beta_residual_mr` had decent gross (+13%) but lost everything to cost drag — the residual is too noisy at 1h for naive mean-reversion.

5. **P1 leaders did NOT carry over to 1h**:
   - `volume_shock_continuation` at 1h has -1% gross (vs +12% gross at 5m). The 5m gross signal was driven by 5m-specific microstructure events that don't aggregate up.
   - This is a useful negative finding: the 5m gross signal we celebrated in P1 was illusory — it wouldn't have survived even with perfect cost control because the underlying gross edge is interval-specific.

## Headline P2 finding (honest)

After exploring 8 strategy families across perp-specific, cross-asset, vol-regime, and turnover-controlled re-runs:
- **Two survive realistic costs**: `funding_momentum` and `funding_contrarian`.
- Both ride the perp-specific funding mechanism with near-zero turnover.
- **`funding_momentum` is likely a regime-dependent bull proxy** rather than a robust intraday edge — H1 2024 was uniformly favorable.
- **Cross-asset and TA-style intraday signals fail to clear costs** at 1h on this 6-month window.

This is a positive result for the framework (it surfaces survivors and explains the failures), but a cautious result for tradability — we have two signals that "work" only on a single bull regime and need cross-period validation.

## Failure modes / caveats

- **Sample period dependence**: H1 2024 was a BTC bull. Funding signals are highly regime-correlated. Multi-year validation (ideally including chop + bear periods) is essential before any tradability claim.
- **Vol-overlay finding is sample-specific**: the bull happened during high-vol days. In other regimes the conclusion may flip.
- **No multi-symbol generalization**: only BTC + ETH tested. Funding-momentum on a broader set of perp pairs would test whether the mechanism is BTC-specific or general.
- **No purged-CV on these strategies yet**: walk-forward / purged CV would tighten the validation. P3 ML notebook will exercise that machinery.
- **Multiple-testing**: 8 strategies, no formal correction. With α=0.05 we'd expect ~0.4 false positives; the funding family Sharpes are large enough not to be pure noise, but the multiple-testing caveat stands.

## Next steps

- **P3 ML notebook**: train logistic / ridge / LightGBM on the full feature catalog (intraday + funding + cross-asset) with WalkForwardSplit + Standardizer. Must beat `funding_momentum` OOS after costs to be promoted.
- **P4 Final PM report**: assemble the 11 required sections. Headline verdict will likely be `monitor` — the funding signals are interesting enough to deserve more research but the regime caveat is real.
- **Future P2 extensions** (deferred):
  - Validate funding-momentum on 2022 bear + 2023 chop periods.
  - Add OI and basis features (require loader extension to Binance Vision metrics endpoint).
  - Test funding-rate signal on more perp pairs (SOL, BNB, etc.).

## Appendix

- Notebook: [20_p2_strategies.ipynb](../../../notebooks/90_crypto_intraday/20_p2_strategies.ipynb).
- PM holdout audit at notebook end: 27 events, 0 raises. PM Holdout was not accessed.
