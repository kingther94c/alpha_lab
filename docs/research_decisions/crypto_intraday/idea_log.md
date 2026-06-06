# BTC/ETH Intraday Idea Log

Running catalog of strategy ideas considered during this research engagement.
Append-only; never delete an entry, only update `status` / add notes.

Status legend:
- **implemented** — coded + backtested at least once, see decision record
- **accept_monitoring** — implemented + currently considered promising
- **needs_revision** — implemented but a specific issue must be fixed before re-running
- **reject** — implemented and failed; recorded as a negative result
- **reject_cost_killed** — gross signal positive but lost to costs at the tested spec; may revisit with turnover control
- **deferred** — not yet implemented; scheduled for a later phase
- **dropped** — considered and discarded without implementing (e.g., needs data we don't have)

Sources marked `internal prior / not externally verified` are my own assumptions or commonly-discussed-but-uncited mechanisms; sources marked with a name+context are real reads or specific public references.

---

## P0 — Foundation

| # | name | source | intuition | data | horizon | turnover | cost sensitivity | leakage risk | status | notes |
|---|------|--------|-----------|------|---------|----------|------------------|--------------|--------|-------|
| 1 | always_flat | internal prior | null hypothesis | none | n/a | zero | none | none | implemented | exactly 0 across scenarios — confirms wiring |
| 2 | bh_btc | textbook | passive long BTC | close | persistent | minimal | low | none | implemented | gross matches BTC pct exactly |
| 3 | bh_eth | textbook | passive long ETH | close | persistent | minimal | low | none | implemented | -8.79% in June 2024 |
| 4 | equal_weight_btc_eth | textbook | passive 50/50 | close | persistent | minimal | low | none | implemented | midpoint of bh_btc / bh_eth |
| 5 | random_sanity | textbook | null check | none | n/a | extreme at 1m | catastrophic | none | implemented | -100% under base/stress at 1m — proves cost model bites |

## P1 — Rule-based at 5m

| # | name | source | intuition | data | horizon | turnover | cost sensitivity | leakage risk | status | notes |
|---|------|--------|-----------|------|---------|----------|------------------|--------------|--------|-------|
| 6 | rsi_mean_reversion | technical-analysis canon (Wilder 1978, RSI definition; not externally verified for crypto intraday) | overbought / oversold dynamics revert at short horizons | close | 5m–1h | high (continuous weight) | severe | low (Wilder smoothing is causal) | reject_cost_killed | gross Sharpe 0.57; net -97% under stress |
| 7 | bollinger_mr | technical-analysis canon (Bollinger 1980s; internal prior on crypto) | excursions to band extremes revert | close | 5m–1h | high | severe | low | reject_cost_killed | gross Sharpe 0.63; net -99% under stress |
| 8 | bollinger_breakout | technical-analysis canon | breakout from low-vol band signals continuation | close | 5m–1h | medium | severe | low | reject | gross Sharpe -1.19 — anti-edge at this horizon |
| 9 | distma_reversion | internal prior | distance from MA reverts at short horizons | close | 5m–1h | high | severe | low | reject_cost_killed | gross Sharpe 0.31; net -94% under stress |
| 10 | short_term_reversal | crypto microstructure folklore (internal prior; not externally verified) | recent ret reverses next bar | close | 5m | extreme | catastrophic | low | reject | gross Sharpe -0.43 — sign wrong for this spec |
| 11 | ma_crossover | technical-analysis canon | fast-MA > slow-MA = uptrend | close | 5m–1h | low-medium | high | low | reject | gross Sharpe 0.04 — null result |
| 12 | macd_trend | technical-analysis canon | MACD histogram > 0 = uptrend | close | 5m–1h | high | severe | low | reject_cost_killed | gross Sharpe 0.69 — but sample-period sensitive; H1 2024 was a bull |
| 13 | donchian_breakout | technical-analysis canon (Donchian) | new N-bar high / low triggers continuation | OHLC | 5m–1h | medium | severe | low | reject_cost_killed | gross Sharpe 0.47 |
| 14 | volume_shock_continuation | crypto microstructure folklore (internal prior; not externally verified) | spike in volume with positive recent ret implies continuation | volume + close | 5m–30m | medium (event-gated) | high | low | reject_cost_killed | gross Sharpe **1.11** — strongest gross signal; revisit with turnover control |
| 15 | volume_shock_reversal | crypto microstructure folklore (internal prior) | spike with negative ret may be a liquidation flush, then revert | volume + close | 5m–30m | medium | high | low | deferred | symmetric variant of #14 |

## P2 — Perp-specific (deferred)

| # | name | source | intuition | data | horizon | turnover | cost sensitivity | leakage risk | status | notes |
|---|------|--------|-----------|------|---------|----------|------------------|--------------|--------|-------|
| 16 | funding_contrarian | internal prior; commonly-discussed crypto mechanic | crowded-long funding flips, take other side | funding (perp) | 8h–24h | low | low | high if not lagged; LOW with proper shift | **BUILT → accept_monitoring (P7, S4)** | banded daily funding-z fade (enter \|z\|>1, exit<0.3, dormant otherwise); directional contrarian; net Sharpe 0.57; orthogonal (ρ≈−0.29 vs trend) |
| 17 | funding_momentum | internal prior | funding stickiness predicts return continuation | funding | 8h–24h | low | low | low (lagged) | deferred | counterintuitive — folk wisdom prefers contrarian |
| 18 | high_funding_short_filter | internal prior | when funding is extreme positive, longs are overcrowded — short with vol filter | funding + vol | 8h–1d | low | low | low (lagged) | deferred | gate on percentile rank of funding |
| 19 | low_neg_funding_long_filter | internal prior | when funding is extreme negative, shorts are crowded — long with filter | funding | 8h–1d | low | low | low | deferred | mirror of #18 |
| 20 | funding_adjusted_trend | internal prior | weight trend signals by sign of funding | close + funding | 1h+ | medium | medium | low | deferred | combine #12 with #16 |
| 21 | oi_expansion | internal prior | rising OI + rising price = new longs entering | OI (Binance metrics) | 1h+ | medium | medium | low (lagged) | deferred | requires Binance Vision metrics endpoint (TODO loader extension) |
| 22 | oi_contraction | internal prior | falling OI + price = position unwind | OI | 1h+ | medium | medium | low | deferred | mirror of #21 |
| 23 | basis_premium | internal prior; CME literature on cash-and-carry (no specific cite) | spot–perp basis carries information about positioning | spot + perp close | 1h+ | low | low | low (lagged) | **BUILT → accept_monitoring (P6)** | market-neutral cash-and-carry: +15.7% gross-of-financing, ~+7-11% net of cash cost / −1.1% MaxDD, positive every regime incl. 2025 OOS; needed an ms/µs-timestamp loader fix |

## P2 — Cross-asset (deferred)

| # | name | source | intuition | data | horizon | turnover | cost sensitivity | leakage risk | status | notes |
|---|------|--------|-----------|------|---------|----------|------------------|--------------|--------|-------|
| 24 | btc_leads_eth | internal prior; long-discussed in crypto | BTC moves first, ETH catches up | both closes | 5m–1h | medium | high | low (strict shift on BTC return) | deferred | rolling regression of ETH-fwd on BTC-lag-k ret |
| 25 | relative_strength | classical equities (e.g. Jegadeesh-Titman 1993; not retrieved) | long stronger, short weaker | wide panel | 1h–4h | low | low | low | **BUILT → accept_monitoring (P7, S3)** | widened to BTC/ETH/SOL/BNB perps, daily 30d formation, long top2/short bot2; market-neutral, net Sharpe 0.47 |
| 26 | spread_z_pair_trade | classical stat-arb | log-spread reverts | both closes | 1h–4h | medium | medium | low | deferred | z-score of log(BTC/ETH) over rolling window |
| 27 | rolling_beta_residual | classical (CAPM residual) | residual of ETH-on-BTC regression mean-reverts | both closes | 1h–4h | medium | medium | low | deferred | uses `rolling_beta_residual` helper in features/intraday.py |
| 28 | correlation_regime | internal prior | when correlation breaks down, pair trade pays | both closes | 1h–4h | low | low | low | deferred | as overlay, not standalone signal |

## P2 — Volatility / regime (deferred)

| # | name | source | intuition | data | horizon | turnover | cost sensitivity | leakage risk | status | notes |
|---|------|--------|-----------|------|---------|----------|------------------|--------------|--------|-------|
| 29 | vol_scaled_momentum | well-known; e.g. Moskowitz et al 2012 (not retrieved) | scale trend by inverse vol | close | 1h–4h | low–med | medium | low | deferred | reduces tail risk |
| 30 | high_vol_breakout | internal prior | breakouts only matter in high-vol regimes | OHLC + vol | 1h–4h | medium | medium | low | deferred | gate Donchian on vol percentile |
| 31 | low_vol_mean_reversion | internal prior | MR works when vol is calm | close + vol | 1h | medium | medium | low | deferred | gate #6/#7/#9 on vol < median |
| 32 | inverse_vol_sizing | classical | size positions inversely to recent vol | close + vol | persistent | low | low | low | deferred | a portfolio overlay |
| 33 | drawdown_derisk | internal prior; trend-following lore | cut exposure during drawdowns | own equity curve | n/a | low | low | low | deferred | applied at portfolio level after strategy |

## Time-of-day / liquidity / event (deferred)

| # | name | source | intuition | data | horizon | turnover | cost sensitivity | leakage risk | status | notes |
|---|------|--------|-----------|------|---------|----------|------------------|--------------|--------|-------|
| 34 | us_hours_overlay | crypto microstructure folklore | US-overlap hours are most liquid | timestamp + close | intraday | medium | high | low | deferred | filter trading to UTC 12-21 |
| 35 | weekend_avoid | internal prior | weekends are thin / news-light | timestamp | n/a | low | low | low | deferred | gate signals to weekdays only |
| 36 | vwap_distance | classical execution lit | close above VWAP = buying pressure | OHLCV | 5m–1h | medium | high | low | deferred | requires clean intraday VWAP (`taker_buy_base` exists but full-bar volume needs care) |
| 37 | macro_overlay_fomc | internal prior | crypto moves at FOMC announcements | external econ calendar | event-driven | low | low | high (must use only known-at-time) | dropped | needs external event-time dataset; out of scope |
| 38 | liquidation_cascade | internal prior; commonly-discussed crypto mechanic | large liq → flush → bounce | liquidations | 5m | high | high | low (lagged) | dropped | Binance Vision archives do not publish liquidation tape |
| 39 | orderbook_imbalance | classical microstructure | top-of-book imbalance predicts next price | order book L1 | seconds-minutes | extreme | catastrophic | low | dropped | order book data not in Vision; out of scope |

## P3 — ML (deferred)

| # | name | source | intuition | data | horizon | turnover | cost sensitivity | leakage risk | status | notes |
|---|------|--------|-----------|------|---------|----------|------------------|--------------|--------|-------|
| 40 | logistic_regression_direction | classical | binary up/down classifier on lagged features | features | 5m–1h | depends | depends | medium (CV must be purged) | deferred | needs features from #6–14 + #16–23 |
| 41 | ridge_regression_return | classical | linear regression on lagged features | features | 5m–1h | depends | depends | medium | deferred | with `Standardizer` fit on train only |
| 42 | random_forest | classical | tree ensemble | features | 5m–1h | depends | depends | medium | deferred | sklearn already in env |
| 43 | gradient_boosting | classical | sklearn HistGradientBoosting or LightGBM | features | 5m–1h | depends | depends | medium | deferred | LightGBM in env |
| 44 | simple_ensemble | classical | equal-weight or val-ranked combination | strategy returns | n/a | derived | derived | low | deferred | combine surviving rule-based + ML |
| 45 | meta_model_gate | de Prado | classifier on whether to take the underlying signal | strategy signal + features | per-trade | low | medium | high (must respect signal lag) | deferred | classic meta-labeling pattern |

## P7 — Multi-strategy book (five low-correlation sleeves)

Goal: assemble 5 sleeves each anchored to a *different* return source so PnL is orthogonal by
construction, then combine them. Ideas generated via the `idea-generation` skill (random stimuli →
operators → one idea per return source). All daily, leak-safe, excess-of-cash. See `P7-multi-strategy-book.md`.

| # | name | source | intuition | data | horizon | turnover | cost sensitivity | leakage risk | status | notes |
|---|------|--------|-----------|------|---------|----------|------------------|--------------|--------|-------|
| 46 | ts_trend_daily_ls | well-known (Moskowitz/Ooi/Pedersen 2012, TS-momentum) | BTC/ETH above/below 50d MA, long/short on perp | perp close | 1 day | low (~19/yr) | medium | low (lagged) | **BUILT → accept_monitoring (P7, S2)** | net Sharpe 0.67; profits in 2022 bear; the directional sleeve |
| 47 | macro_credit_gate | internal prior; credit-as-risk-appetite (rehab of dropped #37) | hold crypto only when HYG credit regime risk-on (non-price-volume gate) | HYG (yfinance) / BAA10Y (FRED) | 1 day | low (~10/yr) | low | low (lagged a day) | **BUILT → accept_monitoring (P7, S5)** | net Sharpe 0.50; orthogonal (|ρ|≤0.12); driven by exogenous macro, not crypto price |
| 48 | multi_strategy_book | portfolio theory; #44 ensemble realized | combine S1–S5 (carry/trend/XS-mom/funding-contra/macro) by equal-capital + risk-budget | the five sleeve returns | 1 day | derived | derived | low | **BUILT → accept_monitoring (P7)** | mean\|ρ\|=0.11, ΣρPairs=−0.25, DR=2.0; combo Sharpe **1.15** vs BTC 0.51, CAGR 20% vs 14%, MaxDD −15% vs −67%; +ve every year incl. 2025 OOS |

## Caveats

- All "internal prior / not externally verified" entries reflect mechanisms I find plausible but have not verified against a specific published study during this engagement. They are pointers for direction, not citations.
- Sample window for P0+P1 is Q1+Q2 2024 only — a specific market regime (BTC bull early-to-mid 2024). Generalization to other regimes is P2 work.
- The set of ideas above is not exhaustive. The point of this log is auditability of what was tried + what was deferred, not a complete strategy catalog.
