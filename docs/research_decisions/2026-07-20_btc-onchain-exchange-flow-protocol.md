# BTC on-chain exchange-flow strategy protocol

Date: 2026-07-20
Status: pre-registered before inspecting strategy returns

## Question and mechanism

Can persistent withdrawals from known exchange clusters, or a fall in BTC held by
those clusters, identify large-holder accumulation that improves a long-BTC/cash
trend strategy after financing and trading costs?

The mechanism is inventory migration: sustained net withdrawals and falling
exchange reserves reduce immediately saleable inventory. This is only a proxy for
large holders. It is not proof of beneficial ownership, and internal wallet
reorganizations or later changes to address labels can contaminate it.

## Data and point-in-time contract

- Coin Metrics Community API daily BTC metrics: `PriceUSD`, `FlowInExNtv`,
  `FlowOutExNtv`, `SplyExNtv`, and `CapMVRVCur`.
- `^IRX` is used as the daily cash hurdle; missing weekends are forward-filled only.
- Coin Metrics UpperCamelCase daily metrics use beginning-of-interval timestamps.
  A value labelled day *t* is treated as unavailable until day *t+1* close.
- The portfolio target is lagged once more, so a day-*t* on-chain observation can
  earn no return earlier than the *t+2* close-to-close interval.
- Rolling thresholds use trailing observations only and are shifted one full day.
- Current API history does not provide label vintages. Historical exchange-cluster
  revisions are therefore a known backfill risk and must be disclosed.

## Frozen research periods

- Warm-up: 2013-01-01 through 2014-12-31.
- Development: 2015-01-01 through 2019-12-31.
- Validation: 2020-01-01 through 2023-12-31.
- External period: 2024-01-01 through 2026-07-19, withheld until one specification
  is frozen. This is not pristine blind data because the broad BTC price path was
  seen during the earlier ETF-flow study; it remains unseen for this signal family.

## Pre-declared candidate families

All candidates are long BTC or cash and require price above a 100- or 200-day
moving average.

1. Net-outflow event: 7- or 30-day net withdrawals divided by exchange supply are
   above their trailing 730-day 70th percentile; hold the event for 7 or 30 days.
2. Reserve-decline event: 30- or 90-day percentage decline in exchange supply is
   above its trailing 730-day 70th percentile; hold for 7 or 30 days.
3. Dual accumulation: both 30-day net withdrawals and 30-day reserve decline are
   above their trailing 730-day median or 70th percentile; hold 7 or 30 days.
4. Structural scarcity: exchange supply is below its trailing 365-day mean.

The grid is frozen before returns are inspected. Controls are BTC buy-and-hold and
price-only MA100/MA200.

## Costs, selection, and rejection

- Base trading cost: 15 bps per 100% change in BTC weight; stress: 30 bps.
- Cash earns `^IRX`; excess return subtracts that same cash hurdle from all capital.
- Eligibility requires positive total and excess CAGR in development and validation,
  max drawdown no worse than -45%, positive results at 30 bps, and incremental value
  versus the matched price-trend control.
- Select the eligible cell with the largest minimum development/validation excess
  Sharpe, with turnover used only as a deterministic tie-break.
- Before release, attack the selected rule with one extra execution day, validation
  subperiods, circular shifts (including family-wise reselection), and a moving-block
  bootstrap. A profitable external period is required for a paper-monitor candidate.
- No rule from this study authorizes live trading.
