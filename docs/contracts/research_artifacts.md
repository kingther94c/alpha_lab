# Research artifact contracts

Lightweight, enforceable-by-convention shapes for the artifacts that flow
between notebooks and `src/alpha_lab/`. Goal: any helper or notebook can
trust the structure of the things it receives, so studies are composable.

> These are **contracts, not abstractions.** Pandas/numpy types, no base
> classes, no schemas library. If a helper accepts something out of contract,
> it should fail loudly and quickly.

Conventions used below:

- "wide" frame = `DataFrame` with `DatetimeIndex` rows, one column per asset.
- "long" frame = `DataFrame` with explicit columns including a date column.
- All datetimes are timezone-naive in `America/New_York` business hours
  unless explicitly labeled otherwise.
- All percentages stored as **fractions** (0.01 = 1%) unless the column name
  ends in `_bps` (basis points) or `_pct` (already in percent for display).
- Floats are `float64`; ids are `string`; booleans are `bool`.

---

## Universe file

A static list of tradable instruments that defines a study's investable set.
Lives in `configs/` if shared, or `data/private/` if proprietary.

**Shape:** CSV, one row per asset.

**Required columns:**

| column        | dtype  | notes                                                       |
|---------------|--------|-------------------------------------------------------------|
| `ticker`      | string | unambiguous instrument id (Yahoo ticker by default)         |
| `name`        | string | human-readable label                                        |
| `asset_class` | string | `equity` / `etf` / `fx` / `rate` / `commodity` / `crypto`   |
| `start_date`  | date   | first valid trading date for this row (ISO)                 |

**Optional columns** depending on study:

- `region`, `sector`, `country`, `currency`
- `proxy_ticker` — backfill ticker before `start_date` (e.g. DBMF → SG_Trend)
- `weight_cap`, `weight_floor` — per-name constraints
- Family-of-mappings (e.g. `signal_etf`, `long_1x_etf`, `inverse_1x_etf` for
  sector-momentum) follow `alpha_lab.backtest.sector_momentum.UNIVERSE_COLUMNS`.

**Rules:**

- Frozen at study time. If the universe changes mid-study, **add a new file**
  with a date suffix (`us_sector_etf_2026_05.csv`) — don't edit in place.
- Document construction: how were tickers selected, was there a survivorship
  filter, what was the cutoff date.
- The notebook should **assert the universe loaded** matches what it
  documents in the hypothesis cell.

---

## Price panel

The canonical asset-price input to signal construction and backtesting.

**Shape:** wide `DataFrame`.

- Index: `DatetimeIndex`, sorted ascending, no duplicates.
- Columns: tickers (must match the universe file's `ticker`s).
- Values: **total-return** prices when possible (auto-adjusted close). If
  price-only, document it in the notebook and accept the basis risk.
- dtype: `float64`.
- Missing data: `NaN` allowed pre-IPO / post-delisting; never `0` for missing.

**Producer:** `alpha_lab.data.loaders.*` (yfinance, fred, local).
**Consumer:** signal builders, backtest, vol-target, charts.

**Rules:**

- Align cross-asset panels onto a common calendar via
  `alpha_lab.data.align.align_prices` before computing returns.
- If you splice live + proxy (e.g. DBMF + SG_Trend), the splice helper goes
  in `data/loaders/`, the splice date goes in the universe file, and the
  resulting panel is one column — not two.

---

## Return panel

Same shape as the price panel, but pct/log returns. Index aligned to a
trading calendar.

- `simple_returns(prices)` — arithmetic, default in this repo.
- `log_returns(prices)` — for compounding math.
- First row is `NaN` by construction; backtest engines should `fillna(0)`
  at the edge, not in the middle.

**Forward returns**

For factor / IC studies, use
`alpha_lab.data.align.forward_returns(returns, horizon)`. By construction,
the last `horizon` rows are `NaN`. Never compute IC against backward-shifted
returns — that's the same as using the future to score the present.

---

## Signal panel

A wide frame interpreted as either a **factor score** (for IC / quantile
analysis) or a **target weight** (for backtesting).

**Shape:** wide `DataFrame` matching the return panel's index and a subset of
its columns.

**Two flavors:**

1. **Factor score** — arbitrary scale (e.g. raw momentum). To analyze, pair
   with `forward_returns` and use `alpha_lab.analytics.factor.ic` /
   `quantile_buckets`.
2. **Target weight** — already normalized to a portfolio (sums to 1 for
   long-only, ~0 for dollar-neutral L/S, may exceed 1 with leverage). To
   backtest, feed to `alpha_lab.backtest.vector.run_backtest`.

**Rules:**

- **Leak-safe by construction.** Any value at time `t` must be computable
  using only data with timestamp ≤ `t`. Rolling stats, rebalancing dates,
  vol estimates, regime classifiers all obey this.
- Signal rows on non-rebalance dates may carry the previous value or `NaN` —
  both are supported by `run_backtest`, which forward-fills and lags by 1
  period before applying.
- Don't normalize cross-sectionally over the full sample (that's a giant
  hidden lookahead). Use rolling / expanding stats in
  `alpha_lab.features.transforms`.

---

## Weight panel

The output of portfolio construction; the input to the backtest engine.

**Shape:** wide `DataFrame`.

- Index: rebalance-dense `DatetimeIndex` (typically a subset of the price
  panel's index — daily, weekly, or monthly).
- Columns: tradable tickers (not signal tickers — if the strategy buys
  3× leveraged proxies, the columns should be those proxy tickers).
- Values: target weights, fractions of capital. Long-only sums to 1; dollar-
  neutral sums to ~0; gross may be reported separately.
- dtype: `float64`. `NaN` is **not** allowed — fill with `0.0` for "no
  position".

**Rules:**

- Weight panel is **target**, not held. The backtest engine lags by one
  period to model decision-vs-execution timing.
- If the universe changes through time (e.g. new ETF launched in 2018),
  pre-launch rows for that column are `0.0`, not `NaN`.
- Per-asset capping (max position, sector cap) happens in the portfolio
  helper, not the backtest engine.

---

## Backtest result

`alpha_lab.backtest.vector.BacktestResult` dataclass.

**Fields:**

| field           | type                | shape                                         |
|-----------------|---------------------|-----------------------------------------------|
| `weights`       | wide DataFrame      | same index as `asset_returns`, lagged 1      |
| `asset_returns` | wide DataFrame      | per-asset returns over the backtest window    |
| `returns`       | Series              | net portfolio return (after costs)            |
| `gross_returns` | Series              | gross portfolio return (before costs)         |
| `turnover`      | Series              | per-period one-way turnover (fraction)        |
| `costs`         | Series              | per-period cost drag (fraction)               |
| `equity`        | Series (property)   | wealth index from `returns`, starts at 1      |

**Rules:**

- `returns` and `gross_returns` are simple returns (compoundable via
  `(1 + r).cumprod()`).
- `turnover` is **one-way** (half-sum of |Δw|); a 100% portfolio swap is
  `1.0`, not `2.0`.
- If you build a new portfolio helper, make sure the resulting weight panel
  yields a sensible turnover after `run_backtest` — the engine does
  forward-fill between rebalance dates.

---

## Performance summary

`dict[str, float]` from `alpha_lab.backtest.metrics.summary(returns)`.

**Required keys:**

`CAGR`, `AnnVol`, `Sharpe`, `Sortino`, `MaxDD`, `Calmar`, `HitRate`,
`NPeriods`.

All values are fractions (e.g. `Sharpe = 0.84`, `MaxDD = -0.18`,
`HitRate = 0.54`). `NPeriods` is the count of non-NaN return observations.

**Rules:**

- Pair every reported `Sharpe` with the **frequency** the returns were
  computed at and the `NPeriods`. A Sharpe with 60 monthly observations is
  not the same artifact as one with 5,000 daily observations.
- Net-of-cost is the headline; gross is a diagnostic. Don't quote gross
  Sharpe without saying so.
- **"Net" includes the cost of cash.** For any capital-consuming or leveraged
  strategy (long/short, futures/perp basis, cash-and-carry, levered overlay), net
  must also subtract financing — the risk-free rate (3M T-bill / SOFR) on the
  deployed/borrowed capital — not only commissions, slippage, and funding.
  Otherwise the headline overstates the edge (see `research_decisions/crypto_intraday/P6` §6b).
- Confidence intervals / t-stats for overlapping-return strategies need
  Newey-West (todo: `stats/tests.py`). Until then, report `NPeriods` honestly
  and call out the overlap.

---

## Monthly performance table

`pd.DataFrame` from `alpha_lab.backtest.metrics.monthly_table(returns)`.

- Index: `year` (int).
- Columns: `Jan` … `Dec`, then `YTD`.
- Values: compounded simple monthly returns (fractions).
- Missing months are `NaN`.

Used by `alpha_lab.reporting.charts.heatmap_monthly`.

---

## Research report

A decision-bearing artifact saved when a study reaches a verdict. **Two
parts:**

1. **Decision record** — markdown under `docs/research_decisions/` using the
   [decision-record template](../research_decisions/template.md). One file
   per study. Status: `accept`, `accept_monitoring`, `needs_revision`,
   `reject`, `park`.
2. **Rendered report** — optional HTML / PDF under `reports/`, generated
   from the source notebook (use `jupyter nbconvert` or the (future)
   `reporting.render.render_html` helper). Treated as derived and
   gitignored.

**Rules:**

- The decision record is the **source of truth** for the verdict. The
  notebook is the working paper.
- A study that never reaches a decision still gets a decision record with
  status `park` or `needs_revision`. Silent abandonment is the worst
  outcome.
- If a strategy moves from `accept_monitoring` to `reject`, append a new
  entry rather than overwriting — the history of conviction is itself
  signal.

---

## Quick lookup: producer → consumer

| Artifact          | Produced by                                  | Consumed by                                       |
|-------------------|----------------------------------------------|---------------------------------------------------|
| Universe file     | hand-curated CSV in `configs/`               | loaders, signal builders                          |
| Price panel       | `data/loaders/*`                             | feature transforms, signal builders, charts       |
| Return panel      | `analytics.returns.simple_returns`           | analytics, portfolio, backtest                    |
| Forward returns   | `data.align.forward_returns`                 | `analytics.factor.ic`, IC studies                 |
| Signal panel      | notebook + `features.transforms` + `analytics.factor` | `portfolio.*`                            |
| Weight panel      | `portfolio.long_only` / `active_mv` / etc.   | `backtest.vector.run_backtest`                    |
| BacktestResult    | `backtest.vector.run_backtest`               | `backtest.metrics`, `reporting.charts`            |
| Performance summary | `backtest.metrics.summary`                 | decision record, notebook table                   |
| Monthly table     | `backtest.metrics.monthly_table`             | `reporting.charts.heatmap_monthly`                |
| Decision record   | the researcher                               | future-you, future agents                         |

When a notebook breaks one of these contracts, fix the contract or fix the
notebook — but don't propagate the leak silently.
