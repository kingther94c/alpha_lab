# Research decision — QQQ put-write with BOXX-like cash yield

| field | value |
|---|---|
| **Slug** | `qqq-putwrite-cash-yield` |
| **Date** | 2026-07-03 |
| **Status** | `accept_monitoring` for paper advisory only |
| **Runner** | `scripts/qqq_putwrite_study.py` |
| **Report** | `reports/qqq_putwrite_cash_yield.html` |

## Research question

Can a BOXX-like cash sleeve plus repeated OTM QQQ puts earn a robust premium above
cash when an in-the-money expiry is accepted as QQQ stock and held temporarily? Do
selective IV entry rules or price/IV/time assignment exits improve on a simple policy?

## Hypothesis

The cash sleeve earns the short-rate carry while the put earns the volatility risk
premium for warehousing crash risk. Assignment is not a repair trick: it converts the
strategy into temporary QQQ beta and can strand capital after a persistent decline.
IV selection may reduce badly priced insurance sales, but waiting for rare high-IV
conditions also gives up premium time and creates threshold-selection risk.

## Data and model

- Daily adjusted QQQ/SPY and VXN/VIX/PUT data from Yahoo Finance, 2007-01-04
  through 2026-06-30; DTB3 from FRED.
- BOXX-like cash return is lagged DTB3 accrual less 19.49 bp/year.
- Historical QQQ options are a Black-Scholes/VXN proxy, calibrated against Cboe
  PUT on 2007-2014 and checked on 2015-2026. It is not an option-chain backtest.
- Discovery: 2007-2016; validation: 2017-2021; sealed holdout: 2022-2026H1.
- Entry strike and IV filters use the prior close; execution occurs at the next close.
  Price/IV assignment-exit signals also execute at the following close. Short-option
  liabilities are marked daily.
- Cash is the economic hurdle on the full collateral capacity, not merely a benchmark
  displayed beside the strategy.

## Selected model frontier

25-delta, 42 trading days, 75% collateral, no trend gate; after assignment, hold
until premium-adjusted breakeven or 126 trading days. Entry premium receives a
10% theoretical-price haircut plus $0.65/contract; stock exit costs 2 bp.

| period | CAGR | cash CAGR | excess CAGR | vol | Sharpe | max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Discovery | 5.68% | 0.48% | 5.21% | 9.48% | 0.63 | -17.95% |
| Validation | 7.25% | 0.89% | 6.36% | 9.90% | 0.76 | -12.51% |
| Holdout | 10.18% | 3.97% | 6.22% | 10.60% | 0.97 | -11.31% |
| Full | 7.10% | 1.37% | 5.73% | 9.85% | 0.75 | -17.95% |

QQQ buy-and-hold is a risk reference, not the economic hurdle; cash is the hurdle.

## Policy comparison at practical 50% collateral

All comparison rows freeze 25-delta, 42-day puts, the calibrated premium proxy and
the same cash hurdle. Only entry selection or assigned-stock exit changes.

- Simple baseline (always write; exit assigned QQQ at breakeven or 126 days):
  full CAGR 5.17%, 3.79% above cash, MaxDD -12.00%; holdout excess CAGR 4.11%
  with -7.50% MaxDD.
- The eligible pre-holdout challenger was `breakeven AND VXN <= 25% OR 126d`.
  Its weakest pre-holdout excess CAGR was 3.62%, but sealed-holdout excess CAGR
  was only 2.49%: 1.62 percentage points behind baseline, with -10.54% MaxDD.
- The best IV-entry rule by pre-holdout rank was `VXN - 21d RV >= 5pp`.
  It delivered 3.78% holdout excess CAGR versus 4.11% for baseline, while reducing
  MaxDD to -5.98%. That is a risk-reduction trade, not proven return enhancement.
- `VXN - 21d RV >= 0pp` happened to beat baseline by 0.13 points in holdout and
  reduced MaxDD to -6.75%, but it was not the preselected IV winner and is not
  promoted on this post-holdout observation.
- Fixed 126-day holding and breakeven+5% had high raw pre-holdout scores but failed
  the prespecified 20% drawdown eligibility gate. Strong absolute/percentile IV
  filters generally traded too little and lagged baseline out of sample.

Decision: retain the simple baseline as the operational reference. No alternative
is promoted without real option-chain paper data and another forward holdout.

## Diagnostics and robustness

- 107 selected-frontier put entries, 8 assignments, and 5.43% of days holding
  assigned stock. Completed stock episodes had a 23.5-day median, 33.3-day mean,
  102-day maximum, and no 126-day timeout. Eight episodes are too few for a stable
  exit-rule conclusion.
- A 21-day block bootstrap put the annualized arithmetic excess-return 95% interval
  at 2.97% to 9.03%.
- With the IV proxy reduced 10%, holdout excess CAGR fell from 6.22% to 1.68% and
  MaxDD worsened to -20.12%. Premium mapping remains first-order model risk.
- The proxy calibration selected the edge of the tested grid (1.15x IV and a 10%
  entry haircut), another reason not to treat the headline as execution-grade.
- Static leakage scan: 0 blockers. It flagged the trailing rolling quantile and the
  bootstrap diagnostic quantile heuristically. Manual review confirmed the former
  is a 252-day trailing window read at t-1, while the latter never feeds a signal,
  portfolio weight or selection. FRED accrual is shifted one observation and all
  price/IV entry and exit signals trade on the following close.

## Decision

**Status: `accept_monitoring` for a paper advisory; not accepted for live trading.**

Keep the selected timing/delta/assignment rule as the research frontier, but begin
paper monitoring at **50% collateral** rather than 75%. This is an operational
model-risk haircut, not a second holdout-selected policy. For a model snapshot based
on a $200k account and QQQ at $736.40, the frontier rounded to two $685-strike
contracts; the paper starter remains one contract. Neither is executable advice
without a live chain and broker collateral rules.

## Monitoring and kill conditions

- Capture timestamped QQQ bid/ask, IV, delta, open interest, quoted buying power,
  and next-day executable quotes for 3-6 months.
- Pause new advice if the quoted bid is more than 15% below model mid, bid/ask
  exceeds 10% of mid, or post-trade buying-power buffer would fall below 25% NAV.
- Pause and review if assigned-stock time exceeds 15%, any assignment hits the
  126-day timeout, or live premium capture is below 70% of modeled premium.
- Track baseline, `VXN-RV >= 5pp`, and price/IV exit variants in shadow mode; do not
  switch the advisory default based on this historical proxy comparison.
- Re-run with a paid historical QQQ option chain before any execution adapter.
- Do not automate live orders; preserve all `quant_bot_manager` live-money gates.

## Supporting artifacts

- `src/alpha_lab/backtest/put_write.py`
- `tests/test_put_write.py`
- `tests/test_put_write_policy_rules.py`
- `scripts/qqq_putwrite_study.py`
- `data/results/qqq_putwrite_cash_yield/policy_comparison.csv`
- `data/results/qqq_putwrite_cash_yield/policy_returns.parquet`
- `data/results/qqq_putwrite_cash_yield/policy_events.parquet`
- `reports/qqq_putwrite_cash_yield.html`