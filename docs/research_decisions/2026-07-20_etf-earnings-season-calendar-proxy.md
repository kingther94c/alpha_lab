# Research decision record — ETF earnings-season calendar proxy

## Header

| field | value |
|---|---|
| **Slug** | `etf-earnings-season-calendar-proxy` |
| **Date** | `2026-07-20` |
| **Researcher** | Codex |
| **Status** | `reject` |
| **Notebook** | n/a — reproducible script study |
| **Report** | `reports/etf_earnings_season_pre2022.html` |
| **Supersedes** | n/a |
| **Superseded by** | n/a |

## 1. Research question

Do fixed, knowable windows in January, April, July, and October improve ETF returns or ETF allocation decisions, net of trading costs, using data only through 2021?

## 2. Hypothesis & economic rationale

Scheduled earnings releases can attract investor attention and concentrate information arrival. Prior work reports an individual-stock earnings-announcement premium and a market premium immediately before clusters of famous firms announcing after the close. A fixed month-position proxy could capture part of that return source, but it may instead proxy unrelated quarter-cycle flows or dilute the effect because actual announcement clusters move from quarter to quarter.

## 3. Universe & data

- **Universe:** SPY, QQQ, RSP, IWM, nine Select Sector SPDR ETFs, SHY, and IEF.
- **Date range:** 2004-01-02 through 2021-12-31. Development: 2004–2012; validation: 2013–2021.
- **Data source:** frozen Yahoo Finance adjusted-close cache at `data/results/etf_strategy_50plus_pre2022/market_prices_adjusted_pre2022.parquet`.
- **Known gaps:** no point-in-time company announcement date, BMO/AMC session, historical index membership, surprise, or attention data. The ETF list is surviving products that existed throughout the tested sample; this is a limited survivorship bias, but it does not reconstruct historical sector constituents.

## 4. Signal & portfolio construction

- **Primary calendar proxy:** the 7th through 14th trading days of January, April, July, and October; the dates are known before trading.
- **Exploratory grid:** 195 asset × window event tests and 516 strategy variants. Families were season-only ETF/SHY, ETF/SPY season tilts, QQQ/SPY prior-close relative momentum, prior-close sector momentum fixed at window entry, and an SPY/SHY prior-close trend gate.
- **Execution:** calendar positions are known before the return interval. Every price-derived signal uses prices through `t-1`; no same-close signal earns the same close-to-close return.
- **Costs:** 5 bps per unit of L1 turnover. A complete ETF-to-ETF switch has L1 turnover 2 and costs 10 bps. SHY is the funded cash substitute; there is no leverage or borrowed capital.
- **Benchmark:** SPY. Dynamic rules were also compared with the same signal run all year to isolate whether the earnings-season conditioner added value.

## 5. Headline performance

Daily net returns, 252 periods per year.

| metric | primary SPY/SHY window | primary QQQ/SHY window | SPY buy-and-hold |
|---|---:|---:|---:|
| CAGR | 2.80% | 3.47% | 10.55% |
| AnnVol | 7.37% | 7.77% | 18.93% |
| Sharpe | 0.41 | 0.48 | 0.62 |
| MaxDD | -14.20% | -19.22% | -55.19% |
| Worst ≥5% trough-to-recovery | 565 days | 752 days | 869 days |
| Share ≥5% drawdowns recovered within 20 days | 0% | 0% | 30% |

The lower drawdown is mainly lower equity exposure, not earnings-season alpha. The primary QQQ-in-window/SPY-otherwise tilt had 10.38% CAGR versus 10.55% for SPY, with development active return -1.30% annualized and validation active return +1.05%.

## 6. Diagnostics

- The primary eight-day window averaged 0.53% for SPY and 0.71% for QQQ per quarter.
- Relative to the same trading-day positions in the next two months of each quarter, the mean differences were +0.51% for SPY (`t=1.22`) and +0.51% for QQQ (`t=1.05`).
- QQQ's relative-to-SPY earnings-window advantage over the placebo months was effectively zero (`0.003%`, `t=0.01`).
- No absolute or relative event-window result survived Benjamini–Hochberg FDR at `q<10%` across 195 tests.
- The scanned late-month XLF tilt (`start=13`, `length=8`) produced 12.14% CAGR, +1.80% annual active return, and -46.07% MaxDD. Its quarter-window relative test had `t=3.05`, but FDR `q=0.62`; neighboring start dates were unstable, and the timing does not cleanly match large-bank reporting clusters.

## 7. Robustness

- [x] Time-ordered development/validation split — primary market result strengthened only in validation; QQQ changed sign.
- [x] Neighboring calendar windows — the best scanned parameters were not a broad stable surface.
- [x] Multiple-testing control — zero of 195 absolute and zero of 195 relative tests passed FDR `q<10%`.
- [x] Same-day signal/execution check — automated audit: 0 blocker, 0 warning; manual review confirmed all price signals use `t-1`.
- [x] Same-factor attribution — earnings-conditioned QQQ momentum and SPY trend gates did not robustly beat their all-year versions in both subperiods.
- [x] Trading costs and funded cash — ETF turnover charged; SHY earns the cash return.
- [ ] Exact announcement-cluster holdout — blocked by missing point-in-time event/session data.
- [ ] Intraday decomposition — daily closes cannot isolate the documented pre-close cluster premium.

## 8. Failure modes

- Fixed quarter-start months overlap macro releases, option expiries, tax/flow effects, and ordinary month seasonality; causal attribution to earnings is impossible.
- Only 72 quarters are available, and three-to-eight-day windows have low statistical power.
- Actual earnings clusters shift dates and announcement sessions. A wide fixed window dilutes the hypothesized effect; scanning a narrow window overfits it.
- Current adjusted-close data cannot separate overnight gaps from regular-session returns or model close-auction execution.
- Sector ETFs can speed common-information transmission and reduce sector-level post-announcement drift.

## 9. Decision

**Status: `reject`.**

Reject fixed January/April/July/October calendar windows as a standalone ETF strategy or allocation overlay. Positive average market returns do not survive the placebo, subperiod, factor-attribution, and multiple-testing hurdles. The best scanned XLF result is a follow-up hypothesis, not a tradeable conclusion.

## 10. Next steps

- Obtain point-in-time historical earnings dates with BMO/AMC session and confirmation timestamps.
- Reproduce the high-attention after-close earnings-cluster design: select cluster days only from information known before that day's close, enter SPY or QQQ before the close, and exit the next close.
- Freeze attention weights using prior-year media counts or prior-year-end market capitalization; never rank firms using full-sample fame.
- Preserve a final holdout after all event thresholds, fame cutoffs, and execution assumptions are frozen.
- If historical sessions cannot be sourced reliably, stop; SEC filing dates are not a valid substitute for earnings-release timestamps.

## 11. Supporting artifacts and sources

- Script: `scripts/etf_earnings_season_study.py`
- Results: `data/results/etf_earnings_season_pre2022/`
- Report: `reports/etf_earnings_season_pre2022.html`
- Frazzini and Lamont, *The Earnings Announcement Premium and Trading Volume*: https://www.nber.org/papers/w13090
- Chen, Cohen, and Wang, *Famous Firms, Earnings Clusters, and the Stock Market*: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3685452
- Hirshleifer, Hou, Teoh, and Zhang, *Stock Returns, Aggregate Earnings Surprises, and Behavioral Finance*: https://doi.org/10.1016/j.jfineco.2004.06.016
- Bhojraj, Mohanram, and Zhang, *ETFs and Information Transfer Across Firms*: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3175382
