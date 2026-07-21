# ETF drawdown-recovery addendum — pre-2022 result

## Decision

The requested **20-business-day recovery ceiling is rejected by the historical
evidence**. None of the original 104 strategies and none of the 100 frozen
recovery-oriented variants combined the return/risk objective with recovery of every
5% drawdown within 20 observed trading sessions. Resetting the high-water mark after
20 days would only relabel an unrecovered loss and was not allowed.

The original data boundary remains intact: every price and strategy return ends no
later than 2021-12-31. Duration is an ex-post evaluation field only and never enters a
signal or weight.

## Definitions

- `max_underwater_days`: first session below the previous equity high through the last
  underwater session before regaining that high.
- `max_5pct_trough_to_recovery_days`: worst trough-to-prior-high recovery leg among
  episodes that reached -5% or worse.
- `median_5pct_trough_to_recovery_days`: median recovery leg for the same episodes.
- `share_5pct_recovered_within_20d`: share of material episodes fully recovered no more
  than 20 sessions after the trough. A still-open episode is a failure and is censored
  at the sample end.

## Original primary candidates

| Candidate | CAGR | Vol | MaxDD | Max underwater | Worst 5% recovery | Median 5% recovery | <=20d share |
|---|---:|---:|---:|---:|---:|---:|---:|
| HYG/IEF credit canary | 11.51% | 11.11% | -12.90% | 453d | 295d | 28d | 37.5% |
| QQQ 12% vol target | 11.36% | 12.90% | -21.72% | 515d | 230d | 55d | 25.0% |
| Cross-asset 6m top-3 momentum | 10.21% | 12.60% | -22.15% | 727d | 499d | 47d | 21.4% |
| Sector low-Ulcer positive trend | 10.19% | 12.49% | -16.37% | 573d | 352d | 31d | 36.8% |
| Sector downside/trend | 10.00% | 12.90% | -21.97% | 519d | 256d | 26d | 36.8% |
| SPY 150d trend | 9.16% | 12.02% | -20.89% | 540d | 381d | 49d | 23.5% |

No original primary candidate was close to a worst-case 20-session recovery. The
credit canary had the smallest drawdown but still took as long as 295 sessions from a
material trough to its previous high. Reducing loss depth and accelerating recovery
are different objectives.

## Frozen 100-variant recovery sweep

The addendum tested 100 additional long-only rules at weekly or monthly frequency:

- SPY/QQQ 20/50/75/100/150-day trend controls;
- SPY/QQQ 8%/10%/12% volatility targets combined with 20/50/100-day trends;
- defensive/cyclical sector barbells across two rebalance frequencies, three cyclical
  budgets, two momentum windows, three trend windows and top-2/top-3 selection.

All signals use trailing data through the decision close and execute at the next close.
The headline uses 5 bp one-way trading cost and the stress path uses 10 bp.

| Frozen result | Count |
|---|---:|
| New variants | 100 |
| Original return/vol/drawdown/subperiod objective passes | 21 |
| Every >=5% drawdown recovered within 20d | 0 |
| Objective pass with median recovery <=20d | 0 |
| Objective pass with majority of material episodes recovered within 20d | 0 |
| Joint objective + hard 20d pass | 0 |

## Honest near-misses

These three rows are parameter neighbours from one sector-barbell family, not three
independent strategies.

| Role | Exact frozen rule | CAGR | 10 bp CAGR | Vol | MaxDD | Worst recovery | Median | <=20d share | Turnover/year |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Duration-first rank winner | Monthly; 50% cyclical; 63d momentum; 200d trend; top 3 | 10.98% | 10.74% | 12.95% | -24.10% | 218d | 23d | 45.5% | 4.35x |
| Best return/drawdown compromise | Monthly; 65% cyclical; 63d momentum; 100d trend; top 2 | 12.05% | 11.66% | 13.58% | -19.57% | 219d | 25d | 40.0% | 6.83x |
| Best typical recovery share | Monthly; 65% cyclical; 63d momentum; 200d trend; top 3 | 11.06% | 10.75% | 13.10% | -20.54% | 228d | 22d | 48.1% | 5.49x |

The first row wins the frozen duration ordering, but its MaxDD sits close to the -25%
limit and its 2007-2012 CAGR is only 6.82%. The second row gives up one worst-case
recovery day and two median days for materially better return, MaxDD and early-period
performance; it is the more balanced paper candidate. The third row is closest to the
20-day intent in a typical episode, but still repairs fewer than half of material
drawdowns within 20 sessions.

## What creates the long recovery tail

For the duration-first rule, the four longest material recovery legs were:

| Prior peak | Trough | Recovered | Depth | Peak-to-trough | Trough-to-recovery |
|---|---|---|---:|---:|---:|
| 2007-12-26 | 2009-03-09 | 2010-01-19 | -24.10% | 301d | 218d |
| 2010-04-23 | 2010-07-02 | 2011-01-03 | -12.40% | 49d | 127d |
| 2018-01-26 | 2018-03-23 | 2018-09-20 | -9.63% | 39d | 125d |
| 2015-07-20 | 2016-01-20 | 2016-07-14 | -10.17% | 127d | 122d |

The tail is not only the GFC. Ordinary slow recoveries in 2010, 2015-2016 and 2018 also
break the 20-day ceiling. Weekly rebalancing did not solve the problem: it raised
turnover and often whipsawed entry without guaranteeing faster capital repair.

## Recommendation

Treat 20 sessions as a **monitoring aspiration for the median episode**, not as a
credible hard guarantee for an unlevered long-only ETF portfolio targeting 10% return.
The balanced sector-barbell row above can enter the paper-monitoring set, but it should
not replace the other independent mechanisms and it must not be advertised as a
20-day-recovery strategy.

A true hard recovery deadline would require one of three unacceptable shortcuts:
resetting the high-water mark, realizing the loss and calling the new NAV a recovery,
or adding enough leverage/option convexity after losses to breach the risk budget.
Permanent SPY collars tested in the original study reduced volatility but also missed
the return target and did not solve duration.

## Leakage verdict and artifacts

### Ensemble and relaxed-return follow-up

The next addendum combined previously frozen sleeves and separately lowered the return
floor to 5%. Combinations improved typical recovery to an 18-session median at roughly
11.3% CAGR, while a 45%-risk / 55%-SHY allocation avoided a -5% drawdown at roughly
5.2% CAGR. Neither produced a hard 20-session worst-case recovery guarantee. See
`docs/research_decisions/2026-07-19_etf-ensemble-recovery-result.md`.

Static scan: zero blockers. The one warning remains the full-sample 5% quantile used
only for ex-post CVaR reporting. Manual review confirms trailing windows, next-close
execution, no backward fill or forward-return signal, and no use of future trough or
recovery dates in portfolio construction. Verdict: trustworthy as a candidate screen,
not proof of forward recovery time.

- Protocol: `docs/research_decisions/2026-07-18_etf-drawdown-recovery-protocol.md`
- Runner: `scripts/etf_drawdown_recovery_study.py`
- Reusable duration helper: `src/alpha_lab/analytics/returns.py`
- Existing metrics: `data/results/etf_strategy_50plus_pre2022/drawdown_recovery_existing_metrics.csv`
- Variant metrics: `data/results/etf_strategy_50plus_pre2022/drawdown_recovery_variant_metrics.csv`
- Combined frontier: `data/results/etf_strategy_50plus_pre2022/drawdown_recovery_frontier.csv`
- Episode ledger: `data/results/etf_strategy_50plus_pre2022/drawdown_recovery_episodes.csv`
