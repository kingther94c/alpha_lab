# ETF ensemble and 5% return-floor addendum — pre-2022 result

## Decision

**Combining sleeves materially improves recovery, but it does not make a hard
20-business-day recovery guarantee credible.** The answer depends on whether “20 days”
means the typical material drawdown or the single worst historical drawdown:

- At the original return objective, a simple two-sleeve ensemble achieved 11.31% CAGR,
  11.30% volatility and -16.22% MaxDD. Its median recovery after a drawdown reaching
  -5% was 18 sessions and 55% of such episodes recovered within 20 sessions. Its worst
  recovery still took 266 sessions.
- A different two-sleeve ensemble shortened the worst material recovery to 116 sessions
  while retaining 10.24% CAGR, but its median recovery was 56 sessions. Optimizing the
  worst tail and optimizing the typical episode select different portfolios.
- Lowering the return floor to 5% creates a full-GFC allocation with 5.21% CAGR, 3.48%
  volatility and -4.32% MaxDD. It avoided a -5% drawdown entirely in this sample. That
  is loss avoidance, not proof that a future -5% loss would recover in 20 days.

Every result ends at 2021-12-31. No post-2021 observation was read.

## Frozen sweep result

The protocol enumerated monthly equal-weight subsets of eight previously frozen ETF
strategy sleeves, four risk-budget scales, a separate credit-canary tier and limited
inverse-volatility diagnostics.

| Result | Count |
|---|---:|
| Total ensemble trials | 1,504 |
| Full-GFC tier | 848 |
| Credit-canary 2008+ tier | 656 |
| Original 9%/8.5%-stress return and risk gate | 329 |
| Relaxed 5%/4.5%-stress return and risk gate | 1,122 |
| No -5% drawdown in sample | 442 |
| Every material recovery <=20 sessions | 0 |
| Original gate plus no -5% drawdown / hard 20d recovery | 0 |
| Relaxed gate plus no -5% drawdown / hard 20d recovery | 64 |

All 64 relaxed joint passes are avoidance cases; none experienced a -5% drawdown and
then repaired it within 20 sessions.

## Three distinct portfolio choices

| Objective | Research sleeve weights | CAGR | 20 bp meta-cost CAGR | Vol | MaxDD | Max underwater | Worst >=5% recovery | Median | <=20d share |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Typical recovery near 20d | 50% cross-asset 6m top-3 momentum; 50% recovery sector barbell | 11.31% | 11.26% | 11.30% | -16.22% | 502d | 266d | 18d | 55.0% |
| Shorter worst recovery | 50% fixed retail all-weather; 50% recovery sector barbell | 10.24% | 10.19% | 8.90% | -17.17% | 339d | 116d | 56d | 11.1% |
| 5% return / avoid material loss | 55% SHY; 11.25% each in QQQ vol target, sector low-Ulcer, fixed all-weather and trend-gated all-weather | 5.21% | 5.18% | 3.48% | -4.32% | 299d | none occurred | n/a | n/a |

The percentages above are strategy-sleeve research budgets, not final direct ETF
holdings. Each sleeve produces its own ETF targets; implementation must aggregate and
net the underlying ETF orders.

## Preferred 10% candidate

The pre-specified equal-weight combination of:

1. `cross_asset_mom_126_top3`; and
2. `recovery_sector_barbell_M_cyc65_mom63_ma100_top2`

is the best answer if “尽量 20 biz day” means a median or majority objective rather
than a worst-case guarantee. It rebalances the two virtual sleeves monthly and trades
at the next close.

| Stress window | Return | Window MaxDD |
|---|---:|---:|
| GFC | -1.79% | -13.16% |
| 2011 euro / US downgrade | +1.71% | -4.42% |
| 2018 Q4 | -14.13% | -14.46% |
| COVID crash | -10.98% | -11.65% |

Its 2007-2012 CAGR was 11.23% and its 2013-2021 CAGR was 11.36%, a much smaller split
than the individual sector rules. The weakness is duration tail risk: one recovery took
266 sessions and the longest full underwater episode lasted 502 sessions.

Post-selection weights from 40%/60% through 50%/50% retained 11.31%-11.46% CAGR,
11.30%-11.49% volatility, approximately -15.6% to -16.2% MaxDD, an 18-19 session
median recovery and 52%-55% recovered within 20 sessions. Quarterly meta rebalancing
gave nearly identical results. These variants are robustness evidence, not new
independent candidates; the exact 50%/50% frozen rule remains the cleaner choice.

## Preferred 5% candidate

Use 45% total risk budget, equally divided across four strategy sleeves, with 55% in
SHY:

- 11.25% QQQ 12% volatility-target sleeve;
- 11.25% low-Ulcer positive-trend sector sleeve;
- 11.25% fixed retail all-weather sleeve;
- 11.25% trend-gated all-weather sleeve;
- 55.00% SHY.

At 5 bp meta cost it produced 5.21% CAGR, 3.48% volatility and -4.32% MaxDD. At 20 bp
meta cost CAGR was 5.18%. Monthly and quarterly meta rebalancing were almost identical.
The stress windows were +1.57% over the GFC, -3.68% in 2018 Q4 and -2.59% in the COVID
crash window; worst calendar year was -0.54%.

Because no -5% episode occurred, smaller thresholds are more informative:

| Drawdown threshold | Episodes | Worst trough recovery | Median | <=20d share |
|---|---:|---:|---:|---:|
| -1% | 38 | 134d | 18.5d | 63.2% |
| -2% | 14 | 134d | 39.5d | 14.3% |
| -3% | 6 | 134d | 87.5d | 0.0% |
| -4% | 2 | 50d | 42.5d | 0.0% |
| -5% | 0 | n/a | n/a | n/a |

This demonstrates the distinction between depth and duration: a portfolio can remain
within a shallow -5% loss budget yet stay slightly below its old high for 299 sessions.
If the user's discomfort is loss magnitude, the 5% candidate is a meaningful
improvement. If any time below the previous high is unacceptable after 20 sessions, it
does not solve the problem.

The 45%-50% risk-budget range is the relevant boundary. At 40%, CAGR fell below 5%.
At 55%, MaxDD crossed -5% and the worst material recovery immediately increased to
200 sessions. The 45% version has more safety margin than the sample-optimal 50% row.

## Shorter-worst-recovery alternative

The 50% fixed all-weather / 50% recovery-sector-barbell ensemble produced 10.24% CAGR,
8.90% volatility and -17.17% MaxDD. It cut the worst material recovery from 218 days
for the best individual recovery variant to 116 days. It is useful when the worst
historical recovery matters more than the typical one, but only one of nine material
episodes recovered within 20 sessions.

Weight sensitivity from 40%-60% in the all-weather sleeve kept the worst material
recovery near 116-120 sessions and CAGR near 9.9%-10.6%. Quarterly results were nearly
identical. This is a genuine local plateau, not a single 50/50 spike.

## Audit and limitations

- Automated leakage scan: zero blockers, zero warnings across both ensemble scripts.
- The optimized ensemble engine was compared return-for-return with the canonical
  drift-aware engine before the sweep. Month-end decisions trade at the next close.
- Fixed weights use no fitted statistics. Inverse-volatility diagnostics use only the
  trailing 63 or 126 sessions and were not selected as the headline.
- SHY is an explicit ETF return source. There is no uncharged synthetic cash and no
  leverage, so no omitted financing charge.
- Sleeve returns already contain their own trading costs; the ensemble adds another
  5 bp meta-level cost. The 20 bp diagnostic stresses meta-level costs, while underlying
  sleeve cost stress is inherited from the earlier individual studies rather than
  reconstructed order by order.
- Treating sleeves as virtual subaccounts conservatively double-counts some turnover.
  A production implementation must aggregate overlapping ETF targets and recalculate
  exact net trades.
- The 1,504 combinations create substantial selection risk. Only the simple frozen
  equal-weight rows receive candidate status; neighbouring weights are diagnostics.
- The 5% result benefited from the pre-2022 stock/bond/cash environment. It has not seen
  the later joint stock-bond inflation shock because the user explicitly withheld
  post-2021 data.
- Credit-canary combinations are excluded from the primary recommendation despite some
  smoother results because their common sample starts only in 2008 and misses the early
  GFC decline.

Verdict: trustworthy as a pre-2022 candidate screen. Combination improves the feasible
frontier, and a 5% floor permits material-drawdown avoidance, but neither creates a
credible hard 20-session recovery guarantee.

## Artifacts

- Protocol: `docs/research_decisions/2026-07-19_etf-ensemble-recovery-protocol.md`
- Main sweep: `scripts/etf_strategy_ensemble_recovery_study.py`
- Robustness: `scripts/etf_strategy_ensemble_robustness.py`
- All 1,504 metrics: `data/results/etf_strategy_50plus_pre2022/ensemble_recovery_metrics.csv`
- Robustness rows: `data/results/etf_strategy_50plus_pre2022/ensemble_recovery_robustness.csv`
- Selected return paths: `data/results/etf_strategy_50plus_pre2022/ensemble_recovery_selected_returns.parquet`

