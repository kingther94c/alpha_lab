# US sector target-8 study — 2022 holdout result

## Verdict

**Reject the frozen pure-sector candidates for the stated 8% annual-return objective.** None of the
five candidates produced a positive 2022 return, so none passed the preregistered single-year gate.
Calendar year 2023+ remains sealed.

## Primary results

| Strategy | 2022 return | 20 bp stress | Volatility | Maximum drawdown | Ulcer Index |
|---|---:|---:|---:|---:|---:|
| C1 smelted trend/downside | -9.29% | -9.92% | 15.17% | -13.99% | 8.13% |
| C2 asymmetric sector ferry | -3.74% | -4.25% | 9.22% | -10.15% | 6.20% |
| C3 woven defensive/upside | -6.61% | -6.89% | 15.18% | -13.39% | 7.58% |
| C4 downside-vol budget | -8.72% | -9.10% | 9.99% | -11.57% | 7.80% |
| C5 low-Ulcer positive trend | -7.43% | -8.10% | 15.20% | -11.54% | 6.57% |
| Synthetic SPY collar | -15.52% | — | 13.90% | -16.63% | 11.24% |
| SPY | -18.18% | — | 24.24% | -24.50% | 15.06% |

## Interpretation

- C2 was the strongest drawdown-first result. It held an average 53.5% equity exposure, stayed near
  the desired volatility range, and materially beat both SPY and the synthetic collar on return and
  drawdown. It nevertheless missed the central return objective by more than eleven percentage points.
- C4 hit the 10% volatility ceiling and kept drawdown near -12%, but its underlying sector sleeve lost
  enough that scaling exposure to an average 50.4% did not preserve capital.
- The higher-return development candidates C1, C3, and C5 all experienced a sharp volatility increase
  in 2022. Their defensive sector selection reduced SPY's loss but did not create positive absolute return.
- Every candidate beat the synthetic collar's return, maximum drawdown, and Ulcer Index in 2022. This
  supports sector/cash risk control as a useful defensive sleeve, but not as a standalone 8% target strategy.
- The synthetic collar returned -6.8% in 2022Q2 versus Cboe's published -6.7% for CLL, a close diagnostic
  match. It remains a modeled benchmark rather than a historical option-chain backtest.

## Audit

- Data boundary checks reject every price, VIX, rate, or option-proxy observation after 2022-12-31.
- Target weights use trailing inputs and execute at the next close.
- The fixed legacy nine-sector universe is unchanged.
- Static scan: zero blockers. The quantile warnings are ex-post CVaR diagnostics and do not enter a
  signal, weight, or trade. Forward-fill findings are backward carries of already observed data.
- Full test suite passed. The files touched by this study pass Ruff; the repository-wide Ruff run still
  reports 162 pre-existing findings outside this study's scope.

## Next decision

Do not tune these five candidates using 2022 and do not open 2023+. If research continues, register a
new study whose return source combines C2-like defensive sector/cash allocation with an explicit option
overlay or another diversifying return source. The 2022 result should remain part of that study's prior.
