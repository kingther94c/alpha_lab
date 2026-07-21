# US sector rotation and XLK hardware/software — pre-holdout decision

## Header

| field | value |
|---|---|
| **Slug** | `us-sector-rotation-preholdout` |
| **Date** | `2026-07-17` |
| **Researcher** | Codex team |
| **Status** | sector price/volume `reject`; hardware lead/direct rotation/reversal `reject`; breadth, PE and options `park` |
| **Source** | `scripts/us_sector_rotation_study.py` |
| **Report** | `reports/us_sector_rotation_preholdout.html` |
| **Holdout** | this script did not read 2022+; the interval remains a sealed pseudo-OOS set |

## 1. Research question

Can monthly US sector rotation, or SOXX-versus-IGV hardware/software information,
produce a fully invested long-only portfolio that beats SPY net of costs while its
21/63-day geometric volatility remains within ±10% of SPY?

## 2. Hypothesis and mechanism

Sector momentum may arise from slow industry information diffusion. ETF dollar-volume
surprise may distinguish broad participation from weak trends. Inside technology,
semiconductor and software returns may lead one another through supply-chain information
diffusion or limited attention. These are priors, not evidence of ETF-level predictability.

## 3. Universe and data

- Legacy history: XLK, XLF, XLV, XLI, XLY, XLP, XLE, XLU, XLB.
- Full bridge: the legacy nine plus XLC and XLRE; effective evaluation begins in 2019
  after common-history and signal warm-up requirements.
- Technology sub-study: XLK, SOXX, IGV, with SPY as benchmark/overlay asset.
- Yahoo adjusted close and volume; DTB3 from FRED for prior-known cash accrual.
- Download end was hard-coded as `2022-01-01` exclusive. Separate hard assertions cover
  both price and volume indexes; this script read no 2022+ observations. This proves the
  behavior of the study script, not that no researcher has ever seen post-2021 markets.
- XLC starts 2018-06-19 and XLRE starts 2015-10-08. Today's 11-sector taxonomy was not
  backfilled into earlier history.

## 4. Signal and portfolio construction

- Primary sector signal: average cross-sectional rank of 12-1, 6-1, and 3-1 momentum;
  hold the top three.
- Volume candidate: 75% momentum rank plus 25% trailing own-history log dollar-volume
  surprise rank. ETF volume is not described as fund flow.
- Rebalance: actual last trading day of each month; trade at the following close; the new
  target first earns the next close-to-close return.
- Portfolio: long-only, weights sum to 1. At each decision, minimize distance from the raw
  target subject to the portfolio's geometric mean of 21- and 63-day covariance volatility
  being 0.9–1.1 times the same SPY measure.
- Backtest: shares drift between trades. Cost is 5 bp one-way on actual non-cash buy plus
  sell notional; 10 bp is the stress case.
- Sector positions are capped at 60%. A small turnover penalty is applied even when the raw
  target is already inside the volatility band.
- Regimes: prior-known SPY above/below 200DMA × prior-known 21/63-day geometric volatility
  above/below its trailing 756-day median.

## 5. Headline validation performance

Daily returns, 2013-01-01 through 2021-12-31:

| strategy | CAGR | active CAGR vs SPY | active IR | 10bp active CAGR | MaxDD |
|---|---:|---:|---:|---:|---:|
| Risk-matched 9-sector equal weight | 14.66% | -1.84% | -0.450 | -1.86% | -36.72% |
| Multi-horizon momentum | 15.97% | -0.53% | -0.072 | -0.97% | -32.44% |
| Momentum plus volume | 15.71% | -0.79% | -0.115 | -1.24% | -32.44% |

The full 11-sector bridge was also negative versus SPY: momentum active CAGR was -4.88%
and momentum-plus-volume was -4.38% from 2019-08 through 2021-12.

## 6. Technology sub-study

| XLK/SPY overlay | active CAGR | active IR | 10bp active CAGR |
|---|---:|---:|---:|
| SOXX hardware lead | -0.45% | -0.089 | -0.73% |
| SOXX+IGV breadth | +2.08% | 0.520 | +1.70% |
| Relative reversal | +3.47% | 0.960 | +3.16% |
| Static risk-matched XLK | +3.39% | 0.675 | +3.18% |
| Breadth train-exposure baseline | +1.88% | 0.837 | +1.85% |

Relative reversal was proposed only after seeing the negative training relationship, so it
was exploratory. Its paired incremental result versus static risk-matched XLK was effectively
zero: 5 bp CAGR difference +0.08%, incremental IR -0.008, six-month block-bootstrap 90%
interval [-1.35%, +1.37%], p=0.472. At 10 bp the CAGR difference was -0.02%, p=0.517.

Direct SOXX/IGV winner and contrarian rotation also failed to beat simple SOXX/IGV equal
weight: their validation CAGRs were 26.74% and 26.71%, versus 27.48% for equal weight.

Breadth is not rejected outright. Its two validation subperiods were positive (+0.41% and
+3.48% active CAGR), and its six-month block-bootstrap annual mean was +1.88% with a 90%
interval of [+0.02%, +4.06%]. However, 57.28% of absolute yearly log-active contribution
came from one year. More importantly, versus a static SPY/XLK mix frozen from the training
period's breadth frequency, breadth added only +0.20% CAGR at 5 bp (incremental IR 0.074,
90% interval [-1.32%, +1.92%], p=0.387) and lost -0.16% at 10 bp. It is therefore
`park / needs revision`, not evidence of tradable timing alpha.

## 7. Diagnostics and robustness

- Leakage scan found zero blockers. Its sole warning is an unrelated, pre-existing full-sample
  quantile inside the CVaR diagnostic, which is not called by this study. Manual inspection
  confirmed next-close execution, trailing covariance/volume windows, lagged regime labels,
  and forward returns used only for scoring.
- All matched validation-period decision weights summed to one within numerical tolerance;
  their ex-ante vol ratios were inside [0.9, 1.1]. This is target-date compliance, not a
  realized-vol guarantee. After holdings drifted, the legacy momentum realized-vol ratio had
  median 1.073, 10th/90th percentiles 0.924/1.186, and spent 56.95% of validation days inside
  the band. SPY and strategy realized-vol medians are both shown in the HTML.
- Sector momentum was -1.44% active CAGR in 2013–2016 and +0.23% in 2017–2021; its
  full-validation block-bootstrap annual mean was -0.55%, with 90% interval
  [-2.38%, +1.27%]. The volume candidate was negative in both subperiods and had a
  -0.77% bootstrap estimate.
- Momentum and momentum-plus-volume both lost heavily in bear/high-vol observations.
- There were no bear/low-vol observations under the frozen regime definition, so no
  four-regime stability claim is possible.
- XLK reversal and the static XLK baseline had similar positive regime profiles. This is
  consistent with technology exposure explaining the result; the paired test, rather than
  the raw CAGR comparison, is the basis for rejecting reversal timing.

## 8. Data feasibility decisions

- Historical sector P/E/value regression: `park`. Free current P/E is not a reliable
  point-in-time history; current holdings or revised earnings cannot be backfilled.
- Historical call/put spreads: `park`. Free data lacks complete timestamped bid/ask chains,
  strikes, expiries, and executable multi-leg fills. Aggregate options volume may only be a
  future diagnostic.
- Volume confirmation: feasible from free data, but the tested signal failed validation.

## 9. Decision

**Status: sector momentum/volume, hardware lead, relative reversal and direct SOXX/IGV
rotation are `reject`; breadth is `park / needs revision`.**

Neither multi-horizon sector momentum nor its volume-confirmed variant produced positive
active validation performance after costs. The apparent XLK relative-reversal edge disappeared
against the correct static risk-matched XLK baseline. Breadth's positive raw active return was
largely reproduced by a training-frozen static technology-exposure baseline, and its small
incremental return was not robust to 10 bp costs. The 2022+ holdout will not be spent on these
candidates yet.

The static risk-matched XLK allocation is a different long-term asset-allocation question. It
may be studied later only under a newly frozen hypothesis and may not be relabeled as a successful
hardware/software signal.

## 10. Next steps

- Keep 2022+ sealed for these candidates.
- If breadth is revisited, pre-register an equal-beta/exposure attribution test and a maximum
  acceptable single-year contribution before changing its status.
- If revisiting value, acquire or construct timestamped historical sector fundamentals and
  freeze availability timestamps before any return test.
- If revisiting options, obtain historical executable chains and include bid/ask, assignment,
  collateral yield, and financing.
- Treat weekly or threshold variants as new pre-registered hypotheses; do not search the
  sealed period for a better rebalance rule.

## 11. Supporting artifacts

- `scripts/us_sector_rotation_study.py`
- `src/alpha_lab/backtest/vector.py`
- `src/alpha_lab/backtest/sector_momentum.py`
- `src/alpha_lab/portfolio/vol_target.py`
- `reports/us_sector_rotation_preholdout.html`
