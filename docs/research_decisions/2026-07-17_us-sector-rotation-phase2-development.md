# Decision record: US sector rotation / hardware-software phase 2

Date: 2026-07-17  
Decision: **reject the frozen phase-2 batch; stop for user review**  
Holdout: **sealed; no 2022+ observation was loaded**

## Question

After the phase-1 failures, can eight mechanism-led candidates produce credible
timing alpha in 2013-2021 after 10 bp one-way costs, correct static-exposure
benchmarks, SPY volatility matching, regime checks and multiple-testing penalties?

## Frozen batch

Sector: residual momentum conditional on dispersion; credit/breadth state machine;
high-volume residual-loser liquidity provision; dynamic macro sensitivity.

Hardware/software: macro-confirmed hardware lead; credit-stable short-term reversal;
negative semiconductor shock diffusion; residual breadth with dispersion compression.

The exact pre-result definitions and gates are recorded in
`2026-07-17_us-sector-rotation-phase2-protocol.md`.  Together with phase 1, 15 of
the 24 lifetime trial budget has been consumed.  The remaining quota cannot be used
after seeing this batch without a new user-reviewed research cycle.

## Result

No candidate passed the all-of-the-above gate.

| Candidate | Correct baseline | 10 bp active CAGR | Active IR | 20 bp active CAGR | Key failure |
|---|---|---:|---:|---:|---|
| S1 residual momentum/dispersion | sector equal weight | -0.27% | -0.07 | -1.08% | negative after primary cost |
| S2 credit/breadth state | sector equal weight | -0.37% | -0.11 | -1.15% | negative after primary cost |
| S3 high-volume loser | sector equal weight | -2.91% | -0.88 | -6.01% | strongly negative and high turnover |
| S4 dynamic macro beta | sector equal weight | +0.53% | 0.10 | -0.36% | below hurdle; bootstrap and 20 bp fail |
| T1 macro-confirmed hardware | train-static shadow | +0.37% | 0.15 | +0.23% | below hurdle; weak bootstrap/evidence |
| T2 credit-stable reversal | train-static shadow | +0.05% | 0.21 | +0.01% | economically negligible; concentrated |
| T3 negative hardware diffusion | train-static shadow | -0.49% | -0.77 | -0.72% | wrong sign |
| T4 residual breadth/compression | train-static shadow | -0.99% | -0.43 | -1.51% | negative in both subperiod structure |

Family-level stationary-bootstrap SPA p-values were 75.67% for sector and 59.88%
for technology; Holm-adjusted p-values were 100% for both.  No Deflated-Sharpe
probability reached 95%.  All decision-date target volatility checks were compliant,
but realized-vol band share was below the frozen 70% gate for S4 and all technology
candidates, another independent reason not to advance them.

## Leakage and implementation audit

The repository scanner reported 0 blockers and 0 warnings across the phase-2 script,
phase-1 data/cash/report helpers, drift-aware backtest engine and volatility matcher.
The manual checklist confirmed:

- no negative shift, centred window, backfill or validation-wide normalizer;
- rolling beta is lagged one bar and FRED data is lagged two price bars;
- close-formed targets trade at the next close and first earn the following return;
- technology baselines use train-only, time-weighted static exposure;
- the nine-sector universe is frozen and XLC/XLRE launch bridging is excluded;
- adjusted price and volume loaders hard-stop before 2022.

Verdict: the results are trustworthy for **rejecting** this batch.  ETF/index
survivorship and free-data proxy quality remain limitations, but neither can rescue
the weak headline results.

## Fundamental-review decision

Each candidate had an ex-ante economic mechanism, but the protocol required a
statistical survivor before the deeper independent literature and non-price
intermediary review.  Because there was no survivor, no signal is labelled
`fundamentally supported`, and no champion can be proposed for the holdout.

Point-in-time sector PE and executable historical option chains remain unavailable
for free; value regressions and call/put-spread backtests therefore remain parked
rather than being represented by current-data backfills.

## Artifacts

- HTML report: `reports/us_sector_rotation_phase2_development.html`
- Metrics: `data/results/us_sector_rotation_phase2_metrics.csv`
- Reproducible study: `scripts/us_sector_rotation_phase2.py`
- Frozen protocol: `docs/research_decisions/2026-07-17_us-sector-rotation-phase2-protocol.md`

## Next action

Stop and wait for user review.  Do not open 2022+, tune these definitions, reverse a
failed signal, or spend the unused trial quota within this research cycle.
