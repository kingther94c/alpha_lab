# US sector rotation / hardware-software phase-2 protocol

Date: 2026-07-17  
Status: frozen before the phase-2 batch is evaluated  
Holdout status: sealed (`2022-01-03+` must not be loaded)

## Research question

Can a small, mechanism-led set of sector-rotation or hardware/software timing
signals earn positive net active returns after realistic ETF costs while keeping
the portfolio fully invested and its decision-date 21/63-day geometric volatility
within 10% of SPY?

This is not a parameter search.  The 2013-2021 period was already viewed in the
phase-1 study and is therefore labelled **development-validation**, not an untouched
validation sample.  The only pseudo-OOS holdout starts in 2022 and remains sealed.

## Sample and execution contract

- Sector train: available history through 2012-12-31.
- Technology train: available SOXX/IGV history through 2012-12-31.
- Development-validation: 2013-01-01 through 2021-12-31.
- Sealed pseudo-OOS: 2022-01-03 through 2026-06-30.  It may be opened once, for
  one frozen champion, only after a pre-holdout report and explicit user approval.
- Monthly candidates form targets at the final close of the month.  Weekly
  candidates form targets at the Friday close (or the week's last trading close).
  Every target trades at the next close and first earns the following close-to-close
  return.
- Portfolios are long-only, sum to 100% including cash, and have a 60% cap on any
  sector or technology ETF.  They are projected into a `[0.9, 1.1]` band around
  SPY's geometric mean of trailing 21- and 63-day volatility.
- Primary cost: 10 bp one-way on actual non-cash buys plus sells.  Five bp is an
  optimistic display and 20 bp is the failure stress.  Cash earns lagged DTB3 with
  calendar-day accrual.
- Yahoo adjusted prices and volume and daily market-observable FRED series are
  allowed.  FRED observations are shifted two trading bars.  Revised macro series,
  current constituent backfills, current PE backfills, and non-executable option
  chains are forbidden.

## Lifetime search registry

The lifetime cap is 24 selectable strategies: at most 14 sector and 10 technology.
Every signal x direction x lookback x rebalance x threshold x weighting combination
counts as a separate trial.  An ensemble selected after seeing results also counts.

Already consumed in phase 1 (7): sector multi-horizon momentum; sector
momentum+volume; XLK hardware lead; XLK breadth; XLK relative reversal; SOXX/IGV
winner; SOXX/IGV contrarian.  Risk-matched sector equal-weight, static XLK exposure,
the train-frozen breadth exposure mix, and SOXX/IGV equal-weight are benchmarks and
are not selectable trials.

This phase freezes 8 additional trials, bringing the lifetime total to 15.  The
unused quota cannot be filled after this batch's results are seen.  If no candidate
passes, the study stops for user review.

## Frozen sector candidates (S1-S4)

All sector scores use the nine legacy ETFs to avoid the XLC/XLRE launch bridge.
Rolling market beta is `cov(r_i, r_SPY) / var(r_SPY)` over 252 sessions, lagged one
bar before constructing daily residual return `eps_i`.

### S1 — residual momentum conditional on high dispersion

- Mechanism: sector-specific information diffuses gradually; removing market beta
  and requiring high cross-sectional residual dispersion filters common macro noise.
- Signal: residual 12-1, 6-1 and 3-1 cumulative returns are independently ranked and
  averaged.  At month-end, hold the top three only when the cross-sectional standard
  deviation of 21-day residual returns is above its train-only month-end median and
  higher than 21 sessions earlier; otherwise use equal weight.
- Falsification: high-dispersion conditional rank IC must improve on low dispersion;
  otherwise the state gate has no mechanism-consistent evidence.
- Main risk: high dispersion may represent overreaction rather than information
  diffusion.

### S2 — credit/breadth defensive-recovery state machine

- Mechanism: weakening credit conditions plus contracting market breadth identifies
  a financing/deleveraging state; improving credit plus breadth identifies recovery.
- Signal: credit is the 21-day log return of adjusted `HYG/LQD`.  Breadth is the
  share of sectors both above their 126-day moving average and outperforming SPY
  over 63 days; its change is measured over 21 sessions.  At month-end:
  `(credit < 0, breadth_change < 0)` holds the three lowest trailing-126-day downside
  betas; `(credit > 0, breadth_change > 0)` holds the three highest S1 residual
  momentum scores; other states use equal weight.
- Falsification: the low-minus-high downside-beta next-month return must improve in
  the joint stress state, not merely in all months.
- Main risk: HYG/LQD can be coincident and contains rate exposure.

### S3 — high-volume residual-loser liquidity provision

- Mechanism: a sharp sector-specific fall on exceptional trading activity, while
  credit is stable, is more likely to be a price-insensitive flow shock than a new
  economy-wide fundamental deterioration.
- Signal: at each week-end, find sectors in the bottom residual-return tercile over
  five sessions and top cross-sectional tercile of `5-day mean dollar volume /
  trailing-126-day median dollar volume`.  When `HYG/LQD` has a non-negative
  five-day return, put 50% in sector equal weight and 50% equally in qualifiers;
  if there are no qualifiers use equal weight.  The 60% cap still applies.
- Falsification: qualified high-volume losers must outperform matched ordinary
  losers over the following week before costs; 20 bp must not reverse the full
  portfolio edge.
- Main risk: ETF volume is not creation/redemption flow and may reflect genuine news.

### S4 — dynamic macro sensitivity

- Mechanism: sectors differ in cash-flow duration, bank-margin sensitivity,
  inflation pass-through and capital intensity; persistent real-rate, curve and
  inflation-expectation moves may therefore change relative returns.
- Signal: use two-bar-lagged daily changes in `DFII10`, `DGS10-DGS2`, and `T10YIE`.
  At each month-end, regress each sector's SPY-residual daily return on the three
  drivers over the trailing 756 sessions.  Score a sector by the fitted betas times
  each driver's trailing-63-session level change; hold the top three.
- Falsification: the cross-sectional score's next-month rank IC must have the same
  sign in 2013-2016 and 2017-2021.
- Main risk: sensitivities drift and equity/rate co-movement can reflect a third
  shock rather than predictive transmission.

## Frozen hardware/software candidates (T1-T4)

Each dynamic strategy is compared with a **train-frozen static shadow portfolio**:
the average raw asset weights generated through 2012, repeated thereafter and put
through the identical volatility matcher.  The dynamic-minus-shadow result, not
SPY outperformance alone, is the timing-alpha test.  Regressions also control SPY,
`XLK-SPY`, and `SOXX-IGV` exposures.

### T1 — macro-confirmed hardware lead

- Mechanism: semiconductors are more exposed than software to the global durable
  goods, inventory and capital-expenditure cycle.
- Signal: compute 63-day SOXX-minus-IGV momentum after subtracting each ETF's
  lagged-252-day SPY beta.  A hardware state requires positive relative residual
  momentum and at least two of: 63-day copper return positive (`HG=F`), 63-day broad
  dollar change negative (`DTWEXBGS`, lagged two bars), and 21-day `HYG/LQD` return
  positive.  Monthly raw weights are 60% SOXX/40% SPY in hardware state and 60%
  IGV/40% SPY otherwise.
- Falsification: beta-neutral next-month SOXX-minus-IGV return must be concentrated
  in the jointly confirmed state, not one macro component or one year.
- Main risk: copper is China-sensitive and the conditions reduce sample size.

### T2 — credit-stable short-term liquidity reversal

- Mechanism: extreme beta-neutral relative moves with exceptional volume, absent a
  credit shock, are candidate price-insensitive flows for which liquidity provision
  is compensated.
- Signal: weekly rolling-beta-neutral three-day SOXX-minus-IGV residual spread,
  standardized by its trailing-252-day volatility.  When `|z| > 2`, the winning ETF's
  daily dollar volume exceeds 1.5 times its trailing-63-day median, and the five-day
  `HYG/LQD` return is non-negative, hold 60% in the laggard and 40% SPY for the next
  three sessions; otherwise use 50% SPY, 25% SOXX and 25% IGV.  Overlapping events
  do not reset the holding clock.
- Falsification: reversal must be confined to the stable-credit/high-volume bucket
  and survive 20 bp.
- Main risk: a three-day edge is cost-sensitive and HYG/LQD is an imperfect credit
  control.

### T3 — negative semiconductor shock diffusion

- Mechanism: negative hardware demand, inventory and capex information may be
  incorporated into upstream semiconductors before downstream software/technology.
  The negative direction is frozen; a positive-direction variant will not be tried.
- Signal: at each week-end standardize five-day lagged-beta residual returns by their
  trailing-252-day standard deviations.  Trigger when SOXX `< -1.5` and IGV `> -0.5`.
  The default raw allocation is 60% XLK/40% SPY; following a trigger it is 100% SPY
  for five sessions.  Overlapping triggers do not reset the clock.
- Falsification: trigger-following five-day IGV residual return must be negative in
  both fixed development-validation subperiods.
- Main risk: the event count may be small and published lead-lag effects weaken in
  extended samples.

### T4 — residual breadth with dispersion compression

- Mechanism: simultaneous market-beta-adjusted strength in hardware and software,
  with their normalized momentum converging, is more consistent with broad earnings
  improvement than a narrow thematic re-rating.
- Signal: for each ETF average its 21- and 63-day cumulative residual return divided
  by the corresponding trailing-252-day standard deviation.  At month-end, a broad
  coherent state requires both scores positive and their absolute difference below
  the expanding median available at that date.  Raw allocation is 60% XLK/40% SPY
  in state and 100% SPY otherwise.
- Falsification: positive breadth must predict higher next-month XLK residual return
  specifically in the low-dispersion bucket.
- Main risk: two ETFs are a weak proxy for true breadth and have index overlap.

## Profit and robustness gate

The primary comparison uses 10 bp net returns versus the correct benchmark.  A
candidate may enter fundamental review only if all of the following hold:

1. Active CAGR at least +1.0% and active IR at least 0.35.
2. Active CAGR positive in both 2013-2016 and 2017-2021.
3. Six-month block-bootstrap 90% lower bound above zero.
4. Delete-best-year active CAGR remains positive; no year contributes over 40% of
   absolute annual log-active return; at least 7 of 9 leave-one-year-out estimates
   are positive.
5. At least two valid lagged SPY trend/volatility regimes are positive and none has
   active IR below -0.25.
6. Decision-date volatility compliance is 100%; realized volatility-ratio median is
   inside the band and at least 70% of valid days are inside it.
7. Twenty-bp net performance remains positive.
8. Sector strategies also beat the risk-matched sector equal-weight benchmark.
   Technology strategies beat their train-frozen static shadow, with at least half
   of apparent active return surviving exposure controls.

Family-level selection uses monthly 10 bp active returns.  Hansen SPA with a
stationary-bootstrap expected block length of six months (10,000 draws, fixed seed)
is followed by Holm correction across the sector and technology families at 5%.
The selected candidate must also have a Deflated Sharpe probability of at least 95%,
using the full lifetime trial count of 24 rather than a correlation-reduced count.

## Fundamental gate and holdout rule

Only a statistical survivor is reviewed.  Support requires two independent sources
published before 2022, at least one original paper/working paper, matching direction,
horizon and mechanism; one point-in-time non-price intermediary must also agree.
Static beta, broad factor exposure and single-year explanations must be excluded.
The review verdict is `supported`, `plausible-but-unconfirmed`, or `unsupported`.

Only `supported` can be proposed as the single champion.  Before opening 2022+, the
formula, cost, code/config hash and holdout pass/fail rule are frozen, leakage audit
must have no blocker, a pre-holdout HTML report is delivered, and the user must give
explicit approval.  Failure stops the search; it does not authorize trying a second
candidate on the same holdout.
