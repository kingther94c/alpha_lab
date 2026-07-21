# Sector target-8 / low-drawdown research protocol

Date: 2026-07-17  
Status: frozen before candidate results are evaluated  
Holdout: sealed (`2022-01-03+` must not be loaded)

## Objective

Construct a long-only US sector portfolio that aims for approximately 8% annual
return while minimizing volatility and, especially, drawdown.  The primary benchmark
is a synthetic, fully disclosed SPY collar: long SPY total return, long a protective
put, and short a covered call.  This is a drawdown-control study, not a conventional
SPY-alpha study.

The objective is lexicographic:

1. attain a net CAGR of at least 7.5% (the tolerance around the 8% target);
2. among target-attaining candidates, minimize maximum drawdown;
3. then minimize Ulcer Index, downside deviation and ordinary volatility;
4. prefer lower turnover when risk statistics are economically tied.

## Research boundary

- Train / calibration: available history through 2012-12-31.
- Development-validation: 2013-01-01 through 2021-12-31.  This period has been
  viewed in earlier studies and is not an untouched validation sample.
- Sealed pseudo-OOS: 2022-01-03 onward.  This study must stop for user review before
  opening it.
- Nine legacy sector ETFs only: XLK, XLF, XLV, XLI, XLY, XLP, XLE, XLU and XLB.
- Signals form at a close and trade at the next close.  All portfolios are long-only,
  include an explicit cash weight, sum to 100%, use no leverage, and cap each sector
  at 30% (25% where the candidate specifies it).
- Primary sector trading cost is 10 bp one-way on actual non-cash buy plus sell
  notional; 20 bp is the stress case.  Cash earns the previously observable DTB3
  rate with calendar-day accrual.
- A scheduled target is traded only when the maximum absolute difference from the
  drifted pre-trade weight exceeds 2%; this threshold is frozen for all candidates.

## Synthetic option benchmark

The benchmark follows the economic structure of Cboe's S&P 500 95-110 Collar Index:

- hold SPY total return exposure;
- buy a 5% out-of-the-money protective put on the quarterly expiration cycle;
- sell a 10% out-of-the-money covered call on the monthly expiration cycle;
- roll on the third-Friday trading close; option P&L begins after the roll close.

No historical executable SPY/SPX chain is available for free.  The benchmark is
therefore labelled **synthetic VIX/Black-Scholes collar**, never a live-fill backtest.

Frozen pricing assumptions:

- SPY unadjusted close is the option spot; adjusted close supplies the underlying
  total return.
- Daily VIX is the available 30-day ATM implied-volatility proxy.  The model's base
  volatility is `max(VIX, trailing-21-day SPY realised vol + 2 vol points)`.
- Put volatility is base volatility plus 4 vol points for downside skew.
- Call volatility is base volatility minus 2 vol points, floored at 8%, so the model
  does not over-credit covered-call premium.
- European Black-Scholes marks use the previously observable DTB3 annual rate.
- A newly purchased put is paid at model mid plus 10%; a newly sold call receives
  model mid minus 10%.  Expiring options cash-settle at intrinsic value.
- The option overlay is one unit per unit of SPY notional.  Premium financing accrues
  at the cash rate.  American exercise, discrete dividends, tax, strike grids,
  intraday roll prices and changing SPX/SPY basis are not modelled and remain explicit
  model risk.

Where an actual Cboe collar index history is freely retrievable, it may be shown only
as a validation diagnostic; it cannot be used to tune the synthetic assumptions.

## Frozen sector candidates

The new-cycle search budget is five candidates.  No sixth candidate or reversed
variant may be added after results are seen.

### C1 — smelted trend / downside-risk allocation

- Mechanism: time-series trend avoids prolonged sector impairment; inverse downside
  deviation concentrates risk in sectors with more resilient cash flows.
- Monthly eligibility: sector close above its trailing 200-session average.
- Weight eligible sectors inversely to trailing-126-session downside deviation,
  capped at 25% each.  Any unallocated weight is cash.
- Return source: equity premium plus low-volatility anomaly; incremental edge is the
  absolute-trend removal of impaired sectors.

### C2 — asymmetric sector ferry

- Mechanism: leave equity risk quickly during broad stress but require persistent
  evidence before returning, addressing the fast-exit/whipsaw contradiction.
- Weekly risk-on observation: SPY above its 200-session average and VIX below 30.
- Exit risk-on immediately when either condition fails.  Re-enter only after four
  consecutive risk-on weekly observations.
- Risk-on allocation: top four sectors by 126-session return, inverse trailing
  126-session downside deviation, 30% sector cap.
- Risk-off allocation: 50% equally in XLP, XLV and XLU; 50% cash.

### C3 — woven defensive / upside barbell

- Mechanism: defensive-sector equity premium supplies a persistent core while a
  separate cyclical sleeve supplies enough upside to approach 8%; the two sleeves
  need not make the same timing decision.
- Monthly defensive sleeve: 50% distributed inversely to 126-session downside
  deviation across XLP, XLV and XLU.
- Monthly upside sleeve: 50% distributed across the top three 126-session performers
  among XLK, XLF, XLI, XLY, XLE and XLB, but only when each is above its 200-session
  average.  Failed eligibility becomes cash rather than being reassigned.
- Sector cap: 25%.

### C4 — downside-volatility-budgeted sectors

- Mechanism: volatility-managed equity earns the equity premium with less exposure
  during states in which the marginal drawdown risk is highest.
- Monthly base sleeve: C1 eligible sectors and inverse-downside-risk weights.
- Equity budget: `min(100%, 8% / trailing-126-session annualised downside deviation`
  of the unscaled base sleeve).  If SPY is below its 200-session average, equity
  exposure is additionally capped at 50%.  The remainder is cash.
- The 8% number here is a downside-risk budget, not a fitted return forecast.

### C5 — low-Ulcer positive-trend sectors

- Mechanism: maximum drawdown is path-dependent; selecting sectors with shallow,
  short-lived drawdowns attacks the investor's actual loss function more directly
  than variance minimization.
- Monthly eligibility: positive trailing-252-session total return and price above
  the 200-session average.
- Compute each sector's trailing-252-session Ulcer Index from its rolling wealth
  drawdowns.  Hold the four lowest-Ulcer eligible sectors equally, 25% each; absent
  slots remain cash.
- Return source: equity premium plus low-volatility/quality proxy.  It must not be
  described as fundamental quality without point-in-time accounting data.

## Evaluation gate

Primary results use 10 bp sector costs during 2013-2021.  A candidate is eligible for
fundamental review only if all conditions hold:

1. CAGR at least 7.5%; CAGR at 20 bp at least 7.0%.
2. Maximum drawdown no worse than -15%; annual volatility no more than 10%; Calmar
   ratio at least 0.60.
3. Maximum drawdown and Ulcer Index both lower than the synthetic collar, while CAGR
   is no more than one percentage point below it.
4. CAGR at least 6% in both 2013-2016 and 2017-2021.
5. Worst calendar-year return no worse than -10%; no one year contributes more than
   40% of absolute annual log return above cash.
6. At least 60% of rolling five-year windows have annualized return of 7.5% or more.
7. Six-month block-bootstrap 90% lower bound of annualized return above cash is
   positive.
8. Deflated Sharpe probability at least 95%, using five new trials plus all relevant
   earlier sector risk candidates in the stated trial count.

Regime tables must separately report SPY above/below its lagged 200-day average and
high/low VIX using a fixed 20 threshold.  Results are not accepted merely because the 2013-2021 secular bull
market made every long-equity portfolio profitable.

## Stop rule

If no candidate passes, write the report and stop for user review.  If one or more
pass, first perform an independent leakage audit and fundamental-mechanism review;
deliver the pre-holdout report; and wait for explicit permission before opening
2022+.  A synthetic collar that looks attractive does not authorize options trading.
