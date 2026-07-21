# ETF 10% return / 15% volatility / drawdown-control study protocol

Date: 2026-07-18  
Status: frozen before the new 50+ strategy batch is evaluated  
Hard data cutoff: 2022-01-01 exclusive

## Objective

Find a small set of relatively independent, ETF-implementable portfolios that can plausibly target
approximately 10% long-run return with annual volatility below 15%, while treating drawdown control
as a first-class objective. Tradable instruments are ETFs. Option overlays may reference SPY or QQQ
only; no other single-name or index option strategy is eligible for implementation.

This is a candidate-generation study, not a promise that the historical CAGR is the future expected
return. The output must distinguish:

1. long-history ETF implementations;
2. rules built from actual ETF total-return prices;
3. short live product records; and
4. synthetic option or index proxies that are useful only for mechanism screening.

## Information boundary and comparability

- Every market-data download must use `end="2022-01-01"` and assert that no returned observation is
  later than 2021-12-31.
- Core comparison window: 2007-01-03 through 2021-12-31. The 2004-2006 observations are signal
  warm-up only. This gives the common ETF universe enough history and preserves the Global Financial
  Crisis as an evaluated stress, not a fitting-only episode.
- Core development split: 2007-2012 and 2013-2021. Neither is an untouched holdout because strategy
  families are literature-informed and this repository has prior related work. Results must be
  described as historical validation, not fresh out-of-sample evidence.
- Later-launch ETFs are evaluated from inception through 2021-12-31 and cannot outrank a core strategy
  merely because they missed earlier crises.
- No 2022 or later price, return, volatility, yield, product-performance, or stress observation may
  enter a calculation. Current product pages may be read only to understand mechanics and risks.

## Tradable universe

Core sleeves use liquid ETFs available by the common window: SPY, QQQ, IWM, EFA, EEM, VEU, VNQ,
IEF, TLT, SHY, TIP, LQD, HYG, GLD, DBC, AGG, and the nine legacy Select Sector SPDRs (XLK, XLF,
XLV, XLI, XLY, XLP, XLE, XLU, XLB). SHY is the risk-off/cash-like ETF; no unremunerated cash is
credited once a portfolio is invested.

Product screens may include AOA, AOR, SPLV, USMV, MTUM, QUAL, PHDG, TAIL, SWAN, NTSX, RPAR, DBMF,
KMLM, WTMF, UJAN, QVAL, QMOM, VMOT, OMFL, QAI, VIG, VYM, and RSP when pre-cutoff prices exist.

## Return-source map and eight idea cards

### 1. Reservoir equity control — `equity-reservoir-control`

- **Operators:** dam/reservoir analogy; TRIZ segmentation and asymmetry; SCAMPER modify.
- **Question:** Can SPY/QQQ exposure be released or stored in SHY using trailing trend, volatility,
  or portfolio drawdown without sacrificing the 10% return target?
- **Mechanism:** equity premium plus volatility management. Risk is cut when variance or persistent
  impairment rises; SHY is the remunerated reservoir.
- **Incremental edge:** explicit slow re-entry / fast exit and bounded exposure, not raw market timing.
- **Skeptic:** whipsaw and selling after losses can permanently lower compound return.
- **First test:** compare trend-only, vol-only, drawdown-only, and combined rules on identical dates.

### 2. Keystone balanced core — `keystone-balanced-core`

- **Operators:** keystone stimulus; subtraction; morphological recombination.
- **Question:** Which simple stock/bond/gold allocation is the smallest robust structure that remains
  near the return target while materially reducing SPY drawdown?
- **Mechanism:** equity, term, and inflation premia with periodic rebalancing.
- **Incremental edge:** simplicity and low turnover; this is a robustness baseline, not novel alpha.
- **Skeptic:** stock/bond correlation can turn positive and unlevered diversification can miss 10%.
- **First test:** fixed 80/20, 70/30, 60/40, three-asset, global, and sector allocations.

### 3. Four-weather risk web — `four-weather-risk-web`

- **Operators:** spiderweb stimulus; Bridgewater structural analogy; subtraction of return forecasts.
- **Question:** Does fixed or trailing inverse-vol allocation across equities, Treasuries, gold, and
  commodities improve left-tail behavior across growth/inflation regimes?
- **Mechanism:** balance multiple macro risk premia rather than capital dollars.
- **Incremental edge:** ETF-only, unlevered implementation with explicit trend-gated variants.
- **Skeptic:** inverse volatility is not true risk parity, and duration can dominate hidden risk.
- **First test:** fixed, equal, inverse-vol, capped inverse-vol, and sleeve-trend variants.

### 4. Watershed tactical allocation — `watershed-tactical-allocation`

- **Operators:** watershed stimulus; Faber/dual-momentum analogy; morphological recombination.
- **Question:** Can monthly absolute and relative momentum redirect capital among global equity,
  rates, gold, commodities, and real estate with lower crisis drawdown?
- **Mechanism:** behavioral under-reaction and persistent trends across asset classes.
- **Incremental edge:** multiple independent sleeves and an SHY hurdle; no same-close return capture.
- **Skeptic:** momentum is crowded and reversals create sharp losses and timing luck.
- **First test:** GTAA trend, GEM, top-N momentum, multi-horizon momentum, and defensive momentum.

### 5. Checkpoint sector ferry — `checkpoint-sector-ferry`

- **Operators:** checkpoint stimulus; perspective shift toward forced de-riskers; SCAMPER combine.
- **Question:** Do monthly sector trend, downside-risk, or Ulcer screens preserve equity upside while
  avoiding prolonged sector impairment?
- **Mechanism:** sector information diffusion, low-volatility anomaly, and absolute-trend removal.
- **Incremental edge:** long-only legacy universe, explicit SHY residual, and downside-aware sizing.
- **Skeptic:** sector signals are correlated equity beta and prior repo studies rejected plain momentum.
- **First test:** equal sector, top-N momentum, inverse-vol momentum, downside-trend, barbell, and
  low-Ulcer candidates.

### 6. Credit canary allocation — `credit-canary-allocation`

- **Operators:** structural analogy; put-to-other-use; reversal.
- **Question:** Can HYG-versus-IEF trend serve as a prior-known risk gate for equity allocation?
- **Mechanism:** credit spreads and risk appetite often deteriorate before or alongside equity stress.
- **Incremental edge:** credit is a conditioner, not a return forecast or a contemporaneous label.
- **Skeptic:** ETF credit prices can lag or give false alarms, and the sample has few recessions.
- **First test:** fixed risk-on/risk-off weights with one frozen 200-session gate.

### 7. SPY shock absorber — `spy-shock-absorber`

- **Operators:** TRIZ cushion beforehand / taking out; collar and buffer-product analogy.
- **Question:** Can quarterly SPY puts, put spreads, or collars reduce drawdown without pushing CAGR
  too far below target after conservative premium haircuts and financing?
- **Mechanism:** transfer crash risk through option premium; calls finance insurance by surrendering
  part of the right tail.
- **Incremental edge:** compare pure puts, spreads, and collars at common assumptions.
- **Skeptic:** permanent insurance carry and proxy option marks can dominate the result.
- **First test:** frozen VIX/Black-Scholes proxy with doubled entry-spread stress; never call it an
  executable option-chain backtest.

### 8. Packaged-strategy reality check — `packaged-strategy-reality-check`

- **Operators:** subtraction; product-to-mechanism reverse engineering; crowding check.
- **Question:** Did pre-2022 live ETFs implementing low vol, momentum, managed futures, tail risk,
  defined outcomes, risk parity, or capital efficiency add evidence beyond simpler sleeves?
- **Mechanism:** varies by product; the test is whether implementation survives fees and live tracking.
- **Incremental edge:** actual fund NAV/price history, not issuer backtests.
- **Skeptic:** most products have very short histories and strong launch/selection bias.
- **First test:** inception-to-2021 screen with history length shown prominently; use as corroboration,
  not as the main ranking table.

## Frozen search budget

The batch may contain parameter variants, but every row must belong to a named economic family.
Planned minimum counts:

| family | minimum tests |
|---|---:|
| Fixed stock/bond/gold/global/sector allocations | 12 |
| SPY/QQQ trend, volatility and drawdown control | 15 |
| All-weather / inverse-risk / sleeve-trend allocation | 8 |
| Cross-asset trend, GEM and relative momentum | 12 |
| Sector trend / downside / Ulcer allocation | 10 |
| Credit-conditioned allocation | 1 |
| Synthetic SPY option overlays | 6 |
| Live packaged-strategy ETF screens | 15 |

The full batch therefore exceeds 79 rows. Parameter neighbours count as trials for multiple-testing
penalties but do not count as independent final alternatives.

## Execution, cost, and leakage rules

- Monthly or weekly decisions only. A close-formed target trades at the following close and first
  earns the next close-to-close return.
- Rolling means, volatility, downside deviation, Ulcer statistics, momentum, drawdown, and regime
  labels use data available at or before the decision timestamp. Performance regime labels use the
  prior session.
- ETF strategies use 5 bp one-way on actual non-cash traded notional; 10 bp is the stress case.
  Adjusted ETF prices already include fund expense ratios.
- SHY is held explicitly when a rule is risk-off. Any leveraged or synthetic overlay must accrue
  financing on positive or negative overlay cash.
- SPY options use raw SPY as spot, adjusted SPY for the base total return, VIX as an ATM proxy,
  trailing realized-vol floors, skew buffers, and 10% option-entry haircuts; 20% haircuts are the
  stress case. Historical NBBO, early exercise, discrete strikes/dividends, tax, and crisis spreads
  remain unresolved model risk.
- Cboe SPX option indices, if displayed, are external mechanism proxies only because the tradable
  option constraint is SPY/QQQ.

## Evaluation and selection

Headline metrics: CAGR, annual volatility, excess-SHY Sharpe, maximum drawdown, Ulcer Index, Calmar,
CVaR, worst calendar year, longest recovery, turnover, 10 bp stress CAGR, 2007-2012 and 2013-2021
CAGR, rolling three/five-year return distributions, and target-attainment frequency.

Historical stresses: Global Financial Crisis, 2011 euro/US downgrade stress, 2018 Q4, and the 2020
COVID crash/recovery. Regimes are prior-day SPY above/below 200DMA crossed with realized volatility
above/below a trailing three-year median. A hypothetical inflation/rate shock may be shown, but it
must not use realized 2022 data.

A long-history row is target-attaining only if:

1. core CAGR is at least 9% and 10 bp stress CAGR at least 8.5%;
2. annual volatility is no more than 15%;
3. maximum drawdown is no worse than -25%, with Ulcer and recovery reported;
4. 2013-2021 CAGR is at least 8%;
5. it does not rely on a single calendar year for most of its excess return; and
6. its return source and implementation remain credible after the leakage audit.

Ten alternatives will be chosen by economic/return-source cluster first and score second. A cluster
may contribute only one primary representative unless a second version has materially different
left-tail behavior. Monthly correlations and regime/stress fingerprints must be shown. Deflated
Sharpe uses the full rule-trial count; short live products and option proxies cannot be promoted to
long-history evidence by statistical adjustment.

## Stop rule

If fewer than ten strategies genuinely pass the target gate, do not lower the gate or search 2022+.
Deliver a tiered list containing pass, near-target, and research-only alternatives, explain the gap,
and preserve the cutoff. No live orders or execution adapter are authorized by this study.
