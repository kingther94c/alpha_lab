# ETF 10% return / 15% volatility / drawdown-control study — pre-2022 result

## Header

| field | value |
|---|---|
| **Date** | 2026-07-18 |
| **Status** | `needs_revision`; six primary candidates merit paper monitoring, no live allocation accepted |
| **Protocol** | `docs/research_decisions/2026-07-18_etf-strategy-50plus-protocol.md` |
| **Runner** | `scripts/etf_strategy_50plus_study.py` |
| **Report** | `reports/etf_strategy_50plus_pre2022.html` |
| **Data boundary** | 2004 warm-up; core 2007-01-03 to 2021-12-31; 2022-01-01 exclusive download end |
| **Batch** | 104 total rows; 72 long-history ETF rules; seven synthetic SPY option screens; 23 live packaged ETFs; two external Cboe diagnostics |

## Decision

**There is no honest basis yet to promise a 10% forward expected return with volatility below 15%.**
Seven long-history rules passed the frozen mechanical gate, but three are neighbouring SPY moving-average
variants, leaving five genuinely different pass families. A sixth, the HYG/IEF credit-canary allocation,
has the strongest overall profile and survived destructive diagnostics, but starts only in 2008 after HYG
signal warm-up and therefore does not have the common full-GFC history required by the mechanical gate.

The correct action is to paper-monitor a small ensemble of the six primary mechanisms, preserve 2022+
as unseen data for this study, and avoid choosing a single historical winner. Long-only all-weather and
permanent option protection reliably lowered volatility, but neither delivered the return target. Fixed
stock/bond portfolios could deliver 10% in this sample but did not control drawdown tightly enough.

## Primary candidate set

These are the best six mechanism-level candidates. Parameter neighbours are not separate alternatives.

| Candidate | Evidence | CAGR | 10 bp stress CAGR | Vol | MaxDD | Worst year | Main reason to keep | Main reason not to trust yet |
|---|---|---:|---:|---:|---:|---:|---|---|
| HYG/IEF credit canary, weekly | provisional / 2008-2021 | 11.51% | 11.10% | 11.11% | -12.90% | -1.88% | best return/drawdown balance; robust to extra delay and threshold perturbation | no complete GFC window; chosen from a 104-row batch; 7.35x annual traded notional |
| QQQ 12% volatility target | long rule | 11.36% | 11.21% | 12.90% | -21.72% | -15.54% | simplest target-return pass; risk is explicitly bounded without leverage | only 7.05% CAGR in 2007-2012 versus 14.33% in 2013-2021 |
| 6-month top-3 cross-asset momentum | long rule | 10.21% | 9.89% | 12.60% | -22.15% | -12.11% | distinct trend source across equity, rates, gold, commodities and real estate | lost 20.06% in 2018 Q4; 2013-2021 CAGR only 8.33% |
| low-Ulcer positive-trend sectors | long rule | 10.19% | 9.95% | 12.49% | -16.37% | -10.49% | best common-window drawdown among target-return passes | severe regime split: 4.46% in 2007-2012 versus 14.19% in 2013-2021 |
| downside/trend sector allocation | long rule | 10.00% | 9.76% | 12.90% | -21.97% | -13.65% | simple downside-risk sizing plus absolute trend; residual goes to SHY | same regime problem: 4.80% then 13.60%; related to low-Ulcer candidate |
| SPY 150-day trend to SHY | long rule | 9.16% | 8.74% | 12.02% | -20.89% | -11.43% | cheap, transparent P&L control and representative of a flat 100/150/250-day family | misses the 10% target and earned only 3.91% in 2007-2012 |

### Credit-canary post-selection audit

The first run incorrectly entered the defensive sleeve before HYG had 200 observations. That warm-up
behavior was removed; the corrected candidate starts 2008-01-28. The independent scanner and manual
alignment review found no lookahead. A destructive audit then compared all diagnostics on the same
2008-05-01 to 2021-12-31 window:

| Diagnostic | CAGR | Vol | MaxDD | Interpretation |
|---|---:|---:|---:|---|
| Frozen 200d weekly rule | 11.84% | 11.15% | -12.90% | corrected reference |
| Extra one-week execution delay | 11.43% | 11.41% | -14.43% | survives a much harsher timing convention |
| 150d / 250d sensitivities | 12.96% / 12.26% | 11.58% / 10.95% | -13.50% / -15.86% | not a narrow 200d spike; cannot replace the frozen rule |
| Monthly observation | 12.87% | 11.18% | -13.14% | weekly result is not dependent on one rebalance day |
| Credit condition only | 11.79% | 11.33% | -13.13% | almost identical; HYG/IEF is the load-bearing signal |
| SPY condition only | 11.21% | 13.21% | -23.69% | SPY trend alone is materially weaker |
| Static average exposure | 9.98% | 11.43% | -31.01% | timing, not only average beta, explains the drawdown difference |

The sensitivity rows are post-selection evidence and receive no promotion. They show that the result is
not a same-close or single-threshold artifact. They do not create an untouched validation period.

## Additional independent alternatives and honest failure labels

The user requested at least ten relatively independent alternatives. The following seven extend the six
primary mechanisms without pretending that every alternative meets the objective.

| Alternative | CAGR | Vol | MaxDD | Status / role |
|---|---:|---:|---:|---|
| Retail fixed all-weather (SPY/TLT/IEF/GLD/DBC) | 8.15% | 7.55% | -15.17% | stable low-risk anchor; return target miss |
| SPY 60% / TLT 40% | 10.18% | 10.48% | -30.50% | return/vol pass, drawdown fail; duration concentration |
| SPY drawdown ladder, medium | 8.52% | 11.91% | -27.24% | direct P&L-control baseline; sells after loss and misses both return/drawdown gates |
| iShares AOR live allocation ETF | 9.68% | 13.03% | -24.44% | actual packaged ETF near target; began November 2008 and missed most of the GFC |
| Synthetic SPY 95/105 collar | 7.36% | 9.02% | -24.04% | explicit convexity/capped-upside mechanism; 20% option-entry haircut stress CAGR only 5.20% |
| PHDG live downside-hedged ETF | 6.66% | 11.08% | -17.91% | longer live protection record than SWAN/UJAN; insufficient return |
| SWAN live Treasury + SPY-call ETF | 14.49% | 13.37% | -12.78% | useful product-mechanics watchlist; only 3.15 pre-cutoff years, so not rankable |

This yields 13 mechanism/product alternatives. Their monthly returns are not independent in the statistical
sense—most still contain equity beta—but the report's matrix shows materially different clusters. For example,
cross-asset momentum correlates about 0.38 with QQQ volatility targeting and about 0.54 with the credit canary;
all-weather inverse volatility is around 0.26 with QQQ volatility targeting. The two sector candidates are more
related and should share one risk budget rather than be counted as full diversification.

## Packaged strategies and internet recommendations

- Bridgewater's own All Weather description supports balancing assets by their structural reactions to growth
  and inflation surprises, not copying a magic retail weight vector. The unlevered ETF versions here produced
  roughly 6%-8% CAGR with excellent 6%-8% volatility: useful ballast, not a standalone 10% solution.
  Source: <https://www.bridgewater.com/research-and-insights/the-all-weather-story>.
- RPAR expresses risk parity across equities, commodities, Treasuries and TIPS, but its pre-cutoff live record
  was only about two years. It remains a product watch, not long-history evidence.
  Source: <https://www.rparetf.com/rpar>.
- AQR's century study gives a strong mechanism prior for diversified trend following, but an investable ETF's
  own fees, model and live history matter. WTMF lost 1.28% annualized in its 2011-2021 live record; DBMF and KMLM
  had only 2.7 and 1.1 pre-cutoff years. Trend ETFs are diversifying sleeves, not validated standalone champions.
  Source: <https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing>.
- Moreira and Muir motivate reducing exposure when realized volatility is high. The QQQ 12% target is the clean
  ETF implementation in this batch and does not use leverage.
  Source: <https://www.nber.org/papers/w22208>.
- Faber-style GTAA and Antonacci-style dual momentum have credible behavioral mechanisms, but the exact ETF
  implementations here did not all succeed. GTAA trend returned only 4.8%-5.2%; GEM returned 6.8% with a -38.9%
  maximum drawdown. The 6-month top-3 cross-asset rule was the tactical survivor.
  Sources: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461> and
  <https://papers.ssrn.com/sol3/Papers.cfm?abstract_id=2042750>.
- Innovator explicitly describes buffers as point-to-point, capped-upside outcomes and says there is no guarantee
  the desired protection will succeed. UJAN had only three years before the cutoff. “Protected outcome” must not
  be translated into “不会亏损”. Source: <https://www.innovatoretfs.com/education/>.
- Cboe collar/put-write indices are useful option-mechanics references, but they use SPX options. They are shown
  only as external diagnostics because the permitted implementation universe is SPY/QQQ options.
  Source: <https://cdn.cboe.com/api/global/us_indices/governance/Cboe_Collar_Indices_Methodology.pdf>.

## Stress and regime findings

- Credit canary had the cleanest corrected stress behavior in available windows: -10.28% in 2018 Q4 and -3.43%
  in the COVID crash window. Its return remained positive in the lagged bear/high-vol state, but only one major
  credit crisis is present after signal warm-up.
- Cross-asset momentum made +14.74% over the GFC window and +7.22% in the 2011 stress, then lost -20.06% in
  2018 Q4. Slow trend is a crisis diversifier only when the shock develops into a persistent move.
- QQQ volatility targeting reduced exposure but still lost -19.97% in the GFC and -18.31% in the COVID crash.
  Vol targeting is not a put.
- Sector low-Ulcer and downside/trend rules kept individual stress drawdowns mostly in the -12% to -19% range,
  but their weak 2007-2012 CAGR shows that left-tail control was not consistently rewarded.
- Fixed/all-weather approaches were the lowest-volatility cluster, but the pre-2022 sample cannot test the later
  stock/bond joint inflation shock. No realized 2022 observation was opened to fill that gap.

## Leakage and implementation audit

Final static scan: **zero blockers, one warning**. The warning is the full-sample 5% quantile used only to report
ex-post CVaR; it never enters a signal, weight, selection threshold or execution decision. Manual checks confirmed:

- no negative shift, centered window, backward fill, interpolation, or full-sample normalizer in signals;
- targets form from closes at `t`, trade at the next close, and first earn the following return;
- monthly/weekly period ends are actual observed bars; resampling is performance aggregation only;
- the HYG/IEF and SPY rolling means start only after full trailing history exists;
- regime labels use the prior session; option regime ratios use prior VIX;
- adjusted prices include distributions and fund expenses; SHY is held explicitly as the risk-off ETF;
- 5 bp one-way trading cost is headline and 10 bp is stress; option overlays accrue financing and double entry
  haircuts in stress;
- every saved market and strategy series ends no later than 2021-12-31.

Verdict: the historical results are believable as **candidate screens**, not as proof of forward expected return.
ETF/index survivorship, launch bias, one-crisis regime counts, synthetic option marks and the lack of a fresh holdout
remain material.

## Recommended paper portfolio and next gate

Do not allocate from the sample-optimal ranking. For paper monitoring, use equal **research budgets**, not optimized
capital weights, across these six mechanism sleeves: credit canary, QQQ 12% vol target, cross-asset momentum,
low-Ulcer/downside sector family (one shared sleeve), SPY trend, and fixed all-weather ballast. This is a monitoring
design, not a backtested meta-portfolio result.

Before any capital recommendation:

1. freeze one exact implementation per family and one simple ensemble rule without optimizing weights;
2. obtain an independent post-2021 evaluation only after explicit user permission—the current study did not read it;
3. paper-reconstruct monthly/weekly target weights and transaction logs for at least six months;
4. validate SPY/QQQ option candidates with timestamped executable chains before treating them as more than proxies;
5. require the ensemble to retain at least 8% realized annualized return, less than 15% volatility, and less than
   20%-25% drawdown across the next independent regime before promotion.

## Supporting artifacts

### Drawdown-duration addendum

The follow-up recovery study separated full underwater duration from the true
trough-to-prior-high recovery leg and tested 100 additional recovery-oriented rules.
No original or new rule satisfied a hard 20-business-day recovery ceiling while
retaining the return/risk objective. The full result is in
`docs/research_decisions/2026-07-18_etf-drawdown-recovery-result.md`.

- `reports/etf_strategy_50plus_pre2022.html`
- `data/results/etf_strategy_50plus_pre2022/all_strategy_metrics.csv`
- `data/results/etf_strategy_50plus_pre2022/selected_independent_candidates.csv`
- `data/results/etf_strategy_50plus_pre2022/selected_stress_windows.csv`
- `data/results/etf_strategy_50plus_pre2022/selected_regimes.csv`
- `data/results/etf_strategy_50plus_pre2022/selected_bootstrap.csv`
- `data/results/etf_strategy_50plus_pre2022/credit_canary_robustness.csv`
- `scripts/credit_canary_robustness_audit.py`
