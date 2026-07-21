# C2 defensive sector + SPY option overlay — frozen protocol

## Status and information set

- Registered after the 2022 pure-sector holdout was observed.
- Development window: 2013-01-01 through 2021-12-31.
- Calendar year 2022 is a **known stress sample**, not untouched out-of-sample data.
- No observation dated 2023-01-01 or later may be loaded. Calendar year 2023+ remains sealed.
- C2 asymmetric-sector-ferry signals, sector weights, costs, and rebalance rules are unchanged.

## Functional objective

Preserve C2's sector/cash return source while adding explicit convex protection so that long-run net
return remains near 8% with materially lower volatility and drawdown. The overlay is risk
transformation, not a claim that buying options creates alpha.

## Return sources and design operators

- **Base return source:** equity risk premium plus defensive sector rotation / asymmetric trend exit.
- **Overlay source:** transfer of crash risk through purchased SPY puts; any call premium is payment
  for surrendering part of the right tail, not free carry.
- **Operators:** TRIZ “cushion beforehand” and “taking out”; SCAMPER combine/substitute/eliminate;
  conceptual blend of C2's slow defensive ferry with an option shock absorber.
- Random stimuli were `flywheel`, `fulcrum`, `echo`, `calving`, and `glacier`: premium recycling,
  exposure-matched hedge sizing, a passive-collar control, and protection against abrupt breaks.

## Synthetic option assumptions

- Options reference raw SPY spot while the C2 base uses adjusted ETF total returns.
- Quarterly puts roll on the third-Friday trading date; monthly calls roll on the same convention.
- ATM volatility proxy: `max(VIX / 100, trailing 21-day realised SPY vol + 2 vol points)`.
- 95% long-put IV: ATM proxy + 4 vol points.
- 85% short-put IV: ATM proxy + 8 vol points.
- 110% call IV: ATM proxy - 2 vol points, floored at 8%.
- Long options are entered 10% above model mid; short options receive 10% below model mid.
- Stress case doubles entry haircuts to 20%.
- Overlay cash earns or pays the previously observable 3-month T-bill rate.
- Hedge ratios use the prior close's C2 equity exposure or VIX state; no same-close state decision.
- European Black–Scholes marks are a scenario proxy, not historical executable option-chain data.

## Frozen five-candidate batch

### O1 Exposure-matched quarterly put spread — `O1_matched_95_85_put_spread`

- **Operator(s):** TRIZ cushion + fulcrum stimulus.
- **Question:** Does a quarterly 95/85 SPY put spread sized to lagged C2 equity exposure reduce C2
  drawdown below 15% while preserving at least 7.5% CAGR?
- **Mechanism:** Crash-risk transfer; the short 85 put lowers insurance carry while retaining
  protection across ordinary bear-market declines.
- **Skeptic:** Protection stops below the 85 strike and may fail in a discontinuous crash.
- **Cheapest test:** compare development and 2022 drawdown/CAGR with unhedged C2.
- **Scores:** Novelty 3/5 · Mechanism 4/5 · Testability 4/5 · Tradability 4/5.
- **Maturity:** Ready.

### O2 Calm-VIX exposure-matched put spread — `O2_calm_vix_put_spread`

- **Operator(s):** SCAMPER modify + glacier stimulus.
- **Question:** Does opening O1 only when prior-close VIX is below 20 avoid expensive insurance carry
  without giving up most crash protection?
- **Mechanism:** Buy convexity before stress, not after implied volatility has repriced.
- **Skeptic:** A high-VIX market can still crash; the gate may remove protection exactly when needed.
- **Cheapest test:** compare premium drag and worst drawdown with O1.
- **Scores:** Novelty 3/5 · Mechanism 3/5 · Testability 5/5 · Tradability 4/5.
- **Maturity:** Developing.

### O3 Premium-flywheel partial overwrite — `O3_put_spread_call_flywheel`

- **Operator(s):** Conceptual blend + flywheel stimulus.
- **Question:** Can O1 plus a monthly 110 call capped at 25% NAV finance enough put-spread carry to
  reach the return target without recreating collar-like downside?
- **Mechanism:** Sell a limited amount of right-tail equity beta to finance left-tail protection.
- **Skeptic:** C2 does not hold SPY itself, so the call is a basis-risk beta overwrite rather than a
  legally covered call.
- **Cheapest test:** attribute option premium/P&L and inspect strong-rebound months.
- **Scores:** Novelty 4/5 · Mechanism 3/5 · Testability 4/5 · Tradability 3/5.
- **Maturity:** Developing.

### O4 Exposure-matched full collar — `O4_matched_95_110_collar`

- **Operator(s):** Echo stimulus + structural analogy to Cboe CLL.
- **Question:** Does a 95 put / 110 call overlay sized to lagged C2 equity exposure outperform the
  standalone SPY collar on drawdown-adjusted return?
- **Mechanism:** The call finances uncapped put protection while C2 supplies sector selection.
- **Skeptic:** Right-tail truncation and SPY/sector basis can overwhelm the funding benefit.
- **Cheapest test:** compare CAGR, upside capture, and 2022 result with C2 and synthetic CLL.
- **Scores:** Novelty 2/5 · Mechanism 4/5 · Testability 5/5 · Tradability 3/5.
- **Maturity:** Ready as a control.

### O5 Static half-notional long put — `O5_half_notional_long_put`

- **Operator(s):** Subtraction: remove short-put and call financing legs.
- **Question:** Is simple 50%-NAV quarterly 95-put protection more robust than the structured
  alternatives after its explicit premium drag?
- **Mechanism:** Pure convex insurance with no short-tail or short-upside leg.
- **Skeptic:** Persistent implied-over-realised premium can consume the full C2 edge.
- **Cheapest test:** compare CAGR loss per percentage-point of max-drawdown improvement.
- **Scores:** Novelty 2/5 · Mechanism 5/5 · Testability 5/5 · Tradability 4/5.
- **Maturity:** Ready as an attribution control.

## Development gates

A candidate advances only if all gates pass on 2013-2021 after modeled option spreads and C2 costs:

1. CAGR at least 7.5%; stress-haircut CAGR at least 7.0%.
2. Annualized volatility at most 10%.
3. Maximum drawdown no worse than -15% and Calmar at least 0.60.
4. Maximum drawdown and Ulcer Index both improve on unhedged C2.
5. CAGR is no more than one percentage point below the synthetic SPY collar.
6. Both 2013-2016 and 2017-2021 CAGR are at least 6%.
7. Worst calendar year at least -10%; rolling five-year target attainment at least 60%.
8. Circular-block-bootstrap 90% lower excess-cash bound is positive.
9. Deflated Sharpe probability is at least 95%, using 16 lifetime related trials.

## Reporting and decision discipline

- Report all five candidates, C2, synthetic SPY collar, and SPY.
- Show option P&L, premium/spread drag, hedge ratios, subperiods, regime results, and the known 2022
  stress sample separately.
- No sixth variant and no parameter change after development results are viewed.
- Do not open 2023+ without a separate user authorization and release record.
