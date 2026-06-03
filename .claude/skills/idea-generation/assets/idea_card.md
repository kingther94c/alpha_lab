# Idea card template

One card per idea. The fields mirror the research-question + hypothesis cells of
`notebooks/_templates/strategy_research_template.md`, so a card drops straight into a study.
Emit cards ranked best-first. Keep each terse — a card is a *what to test next*, not a write-up.

## Template

```markdown
### <Idea title> — `idea-slug`

- **Operator(s):** <which creative move(s) produced it>
- **Research question:** <one falsifiable, scoped sentence — name universe, signal, window, cost, benchmark>
- **Economic mechanism:** <family: trend / carry / reversal / value / seasonality / liquidity /
  behavioural / microstructure / regime / event> — <1–2 sentences on why it might exist *and persist*>
- **Return source & orthogonality:** <which premium / anomaly / flow it harvests, per return_sources.md> — <the *incremental* edge over its crowded default, or "novel source">
- **Signal sketch:** <what you'd compute, leak-safe (data ≤ t), at what horizon>
- **Universe & data:** <assets> via <loader(s)>. <flag if data isn't available yet>
- **Portfolio / trade:** <construction + rebalance/timing>
- **Skeptic's view:** <the single strongest reason it won't work — crowding / cost / capacity / regime / borrow>
- **Cheapest first test:** <the one experiment that would most quickly kill or confirm it>
- **Crowding · capacity · regime:** <how arbitraged the source is · how much size it holds · which regimes it needs>
- **Scores:** Novelty _/5 · Mechanism _/5 · Testability _/5 · Tradability _/5
- **Maturity:** Raw | Developing | Ready  (→ park | needs_revision | accept-candidate)
- **Next step:** new notebook from `strategy_research_template.md` at `notebooks/<folder>/NN_<slug>.ipynb`
```

## Worked example

```markdown
### Funding-carry with a positioning-crowding gate (BTC/ETH perps) — `funding-carry-crowding-gate`

- **Operator(s):** Geneplore (random triple {funding, day-of-week, inverse-vol}) → reframed; SCAMPER "put-to-other-use" (open interest as a risk gate, not a signal)
- **Research question:** Does collecting perp funding carry on BTC/ETH, scaled inverse to realised vol
  and gated off when open-interest crowding is extreme, beat naive always-on funding carry on net
  Sharpe (5m data, 2021–2025, funding + 2 bps slippage)?
- **Economic mechanism:** Carry + liquidity provision. Funding is paid by the crowded side; you earn it
  for providing liquidity. The crowding gate exists because the carry inverts violently during
  positioning unwinds (the carry-crash mechanism), so conditioning on open-interest extremes should cut
  the left tail without giving up much premium.
- **Return source & orthogonality:** Carry (perp funding) + liquidity provision; the gate reads positioning/microstructure (open interest). Incremental edge over naive funding carry = the OI-crowding tail gate, not the carry itself.
- **Signal sketch:** sign(−funding) sized by 1/realised_vol; multiply by 0 when OI z-score > k. All
  inputs computed on data through bar t only; trade on t+1.
- **Universe & data:** BTC, ETH perps via `binance_vision` (klines + funding + open interest). Data on hand.
- **Portfolio / trade:** 2-name inverse-vol book, vol-targeted; rebalance each funding interval.
- **Skeptic's view:** funding carry is well known and crowded; net of funding-cost-to-hold and slippage
  the premium may be thin, and the OI gate adds a fitted threshold that could be overfit.
- **Cheapest first test:** plot net carry conditioned on OI-extreme vs not, before any portfolio — if the
  left tail doesn't shrink in the gated bucket, the whole idea dies in 20 minutes.
- **Crowding · capacity · regime:** funding carry is crowded (high); capacity moderate (2 liquid perps); needs a positioning-unwind / non-trending regime to add value.
- **Scores:** Novelty 3/5 · Mechanism 4/5 · Testability 5/5 · Tradability 3/5
- **Maturity:** Developing (→ needs_revision: the gate threshold is the open question)
- **Next step:** new notebook at `notebooks/90_crypto_intraday/50_funding_carry_crowding_gate.ipynb`
```
