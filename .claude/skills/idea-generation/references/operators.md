# The ten operators — quant playbook

Each operator below is the same cognitive move the creativity literature describes, translated
into a concrete research move for *this* repo, with prompts to ask, a worked example built on a
real notebook here, and the way it typically fails in quant. Read the family you need; you
rarely need all ten in one session.

Three framing reminders from the research:

- **Two axes.** Operators in *Expand* and *Iterate* widen retrieval; operators in *Break* and
  *Cross-domain* re-represent the problem; the project's filter (mechanism, leak-safety,
  testability, skeptic, first-test) is the *persistence* axis that turns novel into novel-and-
  appropriate. Don't skip the second axis.
- **Anchor to a return source.** You're not inventing edges from nothing — you move along the map of
  real return sources in [`return_sources.md`](return_sources.md): to an *adjacent* premium (same
  family, new asset), a *transplanted* one (analogy from elsewhere), or a *blended* one (two sources,
  emergent edge). After generating, run the orthogonality check — a known factor in its crowded form
  isn't an idea until you name the incremental edge.
- **Sequence, don't fixate.** A strong default order is expand → transform → break → blend →
  iterate, with a short reflection after each round. But match the moment: short on time → start
  with random stimulus or SCAMPER; complex problem space → start morphological or TRIZ; goal is
  to break a crowded assumption → start with reversal, subtraction, or perspective shift.

Pick-by-need:

| If you need to… | Reach for |
|---|---|
| Get unstuck fast / a fresh angle | Remote stimulus · SCAMPER |
| Cover a big design space systematically | Morphological recombination · TRIZ |
| Break a crowded default assumption | Reversal · Subtraction · Perspective shift |
| Synthesise a complex, high-novelty concept | Conceptual blending · Structural analogy · Geneplore |

---

## Family A — Expand search

### 1. Remote / random stimulus
**Core move.** Use semantically distant cues to jump out of the near-neighbour association zone.
The benefit comes from genuine distance, so the randomness must be real, not curated.

**Quant translation.** Most idea sessions collapse toward "momentum / value / carry on equities."
A far stimulus — a random word, a far asset class, an underused *data source*, or someone else's
unrelated idea — forces a different retrieval path. The most fertile far jump in this repo is the
**data source you don't normally signal on** (funding rates, open interest, prediction-market
probabilities, macro spreads), because a new input is a new mechanism, not a reskinned old one.

**Prompts.**
- Run `scripts/random_stimulus.py --words 5`. For each word, ask: how does it suggest a *signal*,
  a *structure*, a *timing*, or a *data source*?
- "What is the most distant asset class or data source from the one I'm in, and what is the
  analogous signal there?"
- "Which loader in `data/loaders/` have I never built a signal from? What would its native signal be?"

**Worked example.** Random stimulus *"tide"* → periodic forcing → time-of-day / day-of-week
seasonality in BTC perps (you already brush this in `90_crypto_intraday/40_regime_tod_pnl_ml.ipynb`).
Random data source *Polymarket* (you have `data/loaders/polymarket.py` but no signal on it yet) →
use market-implied event probabilities as a forward-looking macro-regime gate for the DBMF/60-40
overlay — a different mechanism than any backward-looking vol filter.

**Failure mode.** Pseudo-randomness — you "pick" a word that is secretly on-topic, and get a
near-neighbour idea. Use the script. And remote ≠ appropriate: a far idea still has to clear the
mechanism filter afterward.

### 2. Structural analogy transfer
**Core move.** Map the *relational structure* of a base domain onto the target — not surface
similarity. The win is importing a mechanism whose relational shape matches.

**Quant translation.** Take an effect that works structurally in one place and ask where the same
relation holds: another asset class, another timeframe, another literature. "Relative strength
across names, hold winners" is a relation that can move between universes and horizons.

**Prompts.**
- State the target as a functional relation, then find a base domain with the same relation: a
  different asset class, a different timeframe, a paper from an adjacent field.
- "Take an equity-factor mechanism and find its structural analog in crypto / FX / rates."
- "What works at the daily horizon whose relational form should also exist intraday (or weekly)?"

**Worked example.** Cross-sectional equity momentum (relative strength across names) → already
transferred by this repo to country ETFs (`05_cross_sectional_country_etf_momentum`) and crypto
(`09_crypto_cross_sectional_momentum`). The *next* analogy keeps the relation, changes the
timeframe: cross-sectional relative-strength among crypto perps at the **5-minute** horizon — an
intraday XS-momentum book, structurally identical, new microstructure. Or: the managed-futures
*portfolio* trend mechanism (DBMF, `06/07`) transferred to crypto as a multi-asset trend book
rather than the single-asset filter you have in `08_crypto_trend_filter`.

**Failure mode.** Surface analogy ("this *feels* like momentum") with no real relational match; or
transferring a mechanism that doesn't survive the new market's costs/microstructure. Make the
element↔relation mapping explicit before you trust it.

---

## Family B — Transform structure

### 3. Morphological recombination
**Core move.** Decompose the problem into parameters, list values per parameter, then search the
recombination grid — especially the cells you'd normally skip.

**Quant translation.** A strategy ≈ {Universe} × {Data source} × {Signal} × {Horizon} ×
{Conditioner/regime} × {Portfolio construction} × {Timing/rebalance}. Build that box (the menu is
in `domain_seeds.md`), delete genuinely incompatible cells, then deliberately pick a few
unusual-but-feasible cells and ask whether each has a story.

**Prompts.**
- Run `scripts/random_stimulus.py --matrix` to get a random cell across all columns; ask "is this
  combination tradable, and what mechanism would justify it?"
- "Hold the signal fixed and sweep the *conditioner* column" — the conditioner column is the most
  under-explored and where regime edge lives.
- "Hold the universe fixed and sweep the *portfolio* column" (equal → inverse-vol → MV → risk-parity).

**Worked example.** Start from sector-ETF XS momentum (universe = US sectors, signal = 12-1, port =
top/bottom-N, timing = month-end). Recombine: conditioner → "only when cross-sectional dispersion is
high"; portfolio → inverse-vol instead of equal weight; timing → rebalance on a volatility trigger
instead of the calendar. Three new cards, each a different cell of the same box, each with a
testable story.

**Failure mode.** Combinatorial explosion and mechanical enumeration — the classic route into data
mining. Keep only cells with a mechanism; the grid is a search aid, not a license to fit.

### 4. SCAMPER (attribute operations)
**Core move.** Apply fixed transformation verbs to an existing design: Substitute, Combine, Adapt,
Modify, Put-to-other-use, Eliminate, Reverse. Turns a blank page into seven concrete axes.

**Quant gloss of each verb.**
- **Substitute** the signal input (price → realised vol → volume → funding → open interest), the
  universe, or the benchmark.
- **Combine** two signals (trend × carry) or two strategies (momentum + risk-parity overlay).
- **Adapt** a daily strategy to intraday (your crypto 5m work) or an intraday idea up to daily.
- **Modify** (magnify/minify): horizon (12-1 → 6-1 → 1-day), top-N, vol target, leverage.
- **Put to other use**: take a signal built for *selection* and use it as a *regime filter* (e.g.
  XS-momentum dispersion as a risk-on/off gauge).
- **Eliminate** a component — the short leg, the rebalance, a costly feature (→ Subtraction, op 8).
- **Reverse** the trade — mean-reversion instead of momentum (→ Reversal, op 7).

**Prompts.** "Take [existing strategy]; for each verb produce ≥2 variants; then **combine two
variants from different verbs** into one idea." The cross-verb combination is what escapes the
single-tweak trap.

**Worked example.** DBMF overlay (`06/07`): *Substitute* DBMF → a KMLM/CTA basket; *Combine* DBMF +
short-vol carry; *Adapt* → size DBMF by trend strength (you did a version in `07_dbmf_sizing`);
*Put-to-other-use* → use DBMF's own drawdown as a risk-on/off regime signal for the equity sleeve.

**Failure mode.** Stopping at one clever small modification and calling it a new idea — that's a
robustness sweep. Force a cross-verb combination or hand off to a Break-defaults operator.

---

## Family C — Cross-domain blend

### 5. Conceptual blending (dual-domain)
**Core move.** Project two *whole* frameworks into a blended space where a new property emerges.
Stronger than analogy: analogy maps, blending grows structure that's in neither parent.

**Quant translation.** Fuse two strategy frameworks so a rule appears that belongs to neither. Let
one parent supply the organizing frame and the other supply a conflicting new attribute; then run a
scenario to force the emergent property out.

**Prompts.**
- "Blend [framework A] × [framework B]. What is the emergent rule that's in neither parent? How does
  it trade, and what's its *new* failure mode?"
- "Which framework supplies the frame, which supplies the conflicting attribute?"

**Worked example.** *Risk parity × prediction markets*: all-weather risk parity (`11`) supplies the
frame (balance risk across sleeves); Polymarket event probabilities supply a conflicting attribute
(forward-looking, discrete event risk). Emergent rule: a risk-parity book that **tilts sleeve risk by
market-implied event probability** — neither pure RP (backward-looking vol) nor pure event-trading.
New failure mode that's in neither parent: prediction-market illiquidity / manipulation. Or *managed
futures × intraday seasonality*: a trend frame that concentrates risk-taking in the hours that have
historically trended.

**Failure mode.** High cognitive load → "very cool, untradeable." The scenario-run plus the standard
filter (mechanism + data + cost) is what keeps a blend honest.

### 6. Perspective shift / role-play
**Core move.** Generate from a *different objective function* — user, extreme user, adversary,
maintainer, regulator. Not vague empathy; a different goal changes which ideas appear.

**Quant translation.** Markets are near zero-sum, so durable alpha usually means trading against a
participant whose objective is **not** profit. Role-play that counterparty and model their constraint.

**Prompts.**
- "Who is *forced* to trade against me, and why (not for alpha)?" — index rebalancers, ETF
  creation/redemption, margin-called leverage, vol-target / risk-parity funds de-risking on a vol
  spike, CTA stop-outs, tax-loss sellers, options dealers hedging gamma, central banks, retail FOMO.
- "What does that forced/insensitive participant do *predictably*, and how do I provide that
  liquidity for a premium?"

**Worked example.** Vol-target and risk-parity funds mechanically de-lever when realised vol jumps —
role-play them → a signal that anticipates mechanical de-leveraging flow around vol spikes (connects
your vol-target work in `12_crypto_equity_vol_target` and risk-parity in `11`). Options dealers'
gamma hedging → intraday mean-reversion vs. trend in BTC around large open-interest strikes (uses
Binance OI). Index/ETF rebalancers → predictable flow in sector/country ETFs near rebalance dates.

**Failure mode.** Projecting *yourself* onto "the counterparty" instead of modelling their real
mechanical constraint. Anchor to a documented behaviour, then check it's exploitable net of cost.

---

## Family D — Break defaults

### 7. Reversal / Janusian
**Core move.** Hold a thing and its opposite together, or invert the question, to break the default
representation.

**Quant translation.** Surface the assumptions baked into a strategy and negate each; or reverse the
trade; or hold two opposites in one book (trend *and* reversal at different horizons).

**Prompts.**
- "List 5 defaults inside [strategy] — e.g. 'momentum = buy winners', 'rebalance monthly',
  'long-biased', 'use close prices', 'one horizon'. Negate each."
- "Reverse-brainstorm: how would I design this to reliably *lose* money? Then invert each loss-driver
  into a potential edge."
- "Find a paradox pairing: fast signal + slow execution; aggressive entry + defensive exit."

**Worked example.** Default in XS momentum = "buy recent winners." Reverse → short-term reversal (buy
recent losers) at a *shorter* horizon; the Janusian combine holds both — long-horizon trend and
short-horizon reversal in one book, a structure this repo hasn't tested. Reverse-brainstorm the
crypto trend filter: "how to maximise whipsaw losses?" → trade every cross of a noisy MA in chop →
invert → a chop/trend regime gate (links to the regime work in `40_regime_tod_pnl_ml`).

**Failure mode.** Stopping at a rhetorical contradiction ("aggressive but safe") with no mechanism
for partial co-satisfaction. Demand the mechanism that lets both sides hold at once.

### 8. Subtraction / de-fixation
**Core move.** Remove a component, step, or feature and reassign its function. Directly counters the
human additive bias and functional fixedness.

**Quant translation.** Strip a "load-bearing" piece and see if the strategy still stands. Simpler is
often more robust, cheaper, and less overfit — and subtraction is the cleanest *attribution* test
(if it survives the cut, that piece wasn't the edge).

**Prompts.**
- "Draw [strategy] as components → function. Delete the one that looks indispensable (the
  covariance estimate? the short leg? the rebalance? the vol forecast? the universe filter?). Who
  takes over its function?"
- "Remove one degree of freedom (a fitted parameter) entirely. Does performance survive? If yes,
  you just removed overfit."

**Worked example.** Active-MV sector strategy (`03_us_sector_etf_rolling_active_mv`): subtract the
covariance estimate → does equal-risk or plain top-N match it net of cost? If so, you removed
estimation error, not edge. Crypto vol-target (`12`): subtract the vol forecast → fixed leverage;
subtract the trend filter → does vol-targeting alone carry it? Dual-momentum GEM (`10`): subtract the
absolute-momentum (out-of-market) switch → is the relative or the absolute leg doing the work?

**Failure mode.** Subtracting the actual edge → a castrated version. Subtract to *test attribution*,
not blindly, and always pair the cut with the cheapest-first-test.

### 9. TRIZ contradiction
**Core move.** Abstract the problem to "improving A worsens B", then apply cross-domain inventive
principles to seek a resolution that *isn't* a 50/50 compromise.

**Quant contradictions (the real ones).** Responsive signal ↔ turnover/cost. Faster risk cut ↔ more
whipsaw. More diversification ↔ diluted edge. Bigger size ↔ worse capacity/impact. More leverage ↔
ruin risk.

**Inventive-principle analogs for quant.** *Segmentation* (split the book by regime or horizon);
*Asymmetry* (different rule up vs down); *Prior action / pre-positioning* (act before a known flow);
*Cushion beforehand* (carry a cheap tail hedge); *Dynamics* (make a fixed parameter adaptive);
*Taking out* (separate the hedge from the alpha sleeve); *Periodic action* (trade on triggers, not on
a clock).

**Prompts.** "Write the core tension as 'improving ___ worsens ___'. Apply **three** inventive
principles to dissolve it without splitting the difference."

**Worked example.** Crypto trend filter tension = "react fast to trend (less lag) ↔ avoid whipsaw in
chop." *Segmentation* → separate fast and slow signals, blended by a chop/trend gauge. *Asymmetry* →
enter slow, exit fast. *Periodic action* → act only on volatility-confirmed breaks, not every bar.
Three cards from one contradiction.

**Failure mode.** Staying at the lookup-table level and never instantiating a principle into an
actual signal — or forcing TRIZ on what is really just a parameter sweep.

---

## Family E — Iterate

### 10. Generate–explore (Geneplore)
**Core move.** Generate a rough, half-formed, deliberately un-named structure first; *then* interpret
/ rename / reframe it; reflect on what to keep; regenerate. Short generate↔interpret loops beat one
long diverge-then-converge — inserting a brief reflective evaluation between rounds measurably raises
the originality of the next round.

**Quant translation.** Build an uninterpreted signal by combining primitives at random, *then* ask
"what real phenomenon could this proxy — flow, risk, positioning, sentiment?" Attach the mechanism
*after* generation, keep the interpretable part, and seed the next round with the survivor.

**Prompts.**
- Run `scripts/random_stimulus.py --triple` for three primitives. Combine them into one signal.
  **Don't justify it yet.** Then: what could it proxy? Put it in three roles (selection / regime
  filter / risk overlay). Keep what's interpretable; regenerate.
- After each round, 60-second reflection: which ideas have a mechanism, which are noise — then
  generate again seeded by the survivors.

**Worked example.** Random triple *{funding rate, day-of-week, inverse-vol}* → un-named structure
"inverse-vol-weighted funding carry, conditioned on day-of-week." Interpret: funding ≈ crowding /
positioning proxy; day-of-week ≈ settlement/flow seasonality. Reframe as a **carry strategy with a
positioning-crowding risk gate** — fresh, and built entirely on Binance data you already load.

**Failure mode.** Never interpreting → "fascinating noise." The explore/reflect step is mandatory;
that's where mechanism gets attached and where novel becomes appropriate.
