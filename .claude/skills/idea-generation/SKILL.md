---
name: idea-generation
description: >-
  Generate novel, testable investment-research ideas — trading strategies, factors,
  signals, overlays — using a structured pipeline of ten guided-creativity operators
  (morphological recombination, SCAMPER, analogy, random stimulus, conceptual blending,
  reversal, subtraction, perspective-shift, TRIZ, generate-explore iteration) grounded in
  alpha_lab's universes, data loaders, return-source taxonomy, and research discipline. Reach for this whenever the
  user wants new strategy / factor / signal ideas, wants to brainstorm or expand a research
  direction, feels stuck or "out of ideas", wants fresh variations on an existing study, or
  asks "what else could I test / research / trade here". Produces ranked idea cards in the
  project's research-question + economic-mechanism format, each with a skeptic's view and a
  cheapest-first-test, ready to drop into the strategy-research notebook template. Not for
  backtesting an already-chosen idea — that is the notebook workflow this hands off to.
---

# Idea generation (alpha_lab)

A guided way to manufacture research ideas instead of waiting for them. It adapts ten
well-studied creativity operators to *this* repo — its universes, its data loaders, its
no-lookahead discipline — and routes every surviving idea into the project's existing
research workflow (research question → economic mechanism → leak-safe signal → backtest →
robustness → verdict).

## Mental model: novelty bounded by tradeability

Creativity research converges on two cognitive moves: **expand the search** (escape
near-neighbour associations, change how the problem is represented) and **persist /
evaluate** (push a "novel but vague" idea into "novel *and* appropriate"). Good ideation is
not "think harder" — it is operating these two moves in sequence with short feedback loops.

For quant research the second move is unusually well-defined, and the repo already encodes
it. An idea here is "appropriate" only if it has an **economic mechanism**, is **leak-safe /
tradable**, is **testable with data we can actually get**, and **survives a skeptic**. So
this skill is divergence on a leash: generate widely with the operators, then filter hard
through the project's own research standards. Pure novelty with no mechanism is just data
mining wearing a costume — name it and drop it.

Concretely, every idea must trace to a **real return source** — a risk premium, an anomaly, or a flow
you're paid to absorb — that isn't already arbitraged away in its crowded form. The taxonomy of those
sources, and where this repo can test each, lives in
[`references/return_sources.md`](references/return_sources.md): you *generate* by moving along that map,
and you *filter* by checking the idea isn't a known premium in disguise.

## The operator map

Ten operators in five families. You rarely run all ten — diagnose what the moment needs
(expand / transform / break / blend / iterate) and pull the matching tools. Full playbook
with quant translations and worked examples grounded in this repo's real notebooks lives in
[`references/operators.md`](references/operators.md) — read it when you run a session.

| Family | Operator — *investment move* | Use it when… |
|---|---|---|
| **Expand search** | Remote / random stimulus — *jump to a far or underused return source / data input* | You keep landing near "momentum/value/carry on equities"; need a far jump |
| | Structural analogy — *transplant a premium that works elsewhere* | A mechanism works *elsewhere* with the same relational shape (other asset, timeframe, literature) |
| **Transform structure** | Morphological recombination — *recombine the strategy stack* | The problem decomposes into {universe × signal × horizon × regime × portfolio × timing}; search odd cells |
| | SCAMPER (attribute ops) — *mutate a working strategy* | You have a working strategy and want disciplined variants, not a blank page |
| **Cross-domain blend** | Conceptual blending — *fuse two edges into one with an emergent rule* | Two whole frameworks could fuse into one with an emergent rule neither parent has |
| | Perspective shift / role-play — *trade the forced, price-insensitive counterparty* | Alpha = trading against someone with a *non-profit* objective (forced seller, rebalancer, de-risker) |
| **Break defaults** | Reversal / Janusian — *attack the crowded default; invert the trade* | Surface the assumptions baked into a strategy and negate them; or reverse the trade |
| | Subtraction / de-fixation — *strip to the load-bearing edge (attribution)* | Strip a "load-bearing" component to fight additive bias and overfit |
| | TRIZ contradiction — *resolve a real tradeoff without compromise* | A real tension ("faster signal ↔ more cost") wants a non-compromise resolution |
| **Iterate** | Generate–explore (Geneplore) — *grow a raw signal, then name its return source* | Build a rough un-named structure first, then interpret what it could proxy; loop |

## Running a session

Default flow when invoked. Keep it adaptive and keep the loops short — the research finding
is that *short diverge → short reflect → diverge again* beats one long brainstorm followed by
one big cull.

1. **Frame.** Get the seed: a problem, an asset, a constraint, or an existing study to mutate.
   If the user gave none, ask **one** scoping question, or default to the repo's most active
   thread (currently the crypto-intraday work under `notebooks/90_crypto_intraday/`). Restate
   the target as a *functional relation* ("concentrate signal X while limiting cost Y") — that
   phrasing is what makes analogy and TRIZ work. Then **anchor it on the return-source map**
   ([`references/return_sources.md`](references/return_sources.md)): which premium / anomaly / flow is in
   play, how crowded is it, which loaders can test it? You generate by moving along that map.
2. **Draw stimuli.** Run `scripts/random_stimulus.py` for genuinely random seeds (words,
   analogy domains, a morphological cell, a primitive triple). Use the script rather than
   picking your own — self-chosen "random" words are near-neighbours and kill the remote-
   association effect. See [`references/domain_seeds.md`](references/domain_seeds.md) for the
   primitive menu the matrix samples from.
3. **Expand** with remote stimulus + structural analogy to pull the search space open.
4. **Transform** with the morphological matrix + SCAMPER to enumerate structured variants.
5. **Break defaults** with reversal, subtraction, and TRIZ — this is where most quant edge
   hides, because the crowded version of every factor is the *default* one.
6. **Blend & shift** with conceptual blending and counterparty role-play to synthesise the
   more complex, higher-novelty ideas.
7. **Reflect & filter.** After each round spend a moment sorting "has a mechanism" from
   "noise", then regenerate seeded by the survivors. Apply the filter below, score, rank.
8. **Output** 5–8 ranked idea cards. Offer to scaffold the top pick into a notebook.

## The filter — what makes an idea "appropriate" here

Every surviving idea must clear these, or be explicitly flagged as failing one. This is the
"persistence" half of creativity and it is non-negotiable for tradable research:

- **Mechanism.** Name the family it belongs to — trend, carry, mean-reversion, value,
  seasonality, liquidity provision, behavioural, microstructure, regime, event/flow. "The
  backtest looks good" is not a mechanism. If you can't articulate one, the card's verdict is
  `reject` or `park`, exactly as in the notebook template.
- **Return source & orthogonality.** Name which return source it harvests (see
  [`references/return_sources.md`](references/return_sources.md)). If it's market beta or a known factor
  — momentum, value, low-vol, carry — in its *crowded default* form, it isn't an idea yet: state the
  **incremental** edge (the conditioner, horizon, universe, or flow that isn't already arbitraged), or
  downgrade it. This is the check that separates a new hypothesis from a repackaged factor.
- **Leak-safe / tradable.** The signal at time `t` uses only data with timestamp ≤ `t`, and
  the trade happens after signal formation. (See `AGENTS.md` — no future information, trade
  after signal.) If an idea needs forward returns to work, it is a diagnostic, not a strategy.
- **Testable with available data.** Map it to a real loader: yfinance, FRED, Binance Vision,
  Polymarket, or local. If the data doesn't exist yet, say what would be needed instead of
  pretending it's runnable.
- **Skeptic's view.** State the single strongest reason it won't work — crowding, costs,
  capacity vs ADV, regime dependence, borrow/short availability. An idea with no stated
  weakness hasn't been thought about.
- **Cheapest first test.** Name the one experiment that would most quickly kill or confirm it.
  Ideas are cheap; the deliverable is *what to run next*.

## Scoring & ranking

Score each surviving idea 1–5 on four axes and rank by them. Show the scores on the card.

- **Novelty** — distance from the *crowded default* version of its return source (and from what this
  repo already did). Repackaged beta or a textbook factor scores low.
- **Mechanism plausibility** — how believable the economic story is.
- **Testability** — how cheaply it can be tested with data on hand.
- **Tradability** — prospects of surviving costs, capacity, and the no-lookahead rules.

Alongside the scores, tag each card with a one-line **Crowding · Capacity · Regime** read — how
arbitraged the source is, how much size it holds, which regimes it needs. These are the investment
realities that decide whether a high-scoring idea survives contact.

Call out the two danger quadrants explicitly: **high-novelty / low-tradability** ("cool but
probably untradeable" — park it honestly) and **high-tradability / low-novelty** ("a
robustness sweep, not a new hypothesis"). The target zone is novel *and* plausibly tradable.

## Output: idea cards

Emit ranked **idea cards** using [`assets/idea_card.md`](assets/idea_card.md). Default 5–8.
Each card is a pre-filled stub of the notebook template's research-question + hypothesis cells,
so a card drops straight into a study. Each card also names its **return-source family** and a
**crowding / capacity / regime** tag, so the investment reality is visible at a glance. Tag each
card's maturity, mapped to the repo's own verdict words so the handoff is seamless:

- **Raw** ↔ `park` — interesting, not yet shaped.
- **Developing** ↔ `needs_revision` — promising; the next experiment is defined.
- **Ready** ↔ `accept`-candidate — mechanism + test + data all line up; worth a notebook now.

## Handoff & persistence

- To act on a card: create a notebook under the right topic folder from
  `notebooks/_templates/strategy_research_template.md`, copying the card into the
  research-question and hypothesis cells. Offer to scaffold this for the top-ranked idea.
- To save a session: only if asked, write the card set to a backlog (e.g.
  `docs/idea_backlog.md`) or drop the strongest as `park`-status stubs under
  `docs/research_decisions/`. Don't auto-create files otherwise — keep the session in-chat.

## Anti-patterns

- **Dumping 30 vague ideas.** Structured operators + a hard filter beat volume. Quality of the
  *filter* is the product.
- **Data mining in a creativity costume.** Novelty with no mechanism. The filter exists to catch
  exactly this.
- **Fake-random stimulus.** Hand-picking "random" words pulls toward near-neighbours. Use the
  script.
- **Stopping at a clever tweak.** "6-1 instead of 12-1 momentum" is a parameter sweep. Push the
  operator further — combine across SCAMPER verbs, or break a default.
- **Untradeable beauty.** Elegant ideas that need lookahead, infinite shorting, or zero capacity.
  Tradability is a scored axis for this reason.

## Bundled resources

- [`references/operators.md`](references/operators.md) — the ten operators, each translated to
  quant research with prompts, a worked example from this repo, and its failure mode. Read at
  session start.
- [`references/domain_seeds.md`](references/domain_seeds.md) — the raw material: this repo's
  universes, data loaders, signal/portfolio/regime/friction primitives, and the strategy
  morphological matrix. Grounds the combinatorial operators in what alpha_lab can actually test.
- [`references/return_sources.md`](references/return_sources.md) — the taxonomy of real return sources
  (risk premia, anomalies, flow, microstructure, macro/regime), each with mechanism, crowding, capacity,
  and which loader can test it. Anchor generation here; run the orthogonality check against it.
- [`assets/idea_card.md`](assets/idea_card.md) — the idea-card output template, with a worked
  example.
- [`scripts/random_stimulus.py`](scripts/random_stimulus.py) — genuinely random draws (words,
  analogy domains, morphological cells, primitive triples) to seed the expand/iterate steps.
  Run with the project interpreter (see `CLAUDE.md` / `AGENTS.md`); stdlib-only, no deps.
