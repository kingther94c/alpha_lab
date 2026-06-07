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

Manufacture research ideas instead of waiting for them. Generate widely with creativity
operators, then filter hard through the repo's own research standards. The product is not idea
*volume* — it is a handful of filtered, testable hypotheses, each ready to drop into a notebook.

**Core principle — novelty bounded by tradeability.** This is divergence on a leash: two moves
in sequence, expand the search then evaluate hard. An idea is "appropriate" here only if it has
an economic mechanism, is leak-safe, is testable with data we can get, and survives a skeptic.
Pure novelty with no mechanism is data mining in a creativity costume — name it and drop it.
Every idea must trace to a **real return source** ([`references/return_sources.md`](references/return_sources.md))
that isn't already arbitraged away in its crowded form.

## When to use / not use

Use it when the user wants new strategy / factor / signal / overlay ideas, wants to expand or
mutate a direction, is stuck, or asks "what else could I test here".

**This skill stops at notebook-ready idea cards.** It does **not** run backtests, write full
strategies, or draw performance conclusions. Once an idea is chosen, hand off to the notebook
workflow — don't start testing inside this skill.

## Default behavior — proceed, don't stall

- **Seed given** (asset, strategy to mutate, constraint, observation): use it.
- **No seed:** do not stop to ask. Default to the repo's most active thread — check
  `docs/research_decisions/` recency or the suite roadmap (at time of writing, the crypto-intraday
  work under `notebooks/90_crypto_intraday/`) — and proceed. Only ask **one** question if the
  asset / market / objective is genuinely absent and can't be defaulted.
- Restate the target as a **functional relation** ("concentrate signal X while limiting cost Y") —
  that phrasing is what makes analogy and TRIZ work.
- Default output: **5–8 ranked idea cards**.

## Repo anchors — read what you need, don't hallucinate

- [`references/return_sources.md`](references/return_sources.md) — the return-source map. Anchor generation here; run the orthogonality check against it.
- [`references/operators.md`](references/operators.md) — full operator playbook (read the families you'll use).
- [`references/domain_seeds.md`](references/domain_seeds.md) — universes, loaders, signal/regime/portfolio/friction primitives, morphological matrix.
- [`assets/idea_card.md`](assets/idea_card.md) — the output card format.
- **What's already been tried** — before generating, learn the thread's prior art. If it has an
  idea log / decision records (e.g. `docs/research_decisions/<thread>/idea_log.md`, `FINAL.md`),
  skim them. **Most threads have none** — then reconstruct prior art from the topic folder's
  notebook markdown + parameter cells (grep titles / headers / strategy parameter args rather than
  reading each notebook in full). Re-proposing a rejected idea wastes a card; knowing what failed
  *is* the orthogonality check in practice.

If a referenced file is missing, say so in one line and proceed from the closest available
context — never invent repo structure or fabricate a loader.

## Process — short loops beat one big brainstorm

1. **Frame & anchor.** State the functional relation; locate it on the return-source map (which
   premium / anomaly / flow, how crowded, which loader). Skim what's already been tried.
2. **Draw stimuli.** Run [`scripts/random_stimulus.py`](scripts/random_stimulus.py) for genuinely
   random seeds (via the project interpreter — on git-bash use a forward-slash exe path).
   Self-picked "random" words are near-neighbours and kill the effect. Draws are *prompts, not
   commitments* — discard any with no tradable mechanism; don't force a card from every cell.
3. **Pick 3–5 operators** that fit the moment (table below). **Do not mechanically run all ten.**
4. **Generate → reflect → regenerate.** After each short round, sort "has a mechanism" from
   "noise", then generate again seeded by the survivors.
5. **Filter, score, rank** → emit cards.

## Operators — pick 3–5 (full playbook: [`references/operators.md`](references/operators.md))

Ten operators in five families (Expand search · Transform structure · Cross-domain blend ·
Break defaults · Iterate). Diagnose what the moment needs and pull only the matching tools:

| When you need to… | Reach for |
|---|---|
| Get unstuck fast / a fresh angle | Remote random stimulus · SCAMPER |
| Cover a big design space systematically | Morphological recombination · TRIZ |
| Break a crowded default assumption | Reversal · Subtraction · Perspective-shift (trade the forced counterparty) |
| Synthesise a complex, high-novelty concept | Conceptual blending · Structural analogy · Generate-explore |

Most quant edge hides under **Break defaults** — the crowded version of every factor *is* the
default one. Read [`references/operators.md`](references/operators.md) for each operator's quant
translation, a worked example from this repo, and its failure mode.

## The filter — every card clears these, or is flagged as failing one

- **Mechanism.** Name the family — trend / carry / mean-reversion / value / seasonality /
  liquidity provision / behavioural / microstructure / regime / event-flow. "The backtest looks
  good" is not a mechanism. No mechanism → verdict `park` or `reject`.
- **Return source & orthogonality.** Name the source (`return_sources.md`). If it's beta or a
  known factor (momentum / value / low-vol / carry) in its *crowded default* form, state the
  **incremental** edge (conditioner / horizon / universe / flow) or downgrade it. Check it isn't
  already in the thread's idea log.
- **Leak-safe / tradable.** Signal at `t` uses only data timestamped ≤ `t`; trade *after* signal
  formation (see `AGENTS.md`). If it needs forward returns, it's a diagnostic, not a strategy.
- **Testable with available data.** Map to a real loader — yfinance / FRED / binance_vision /
  polymarket / local. **Verify against the loader source, not an idea log's "deferred / needs
  loader" label** (the data may already be loadable, or a "available" note may be stale). If the
  data truly doesn't exist, say what's needed; don't fake it.
- **Skeptic's view.** The single strongest reason it won't work — crowding, cost, capacity vs
  ADV, regime dependence, borrow/short availability.
- **Cheapest first test.** The one experiment that most quickly kills or confirms it. Ideas are
  cheap; the deliverable is *what to run next*.

## Scoring & ranking

Score each surviving idea 1–5 and rank by these axes (show them on the card):

- **Novelty** — distance from the crowded default of its return source and from what this repo
  already did. Repackaged beta or a textbook factor scores low.
- **Mechanism plausibility** — how believable the economic story is.
- **Testability** — how cheaply it can be tested with data on hand.
- **Tradability** — prospects of surviving costs, capacity, and the no-lookahead rules.

Anchor the endpoints so two sessions agree:

- **Novelty** 5 = a new return source for this repo · 1 = a textbook factor in its crowded form.
- **Mechanism** 5 = a specific, documented economic force · 1 = "the backtest looks good".
- **Testability** 5 = runnable now from existing helpers · 1 = a data gap.
- **Tradability** 5 = survives stress costs at usable capacity · 3 = moderate turnover / capacity drag but plausibly clears base costs · 1 = needs lookahead / infinite shorting / zero capacity.

**Rank by target-zone fit** — the lead cards are the ones strong on novelty, mechanism, *and*
tradability together (testability gates: an untestable idea can't lead). A tradable-but-unoriginal
idea is a robustness sweep — flag it in the danger quadrant below, don't headline it.

Tag each card with a one-line **Crowding · Capacity · Regime** read. Call out the two danger
quadrants explicitly: **high-novelty / low-tradability** ("cool but untradeable" — park it
honestly) and **high-tradability / low-novelty** ("a robustness sweep, not a new hypothesis").
The target zone is novel *and* plausibly tradable.

## Output — idea cards

Emit ranked cards using [`assets/idea_card.md`](assets/idea_card.md) (read it for the field list).
**5–8 is a soft target — fewer genuinely strong cards beat padding;** surface near-misses (sweeps,
already-claimed ideas) as one-line danger-quadrant callouts, not full cards. Tag each card's
maturity (a one-word summary of the card, not a fifth score), mapped to the repo's verdict words:

- **Raw** ↔ `park` — interesting, not yet shaped.
- **Developing** ↔ `needs_revision` — promising; the next experiment is defined.
- **Ready** ↔ `accept`-candidate — mechanism, cheapest-test, and data all line up; worth a notebook
  now (an open *empirical* question is what the notebook answers — that's expected, not a blocker).

If `idea_card.md` is unavailable, fall back to these fields per card: title · operator(s) ·
research question · mechanism · return source & orthogonality · signal sketch (leak-safe) ·
universe & data · portfolio/trade · skeptic's view · cheapest first test · crowding·capacity·regime ·
scores · maturity · next step.

## Handoff & persistence

- **Act on a card:** create a notebook from `notebooks/_templates/strategy_research_template.md`
  under the right topic folder, copying the card into the research-question and hypothesis cells.
  Offer to scaffold this for the top-ranked idea.
- **Save a session** *only if asked:* write the card set to `docs/idea_backlog.md`, or drop the
  strongest as `park`-status stubs under `docs/research_decisions/`. Don't auto-create files
  otherwise — keep the session in-chat.

## Anti-patterns

- **Dumping many vague ideas.** Operators + a hard filter beat volume. The filter is the product.
- **Data mining in a creativity costume.** Novelty with no mechanism.
- **Re-proposing what the idea log already rejected.** Check first.
- **Fake-random stimulus.** Hand-picked "random" words pull to near-neighbours — use the script.
- **Stopping at a parameter tweak.** "6-1 instead of 12-1 momentum" is a sweep; push the operator
  further (combine across SCAMPER verbs, or break a default).
- **Untradeable beauty.** Elegant ideas that need lookahead, infinite shorting, or zero capacity.
