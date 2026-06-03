---
name: using-alpha-lab-skills
description: >-
  Orientation and standards for alpha_lab's agent layer — the map of which skill to use for a
  research task, how the knowledge layers divide responsibility (CLAUDE.md facts, AGENTS.md rules,
  on-demand skills, docs/, cross-session memory), the skill-suite roadmap, and the conventions for
  authoring new skills in this repo. Consult this whenever choosing how to approach a research task,
  deciding whether something belongs in a skill vs CLAUDE.md vs a notebook cell, authoring or
  reorganizing skills, or when unsure which skill applies. This is the standards/ordering layer;
  actual skill selection is description-based.
---

# Using alpha_lab's agent skills

This is the orientation skill for the repo's agent layer. Skill *selection* happens automatically
from each skill's description — this doc owns what descriptions can't: the **layer map** (who owns
which knowledge), the **ordering/precedence** of the research pipeline, the **suite roadmap**, and
the **conventions** for writing new skills so the suite stays coherent as it grows.

## The knowledge layers — who owns what

Keep each fact/rule/procedure in exactly one layer and reference it from the others. The failure mode
to avoid is the same rule drifting across three files.

| Layer | Owns | Loaded |
|---|---|---|
| `CLAUDE.md` | Always-true **facts** + mechanical rules: interpreter path, "never touch `data/raw/`", cost/benchmark defaults, where code goes | Always |
| `AGENTS.md` | The agent **operating contract**: no-lookahead, trade-after-signal, extend-don't-proliferate, what "done" means | Always |
| `.claude/skills/<name>/` | Reusable **procedures & techniques** that load only when relevant (ideate, backtest, audit, build a factor) | On trigger |
| `references/` inside a skill | Heavy domain **reference** (taxonomies, formula tables) | On demand |
| `docs/` | Durable **knowledge**: architecture, artifact contracts, research decisions, roadmap | By link |
| `memory/MEMORY.md` | Cross-session **learned facts / corrections** | Curated |

**De-dup rule:** a *fact* or *mechanical rule* lives in CLAUDE.md/AGENTS.md; a *procedure* lives in a
skill; an *ephemeral research knob* (date window, hyperparameter) lives in a notebook cell. Don't
restate one in another layer — link to it.

## The research pipeline — skill ordering & precedence

Studies in this repo follow a natural order; skills slot into it. Respect the precedence even when a
later step looks more interesting.

1. **Ideate** → `idea-generation` produces ranked idea cards (what is worth testing).
2. **Study** → scaffold and drive a notebook from `notebooks/_templates/strategy_research_template.md`
   *(planned skill: `running-a-study`)*.
3. **Trust it** → audit the signal/notebook for leakage before believing any backtest, via the
   **`auditing-for-leakage`** skill.
4. **Decide** → run the robustness checklist and write the decision record
   *(planned skill: `writing-a-decision-record`)*.
5. **Lift** → move reusable logic from notebook to `src/alpha_lab/` *(planned skill: `lifting-to-package`)*.

Cross-cutting precedence, always on:
- **Never trust a backtest before the leakage audit** — same-day execution, full-sample normalization,
  survivorship, and forward-information are the default suspects (see `AGENTS.md`).
- **Honor the always-on rules** in every skill: `data/raw/` is read-only; signals use only data ≤ `t`;
  trade after signal formation.
- **Anchor ideation in real return sources** before generating
  (`idea-generation/references/return_sources.md`) — novelty without a return source is data mining.

## Skill suite — status & when to use

Built on demand, not scaffolded speculatively (empty `helper`-style skills are an anti-pattern). One
capability per skill; selection is by description.

| Skill | Status | Use when |
|---|---|---|
| `idea-generation` | **built** | Want new strategy/factor/signal ideas; stuck; expanding a research direction |
| `running-a-study` | planned | Start a study: scaffold the template, wire loaders/contracts, drive to a verdict |
| `auditing-for-leakage` | **built** | Sanity-check a notebook/signal for lookahead, same-day execution, full-sample stats, survivorship |
| `building-a-factor` | planned | Construct a factor score (winsorize/zscore/rank) leak-safely for IC analysis |
| `computing-factor-ic` | planned | Standardized IC / rank-IC / quantile-bucket evaluation of a signal |
| `running-a-backtest` | planned | Vectorized backtest with cost/slippage; thin wrapper over `backtest.vector` |
| `classifying-regimes` | planned | Produce rolling regime labels (vol/trend/risk-on-off) to use as conditioners |
| `constructing-a-portfolio` | planned | Apply a weighting scheme + constraints; wraps `portfolio.*` |
| `adding-a-data-loader` | planned | Scaffold a new source loader under `data/loaders/` per the artifact contracts |
| `lifting-to-package` | planned | Move reusable notebook logic into `src/alpha_lab/` per the backlog |

To build the next one, use the `anthropic-skills:skill-creator` skill and the conventions below.

## Authoring a new skill here (conventions)

Distilled from frontier practice (anthropics/skills, obra/superpowers, wshobson/agents
`quantitative-trading`) and this repo's "notebook-first, package-backed, don't-overengineer" ethos.

- **Name** — gerund/verb-y, hyphenated, ≤64 chars, **one capability**: `running-a-backtest`, not
  `backtest-helper`. Never ship `helper` / `utils` / `tools` skills, or a description like "helps with data".
- **Description** — third person, `"<what it does>. Use when <concrete triggers/keywords>."` Keyword-rich
  and slightly pushy (Claude under-triggers skills). ≤1024 chars.
- **Body** — under 500 lines. Structure that works for procedural skills:
  `## When to use` (bullet triggers) → `## Workflow` (numbered; include a copyable checklist for
  multi-step jobs) → `## Do / Don't` → `## References`.
- **Progressive disclosure** — heavy reference → `references/*.md` (one level deep; add a TOC if >100
  lines; name by domain, not `doc1.md`); executable logic → `scripts/` (these **`import alpha_lab`** and
  call the package — don't re-implement analytics, that's the notebook-first/package-backed rule);
  output templates → `assets/`.
- **Respect the layers** — a skill encodes a *procedure*. Facts and mechanical rules stay in
  CLAUDE.md/AGENTS.md; ephemeral knobs stay in notebook cells.
- **Side-effects** — for skills that pull data or write files, prefer `disable-model-invocation: true`
  (user-invoked only) so they don't fire unintentionally.
- **Iterate** — draft → test on a couple of realistic prompts → review → improve, via `skill-creator`.

## Pointers

- Skill spec & examples: [anthropics/skills](https://github.com/anthropics/skills). Quant exemplar:
  [wshobson/agents `quantitative-trading`](https://github.com/wshobson/agents/tree/main/plugins/quantitative-trading).
- Repo knowledge: [`docs/README.md`](../../../docs/README.md) (architecture, artifact contracts,
  decision records, roadmap); the [strategy-research template](../../../notebooks/_templates/strategy_research_template.md).
