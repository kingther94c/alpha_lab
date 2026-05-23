# alpha_lab architecture

Durable map of where things live and why. Read this once before adding code.
It restates the **notebook-first, package-backed** philosophy in CLAUDE.md
with concrete ownership rules per directory.

> If a rule here conflicts with CLAUDE.md, CLAUDE.md wins — patch this doc.

---

## One-sentence layout

```
configs/     reusable settings           — YAML, small, project-wide knobs
data/        the data lake               — raw / interim / features / results / private
notebooks/   the research interface      — one story per notebook, thin orchestration
src/alpha_lab/   the reusable package    — small, pure, typed, tested helpers
reports/     rendered artifacts          — gitignored HTML / PDF outputs
docs/        durable project knowledge   — architecture, contracts, decisions, backlog
tests/       safety net for src/         — fast pytest, no I/O, no networks
```

---

## What belongs where

### `configs/`

YAML and CSV registries that are **shared across notebooks** and would be
annoying to re-specify per study.

- ✅ Date defaults, base currency, trading days/year (`default.yaml`)
- ✅ Cost & rebalance defaults for backtests (`backtest.yaml`)
- ✅ Chart defaults (`reporting.yaml`)
- ✅ Data source registry: which loaders exist, which env-var holds the key
  (`data_sources.yaml`)
- ✅ Static universes (`us_sector_etf.csv`, `country_etf_universe.csv`) — see
  the [universe contract](../contracts/research_artifacts.md#universe-file)
- ❌ Per-study hyperparameters (lookback, top-N, vol target). Those live in
  the notebook cell that runs the study.
- ❌ Anything secret (`.env`). Anything you wouldn't want on GitHub
  (`data/private/`).

Read via `alpha_lab.utils.config.load_config("name")`. Don't open YAML files
directly inside notebooks — go through the helper so override paths stay
centralized.

### `data/`

Layered data lake. Path constants come from `alpha_lab.utils.paths`.

| Layer        | Purpose                                   | Writable by    | Gitignored?      |
|--------------|-------------------------------------------|----------------|------------------|
| `raw/`       | Original source-of-truth pulls            | loaders only   | yes (contents)   |
| `interim/`   | Cleaned / aligned / cached intermediate   | notebooks, src | yes (contents)   |
| `features/`  | Engineered features ready for modeling    | notebooks, src | yes (contents)   |
| `results/`   | Backtest output, performance summaries    | notebooks, src | yes (contents)   |
| `private/`   | Personal/proprietary data, fixtures, keys | manual only    | yes (everything) |

- Never modify `raw/` in place. If you need to clean, write a new file under
  `interim/` and name it for the transform.
- `interim/` and `features/` are caches — assume they can be deleted and
  rebuilt. Don't ship analysis that depends on an interim file that no
  notebook can regenerate.
- `results/` artifacts should follow the
  [research-artifacts contract](../contracts/research_artifacts.md).
- `private/` is gitignored wholesale; that's where secrets, manual broker
  exports, and personal fixtures live.

### `notebooks/`

Topic-numbered folders. The **research interface** — where hypotheses,
diagnostics, and decisions actually happen.

Folder convention so the listing stays browsable:

```
00_sandbox.ipynb            scratch
10_strategy_research/       full strategy studies
20_factor_research/         (reserved) cross-sectional factor labs
30_exposure/                (reserved) portfolio vs benchmark
50_risk/                    vol / drawdown / stress
60_ml/                      (reserved) ML pipelines
70_llm/                     (reserved) LLM extraction / summarization
80_event_study/             event-windowed analyses
_templates/                 reusable starting points (see below)
```

Each notebook should be **thin**:

1. Load config & data via package helpers.
2. State the research question and hypothesis explicitly.
3. Call reusable functions from `src/alpha_lab/`.
4. Plot diagnostics.
5. Record a decision (use the
   [decision-record template](../research_decisions/template.md)).

Notebook do's:
- One research question per notebook.
- Inline any one-off transformation, but pull it into `src/alpha_lab/` the
  second time you copy-paste it.
- **Clear all outputs before committing.** Cell outputs bloat git and leak
  data.

Notebook don'ts:
- No long class definitions, no plot-styling utility functions, no custom
  data loaders, no backtest engines. Those go in `src/`.
- No mutation of files in `data/raw/`.
- No same-bar trading (signal → trade on same close) presented as live
  performance. See AGENTS.md for the rule.

Use the [strategy research notebook template](../../notebooks/_templates/strategy_research_template.md)
as a starting outline for new studies.

### `src/alpha_lab/`

The reusable package. Lifted helpers, organized by submodule. See the table
in CLAUDE.md for **which submodule owns which concept** — extend an existing
file before creating a new one.

Function-level expectations:
- Small, pure where practical, type-annotated, one-line docstring.
- No notebook-only side effects (no `plt.show()` inside helpers — return the
  Figure).
- Leak-safe: signals / weights / normalizers must use only data available at
  the decision timestamp. If a helper computes a full-sample statistic for
  diagnostic purposes, name it accordingly and document it.
- I/O at the edges: loaders read; analytics compute; reporting renders. Try
  not to mix the three in one function.

If you're tempted to add a new top-level submodule (e.g.
`alpha_lab.regimes/`), first check: does it not fit into `analytics/`,
`stats/`, `features/`, or `ml/`? Adding submodules is a one-way door; lifting
a single function into an existing one is reversible.

### `reports/`

Rendered output — HTML, PDF, snapshots of figures. Treat as **derived
artifacts**: gitignored, regenerable from a notebook + an
`interim/`/`features/` cache.

If a report is decision-bearing, also write a corresponding entry under
`docs/research_decisions/`.

### `docs/`

Durable knowledge that doesn't live in code or commit messages.

```
docs/
  ROADMAP.md                          milestone plan
  architecture/                       this directory tree, ownership rules
  contracts/                          shapes / dtypes / units for shared artifacts
  research_decisions/                 accept / monitor / reject / park entries
  backlog/                            notebook-to-package migration backlog
```

Docs should age well. If you're writing a one-day status update, that's a
git commit message, not a doc.

### `tests/`

`pytest`-runnable suite covering the most-depended-on helpers. Today: paths,
cache, returns, calendars, charts, metrics, sector/country momentum,
long-only / active MV / risk. Run with:

```bash
pytest -q
```

Tests should be fast, deterministic, and **never hit the network**. Mock
loaders if behavior matters; otherwise test the analytics that consume the
loader output.

---

## Data flow (the happy path)

```
loader (data/loaders/*)
   │  raw frame
   ▼
align / calendars (data/align, data/calendars)
   │  date-aligned price/return panel
   ▼
features.transforms / analytics.factor
   │  signal panel (or factor + forward returns)
   ▼
portfolio.* (long_only / active_mv / vol_target)
   │  target-weight panel
   ▼
backtest.vector.run_backtest
   │  BacktestResult (returns, weights, turnover, costs)
   ▼
backtest.metrics.summary + reporting.charts
   │  performance summary + figures
   ▼
results/ + reports/ + research_decisions/
```

Every arrow has a shape contract in `docs/contracts/research_artifacts.md`.
When a study breaks the data flow, that's usually a signal the new step
needs a home in `src/`.

---

## Research-discipline guardrails

Every reusable helper or new study must consider:

| Risk                              | Mitigation                                              |
|-----------------------------------|---------------------------------------------------------|
| Look-ahead in signal              | Lag signals 1 period; use `forward_returns` for IC      |
| Same-bar trading                  | `backtest.vector` shifts weights by 1; preserve that    |
| Survivorship / universe selection | Document universe construction; keep a frozen CSV       |
| Full-sample normalization         | Use rolling / expanding stats in `features/`            |
| Data snooping / multi-testing     | Pre-register the hypothesis in the notebook header      |
| Overlapping-return t-stats        | Use Newey-West / block-bootstrap (see `stats/`)         |
| Hidden beta                       | Report active stats vs benchmark, factor decomposition  |
| Turnover / cost blindness         | Always include cost drag in `backtest.vector`; report   |
| Regime dependency                 | Diagnose performance in vol / drawdown regimes          |
| Capacity                          | Report leg-level dollar size & implied ADV utilization  |

These are not bureaucratic checklists — they are the discipline that makes
the difference between a backtest and a decision-ready research artifact.

---

## When in doubt

1. Read CLAUDE.md and AGENTS.md.
2. Check whether the existing `src/alpha_lab/` submodules already own the
   concept.
3. Prefer the smallest patch that keeps notebooks thin.
4. If you're adding a new submodule, write a one-line rationale in the PR /
   commit message.
