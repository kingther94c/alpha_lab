# AGENTS.md

Instructions for any coding agent working in this repo.

## First steps

1. Read `README.md` and `CLAUDE.md` before planning changes.
2. Skim `src/alpha_lab/` (research leg) and `src/quant_bot_manager/` (execution leg) to see which leg/submodule already owns the concept you're touching.

## Environment

- **Python**: 3.13 via conda env `py313`. Interpreter: `D:\conda\envs\py313\python.exe`.
- **Do not rely on `python` / `py` in PATH** — the Microsoft Store `python.exe` stub is a dead alias. Always invoke the full interpreter path above (or activate the env first: `conda activate py313`).
- **Editable install**: `& D:\conda\envs\py313\python.exe -m pip install -e .` from the repo root. Without this, `import alpha_lab` fails inside notebooks and tests.
- **Run tests / scripts through the same interpreter**, e.g. `& D:\conda\envs\py313\python.exe -m pytest -q`.

## Rules

- **Prefer extending existing modules** over creating new ones. Only add a new file when the concept clearly doesn't fit anywhere existing.
- **Reusable logic must live in the right leg.** Research code → `src/alpha_lab/`; live/paper **execution** code (brokers, runner, bots, CLI, UI) → `src/quant_bot_manager/`. **Execution may import research; research must never import execution.** Don't leave reusable code trapped in notebooks.
- **Do not create a `helpers.py` / `utils.py` grab-bag.** Use the existing submodules (`utils/`, `analytics/`, `features/`, etc.).
- **Never modify files under `data/raw/`.** Treat raw data as read-only.
- **Never commit** secrets, `.env`, or anything under `data/private/`.
- **Always clear notebook outputs before committing notebooks.**
- **Avoid future information in backtests and signals.** Signals, weights, filters, normalizers, model fits, and risk estimates must use only data available at the decision timestamp. If a notebook intentionally uses forward returns or full-sample statistics to study a relationship, label it explicitly as research/diagnostic and do not present it as tradable performance.
- **Trade after signal formation.** For reusable backtests, lag target weights or otherwise make the execution timing explicit. Do not let same-close signals earn same-close returns unless the notebook clearly documents that this is a non-tradable diagnostic.
- **Account for the cost of cash (financing).** A backtest's "net" must subtract the financing cost of the capital it consumes — not only commissions, slippage, and funding. Any strategy that ties up cash or uses leverage (long/short, futures/perp basis, cash-and-carry, levered overlays) must charge the risk-free rate (3M T-bill via the `fred` loader, or SOFR) on the deployed/borrowed capital and judge the edge against that hurdle. Omitting it overstates returns — the P6 spot-perp carry looked like +15.7% but was roughly half that after financing.
- **Never start real-money trading autonomously.** Live execution is hard-gated: it requires the user's live API keys, `CONFIRM_LIVE=YES`, and an explicit `--i-understand-live`. Default to `demo` (mock funds); paper-trade first. Never place live orders, fabricate or hunt for API keys, or weaken these gates without an explicit in-context user instruction. A key whose origin/permissions look like real money (e.g. binance.com vs demo.binance.com) must be flagged, not used.
- **Keep functions small and pure** where practical. Add docstrings on public functions. Type annotations where they help.
- **Don't overengineer.** No config systems, plugin registries, or abstract base classes unless justified by real reuse.
- **Backtrader is optional.** Don't centralize the design around any single backtest framework.

## Agent skills

Use repo-relevant skills deliberately. **Start with the project's own skill suite** under
`.claude/skills/` — its map, ordering, and authoring conventions live in the
[`using-alpha-lab-skills`](.claude/skills/using-alpha-lab-skills/SKILL.md) standards skill. The agent
layer is layered: *facts / mechanical rules* live in `CLAUDE.md` and here; *reusable procedures* live
in skills; *ephemeral research knobs* live in notebook cells — keep each in one layer.

- **Project research skills** (`.claude/skills/`): `idea-generation` turns a prompt or a stuck study
  into ranked, mechanism-backed idea cards (anchored to real return sources). More are planned —
  study scaffolding, leakage audit, factor IC, backtest — see the standards skill for the suite and
  the pipeline ordering (ideate → study → audit → decide → lift).

- **GitHub skills**: use when asked to inspect PRs/issues, fix failing CI, address review comments, or publish local changes. Keep the commit scope tight and never include secrets, private data, raw data, or unrelated notebook output churn.
- **OpenAI docs skill**: use for OpenAI API / model / prompt-upgrade work. Rely on official OpenAI docs rather than remembered API details.
- **Google Drive / Sheets / Docs skills**: use only when the user explicitly references connected Drive files or wants data copied from external Sheets/Docs into the research workflow. Keep reusable analysis code in `src/alpha_lab/`, not in Drive-only artifacts.
- **Playwright/browser skills**: use only for browser automation, web-app verification, or inspecting pages that cannot be handled with normal HTTP/data loaders. Do not use browser automation for ordinary notebook or pandas debugging.
- **Notebook/research work**: no special external skill is required by default. Keep notebooks thin, clear outputs before commits, and move reusable data loading, optimization, analytics, and backtest logic into `src/alpha_lab/`.

## Team development mode (optional, opt-in)

For a substantial study or execution-leg change you can run a cross-functional **team** instead of solo —
four project-tuned roles in `.claude/agents/`, each a concentrated project checklist + an adversarial mandate
(not persona role-play):

- `research-lead` — question / mechanism / holdout discipline / go-no-go + decision record (PM for product work).
- `quant-skeptic` — leakage · cost-of-cash · full-sample stats · "is this Sharpe believable" (cardinal-sin guardian; read-only).
- `engineer` — build + right-submodule placement + small/typed/tested, don't-overengineer, two-leg decoupling.
- `execution-safety` — real-money gates for `quant_bot_manager` (never-auto-live, kill-switch, demo default, secrets/raw hygiene; read-only).

- **Trigger:** `/team <task>` (runs the `product-team` workflow in `.claude/workflows/`), or just ask for "the team".
- **Default is still solo + the checklist.** The cheap ~80%: apply the four role `.md` files as a *standing
  checklist* in the main loop — no extra agents. Reserve the real parallel fan-out for the two cases where
  independence from the author actually earns its cost (fan-out is also rate-limit-fragile, ~9 agents/feature):
    - a result/backtest that looks **too good** → `quant-skeptic` (leakage, cost-of-cash, full-sample, survivorship);
    - an **execution-leg change near money/gates** → `execution-safety` (never-auto-live, kill-switch, de-faucet).
  These are exactly where solo has slipped before — the P7 **cost-of-cash** miss (a written non-negotiable) was
  caught by the user, not the author. **Don't** fan out for trivial / conversational / well-specified mechanical
  work — solo is better there. (Validated by a 2026-06 solo-vs-tuned-team-vs-generic-team A/B/C experiment; the
  generic-team arm confirms a non-tuned reviewer misses this repo's cardinal sins, so keep the lenses project-tuned.)
- Each role file is a **living checklist**: when a lens misses something, edit its `.md` so it catches it next time.
  Per-role `tools` and `model` are tunable (e.g. `engineer` runs on a cheaper model by default).

## Making changes

- Keep each change focused. If you spot unrelated issues, note them in your summary rather than fixing them inline.
- When making a non-obvious choice (algorithm, library, placement), add a one-line note in your response explaining the tradeoff.
- Add a test only when the change introduces or relies on a core reusable helper.
- Run `pytest -q` and `ruff check .` before declaring a task done.

## What "done" looks like

- Code lives in the right submodule.
- Notebook is thin; logic lives in `src/alpha_lab/`.
- No stray debug prints, no dead code, no speculative abstractions.
- Tests (if any) pass.
