# CLAUDE.md

Guidance for Claude when working in this repo.

## Purpose

`alpha_lab` is a personal investment-research workbench spanning backtesting, cross-sectional factor research, exposure/regime/risk analysis, stats/ML, and occasional LLM experimentation. Asset coverage is ETFs, futures, FX, rates.

The project has **two legs**:

- **Research — `src/alpha_lab/`** — *finds* edges (data, features, backtests, strategies). Notebook-first.
- **Execution — `src/quant_bot_manager/`** — *runs* edges as live/paper trading bots (brokers, runner, CLI) and monitors/controls them via a Streamlit cockpit.

The handoff is a **strategy's target-weight function**: research produces it (e.g. `alpha_lab.backtest.crypto_book`), a `quant_bot_manager` *strategy* adapts it to live data, and a *bot* (strategy + broker) trades it. Keep the legs decoupled — **execution imports research, never the reverse.**

**Operator's guide:** [docs/research_to_execution.md](docs/research_to_execution.md) — the end-to-end research→execution workflow. One-click mock stack (demo funds): `make demo-up` (or `& D:\conda\envs\py313\python.exe scripts\demo.py up`) → bot + cockpit at http://localhost:8501; `make demo-down` to stop.

## Environment

- **Python**: 3.13 via conda env `py313`. Interpreter: `D:\conda\envs\py313\python.exe`.
- Do NOT rely on `python` / `py` in PATH — they resolve to a dead Microsoft Store stub on this machine. Use the full interpreter path (or `conda activate py313`).
- Editable install from repo root: `& D:\conda\envs\py313\python.exe -m pip install -e .`. Without this, `import alpha_lab` fails in notebooks and tests.
- Run tests / scripts through the same interpreter, e.g. `& D:\conda\envs\py313\python.exe -m pytest -q`.

See [AGENTS.md](AGENTS.md) for the full agent rule set.

## Philosophy: notebook-first, package-backed

- **Notebooks are the primary entry point** for exploration. Don't fight the notebook workflow.
- **Reusable logic belongs in `src/alpha_lab/`.** Once a function is copy-pasted into a second notebook, lift it into the package.
- **Keep notebooks thin.** Import from the package; don't hide important logic inside ipynb JSON forever.
- **Don't overengineer.** This is a solo research repo, not a production platform.
- **Execution is package-first, not notebook-first.** Trading-bot code lives in `src/quant_bot_manager/`, never in notebooks. It still favors small/pure/typed functions and "don't overengineer", but it carries real-money safety gates (see AGENTS.md).

## Where reusable logic goes

| If it is…                                        | Put it in…                          |
|--------------------------------------------------|-------------------------------------|
| Path / env / config / cache / logging helper     | `src/alpha_lab/utils/`              |
| Source-specific data loader                      | `src/alpha_lab/data/`               |
| Feature transform (zscore, rank, winsorize, …)   | `src/alpha_lab/features/`           |
| Return / factor IC / risk analytics              | `src/alpha_lab/analytics/`          |
| Backtest engine or signal runner                 | `src/alpha_lab/backtest/`           |
| Portfolio construction / optimization            | `src/alpha_lab/portfolio/`          |
| Rolling regression / regime classifier           | `src/alpha_lab/stats/`              |
| ML pipelines / feature engineering for models    | `src/alpha_lab/ml/`                 |
| LLM prompts / extractors / summarizers           | `src/alpha_lab/llm/`                |
| Chart or report templates                        | `src/alpha_lab/reporting/`          |

Prefer **small, pure, typed functions** with short docstrings. Avoid giant `helpers.py`-style dumping grounds.

**Execution leg (`src/quant_bot_manager/`)** — where live/paper trading code goes:

| If it is…                                          | Put it in…                            |
|----------------------------------------------------|---------------------------------------|
| Live strategy adapter (research → target weights)  | `src/quant_bot_manager/strategies/`   |
| Exchange/broker integration (orders, positions)    | `src/quant_bot_manager/brokers/`      |
| Bot definition, runner loop, config/paths          | `src/quant_bot_manager/core/`         |
| Monitoring / control UI                            | `src/quant_bot_manager/ui/`           |
| CLI entrypoint (`run` / `rebalance` / `plan`)       | `src/quant_bot_manager/cli.py`        |

## Configs vs notebook parameters

- `configs/*.yaml` — **shared, reusable** settings (base currency, cost assumptions, benchmark, chart defaults). Edit these when the change should apply to future work too.
- **Notebook cells** — **one-off or research-specific** parameters (date windows, universes under study, hyperparameters being swept). Don't promote ephemeral knobs into YAML.

Use `alpha_lab.utils.config.load_config("default")` to read YAML.

## Private data and secrets

- API keys → `.env` (copy from `.env.example`). Load via `python-dotenv`. Never hardcode.
- Anything you wouldn't want on GitHub (private fixtures, proprietary data, API pulls tied to personal accounts) → `data/private/`. The contents are gitignored by design.
- Never commit files from `data/raw/`, `data/private/`, or `.env`.
- Never modify files in `data/raw/` — treat as read-only source-of-truth.

## Adding a new helper / module cleanly

1. Find the right submodule from the table above. **Extend an existing file** if one fits.
2. Keep the function small and pure where practical. Add a short docstring.
3. Add a test only if the helper is core / easy to break / subtle. No need to test plotting or I/O stubs.
4. If the helper replaces notebook code, update or delete the notebook copy.

## Avoid overengineering

- No premature abstractions. Three similar lines beat a bad base class.
- No fallbacks/validation for scenarios that can't happen. Trust internal code.
- No feature flags or backwards-compat shims — just change the code.
- No CI, no Docker, no pre-commit unless genuinely useful.
- Backtrader is **optional**; don't bend the architecture around it. Vectorized pandas/numpy backtests are the default.

## Tests

Tests live in `tests/` and cover only the most-depended-on helpers (paths, cache, returns). Keep them minimal and fast. Run with `pytest -q`.
