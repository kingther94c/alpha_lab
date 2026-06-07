# alpha_lab

A personal investment-research workbench. **Notebook-first, package-backed.**

Scope:
- Backtesting, cross-sectional factor research, exposure comparison
- Regime analysis, risk decomposition
- Stats / ML experimentation, occasional LLM experiments
- Assets: ETFs, futures, FX, rates
- Data: public APIs / public web, IBKR workflows later, occasional PDF / fund-page scrapes

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[backtesting]"    # backtrader, vectorbt
pip install -e ".[optimization]"   # cvxpy, riskfolio-lib
pip install -e ".[llm]"            # anthropic, openai, tiktoken
```

## Layout

```
configs/         YAML configs for reusable settings
data/
  raw/           untouched source data (gitignored contents)
  interim/       cleaned / intermediate
  features/      engineered features
  results/       backtest / research outputs
  private/       local-only secrets, fixtures, private downloads (fully gitignored)
notebooks/       exploratory work (00_sandbox.ipynb + topic subfolders)
reports/         rendered reports (gitignored contents)
src/alpha_lab/   reusable package code
tests/           minimal tests for core helpers
```

## Quickstart

1. Copy `.env.example` → `.env` and fill in keys you need.
2. Open `notebooks/00_sandbox.ipynb`.
3. Start exploring. When a helper feels reusable, lift it into `src/alpha_lab/`.

```python
from alpha_lab.utils.paths import PROJECT_ROOT, DATA_DIR
from alpha_lab.analytics.returns import simple_returns, sharpe
```

## Research → execution (mock trading)

`alpha_lab` *finds* edges; the **`quant_bot_manager`** leg *runs* them as mock/live bots (brokers,
runner, CLI, Streamlit cockpit). End-to-end workflow — research loop, the target-weight handoff,
running & monitoring a bot, risk/kill-switch, going live:
**[docs/research_to_execution.md](docs/research_to_execution.md)**.

One-click mock stack (Binance **demo** funds) — starts the bot + the cockpit:

```bash
make demo-up        # no make? -> D:/conda/envs/py313/python.exe scripts/demo.py up
# cockpit -> http://localhost:8501   ·   make demo-down (stop)   ·   make demo-status (health)
```

## Where things go

| Kind of code                                   | Where                              |
|------------------------------------------------|------------------------------------|
| One-off exploration                            | `notebooks/`                       |
| Reusable transformations / analytics           | `src/alpha_lab/<submodule>/`       |
| Shared reusable parameters                     | `configs/*.yaml`                   |
| Temporary research params                      | Inline in the notebook             |
| Secrets / API keys                             | `.env` (gitignored)                |
| Private data, private fixtures                 | `data/private/` (gitignored)       |

## Package modules

- `utils/` — path, config, cache, logging helpers
- `data/` — data loaders (stubs; fill in per-source later)
- `features/` — transforms (zscore, winsorize, rank)
- `analytics/` — returns, factor (IC / quantile), risk
- `backtest/` — placeholder for signal backtests
- `portfolio/` — placeholder for optimization
- `stats/` — rolling regression etc.
- `ml/`, `llm/`, `reporting/` — placeholders for future growth

## Testing

```bash
pytest -q
ruff check .
```

## Philosophy

Fast iteration first. Reusable helpers second. Reports optional later. Don't dump everything into notebooks forever — once logic is reused twice, move it into `src/alpha_lab/`.

## Docs

Durable knowledge lives under `docs/` — start at [`docs/README.md`](docs/README.md):

- [Research → execution guide](docs/research_to_execution.md) — operate the full pipeline as a mock bot (CLI, cockpit, risk, going-live gates).
- [Architecture](docs/architecture/alpha_lab_architecture.md) — directory ownership.
- [Research artifact contracts](docs/contracts/research_artifacts.md) — shapes of universes, panels, weights, backtest results.
- [Strategy research notebook template](notebooks/_templates/strategy_research_template.md) — start every new study from this outline.
- [Decision record template](docs/research_decisions/template.md) — close every study with a verdict.
- [Notebook → package backlog](docs/backlog/notebook_to_package_backlog.md) — running list of patterns to lift.
- [Roadmap](docs/ROADMAP.md) — milestone plan for the package.
