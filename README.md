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
