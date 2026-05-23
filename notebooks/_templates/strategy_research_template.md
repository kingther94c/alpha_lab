# Strategy research notebook template

> **How to use:** create a new notebook under the right topic folder
> (e.g. `notebooks/10_strategy_research/NN_my_idea.ipynb`) and reproduce
> these sections as markdown / code cells in order. Delete sections that
> truly don't apply, but **don't skip "Failure modes", "Robustness", or
> "Decision"** — those are the difference between a backtest and a
> research artifact.
>
> Companion artifacts:
> - [`docs/architecture/alpha_lab_architecture.md`](../../docs/architecture/alpha_lab_architecture.md)
> - [`docs/contracts/research_artifacts.md`](../../docs/contracts/research_artifacts.md)
> - [`docs/research_decisions/template.md`](../../docs/research_decisions/template.md)

---

## Cell 1 — Header (markdown)

```markdown
# <Strategy name> — research notebook

**Slug:** `my-idea-slug`
**Date started:** YYYY-MM-DD
**Researcher:** <name>
**Status (current):** `in_progress`
**Decision record:** `docs/research_decisions/YYYY-MM-DD_my-idea-slug.md` (create when complete)
```

---

## Cell 2 — Research question (markdown)

One sentence. Specific, falsifiable, scoped.

Bad: "Is momentum still alive in ETFs?"
Better: "Does a top-3 / bottom-3 12-1 cross-sectional momentum strategy on
US sector SPDR ETFs (1999-2025), rebalanced monthly with 3 bps round-trip
costs, beat an equal-weight benchmark on net Sharpe?"

---

## Cell 3 — Hypothesis & economic rationale (markdown)

Two-to-four sentences:

1. **Mechanism.** Why might this work? Behavioural under-reaction? Carry?
   Liquidity premium? Regime-conditional? Microstructure?
2. **Prior evidence.** Cite (paper, blog post, prior internal study).
3. **Skeptic's view.** What would the strongest critic say? Crowded factor,
   already arb'd out, costs eat it, regime change.

If you can't articulate a mechanism beyond "the backtest looks good," route
this study to `reject` or `park` after the diagnostics step.

---

## Cell 4 — Imports & config (code)

```python
from alpha_lab.utils.config import load_config
from alpha_lab.utils.paths import CONFIGS_DIR, DATA_DIR, RESULTS_DIR
from alpha_lab.data.loaders.yfinance import load_prices
from alpha_lab.data.calendars import trading_days, rebalance_dates
from alpha_lab.data.align import align_prices, forward_returns
from alpha_lab.analytics.returns import simple_returns, sharpe, drawdown
from alpha_lab.analytics.factor import ic, rank_ic, quantile_buckets
from alpha_lab.features.transforms import zscore, winsorize, cross_sectional_rank
from alpha_lab.backtest.vector import run_backtest
from alpha_lab.backtest.metrics import summary, monthly_table
from alpha_lab.reporting.charts import equity_curve, drawdown_chart, heatmap_monthly

cfg = load_config("default")
bt_cfg = load_config("backtest")

# Notebook-local parameters (one-off knobs go HERE, not in YAML)
START, END = "2010-01-01", None
LOOKBACK_MONTHS = 12
SKIP_MONTHS = 1
TOP_N, BOTTOM_N = 3, 3
REBALANCE = "ME"
```

> Rule: shared defaults live in YAML and are loaded from configs. Study-
> specific knobs (lookback, top-N, vol target) live inline. Don't promote
> ephemeral knobs into YAML.

---

## Cell 5 — Universe & data (code + markdown)

```python
import pandas as pd
universe = pd.read_csv(CONFIGS_DIR / "us_sector_etf.csv")
tickers = universe["signal_etf"].tolist()
prices = load_prices(tickers, start=START, end=END)
prices.head()
```

In a markdown cell, document:

- **Universe** — link to the CSV, name the construction date.
- **Selection bias** — was this universe known in advance for the entire
  backtest window? Are delisted tickers represented? If you can't answer
  yes, name the bias and accept it.
- **Splices / proxies** — DBMF spliced with SG Trend pre-2019, etc.
  Link to the splice helper.
- **Calendar choice** — US business days, 24/7 (crypto), exchange-specific.

---

## Cell 6 — Signal definition (code + markdown)

```python
returns = simple_returns(prices)

monthly = prices.resample("ME").last()
momentum = monthly.shift(SKIP_MONTHS) / monthly.shift(LOOKBACK_MONTHS + SKIP_MONTHS) - 1
```

Markdown: state the exact formula, and **state which timestamps the signal
uses.** If your momentum at month-end `t` uses data through `t` itself, you
are leaning on the close-to-close convention — make sure execution is on
`t+1` (the backtest engine handles this).

If the helper already exists in `alpha_lab.backtest.sector_momentum` or
`country_momentum`, **call the helper** rather than re-deriving. If the
signal is new and you write it inline, file a backlog entry to migrate it
later (see
[`docs/backlog/notebook_to_package_backlog.md`](../../docs/backlog/notebook_to_package_backlog.md)).

---

## Cell 7 — Portfolio construction (code + markdown)

```python
from alpha_lab.backtest.sector_momentum import top_bottom_view_weights
weights = top_bottom_view_weights(
    signal=momentum.reindex(prices.index, method="ffill"),
    top_n=TOP_N,
    bottom_n=BOTTOM_N,
)
```

Markdown: weighting (equal / inverse-vol / MV), capping, leverage cap,
turnover buffer. Note any cross-sectional normalization and confirm it is
rolling, not full-sample.

---

## Cell 8 — Costs & benchmark (code + markdown)

```python
bench_prices = load_prices(bt_cfg["benchmark"], start=START, end=END).iloc[:, 0]
bench_returns = simple_returns(bench_prices)
```

Markdown: which benchmark, why. Costs should be **conservative** —
`backtest.yaml` defaults are a fine starting point; be willing to double
them in a sensitivity check.

---

## Cell 9 — Backtest (code)

```python
result = run_backtest(
    signals=weights,
    prices=prices,
    rebalance=REBALANCE,
    costs_bps=bt_cfg["costs"]["commission_bps"],
    slippage_bps=bt_cfg["costs"]["slippage_bps"],
)
perf = summary(result.returns)
perf
```

Confirm:

- `result.returns` is net of cost (see
  [contracts](../../docs/contracts/research_artifacts.md#backtest-result)).
- `result.turnover.sum()` looks sane for the rebalance freq.

---

## Cell 10 — Diagnostics (code)

```python
equity_curve(pd.DataFrame({"strategy": result.returns, "benchmark": bench_returns}))
drawdown_chart(result.returns)
heatmap_monthly(result.returns)
```

Plus, depending on the study:

- IC time series with `forward_returns(returns, horizon)` + `factor.ic`.
- Quantile-sorted return curves to confirm monotonicity.
- Cost-sensitivity sweep: rerun at 2× and 0.5× costs.

---

## Cell 11 — Robustness checks (code + markdown)

Pick the ones that matter; tick them off honestly. Suggested defaults:

- Shift `START` by ±1y and rerun → headline Sharpe ranges.
- Bootstrap or block-bootstrap the return stream for a Sharpe SE.
- Sweep `LOOKBACK_MONTHS` (e.g. 6, 9, 12, 15) — flat surface, not a spike.
- Out-of-sample window (post-`2023-01-01`) — direction matches in-sample.
- Cost sensitivity (above).
- Regime conditioning (high-vol vs low-vol, bull vs bear).
- Capacity: rough notional vs ADV at typical and stressed liquidity.

---

## Cell 12 — Failure modes (markdown)

Where might this break? Be specific:

- Decay risk (crowding, factor saturation).
- Regime risk (e.g. trend-following dies in choppy markets).
- Implementation risk (borrow cost for shorts, leveraged-ETF compounding
  drag, futures roll yield).
- Data risk (proxy delisting, vendor field change, splice misalignment).

If you'd write any of these in a postmortem after the strategy stops
working, write them here first.

---

## Cell 13 — Decision (markdown)

State the verdict in one sentence. Pick a status:

- `accept` — clear edge, robust, ready.
- `accept_monitoring` — meaningful but fragile; define monitoring + kill.
- `needs_revision` — promising; define next concrete experiment.
- `reject` — does not work; record *why*.
- `park` — out of scope right now; note when to revisit.

Then **create the decision record** at
`docs/research_decisions/YYYY-MM-DD_<slug>.md` using the
[decision-record template](../../docs/research_decisions/template.md).

---

## Cell 14 — Next steps (markdown)

Three-to-five concrete bullets. Reference the backlog for any helpers
worth lifting from this notebook into `src/alpha_lab/`.

---

## Cell 15 — Persist artifacts (code, optional)

```python
out_dir = RESULTS_DIR / "<slug>"
out_dir.mkdir(parents=True, exist_ok=True)
result.returns.to_frame("ret").to_parquet(out_dir / "returns.parquet")
pd.Series(perf).to_frame("value").to_csv(out_dir / "summary.csv")
```

> Reminder before commit:
> - Clear all notebook outputs.
> - Don't commit anything under `data/raw/`, `data/private/`, or `.env`.
> - Commit the decision record alongside the notebook.
