# alpha_lab helper roadmap

Each milestone is a thin, useful layer. Only build the next one when the previous one has actually been used in real research. Skip anything that doesn't earn its keep.

Guiding rules (from CLAUDE.md):
- Small pure functions > frameworks.
- Extend existing submodules before adding new ones.
- Every new helper gets a one-line docstring; tests only for load-bearing ones.

---

## v0.2 — Make real research possible

**Theme:** go from "empty package" to "can run a full ETF signal backtest end-to-end."

### `data/`
- `loaders.yfinance.load_prices(tickers, start, end)` → wide adj-close DataFrame. Use as default source first.
- `loaders.fred.load_series(codes, start, end)` → wide macro DataFrame (rates, spreads, CPI).
- `loaders.local.load_parquet_dir(path, pattern)` → concat local files (for manual pulls / IBKR exports later).
- `calendars.py`: `trading_days(start, end, calendar="NYSE")`, `rebalance_dates(index, freq)`. Use `pandas_market_calendars` only if a simple pandas `bdate_range` isn't enough.
- `align.py`: `align_prices(df, calendar)`, `forward_returns(returns, horizon)`.

### `backtest/`
- `vector.py`:
  - `run_backtest(signals, prices, *, rebalance, costs_bps, slippage_bps, initial_capital) -> BacktestResult`
  - Signals → target weights → realized pnl with costs. Vectorized pandas. No loops over dates unless needed.
- `metrics.py`: `summary(returns)` → dict of CAGR, ann vol, Sharpe, Sortino, Calmar, max DD, hit rate, turnover. Plus `monthly_table(returns)`.
- `BacktestResult` dataclass: `weights`, `returns`, `turnover`, `pnl_by_asset`, `summary`.

### `reporting/` (keep thin)
- `charts.py`: `equity_curve(r)`, `drawdown(r)`, `heatmap_monthly(r)` — plotly defaults from `configs/reporting.yaml`.

### Tests
- `test_backtest_vector.py`: buy-and-hold on flat/upward prices matches expected pnl; rebalance + cost reduces pnl correctly.
- `test_metrics.py`: known return streams give expected Sharpe / DD.
- `test_calendars.py`: rebalance_dates honors monthly/quarterly freq.

### Exit criteria
- `notebooks/10_backtest/01_simple_cross_asset_momentum.ipynb` exists and runs top-to-bottom using only package helpers.

---

## v0.3 — Factor depth, exposure comparison, portfolio construction

**Theme:** beyond one backtest — systematic factor evaluation + exposure/risk attribution + constrained portfolios.

### `analytics/factor.py` (upgrade)
- `ic_timeseries(factor_df, fwd_returns_df, method="pearson"|"spearman")` → daily IC Series.
- `quantile_returns(factor_df, fwd_returns_df, n=5, weighting="equal"|"cap")` → wide DataFrame of per-quantile returns + long-short spread.
- `turnover(factor_df, n=5)` → per-period portfolio turnover by quantile.
- `information_decay(factor_df, returns_df, horizons=[1,5,21,63])` → DataFrame of IC vs horizon.
- `neutralize(factor_df, exposures_df)` → residual factor after regressing out exposures (sector, beta).

### `analytics/exposure.py` (new)
- `active_weights(portfolio_w, benchmark_w)` → weight gap.
- `active_exposure(portfolio_w, factor_loadings)` → portfolio × factor exposures.
- `active_exposure_vs_bench(port_w, bench_w, factor_loadings)` → active factor bets.

### `analytics/risk.py` (upgrade)
- `ledoit_wolf_cov(returns)` via sklearn.
- `pca_factor_cov(returns, n_factors=5)` → statistical factor cov decomposition.
- `factor_risk_decomp(weights, loadings, factor_cov, specific_var)` → % risk from each factor + specific.
- `stress_pnl(weights, shocks)` → shock dict → pnl.

### `portfolio/` (new content)
- `weights.py`: `equal_weight`, `inverse_vol`, `risk_parity` (simple Newton iteration), `min_variance`.
- `optimize.py` (opt extra `optimization`): `mean_variance(mu, cov, *, bounds, sector_caps, turnover_budget)` via cvxpy. Returns weights + diagnostics.
- `constraints.py`: tiny helpers that build cvxpy constraint lists (`box`, `sector_cap`, `turnover_cap`, `beta_neutral`).

### `stats/` (extend)
- `regression.newey_west(y, X, lags)` → HAC SE wrapper.
- `regime.py`: `vol_regimes(returns, n=3)` via `GaussianMixture` on rolling vol; `regime_conditional_stats(returns, regimes)`.

### Tests
- `test_factor_pipeline.py`: synthetic factor with known correlation → IC and quantile spread match expectation.
- `test_risk_decomp.py`: components sum to portfolio variance.
- `test_portfolio_weights.py`: equal/inverse-vol sum to 1; risk parity equalizes RC within tolerance.

### Exit criteria
- `notebooks/20_factor_research/01_value_momentum_pipeline.ipynb` runs a full IC + quantile + decay study.
- `notebooks/30_exposure/01_portfolio_vs_benchmark.ipynb` produces an active-exposure chart.

---

## v0.4 — ML pipelines, LLM tooling, reports

**Theme:** scale up repeatable workflows — ML experiments, LLM-assisted extraction, one-command reports.

### `ml/` (new content)
- `splits.py`: `expanding_window_splits(index, n_splits, min_train)`, `purged_kfold(index, k, embargo)`.
- `features.py`: leak-safe builders — `lag(df, k)`, `rolling_stat(df, win, fn)`, `cs_rank(df)` (wraps `features.transforms`).
- `pipelines.py`: `fit_predict_oos(model, X, y, splits)` → OOS predictions aligned to index.
- `eval.py`: `ic_by_fold`, `stability_report`, `feature_importance_summary` (works with sklearn / LightGBM-compatible estimators).

### `stats/` (extend)
- `tests.py`: batch `adfuller`, `kpss`, Ljung–Box on a DataFrame; returns tidy results DataFrame.
- `cointegration.py`: pairwise Engle–Granger / Johansen shortcut.

### `llm/` (new content; all opt-in via `[llm]` extra)
- `clients.py`: `anthropic_client()` / `openai_client()` with default prompt caching headers, retry, token counting via tiktoken.
- `extract.py`:
  - `extract_from_pdf(path, schema)` — PDF → structured dict using a caching-aware prompt.
  - `extract_from_url(url, schema)` — fetch + clean HTML + extract.
- `summarize.py`: `summarize_notes(texts, style="bullet"|"memo")` with citation preservation.
- `prompts/` directory: versioned prompt templates as plain `.txt` / `.md`.

### `reporting/` (upgrade)
- `tables.py`: `summary_table(result)` styled DataFrame; `regime_table(stats_df)`.
- `render.py`: `render_html(title, sections)` → single-file HTML with embedded plotly figs. Writes to `reports/`.
- `templates/`: one minimal factor-study template and one backtest template.

### `utils/` (extend)
- `cache.py`: add `cached_json(key, builder)` + TTL option for LLM calls.
- `io.py` (new): `read_any(path)` dispatching on extension (parquet / csv / json); `atomic_write` helper.

### Tests
- `test_splits.py`: purged K-fold actually embargoes; train/test don't overlap.
- `test_llm_clients.py`: monkeypatched SDK call — verifies cache-control headers are set.
- `test_report_render.py`: `render_html` emits a file with expected sections.

### Exit criteria
- `notebooks/60_ml/01_xgb_signal_oos.ipynb` runs an OOS ML signal with purged K-fold and emits an HTML report.
- `notebooks/70_llm/01_fund_page_extract.ipynb` pulls structured data from a fund fact sheet PDF.

---

## What is deliberately **not** on this roadmap

- Security master / ticker mapping service.
- Live order routing.
- CI / pre-commit / Docker.
- A config framework beyond YAML + `load_config`.
- A plugin or registry system.
- Unified "BaseStrategy" or "BaseFactor" abstract classes.

Add any of these only if a concrete research need actually forces them.

---

## Sequencing note

Each version should ship alongside **one notebook in the matching topic folder** that exercises the new helpers end-to-end. That notebook doubles as living documentation and a regression canary — if it breaks on future refactors, the helper surface has drifted and needs attention.
