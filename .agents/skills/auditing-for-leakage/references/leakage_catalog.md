# Leakage catalog

Every pattern `scan_leakage.py` flags, why it leaks, and the leak-safe fix (with the `alpha_lab` helper
to use instead). The scanner is heuristic — it finds *suspects*; this catalog tells you whether a
suspect is real and how to fix it. Severities: **BLOCKER** (almost always a true leak), **WARN**
(leak unless you can show otherwise), **INFO** (usually fine; confirm intent).

## Contents
1. Future information in signals
2. Full-sample statistics
3. Signal → execution timing
4. Train/test split & model fitting
5. Alignment & resampling
6. Universe / survivorship
7. Forward returns misuse

---

## 1. Future information in signals  (BLOCKER)

The value at `t` must be computable from data with timestamp ≤ `t`.

- **Negative shift** — `px.shift(-1)` puts tomorrow's value on today's row. Fix: only `shift(k>=0)`; if
  you need a target, build it with `data.align.forward_returns` and keep it on the *evaluation* side.
- **Centered window** — `rolling(w, center=True)` averages points from the future half of the window.
  Fix: `center=False` (the trailing default).
- **Backfill** — `fillna(method="bfill")` / `.bfill()` copies a future observation backward. Fix:
  `ffill`, or leave `NaN` and handle only at the panel edge.
- **Negative-period change** — `pct_change(-1)` is a forward return. Fix: positive periods only.

## 2. Full-sample statistics  (WARN)

Any statistic computed over the *whole* sample and then used to transform each row leaks the
distribution of the future into the past.

- **Full-sample z-score** — `(x - x.mean()) / x.std()`. Fix: rolling/expanding —
  `features.transforms.zscore(x, window=...)`.
- **Full-sample quantile/rank threshold** — `x.quantile(0.9)` as a cutoff. Fix: expanding/rolling
  quantile through `t`.
- **Scaler fit on everything** — `StandardScaler().fit(X)` over the full panel. Fix: fit on the train
  fold only (`ml.cv`, `data.holdout`), transform out-of-sample.
- **Full-sample corr/cov feeding weights** — `df.cov()` used by an optimizer. Fine as a diagnostic;
  for weights use a rolling estimate (`analytics.risk`).

## 3. Signal → execution timing  (BLOCKER if same-bar)

- **Same-close signal earns same-close return** — a signal formed from the `t` close must not be
  applied to the `t` return. Fix: lag the weight panel ≥1 period. `backtest.vector.run_backtest`
  forward-fills and **lags by 1** for you; a hand-rolled `(weights * returns).sum()` must do the same.
- **`shift(0)`** — a no-op that often marks a missing lag. Confirm execution is actually lagged.
- The contract: the weight panel is a **target**, the engine models decision-vs-execution by lagging
  (see `docs/contracts/research_artifacts.md#weight-panel`).

## 4. Train/test split & model fitting  (WARN)

- **Random split** — `train_test_split(X, y)` shuffles, training on future rows. Fix: time-ordered /
  purged split — `data.holdout`, `ml.cv`.
- **`fit_transform` on the whole array** — leaks if it spans the test window. Fix: `fit` on train,
  `transform` on test, inside the time-ordered split.
- Feature engineering (scaling, imputation, target encoding) belongs **inside** the CV fold.

## 5. Alignment & resampling  (INFO → verify)

- **`resample(label="right")`** can stamp a bar with its closing (future-of-bar) timestamp; combined
  with same-stamp execution that's a leak. Verify `label`/`closed` and that you trade after close.
- **`reindex(...).ffill()`** across a coarse→fine join can carry a later observation onto an earlier
  index if the source frame wasn't lagged. Align with `data.align.align_prices` and lag inputs.
- **`interpolate()`** on a signal input blends future into past. Avoid on inputs.
- **`merge_asof`** must use `direction="backward"` for point-in-time joins.

## 6. Universe / survivorship  (manual)

- The investable set must be **frozen at the start** of the window and include names that later
  delisted, or the bias must be named and accepted. Universe files live in `configs/` and are frozen
  by date suffix (`docs/contracts/research_artifacts.md#universe-file`). The scanner can't see this —
  check it by reading.

## 7. Forward returns misuse  (INFO → verify)

- `data.align.forward_returns(returns, horizon)` is correct for **scoring** a signal (IC, quantiles).
  It becomes a leak the moment a forward return is shifted back and used as a *feature*. Keep forward
  returns strictly on the evaluation side; never let one enter the signal.
