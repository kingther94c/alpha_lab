---
name: auditing-for-leakage
description: >-
  Audit a strategy notebook, signal, or backtest for lookahead / leakage before its results are
  trusted — the cardinal sins this repo cares about: future information in signals, same-bar
  signal-to-execution, full-sample normalization, survivorship, and forward returns used to score the
  present. Runs a heuristic static scan (scripts/scan_leakage.py over .ipynb / .py) and a manual
  leak-safe checklist, then returns per-finding severity + the exact fix and a verdict on whether the
  headline performance is believable. Reach for this whenever a backtest looks "too good", before
  accepting a study, when reviewing a notebook or a new signal/feature, when wiring weights to returns,
  or any time someone asks "is this leaking / is this lookahead-free / can I trust this Sharpe". Use it
  before `running-a-backtest` results are believed and before writing a decision record.
---

# Auditing for leakage

A backtest is only a research artifact once you can trust it didn't peek at the future. This skill is
the gate between "the equity curve looks great" and "the edge is real". It pairs a fast **heuristic
scan** (flags suspicious patterns) with a **manual leak-safe checklist** (the contract points the
scanner can't prove), and ends with a blunt verdict.

The standard it enforces is the repo's own: signals use only data with timestamp ≤ `t`, trades happen
after signal formation, normalization is rolling not full-sample, the universe is pre-frozen, and
forward returns are used *only* to score a signal — never to build it (see `AGENTS.md` and
`docs/contracts/research_artifacts.md`).

## When to use

- A backtest Sharpe looks suspiciously high, or an equity curve is implausibly smooth.
- Before accepting a study / writing a decision record, or before believing a `running-a-backtest` result.
- Reviewing a notebook, a new signal/feature, or a portfolio→returns wiring.
- Any "is this lookahead-free / is this leaking / can I trust this number" question.

## Workflow

1. **Scope the surface.** Identify where the signal/feature is built, where weights are formed, and
   where returns/IC are computed. Most leaks live at three seams: feature normalization, the
   signal→execution lag, and return alignment.
2. **Run the heuristic scan.** Point the scanner at the notebook(s)/module(s):

   ```bash
   "D:\conda\envs\py313\python.exe" .Codex/skills/auditing-for-leakage/scripts/scan_leakage.py <path.ipynb|path.py> [more...]
   ```

   It flags BLOCKER / WARN / INFO patterns with file:cell:line. Treat it as a smoke detector — it finds
   *suspects*, not proof. Read [`references/leakage_catalog.md`](references/leakage_catalog.md) for what
   each pattern means and the fix.
3. **Walk the manual checklist** (the scanner can't see intent — verify these by reading):

   - [ ] **No future information.** Every value at `t` is computable from data with timestamp ≤ `t`.
         No `shift(-k)`, no `center=True` windows, no `bfill`/back-interpolation into signal inputs.
   - [ ] **Trade after signal.** Weights are lagged ≥1 period before multiplying by returns
         (`run_backtest` lags by 1; manual loops must replicate it). No same-close signal earning the
         same-close return unless explicitly labelled a non-tradable diagnostic.
   - [ ] **Rolling, not full-sample.** Z-scores / ranks / vol / betas / model fits use rolling or
         expanding windows — never a statistic computed over the whole sample (a giant hidden lookahead).
   - [ ] **Universe pre-frozen.** The investable set was known at the start of the window; delisted /
         survivorship handled or named as an accepted bias.
   - [ ] **Forward returns scoring only.** `data.align.forward_returns` is used to *evaluate* a signal,
         never shifted back into the signal itself.
   - [ ] **Alignment is honest.** `reindex`/`resample`/`merge_asof` don't pull a later observation onto
         an earlier timestamp; resample `label`/`closed` conventions checked.
4. **Verdict.** For each finding: severity, the offending line, and the concrete fix. Then one call:
   **trustworthy / fix-then-rerun / not-tradable-as-built**. If any BLOCKER stands, the headline
   performance is not believable until fixed and re-run.

## Do / Don't

- **Do** run the scan early and cheaply, then spend human attention on the checklist items the scanner
  can't judge (intent, alignment, universe construction).
- **Do** quantify when possible: a leak usually *shrinks* edge when removed — re-run after the fix and
  report the before/after Sharpe so the cost of the leak is explicit.
- **Don't** treat a clean scan as a pass — the scanner is heuristic; the checklist is the real audit.
- **Don't** rewrite the strategy here; this skill *finds and explains* leaks. Hand the fix back to the
  notebook / `running-a-backtest` flow.
- **Don't** flag intentional research diagnostics (a notebook deliberately using forward returns to
  study a relationship) as leaks — confirm whether the author labelled it non-tradable first.

## References

- [`references/leakage_catalog.md`](references/leakage_catalog.md) — every pattern the scanner flags,
  why it leaks, and the leak-safe fix (with the `alpha_lab` helper to use instead).
- Repo contracts: `AGENTS.md` (no-lookahead, trade-after-signal), `docs/contracts/research_artifacts.md`
  (signal/weight/forward-return shapes).
