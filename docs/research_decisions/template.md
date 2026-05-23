# Research decision record — TEMPLATE

> **How to use:** copy this file to
> `docs/research_decisions/YYYY-MM-DD_short_slug.md`, fill in the sections,
> and commit alongside the notebook that produced the decision. One file per
> study. Keep entries terse and skim-able.
>
> Status legend: `accept` · `accept_monitoring` · `needs_revision` · `reject` · `park`

---

## Header

| field            | value                                                    |
|------------------|----------------------------------------------------------|
| **Slug**         | `short-kebab-case-handle`                                |
| **Date**         | `YYYY-MM-DD`                                             |
| **Researcher**   | name                                                     |
| **Status**       | one of `accept` / `accept_monitoring` / `needs_revision` / `reject` / `park` |
| **Notebook**     | `notebooks/<path>.ipynb`                                 |
| **Report**       | `reports/<path>.html` (if rendered)                      |
| **Supersedes**   | prior decision slug, if any                              |
| **Superseded by**| later decision slug, if any (fill in when reversed)      |

---

## 1. Research question

One sentence. What did this study try to answer?

> Example: "Does adding DBMF (managed-futures ETF) to a 60/40 stock-bond
> portfolio improve risk-adjusted return at a 10% vol target, net of costs,
> from 2010-2025?"

## 2. Hypothesis & economic rationale

Two-to-four sentences. Why might this work? Cite the mechanism (carry, trend,
mean-reversion, regime, liquidity, behavioral, fundamental) — not just "it
backtests well." If the rationale is "I tried 30 things and this one
backtested best," say so and route to `reject` or `park`.

## 3. Universe & data

- **Universe:** filename or notebook cell. Document selection cutoff and any
  survivorship filter.
- **Date range:** start / end of in-sample. Note any out-of-sample window.
- **Data sources:** which loaders, which fields. Splice cutoffs if any.
- **Known gaps:** corporate actions, holiday calendars, missing days, FX
  conversion, etc.

## 4. Signal & portfolio construction

- **Signal:** one-line definition + reference to the `src/alpha_lab/`
  function that builds it. If still inline in the notebook, flag for
  migration in [the backlog](../backlog/notebook_to_package_backlog.md).
- **Portfolio:** weighting scheme (equal / inverse-vol / MV / risk-parity /
  vol-target), capping rules, leverage cap, rebalance frequency, turnover
  buffer.
- **Costs:** commission_bps, slippage_bps used. Whether they are
  conservative or "typical."
- **Benchmark:** the comparison series. Choosing the wrong benchmark
  (e.g. SPY for a global L/S) is the most common way to mis-decide.

## 5. Headline performance

Net of costs, against the benchmark. Fill in real numbers.

| metric       | strategy | benchmark | active |
|--------------|----------|-----------|--------|
| CAGR         |          |           |        |
| AnnVol       |          |           |        |
| Sharpe       |          |           |        |
| Sortino      |          |           |        |
| MaxDD        |          |           |        |
| Calmar       |          |           |        |
| HitRate      |          |           |        |
| AnnTurnover  |          | n/a       |        |
| AnnCostDrag  |          | n/a       |        |
| NPeriods     |          |           |        |

Quote the return-frequency (daily / weekly / monthly) and Periodicity once,
not per metric.

## 6. Diagnostics

- Equity curve & drawdown — sanity check vs. benchmark.
- Monthly / yearly heatmap — any single year carrying the whole result?
- Turnover plot — what % of trading happens around regime changes?
- Cost sensitivity — performance at 2× and 0.5× cost assumption.
- Conditional performance — Sharpe in high-vol vs low-vol regimes (or
  bull / bear), bond / equity stress windows.
- Factor / beta decomposition — what's the active alpha after removing the
  benchmark and obvious style factors?

## 7. Robustness

Tick every box you checked. Honesty > completeness.

- [ ] Re-ran with shifted start date (±1y) — direction stable.
- [ ] Re-ran on holdout / out-of-sample window — direction stable.
- [ ] Re-ran with neighbouring parameter values — no single-point optima.
- [ ] Block bootstrap or Newey-West SE on the Sharpe — t-stat ≥ 2 or noted.
- [ ] Survivorship check — universe documented and pre-frozen.
- [ ] Same-day signal/execution check — weights are lagged.
- [ ] Cost sweep — break-even cost > assumed cost with comfortable margin.
- [ ] Capacity estimate — implied trade size vs. ADV.

## 8. Failure modes

Where would this stop working? Concretely:

- Regimes that historically hurt the strategy (sample evidence).
- Mechanism-level reasons it could decay (crowding, regulation, market
  microstructure change).
- Data dependencies that could silently break (vendor change, splice
  proxy delisted).
- Implementation gotchas (liquidity, borrow cost for shorts, dividend
  treatment for leveraged ETFs).

## 9. Decision

State the verdict in one sentence, then justify in two-to-four. Tie it back
to the hypothesis and the failure modes.

**Status:** `accept` / `accept_monitoring` / `needs_revision` / `reject` / `park`

When to use which:

- `accept` — clear edge, robust, ready for paper / live consideration.
- `accept_monitoring` — meaningful but fragile or marginal. Define the
  monitoring metric & frequency. Define the kill switch.
- `needs_revision` — promising but a specific issue blocks acceptance.
  Define the next concrete experiment.
- `reject` — does not work, or fails a critical robustness check. Record
  *why* — future-you will want to remember the dead end.
- `park` — out of scope or low priority right now. Note when to revisit.

## 10. Next steps

Three-to-five concrete bullets. Owner + timeframe optional.

- Migrate `X` from the notebook to `src/alpha_lab/<module>` (link to backlog).
- Run additional robustness check `Y`.
- If `accept_monitoring`: add to monitoring set, dashboard at `Z`.
- If `reject` / `park`: archive the notebook, retain the data cache for `N`
  weeks.

## 11. Appendix / supporting artifacts

- Notebook path & last-run commit SHA.
- Path to `data/results/` outputs.
- Path to `reports/*.html` (if rendered).
- Any external links (paper, blog, market data vendor docs).
