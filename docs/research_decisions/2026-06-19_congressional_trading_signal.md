# Research decision record — Congressional trading signal

> Status legend: `accept` · `accept_monitoring` · `needs_revision` · `reject` · `park`

---

## Header

| field            | value                                                              |
|------------------|--------------------------------------------------------------------|
| **Slug**         | `congressional-trading-signal`                                     |
| **Date**         | 2026-06-19                                                         |
| **Researcher**   | Kelvin                                                             |
| **Status**       | `reject` (as a tradable ETF strategy) · tools retained for reuse   |
| **Notebook**     | `notebooks/10_strategy_research/03_congressional_trading_signal.ipynb` |
| **Report**       | `reports/congress_signal_report.html`                              |
| **Supersedes**   | —                                                                  |
| **Superseded by**| —                                                                  |

---

## 1. Research question

Is there an extractable, **tradable** signal in US congressional STOCK-Act disclosures (PTRs) once the
single-stock flow is aggregated to the GICS-sector / macro level and expressed through **ETFs/futures/options only**
(the mandate forbids individual names), net of costs and the cost of cash, 2018–2026 — and does it beat SPY and the
NANC/KRUZ congress-copy ETFs?

## 2. Hypothesis & economic rationale

Mechanism: members (and spouses) may trade on soft information; persistent, multi-member, same-direction sector
accumulation could be a medium-term sector-tilt signal expressible in sector ETFs. **Skeptic's prior (correct, as it
turned out):** the 45-day disclosure lag eats the sharpest drift; post-STOCK-Act studies (Karadas 2021, Belmont 2022)
find the edge largely gone; NANC/KRUZ outperformance is **tech beta**, not selection. This study set out to *falsify*.

## 3. Universe & data

- **Universe:** 11 SPDR sector ETFs (`configs/us_sector_etf.csv`) for Angle A; QQQ/IWM for Angle C. Single-stock PTRs
  mapped to GICS sectors via curated `configs/congress_ticker_sector.csv` (+ yfinance tail).
- **Date range:** trades 2014→2026 (filing dates); **eval 2018-06-19→2026** (XLC ETF inception is the binding
  constraint — we never trade an ETF before it listed).
- **Data sources:** **kadoa** per-filer JSON aggregated to 52,905 disclosures (26,324 single-stock, 207 members) via
  `data/loaders/congress.py`; official **House Clerk XML index** for a coverage audit; Senate Stock Watcher as a
  cross-check. Prices via `yfinance`. Cost of cash via FRED `DTB3`.
- **PIT discipline:** signal date = `filing_date` (public), never `transaction_date`. Median days-to-file 29,
  **p90 = 245, 19% filed past the 45-day deadline** — so this matters a lot.
- **Known gaps:** ~16% of |flow| unmapped (mostly mislabeled ETFs / foreign ADRs); sector labels are current-not-PIT
  (mild for a sector signal); official-audit coverage ~55% (the official count includes bond/option/fund-only PTRs and
  scanned PDFs no parser ingests). Delisted tickers retained (no survivorship trimming).

## 4. Signal & portfolio construction

- **Signal (Angle A, core):** rolling 63-day net log-mid $ flow per GICS sector → trailing 252-day z-score → long
  top-3 / short bottom-3 sector ETFs, dollar-neutral. Built by `backtest/congress_signal.py` + `backtest/congress_book.py`.
- **Portfolio:** dollar-neutral (long_gross=short_gross=1), weekly (W-FRI) rebalance, ~16.6× annual turnover.
- **Costs:** 1 bp commission + 3 bp slippage one-way (conservative for liquid sector ETFs). **Cost of cash:** 3M
  T-bill (FRED DTB3) on long-gross exposure, netted in the headline ("excess of cash"). *Note:* for a self-funding
  market-neutral book this is a slightly **over-harsh** hurdle (the skeptic flagged it) — i.e. conservative; the verdict
  holds without it (net Sharpe 0.27 still ≪ SPY).
- **Benchmarks:** SPY buy-hold, **NANC** (Dem copy) + **KRUZ** (Rep copy) ETFs, and a sector equal-weight book.

## 5. Headline performance

Daily returns, 2018-06-19→2026 (excess of cash). Active vs SPY.

| metric       | strategy (L/S) | SPY      | active |
|--------------|----------------|----------|--------|
| CAGR         | 0.7%           | 15.1%    | −14.4% |
| AnnVol       | 12.2%          | 19.3%    |        |
| Sharpe       | **0.12**       | **0.82** | −0.70  |
| MaxDD        | −45.9%         | −33.7%   |        |
| HitRate      | ~0.50          | ~0.54    |        |
| AnnTurnover  | 16.6×          | n/a      |        |
| NPeriods     | 2011           | 2010     |        |

Reference Sharpes (same window where available): **NANC 1.30, KRUZ 1.28, SectorEW 0.74.** The strategy loses to all.

## 6. Diagnostics

- **Leg decomposition (the story):** Long-top-3 Sharpe **0.75** (≈ market beta, < SPY 0.82, < SectorEW 0.74); Short-
  bottom-3 Sharpe **−0.54** (shorting a rising market). Dollar-neutral (beta removed) ⇒ **~0 alpha (0.12)**.
- **IC (sector flow → forward sector-ETF return):** rank-IC **+0.018 @5d, +0.046 @21d (t=6.5), −0.022 @63d (t=−3.1)** —
  a faint short-horizon signal that **decays and reverses** by ~3 months; the 63-day construction sits where it fades.
- **Event study (Angle D, single names):** after a member **buys**, market-adjusted CAR drifts **+0.74%/42d (t=4.5)**
  around the *transaction* date and **+0.37%/42d (t=2.4)** around the *filing* date (≈ half survives the lag).
  **Sells carry no signal** (t≈0). Real, but buy-side-only and tiny — and it dilutes to ~0 once aggregated to sector ETFs.
- **Sign flip:** contrarian version is worse (Sharpe −0.53) → no hidden right-signed edge.

## 7. Robustness

- [x] Regime split — 2018-21: L/S 0.06, Long 0.87, SPY 0.91; 2022-26: L/S 0.16, Long 0.45, SPY 0.74. No alpha in either.
- [x] Neighbouring parameters — 3×3 grid (flow window × top_n) Sharpe ∈ [−0.79, +0.21]; skeptic's 36-config sweep:
      **no config has a significant positive alpha (max alpha t = 0.85)**. Robust null, not a single bad pick.
- [x] Block bootstrap — 95% Sharpe CI **[−0.58, 0.85]**, P(Sharpe>0)=62% (spans 0).
- [x] Newey-West SE — mean-return **t = 0.32** (overlap-adjusted; far from significant).
- [x] Deflated Sharpe — **DSR = 0.15** (need >0.95; expected max Sharpe under the null over 9 trials = 0.58).
- [x] Same-day signal/execution — verified: filing bucketed to next session, weights lagged 1 day (skeptic-audited, leak scan clean).
- [x] Cost of cash charged — yes (conservatively over-charged on a neutral book; verdict unchanged without it).
- [x] Survivorship — delisted tickers retained in the curated map.
- [x] Adversarial audit — quant-skeptic reproduced every number; verdict "trustworthy NO-GO".

## 8. Failure modes

- **Legislative (highest):** H.R.7008 (Union Calendar 2026-02) / S.1498 (Senate HSGAC 2025-12) would force divestiture
  into blind trusts → the single-stock PTR signal dries up. Pivot: lobbying / government-contract alt-data.
- **Alpha decay / already-priced:** 45-day lag + the existence of NANC/KRUZ.
- **Single-name → ETF dilution:** the faint per-name buy drift washes out at the sector level (quantified above).
- **Data:** unmapped tail, current-not-PIT sectors, kadoa vs scanned-PDF parsing gaps.

## 9. Decision

**Status: `reject`** (as a tradable ETF strategy).

The hypothesis is falsified at the Phase-2 gate. A small, real, **buy-side-only, short-horizon** information signal
exists in the disclosures (event study + IC), but it **does not survive** translation to sector-ETF expression +
beta-neutralization + costs: the strategy returns a Sharpe of 0.12 vs SPY 0.82 / NANC 1.30, with a bootstrap CI that
spans zero and a Deflated Sharpe of 0.15. The NANC/KRUZ edge is tech beta, exactly as the prior warned. This is a clean,
robust, adversarially-audited null — valuable because it saves capital and effort, and the toolkit it produced is reusable.

## 10. Next steps

- **Angle B (committee overlap)** is the one untested place a concentrated-subset edge could hide. Blocked on
  point-in-time committee rosters (GovTrack / Senate.gov / `unitedstates/congress-legislators`). Scaffolded at
  `congress_book.committee_weighted_flow`. *This is the highest-value follow-up* before fully closing the topic.
- **Idea-flag, not a strategy:** wire the sector-flow z-score into the daily idea pipeline as a *low-correlation context
  flag*, not a standalone book.
- **Do NOT** advance to paper trading; do NOT wire `latest_target_weights` to a live `quant_bot_manager` bot.
- Migrate the inline notebook knobs already lifted into `src/alpha_lab/` (done); retain the data cache.
- Fix the shared `top_bottom_view_weights` net-short edge case (separate task; doesn't affect this result).

## 11. Appendix / supporting artifacts

- Code: `data/loaders/congress.py`, `data/congress_universe.py` + `configs/congress_ticker_sector.csv`,
  `backtest/congress_signal.py`, `backtest/congress_book.py`, `analytics/event_study.py`, `stats/tests.py`.
- Report: `reports/congress_signal_report.html` (regenerate: `PYTHONPATH=src python scripts/congress_signal_report.py`).
- Tests: `tests/test_congress_signal.py`.
- References: Ziobrowski 2004/2011; Eggers & Hainmueller 2013; Karadas 2021; Belmont 2022. Data: kadoa
  congress-trading-monitor (MIT), House Clerk eFD, Senate eFD, Senate Stock Watcher. Bills: H.R.7008, S.1498.
