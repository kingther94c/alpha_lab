# P6 — Spot–perp cash-and-carry (market-neutral funding harvest)

**Slug**: `crypto_intraday/P6-spot-perp-carry`
**Date**: 2026-06-03
**Researcher**: Kelvin Chen (+ agent)
**Status**: **accept_monitoring**
**Notebook**: [50_spot_perp_carry.ipynb](../../../notebooks/90_crypto_intraday/50_spot_perp_carry.ipynb)
**Supersedes**: extends P2/P5 — the first *market-neutral* crypto strategy in this engagement
**Skills used**: `idea-generation` (return-source map → flow/carry vein), `auditing-for-leakage`

## 0. Why this exists — the v1 lesson

P0–P5 concluded **MONITOR, nothing deployable**. The single lesson: every "edge" was either
(a) annihilated by cost at 5m, or (b) a **directional bull-beta proxy** (`funding_momentum`: +40.8%
in the 2024 bull → **−44.6% in bear**). A grid re-run this round confirmed it — a directional
funding-fade is net-negative in the **2025 OOS** in almost every setting.

So v2 changed the *kind* of edge: stop betting on direction, **harvest a market-neutral carry**.
The spot–perp basis (idea #23 in `idea_log.md`) was deferred in v1 and never built. It is the
canonical robust crypto carry, and it directly answers all three v1 failure modes (turnover, regime
dependence, single-period validation).

## 1. Research question

Does a delta-neutral **long-spot / short-perp** book on BTC & ETH, that harvests positive perp
funding and rebalances daily, earn a **positive net-of-cost return in every regime — bear, bull, and
the 2025 out-of-sample window** — at conservative (perp+spot "stress") costs?

## 2. Hypothesis & economic rationale

**Mechanism: liquidity provision / carry.** On USD-M perps, positive funding is paid by longs to
shorts. Holding **short perp + long spot of the same asset** is delta-neutral (the two price legs
cancel), so the position **collects the funding stream with ~zero directional exposure**. Because the
edge is the carry — not a price view — it is regime-independent by construction: it should be positive
in a bear and a bull alike, differing only in *how much* funding is on offer.

**Skeptic's view.** Funding carry is well known and has compressed since 2021; the absolute yield is
modest; in bear regimes funding can sit at/below zero (little to harvest); and naive bar-by-bar
implementation churns the four legs and pays it all back in fees (our first attempt did exactly this:
−25% from a 45% cost drag). The edge only exists if the position is *held*.

## 3. Universe & data

- **Universe**: BTCUSDT, ETHUSDT — **perp and spot** (4 legs). `configs/crypto_intraday_universe.csv`.
- **Data**: Binance Vision 1h klines (perp + spot) + perp funding (8h), 2021-10 → **2025-12**.
  `research_test` = 2025 (OOS); **PM holdout 2026 was never loaded** (`END="2026-01-01"`, exclusive).
- **Loader bug fixed**: Binance switched **spot** kline timestamps to **microseconds in 2025** while
  perp stayed milliseconds; the loader hard-coded `unit="ms"`, silently dropping all 2025 spot bars.
  Patched `binance_vision.parse_kline_zip` / `parse_funding_zip` to auto-detect ms/µs/ns
  (`_epoch_unit`). Without this fix the OOS year could not be tested.

## 4. Signal & portfolio construction

- **Signal (leak-safe)**: per symbol, `active = (7-day trailing mean of funding, ffilled to the 1h
  grid using only events ≤ t) > 0`. No `shift(-k)`; backtester lags weights +1 bar.
- **Portfolio**: when `active`, hold `+0.25 spot` and `−0.25 perp` per symbol (gross ≤ 1 across 4
  legs); flat otherwise. Delta ≈ 0 by construction.
- **Rebalance**: **daily** (`run_backtest(rebalance="D")`). Because the smoothed funding regime flips
  rarely, daily checking yields **turnover of only ~18 one-way over four years** — the position is
  *held*, which is the whole point.
- **Costs**: per-leg fee+slippage folded into `slippage_bps`. **Stress** = perp 8/10 bps, spot 15/17.5
  bps; **base** = perp 6/6.5, spot 12/13. Funding charged at the 8h bars on the held perp leg.
- **No spot borrow needed**: only the positive-funding side is harvested (long spot / short perp);
  the book stands flat when funding ≤ 0.

## 5. Headline performance — net of stress costs

Daily, `thr=0`, 7-day funding smoothing, BTC+ETH, perp+spot **stress** costs:

| metric | value |
|---|---|
| Net total return (2022–2025) | **+15.7%** (~+3.7%/yr) |
| Annualized Sharpe | **+3.75** |
| Max drawdown | **−1.1%** |
| One-way turnover (4y) | 18 |
| % time in market | 97% |

**Per-year (Sharpe / net), across real regimes:**

| year | BTC regime | Sharpe | net |
|---|---|---|---|
| 2022 | −65% (deep bear) | +0.8 | +0.6% |
| 2023 | +156% | +2.5 | +3.7% |
| 2024 | +120% (bull) | +9.9 | +6.1% |
| **2025** | **−7% (OOS)** | **+5.0** | **+2.1%** |

**Positive in every regime, including the bear and the out-of-sample year** — the property v1 never had.

## 6. Diagnostics — attribution proves the mechanism

| component | contribution (4y) |
|---|---|
| Funding received | **+16.8%** |
| Commission + slippage | −2.3% |
| **Price legs (net delta)** | **+0.10%** |
| **Net** | **+15.7%** |

The price legs contribute **+0.10% over four years** → the book is genuinely **market-neutral**; the
return *is* the funding carry minus cost. This is the single most important diagnostic: it confirms the
PnL is not a disguised directional bet (unlike v1's `funding_momentum`).

## 7. Robustness

All variations stay net-positive (these are perturbations of **one** mechanism, not independent bets):

| variation | net | full Sharpe |
|---|---|---|
| BTC-only | +7.8% | 3.87 |
| ETH-only | +7.3% | 2.90 |
| smoothing 72h / 168h / 336h | +11.7% / +15.7% / +16.6% | 2.8 / 3.8 / 4.0 |
| threshold 0 / 1e-5 / 3e-5 / 5e-5 | +15.7% / +13.8% / +10.5% / +6.9% | 3.8 / 3.3 / 2.6 / 1.8 |
| BASE costs (vs stress) | +16.4% | 3.93 |
| **+24h extra signal lag (leak probe)** | **+15.3%** | 3.65 |

- Works on **each symbol independently** → not a single-name fluke.
- **`thr=0` (collect whenever funding > 0) is the simplest setting and the best** → no overfitting.
- **Cost-insensitive** (base ≈ stress) — the v1 cost-killer is defeated by the low turnover.
- **Leak-free**: a full extra day of signal lag costs ~0.4% — the edge does not depend on timing.

## 8. Failure modes

- **Carry compression**: the yield tracks the funding environment (richest in 2024, thinnest in the
  2022 bear). If funding structurally trends to zero, the edge shrinks. Monitor realized funding.
- **Modest absolute return** (~3.7%/yr). The high Sharpe is a low-vol artifact of neutrality — size
  with leverage only with explicit margin/liquidation risk budgeting.
- **Operational realism not fully modeled**: perp margin & liquidation if the basis gaps, spot
  custody, real fills on unwinds during stress, and exchange/counterparty risk. The cost model is
  conservative on fees but does not model stressed-unwind slippage.
- **2022 is the weakest year** and goes slightly negative at higher thresholds — harvest only at
  `thr≈0` and accept being flat in negative-funding bear stretches.

## 9. Decision

**Status: `accept_monitoring`.** This is the first market-neutral, cost-surviving, leak-free,
all-regime-positive strategy in the engagement — a genuine edge, validated on a true bear and the 2025
OOS. It is *not* `accept` because the absolute return is modest and operational/liquidation realism is
unmodeled. Monitor: realized funding yield, basis behavior in stress, and live execution cost vs the
2.3% backtested drag. Kill switch: net carry (funding − cost) below 0 over a trailing quarter, or a
margin/liquidation event on the perp leg.

## 10. Next steps

- **Final gate**: a single, pre-registered run on the **2026 PM holdout** (`allow:false` today) once
  operational modeling is added — must stay net-positive at stress costs.
- **Lift to package**: the carry-weight builder is currently inline in the notebook (used once). On
  second reuse, lift to `src/alpha_lab/backtest/crypto_carry.py` (file a backlog entry).
- **Breadth**: extend to SOL/BNB perps (loader already supports them) — more legs diversify the carry
  and reduce single-name basis risk.
- **Execution model**: maker-order cost assumptions + a stressed-unwind slippage scenario.
- **Add OI** (idea #21) to gate harvesting when positioning is extreme (carry-crash risk).

## 11. Appendix

- Notebook: [50_spot_perp_carry.ipynb](../../../notebooks/90_crypto_intraday/50_spot_perp_carry.ipynb).
- Loader fix: `src/alpha_lab/data/loaders/binance_vision.py::_epoch_unit`.
- Costs / splits / holdout lock: `configs/crypto_intraday.yaml`. PM holdout (2026) **not accessed**.
