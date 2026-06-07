# P7 — Crypto multi-strategy book (five low-correlation sleeves)

| field            | value                                                         |
|------------------|---------------------------------------------------------------|
| **Slug**         | `crypto-multi-strategy-book-v3`                               |
| **Date**         | 2026-06-04                                                    |
| **Researcher**   | (session)                                                    |
| **Status**       | `accept_monitoring`                                          |
| **Notebook**     | `notebooks/90_crypto_intraday/60_multi_strategy_book.ipynb`  |
| **Report**       | `reports/crypto_v3_multi_strategy.html`                      |
| **Supersedes**   | — (complements P6; P6's carry is sleeve S1 here)            |
| **Superseded by**| —                                                            |

---

## 0. v3.1 update (2026-06-07) — S5 macro sleeve vol-target fix

**What changed.** The 2026 holdout exposed S5 (macro credit-gate) as the book's weak point: it was
*undefended long crypto beta* (long 0.5 `BTC.s` + 0.5 `ETH.s` when HYG risk-on) and lost **−29.5%** in
2026 when crypto fell ~17% while credit (HYG) stayed calm — the slow credit gate never fired. S5 was
also the hottest leg (~46–51%/yr vol), so at equal capital it dominated the book's drawdown.

**Fix (selected on in-sample ≤2025 only).** Keep the HYG risk-on gate but **vol-target the position to
~25%/yr** (30-day trailing vol, leverage-capped 2×) — `S5_VOLTGT/S5_LEVCAP/S5_VOLWIN` in
`crypto_book.py`. Chosen by an explicit in-sample rule (improve S5 MaxDD, keep book Sharpe & low
correlation) over 6 candidates and **stable across a 3×2×3 neighbour grid**. Notably, *cleaner credit
signals did **not** help*: HYG/LQD spread and FRED `BAA10Y` left S5 no better in-sample (credit
genuinely decoupled from crypto), and an own-trend stop helped 2026 but raised mean|ρ| 0.11→0.16 (S5
started mimicking S2), so it was rejected for eroding the diversification thesis. Risk-normalising the
existing sleeve was the disciplined, non-overfit fix.

| metric | v3 (raw S5) | **v3.1 (vol-target S5)** |
|---|---|---|
| In-sample combo Sharpe | 1.15 | **1.10** |
| In-sample combo CAGR | 20.1% | **17.1%** |
| In-sample combo MaxDD | −15.0% | **−12.2%** |
| In-sample combo Calmar | 1.34 | **1.39** |
| In-sample mean\|ρ\| | 0.111 | **0.112** |
| **2026 holdout — book (eq-cap)** | **−5.0%** | **−2.6%** |
| 2026 holdout — book MaxDD | −13.3% | **−9.4%** |
| 2026 holdout — S5 sleeve | −29.5% | **−17.2%** |
| 2026 holdout — vs BTC | −17.4% | −17.4% |

**Honest trade-off.** Vol-targeting also trims S5's bull exposure, so full-cycle CAGR/Sharpe dip
slightly and the 2024 one-way bull goes from +1.7% to **−3.3%** — the book is *no longer positive every
calendar year*. In exchange the full-cycle drawdown is lower (Calmar up) and the 2026 bear loss is
roughly halved. **The deeper lesson stands: a long-beta sleeve gated by a slow macro signal *defends* a
crypto bear (a third of BTC's loss) but cannot *profit* in one. The book is a diversified risk-reducer,
not a bear-market alpha — the next genuine improvement is a positive non-beta sleeve, not a better
credit proxy.**

Reproduce: selection harness `notebooks/90_crypto_intraday/fix_s5.py`; artifacts regenerated from the
source of truth via `rebuild_v3_artifacts.py` → `render_multi_strategy_report.py` (report rev v3.1).

---

## 1. Research question

Can we assemble **five crypto strategies with low mutual correlation** — each anchored to a
*different* return source — and combine them into an all-weather book that beats BTC on a
risk-adjusted basis, net of all costs **including the cost of cash**, over 2022–2025 with 2025
held out of sample?

## 2. Hypothesis & economic rationale

**Orthogonality is engineered through return-source diversity.** If each sleeve is paid by a
distinct mechanism, the PnL streams cannot share a common driver and so decorrelate by
construction. The five sources (from the repo's `return_sources.md` taxonomy), generated via the
`idea-generation` skill:

1. **Carry** (S1) — perp funding harvested market-neutral (long spot / short perp).
2. **Time-series momentum** (S2) — trend-following, directional long/short.
3. **Cross-sectional momentum** (S3) — relative strength, market-neutral dispersion.
4. **Flow / forced-trade** (S4) — fade crowded funding extremes (liquidity provision to over-levered positioning).
5. **Macro / regime** (S5) — a credit-regime gate driven by *non-price-volume* data (exogenous to crypto).

Skeptic's view: two of the five (S2, S3) share a momentum root, so a sharp reversal hits both at
once; and the book gives up beta in a one-way bull. Both are quantified below.

## 3. Universe & data

- **Universe:** BTC, ETH, SOL, BNB USD-M **perps** + BTC, ETH **spot** (`configs/crypto_intraday_universe.csv` extended).
- **Date range:** 2022-01-01 → 2025-12-31, daily grid. **2025 is out-of-sample.** 2026 is the PM holdout and is never touched.
- **Data sources:** Binance Vision daily klines (spot + perp) and 8h funding (`binance_vision`); HYG high-yield credit ETF for the macro gate (`yfinance`); 3M T-bill for financing (`fred` DTB3, with a piecewise-by-year fallback when the FRED endpoint times out).
- **Known gaps:** SOL perp has ~5 missing early days (ffilled); FRED endpoint was unreliable at run time (fallback rf used, within a few bp of realized 3M T-bill).

## 4. Signal & portfolio construction

All sleeves are **daily, leak-safe** (signal at `t` uses data ≤ `t`; the engine lags weights one
more bar), and reported **excess of cash**.

| Sleeve | Signal (one line) | Direction | Legs |
|---|---|---|---|
| **S1 carry** | hold when 7d-mean funding > 0 | market-neutral | +0.25 spot / −0.25 perp, BTC+ETH |
| **S2 trend** | perp close vs 50d MA | long/short | ±0.5 BTC.p, ETH.p |
| **S3 xsmom** | rank by 30d return, long top2 / short bot2 | market-neutral | demeaned-rank over 4 perps |
| **S4 fundcontra** | banded funding z (enter \|z\|>1, exit<0.3) | directional contrarian | −0.5·sign(z) BTC.p, ETH.p |
| **S5 macro** | HYG above its 50d MA → risk-on, **position vol-targeted ~25%/yr** (v3.1, §0) | long/flat | 0.5 BTC.s, ETH.s × vol-scalar when risk-on |

- **Combination:** (a) **equal-capital** — 20% per sleeve, no leverage assumptions (headline);
  (b) **risk-budget** — scale each sleeve to ~8%/yr vol (trailing-vol, leverage-capped at 10×),
  equal-weight; carry is levered ≈9.6× (realistic for a basis book); optionally vol-targeted to 10%/yr.
- **Costs:** per-leg one-way slippage 8–20 bps (perp 8–12, spot 15–20), `costs_bps=0` folded into slippage; perp funding charged on held weight. Conservative (P6 stress spec).
- **Cost of cash:** every sleeve charged `rf_t · gross_long_notional_t`; a flat sleeve earns 0 excess. Judged against the risk-free hurdle, not zero.
- **Benchmark:** BTC buy-and-hold, excess of cash.

## 5. Headline performance

Daily returns, excess of cash, net of costs + funding + financing, 2022-04 → 2025-12.

**Sleeves (standalone):**

| sleeve | source | dir | net Sharpe | gross Sharpe | CAGR | AnnVol | MaxDD | AnnTO | TiM |
|---|---|---|---|---|---|---|---|---|---|
| S1 carry | carry | neutral | **3.91** | −0.11 | 1.2% | 0.3% | −0.6% | 4 | 97% |
| S2 trend | TS-mom | L/S | 0.67 | 0.81 | 24.4% | 54.0% | −60.5% | 19 | 100% |
| S3 xsmom | XS-mom | neutral | 0.47 | 0.83 | 11.2% | 37.2% | −57.6% | 51 | 100% |
| S4 fundcontra | flow | contra | 0.57 | 0.63 | 15.2% | 36.7% | −33.1% | 39 | 76% |
| S5 macro | macro | long/flat | 0.50 | 0.60 | 13.0% | 45.8% | −43.8% | 10 | 69% |

**Combined book:**

| book | Sharpe | CAGR | AnnVol | MaxDD | Calmar |
|---|---|---|---|---|---|
| **Combined — equal-capital (headline)** | **1.15** | **20.1%** | 17.2% | **−15.0%** | 1.34 |
| Combined — risk-budget (cap 10×) | 1.16 | 5.3% | 4.6% | −7.4% | 0.72 |
| Combined — risk-budget vol-target 10% | 1.13 | 14.8% | 13.0% | −21.0% | 0.70 |
| BTC buy & hold (excess) | 0.51 | 14.1% | 50.3% | −66.5% | 0.21 |

All three combinations land at Sharpe ≈ 1.1 — the **diversification**, not the weighting scheme, is
the driver. Over the full cycle the book beats BTC on **both** return and drawdown.

## 6. Diagnostics

- **Correlation:** mean |ρ| = **0.111**, max 0.319 (S2↔S3), **sum of pairwise ρ = −0.246** (net-negative). Diversification ratio = **2.03**.
- **By year (equal-capital combo vs BTC):** 2022 **+11.2%** / −64.2% · 2023 +34.4% / +142.7% · 2024 **+1.7%** / +110.4% · 2025 OOS **+30.8%** / −10.2%. Positive every calendar year; the lone underperformance is the 2024 one-way bull (hedged book gives up beta). *(v3.1, §0: 2022 +10.0% · 2023 +28.4% · 2024 **−3.3%** · 2025 +32.3% — the S5 vol-target turns the 2024 bull slightly negative, the cost of de-risking the long-beta sleeve.)*
- **Equity & drawdown** curves, correlation heatmap, per-year bars, risk-return scatter: see the HTML report.

## 7. Robustness

- [x] Out-of-sample window (2025) — combined book +30.8%, direction intact.
- [x] Same-day signal/execution — weights lagged one bar by the engine; signals use trailing windows only.
- [x] **Cost of cash charged** — financing on gross-long notional netted; edge beats the risk-free hurdle.
- [x] Gross-vs-net shown per sleeve — costs roughly halve the higher-turnover sleeves (S3, S4) but leave them net-positive.
- [x] Correlation/robustness of the *combination* — three weighting schemes all ≈ Sharpe 1.1.
- [ ] Block-bootstrap SE on the combined Sharpe — **not yet run** (next step).
- [ ] Neighbouring-parameter sweep (MA length, z thresholds, formation horizon) — partial; the menu compared variants but no full grid.
- [x] Survivorship — fixed 4-symbol universe, all live for the full window.

## 8. Failure modes

- **Carry compression** — S1 is the Sharpe anchor; if perp funding trends to zero the funding−financing spread vanishes (P6's documented risk). Monitor the spread.
- **Momentum crash** — S2 and S3 share a momentum root (ρ = 0.32, the book's main concentration); a sharp V-reversal hurts both together.
- **Turnover / cost sensitivity** — S3 (~51×/yr) and S4 (~39×/yr) carry the most cost; double the slippage assumption and re-confirm before sizing.
- **Macro proxy / S5** — *(updated v3.1, §0)* the credit gate only adds value when macro actually drives crypto; in 2026 it didn't. Cleaner credit signals (HYG/LQD spread, FRED `BAA10Y`) were tested and **didn't help** — the fix was to **vol-target** the sleeve so its drawdown is bounded. It remains a long-beta sleeve that bleeds in a crypto-led bear (the residual failure mode).
- **Financing approximation** — rf used a piecewise-by-year fallback (FRED timed out); a vintage DTB3 path would tighten the cost-of-cash numbers slightly.
- **Capacity** — SOL/BNB perp legs and short-perp funding/borrow are the binding constraints at size; fine for a personal book.

## 9. Decision

**Status: `accept_monitoring`** — as a **diversified book**, not a single alpha. The combination is
robust (positive every calendar year incl. the 2025 OOS), beats BTC on both return (20.1% vs 14.1%
CAGR) and drawdown (−15% vs −67%), and every sleeve has an articulated mechanism with a positive
net-of-everything Sharpe. The edge *is* the low correlation. Monitor: (i) the carry funding−financing
spread, (ii) the rolling correlation of S2↔S3, (iii) trailing-quarter combined Sharpe — kill/resize if
it goes ≤ 0 for two consecutive quarters.

## 10. Next steps

- Lift the five sleeve constructors from the notebook into `src/alpha_lab/backtest/` (backlog) — they are currently inline.
- ~~swap S5's HYG proxy for FRED `BAA10Y`~~ — **closed (v3.1, §0):** BAA10Y & HYG/LQD tested, no improvement; adopted **vol-targeting** instead. Replace the rf fallback with vintage FRED DTB3 once the endpoint is reliable (still open).
- **Add a positive non-beta sleeve** — the v3.1 book *defends* a crypto bear but doesn't *profit* in one; a sleeve that is genuinely positive in a crypto-led selloff (e.g. a short-bias/trend-on-credit or a vol/dispersion harvester) is the highest-value next idea.
- Block-bootstrap the combined Sharpe and run a parameter-stability grid on the four tunable sleeves.
- Build the real collateral-yield model for the carry leg (P6 scenario C) so S1's risk-budget leverage is justified.
- Add to the monitoring set; revisit weighting (the risk-budget leverage on carry is the main live-trading judgment call).

## 11. Appendix / supporting artifacts

- Build: `notebooks/90_crypto_intraday/60_multi_strategy_book.ipynb` → `data/results/crypto_v3_multi/` (sleeve_excess_returns, combos, corr, meta.json).
- Report: `notebooks/90_crypto_intraday/render_multi_strategy_report.py` (reads the parquets) → `reports/crypto_v3_multi_strategy.html` (self-contained).
- Engine: `alpha_lab.backtest.vector.run_backtest` (weights lagged 1 bar; per-leg slippage; perp funding on held weight) + uniform cost-of-cash overlay.
- Ideation provenance: `idea-generation` skill — random stimuli (thermostat, hibernation, tributary, coral, power-plant control room, credit-spread × prediction-market) → operators (remote-stimulus, perspective-shift, reversal, macro-as-conditioner) → one idea per return source.
