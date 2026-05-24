# P5 — Cross-regime + Time-of-day + P&L control + ML ensemble

**Slug**: `crypto_intraday/P5-regime-tod-ml-pnl`
**Date**: 2026-05-24
**Researcher**: Kelvin Chen
**Status**: accept_monitoring with revised verdict — see below
**Notebook**: [40_regime_tod_pnl_ml.ipynb](../../../notebooks/90_crypto_intraday/40_regime_tod_pnl_ml.ipynb)
**ML follow-up script**: `reports/_p5_ml_fix.py` (gitignored convenience)

## Mission

Address the four open questions left by FINAL v1:
1. Cross-regime validation (the largest gap).
2. Asia / US time-of-day exploitation.
3. P&L control on long-holding strategies (`funding_momentum` had 89d mean holding).
4. ML signal ensemble (meta-model on strategy signals, not raw features).

## Regime windows + actual market behavior

| regime label   | window                | BTC return | ETH return | true character           |
|----------------|-----------------------|-----------:|-----------:|--------------------------|
| `bear_2022Q1`  | 2022-01-01→2022-04-01 |    -1.60%  |   -10.93%  | mild-bear / chop drawdown (ETH was the real bear) |
| `chop_2023H1`  | 2023-01-01→2023-07-01 |   +84.18%  |   +61.65%  | **STRONG bull recovery, not chop** (FTX-bottom rebound) |
| `bull_2024H1`  | 2024-01-01→2024-07-01 |   +48.27%  |   +50.48%  | clear bull                |

**Caveat that materially affects the read**: the user-supplied label `chop_2023H1` doesn't match reality — H1 2023 was a strong bull recovery from the FTX-crash bottom (BTC ~$16k → ~$30k). The three windows we tested are effectively `mild_bear / strong_bull / strong_bull`, not the intended `bear / chop / bull`. A future P6 should add a true chop window (e.g., H2 2023 Aug-Dec, where BTC traded $25-45k with multiple reversals).

This caveat does not invalidate the findings below but does mean "the strategies were tested across two bulls and one mild bear", not "across three distinct regimes."

## 1. Per-regime backtest at perp_stress

Sharpe by (strategy, regime):

| strategy                | bear_2022Q1 | chop_2023H1 (bull) | bull_2024H1 |
|-------------------------|------------:|-------------------:|------------:|
| **funding_contrarian**  |    **+0.73** |          **+2.16** |    **+1.12** |
| funding_momentum        |       -3.31 |              +1.75 |       +1.56 |
| rsi_mr_threshold        |       -2.40 |              -2.12 |       -1.43 |
| bollinger_mr_threshold  |       -2.44 |              -3.93 |       -2.27 |
| spread_z_pair_trade     |       -2.49 |              -7.23 |       -5.08 |
| beta_residual_mr        |       -4.89 |              -8.86 |       -5.24 |

Net total by (strategy, regime):

| strategy                | bear_2022Q1 | chop_2023H1 (bull) | bull_2024H1 |
|-------------------------|------------:|-------------------:|------------:|
| **funding_contrarian**  |   **+5.12%** |          **+28.04%** |   **+15.01%** |
| funding_momentum        |     -44.59% |             +44.07% |     +40.82% |
| rsi_mr_threshold        |     -14.78% |             -22.77% |     -15.25% |
| bollinger_mr_threshold  |     -21.04% |             -43.03% |     -29.75% |
| spread_z_pair_trade     |      -8.04% |             -25.82% |     -29.94% |
| beta_residual_mr        |     -22.84% |             -46.62% |     -34.46% |

**Headline finding (re-rates the v1 verdict)**:
- **`funding_contrarian` is positive in EVERY regime** — the only strategy that survives the bear period.
- **`funding_momentum` breaks in bear** (-3.31 Sharpe, -44.6% net) — confirms the v1 caveat. The strategy was a bull-regime carry trade in disguise.
- All TA / cross-asset strategies are negative in all three regimes — robust rejection of the v1 set.

## 2. Time-of-day decomposition (session_Sharpe at perp_stress)

### funding_contrarian (the robust survivor)

| session | bear_2022Q1 | chop_2023H1 (bull) | bull_2024H1 |
|---------|------------:|-------------------:|------------:|
| Asia (0-8 UTC)    | -0.68 | +0.70 | -0.94 |
| Europe (8-13 UTC) | +0.46 | +0.41 | -0.09 |
| **US (13-21 UTC)**| **+2.05** | **+1.48** | **+2.68** |
| Late (21-24 UTC)  | -1.26 | +1.75 | -0.50 |

**`funding_contrarian` is overwhelmingly US-hours driven.** Session-Sharpe in US hours: +2.05 / +1.48 / +2.68 across regimes. Other sessions are mixed or negative.

Plausible mechanism: funding rates are set at 00, 08, 16 UTC. By 13-21 UTC (US session), the most recent rate event has had time to inform positioning; US-session traders are the dominant taker flow in BTC/ETH perp, and contrarian positioning against extreme funding is most "right" when their flow is the marginal participant.

This is a **session-gating opportunity**: a `funding_contrarian × US-hours` variant should improve Sharpe materially without changing the underlying signal. Deferred to P6.

### funding_momentum

| session | bear_2022Q1 | chop_2023H1 (bull) | bull_2024H1 |
|---------|------------:|-------------------:|------------:|
| Asia    | -2.57 | +1.03 | +0.87 |
| Europe  | +0.34 | +0.06 | +1.25 |
| US      | -2.19 | +1.40 | -0.67 |
| Late    | -1.58 | +0.62 | +3.12 |

**funding_momentum is session-AND-regime dependent**. The "Late" session (21-24 UTC) is the only one with a strongly positive Sharpe in bull 2024 (+3.12), but it's negative in bear. No clean session gating works across regimes.

### Mean-reversion (bollinger_mr / rsi_mr)

Both are negative in nearly every (session × regime) cell; the rare positives (Asia/Late in bull 2024) are tiny. The v1 decision (reject mean-reversion at 1h sampling) holds at session-level too.

## 3. P&L control overlay on funding_momentum

Variants tested: stop-loss only (5%, 10%), profit-take only (20%), combined (5% / 20%), with optional 7-day hard time-out, plus pure time-outs (7d, 30d).

Sharpe by (variant, regime):

| variant              | bear_2022Q1 | chop_2023H1 (bull) | bull_2024H1 |
|----------------------|------------:|-------------------:|------------:|
| vanilla              |       -3.31 |              +1.75 |       +1.56 |
| sl_5pct              |       -3.87 |              +1.70 |       +1.48 |
| sl_10pct             |       -3.03 |              +1.68 |       +1.56 |
| pt_20pct             |       -3.38 |              +1.65 |       +1.57 |
| sl5_pt20             |       -3.99 |              +1.28 |       +1.34 |
| sl5_pt20_to168h      |       -3.98 |              +1.31 |       +1.85 |
| timeout_7d           |       -2.78 |              +1.39 |       +1.57 |
| **timeout_30d**      |     **-3.37** |          **+1.82** |     **+1.84** |

**Findings**:
- **Stop-loss is counterproductive** in bull/chop (reduces Sharpe by a few decimal points) and only marginally helps in bear (sl_10pct: -3.03 vs vanilla -3.31).
- **Profit-take is neutral-to-slightly-bad** in every regime (it cuts the right tail of the bull distribution).
- **`timeout_30d` is the best variant**: improves bull (+1.84 vs +1.56) and chop (+1.82 vs +1.75) without making bear meaningfully worse (-3.37 vs -3.31).
- **`timeout_7d`** reduces bear damage (-2.78 vs -3.31) but loses chop/bull alpha (down from +1.75/+1.56 to +1.39/+1.57). A risk trade-off worth considering if bear-period drawdowns are unacceptable.

Mechanism: the long holding period of vanilla `funding_momentum` is both the source of its low-turnover edge AND the source of its bear vulnerability. A 30-day cap forces periodic re-entry, which in bull/chop reduces over-extension; in bear it doesn't help because the strategy keeps re-entering long into a falling market.

**P&L control alone doesn't fix `funding_momentum`'s bear problem.** A regime filter (volatility, funding-rate magnitude, drawdown-derisk) is the right next experiment — but that's P6.

## 4. ML signal ensemble (meta-model on strategy signals)

### Setup

- Features: lagged signal series from each of the 6 strategies in section 1 + hour_of_day + day_of_week (8 features per symbol).
- Target: forward 1h direction.
- Train/test scheme: **leave-one-regime-out**. For each held-out regime, train on the other two combined.
- Two models: logistic regression, LightGBM.
- Threshold 0.55 for class 1, 0.45 for class -1, else flat. Per-symbol prediction → 2-col weight matrix → backtest at perp_stress.

### Results

Sharpe by (model, held-out OOS regime):

| model    | bear_2022Q1 (OOS) | chop_2023H1 (OOS) | bull_2024H1 (OOS) |
|----------|------------------:|------------------:|------------------:|
| logistic |             +0.65 |             -0.09 |             -0.27 |
| lightgbm |             -7.09 |             -9.04 |             -7.30 |

Net total:

| model    | bear (OOS) | chop (OOS) | bull (OOS) |
|----------|-----------:|-----------:|-----------:|
| logistic |     +3.53% |     -0.69% |     -1.80% |
| lightgbm |    -54.71% |    -72.06% |    -67.46% |

Avg holding hours: logistic 0.9-2.7h, LightGBM ~1h across.

### Reading

- **LightGBM massively over-fits.** With 8 features and only ~6.5k training bars (bear+chop or chop+bull combined), the tree ensemble memorizes the training periods and predicts directionally-wrong on held-out data. The negative-7 to -9 Sharpe is "actively losing money fast" rather than "no signal".
- **Logistic regression is near-zero across the board.** Marginally positive on bear-OOS only — it has learned a small bias from the bear+bull training set that happens to look like a "stay flat / lean short" rule, which barely works on the mild bear.
- **No model beats `funding_contrarian`** in any regime:
  - bear: logistic +0.65 < funding_contrarian +0.73
  - chop: logistic -0.09 < funding_contrarian +2.16
  - bull: logistic -0.27 < funding_contrarian +1.12

**Decision: ML ensemble REJECTED.** Same rule as P3: ML must beat simple baselines in every regime, else reject.

The honest story: meta-modeling strategy signals didn't add value here because (a) `funding_contrarian` is already a sharp threshold-based signal, (b) the other strategies in the input set are negative-edge anyway, and (c) the training data is small enough that complex models over-fit and simple models don't have enough handles.

## 5. Revised strategy table (v2)

| strategy                       | decision (v2)         | notes                                                            |
|--------------------------------|-----------------------|------------------------------------------------------------------|
| **funding_contrarian**         | **accept_monitoring** | positive Sharpe in all 3 regimes (0.73 / 2.16 / 1.12). PRIMARY candidate. |
| **funding_contrarian × US-hours** | **deferred (likely accept)** | session decomposition suggests a US-hours gate would lift Sharpe materially. Not yet implemented. |
| funding_momentum               | downgrade to **needs_revision** | breaks in bear (-3.31 Sharpe, -44.6% net). Was v1's headline; v2 demotes it. |
| funding_momentum + timeout_30d | needs_revision (best variant) | improves bull/chop without fixing bear. Recommended over vanilla. |
| funding_momentum + timeout_7d  | risk-managed variant  | trades bull/chop Sharpe for less-bad bear. PM choice. |
| logistic / LightGBM ensemble   | reject                | does not beat funding_contrarian in any regime                   |
| All TA / cross-asset           | reject (reconfirmed)  | negative in all regimes                                          |

## 6. Failure modes / caveats

- **Regime labeling problem**: `chop_2023H1` was actually a strong bull. The three tested windows are mild_bear / strong_bull / strong_bull. Cross-regime conclusions are biased toward "what works in bulls", with one mild-bear data point. **A true chop period must be added in P6.**
- **funding_contrarian's bear edge is small** (Sharpe +0.73 over 3 months = ~1100 1h bars). Wider CI than the bull-regime estimates. The bear positive could be noise; needs validation on additional mild-bear windows.
- **Per-symbol asymmetry in bear**: BTC fell only -1.6% in Q1 2022 but ETH fell -10.9%. `funding_momentum`'s long-bias was a fatal pair-trade-the-wrong-way in that period for ETH. Symbol-by-symbol breakdown is a P6 enhancement.
- **ML ensemble's small training set**: ~6.5k bars per held-out regime. LightGBM clearly over-fits; logistic is starved. A live setting with multi-year continuous training would likely improve both. Out of scope here.
- **Session-gating not yet implemented as a tradable backtest**: section 2 reports session-conditional Sharpes, not a backtested session-gated strategy. Session gating could leak via lookahead if not careful — done correctly, it's the natural P6 next step.
- **No PM holdout was accessed** — 46 audit events, 0 raises.

## 7. Revised final recommendation

**MONITOR with these updates from FINAL v1**:
1. **Primary candidate is now `funding_contrarian`**, not `funding_momentum`. funding_contrarian is the only strategy positive in all three tested regimes (including the mild bear). Net stress: +5% / +28% / +15%. Lower per-window magnitude but real regime stability.
2. **`funding_momentum` is downgraded** from "accept_monitoring" to "needs_revision". It works in bull/chop but loses ~44% in a mild bear. Without a regime filter it is not a safe live candidate.
3. **Recommended next experiment** (P6): a true chop window + a US-hours-gated `funding_contrarian` variant + a vol-regime/drawdown filter on `funding_momentum`. If both signals improve under those overlays and survive a real chop, escalate the candidate to `accept` and consider unlocking PM holdout for a single final-pass evaluation.
4. **ML and TA-style rules remain rejected** with reinforced evidence (now across three regimes, not just one).
5. **Do not unlock PM holdout (2026-01 → 2026-05)** based on this evidence. The discipline harness is intact (46 events, 0 raises).

The framework continues to surface evidence-based verdicts. The strategy story has materially changed (different primary candidate, different mechanism story). The framework's value is exactly this — it lets you walk back a v1 verdict with clean data.

## Appendix

- Main notebook: [40_regime_tod_pnl_ml.ipynb](../../../notebooks/90_crypto_intraday/40_regime_tod_pnl_ml.ipynb).
- ML ensemble fix (leave-one-regime-out): `reports/_p5_ml_fix.py` (gitignored).
- PM holdout audit at end of analysis: 46 events, 0 raises.
