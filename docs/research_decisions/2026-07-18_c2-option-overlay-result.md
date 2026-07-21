# C2 defensive sector + SPY option overlay — result

## Header

| field | value |
|---|---|
| **Date** | 2026-07-18 |
| **Status** | `reject` |
| **Protocol** | `docs/research_decisions/2026-07-18_c2-option-overlay-protocol.md` |
| **Report** | `reports/us_sector_c2_option_overlay.html` |
| **Development** | 2013-01-01 through 2021-12-31 |
| **Known stress** | 2022; not untouched out-of-sample |
| **Sealed** | Every observation dated 2023-01-01 or later |

## Verdict

**Reject all five frozen C2 + SPY-option overlays for the 8% / low-volatility / low-drawdown
objective.** No candidate passed every preregistered development gate. O2 was the closest return
candidate, but its result was neither low-volatility enough nor sufficiently stable after multiple-test
adjustment. No candidate advances to the sealed 2023+ sample.

## Headline results

| Strategy | CAGR | Stressed CAGR | Volatility | Maximum drawdown | Ulcer | Calmar | Rolling 5y target |
|---|---:|---:|---:|---:|---:|---:|---:|
| O1 matched 95/85 put spread | 6.56% | 5.57% | 10.21% | -13.93% | 5.63% | 0.47 | 31.05% |
| O2 calm-VIX 95/85 put spread | 8.49% | 7.92% | 10.42% | -13.71% | 5.09% | 0.62 | 38.10% |
| O3 put spread + partial call | 6.52% | 5.47% | 9.94% | -13.58% | 5.46% | 0.48 | 29.56% |
| O4 matched 95/110 collar | 6.34% | 5.46% | 9.31% | -15.35% | 6.10% | 0.41 | 34.82% |
| O5 half-notional long put | 6.95% | 6.45% | 10.29% | -14.01% | 5.42% | 0.50 | 25.50% |
| C2 unhedged | 10.02% | 8.77% | 12.53% | -20.33% | 4.62% | 0.49 | 61.31% |
| Synthetic SPY collar | 10.64% | 10.64% | 9.39% | -13.92% | 4.06% | 0.76 | 45.83% |

The stressed overlay CAGR doubles modeled option entry haircuts from 10% to 20%. C2's stressed
column uses its previously frozen 20 bp implementation-cost case.

## Why the overlays failed

- Protection improved the deepest development drawdown by 4.98–6.75 percentage points, but every
  overlay had a worse Ulcer Index than C2. The strategies fell less deeply but spent longer below
  their prior peaks because insurance carry slowed recovery.
- Relative to C2, CAGR fell by 1.52 percentage points for O2 and by 3.06–3.67 points for the other
  four structures. Explicit option entry-spread drag was 0.47–0.99 points a year; premium decay,
  surrendered call upside, and SPY/sector basis explain the rest of the modeled return loss.
- O2 failed five frozen gates: volatility (10.42% versus a 10% ceiling), Ulcer improvement, return
  within one point of the collar benchmark, 60% rolling-five-year target attainment, and 95%
  deflated-Sharpe probability. Its DSR was only 69.51% after charging 16 related trials.
- Prior-close regime attribution shows O2 reduced annualized log-return contribution in calm-VIX
  observations from C2's 5.75% to 3.97%, while adding only about 0.31 points in elevated-VIX
  observations. The insurance payment was persistent; the modeled crisis benefit was episodic.
- The call-financed structures did not solve this trade-off. O3 and O4 improved the known-2022
  return, but gave up enough development-period right-tail return to miss the long-run objective.

## Known 2022 stress

| Strategy | 2022 return | Spread-stress return | Volatility | Maximum drawdown |
|---|---:|---:|---:|---:|
| O2 calm-VIX put spread | -3.74% | -3.74% | 9.22% | -10.15% |
| O3 put spread + partial call | -2.66% | -3.76% | 7.21% | -7.71% |
| O4 matched collar | -2.44% | -3.41% | 6.69% | -7.34% |
| C2 unhedged | -3.74% | -4.25% | 9.22% | -10.15% |
| Synthetic SPY collar | -15.52% | — | 13.90% | -16.63% |
| SPY | -18.18% | — | 24.24% | -24.50% |

O2 exactly matched unhedged C2 in 2022 because the prior-VIX gate did not open new quarterly put
spreads once volatility was already elevated. O4 had the best 2022 protection, but its development
CAGR was only 6.34% and its development maximum drawdown breached the -15% gate. The 2022 sample
was already known when this overlay study was designed and cannot be counted as fresh validation.

## Data, costs, and audit

- C2 sector signals, cash allocation, lagged execution, and costs were unchanged from the frozen
  sector study. Dynamic option notional and the VIX gate used the prior close.
- Options used raw SPY spot and European Black–Scholes/VIX proxy marks. Long puts used a conservative
  upside IV buffer, short puts a steeper skew buffer, and calls a lower IV; option entry haircuts and
  financing on overlay cash were charged explicitly.
- Historical option NBBO, discrete strike grids, early exercise, dividends, and exact SPY/sector beta
  matching were unavailable. The study is a model-risk screen, not an executable option-chain backtest.
- Static leakage scan: zero blockers. Quantile warnings are ex-post CVaR metrics only. Rolling and
  forward-fill findings use trailing observations or carry already-observed values to later dates.
- The fixed legacy nine-sector universe avoids changing constituents after seeing the result. No
  2023+ observation was loaded.

## Decision and next step

Stop this frozen batch. Retain O2 as a diagnostic reference, not a tradable recommendation, and do
not tune the VIX threshold, strikes, expiries, or call ratio using development or known-2022 results.
If the 8% objective remains binding, the next preregistered study should add a genuinely diversifying
return source rather than relying on permanent option insurance to create return. Any future option
candidate must first be validated against historical executable option-chain quotes and sector/SPY
basis exposure before paper trading.

## Supporting artifacts

- `reports/us_sector_c2_option_overlay.html`
- `data/results/us_sector_c2_option_overlay_metrics.csv`
- `data/results/us_sector_c2_option_overlay_2022_stress.csv`
- `scripts/us_sector_c2_option_overlay_study.py`
- `src/alpha_lab/backtest/collar.py`
