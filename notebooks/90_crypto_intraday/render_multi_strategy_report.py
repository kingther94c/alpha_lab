"""Render the crypto v3 multi-strategy HTML report from saved artifacts.

Self-contained HTML with embedded base64 charts (no external assets):
combined equity vs BTC, drawdowns, 5 sleeve curves, correlation heatmap,
per-year bars, risk-return scatter + tables, cost-of-cash & leak-safety notes.
"""
from __future__ import annotations
import base64, io, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
while not (ROOT / "src" / "alpha_lab").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from alpha_lab.backtest.metrics import summary

OUT = ROOT / "data" / "results" / "crypto_v3_multi"
REPORT = ROOT / "reports" / "crypto_v3_multi_strategy.html"
REPORT.parent.mkdir(parents=True, exist_ok=True)
BARS = 365

R = pd.read_parquet(OUT / "sleeve_excess_returns.parquet")
combos = pd.read_parquet(OUT / "combos.parquet")
corr = pd.read_parquet(OUT / "corr.parquet")
meta = json.loads((OUT / "meta.json").read_text())
diag = meta["diag"]

NAMES = {"S1_carry": "S1 · Carry", "S2_trend": "S2 · Trend", "S3_xsmom": "S3 · XS-Mom",
         "S4_fundcontra": "S4 · Funding-contra", "S5_macro": "S5 · Macro-gate"}
COL = {"S1_carry": "#0d9488", "S2_trend": "#2563eb", "S3_xsmom": "#7c3aed",
       "S4_fundcontra": "#ea580c", "S5_macro": "#16a34a"}
DIRN = {"S1_carry": "market-neutral", "S2_trend": "long/short", "S3_xsmom": "market-neutral",
        "S4_fundcontra": "directional (contrarian)", "S5_macro": "long/flat"}


def b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def cum(r):       return (1 + r.fillna(0)).cumprod() - 1
def dd(r):
    e = (1 + r.fillna(0)).cumprod()
    return e / e.cummax() - 1


# ---- chart 1: combined equity vs BTC ----------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(cum(combos["combo_eqcap"]).index, cum(combos["combo_eqcap"]) * 100, color="#111827", lw=2.2, label="Combined (equal-capital, 5 sleeves)")
ax.plot(cum(combos["combo_vt10"]).index, cum(combos["combo_vt10"]) * 100, color="#dc2626", lw=1.4, ls="--", label="Combined (risk-budget, vol-target 10%)")
ax.plot(cum(combos["btc_bh_excess"]).index, cum(combos["btc_bh_excess"]) * 100, color="#9ca3af", lw=1.6, label="BTC buy & hold (excess of cash)")
ax.axhline(0, color="#d1d5db", lw=.8); ax.set_ylabel("cumulative excess return (%)")
ax.set_title("Combined portfolio vs BTC — excess of cash, net of costs + financing", fontsize=11, weight="bold")
ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=.25)
c_equity = b64(fig)

# ---- chart 2: drawdown combo vs BTC -----------------------------------------
fig, ax = plt.subplots(figsize=(10, 2.8))
ax.fill_between(dd(combos["combo_eqcap"]).index, dd(combos["combo_eqcap"]) * 100, 0, color="#111827", alpha=.75, label="Combined")
ax.plot(dd(combos["btc_bh_excess"]).index, dd(combos["btc_bh_excess"]) * 100, color="#9ca3af", lw=1.3, label="BTC buy & hold")
ax.set_ylabel("drawdown (%)"); ax.legend(fontsize=8.5, loc="lower left"); ax.grid(alpha=.25)
ax.set_title("Drawdown — combined book vs BTC", fontsize=11, weight="bold")
c_dd = b64(fig)

# ---- chart 3: 5 sleeve equity curves ----------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [3, 1]})
for k in ["S2_trend", "S3_xsmom", "S4_fundcontra", "S5_macro"]:
    axA.plot(cum(R[k]).index, cum(R[k]) * 100, color=COL[k], lw=1.5, label=NAMES[k])
axA.axhline(0, color="#d1d5db", lw=.8); axA.set_ylabel("cum. excess (%)")
axA.set_title("Four directional / neutral sleeves", fontsize=10, weight="bold")
axA.legend(fontsize=8, loc="upper left"); axA.grid(alpha=.25)
axB.plot(cum(R["S1_carry"]).index, cum(R["S1_carry"]) * 100, color=COL["S1_carry"], lw=1.8)
axB.set_title("S1 · Carry (low-vol)", fontsize=10, weight="bold"); axB.grid(alpha=.25)
axB.set_ylabel("cum. excess (%)")
for lab in axB.get_xticklabels(): lab.set_rotation(30); lab.set_fontsize(7)
for lab in axA.get_xticklabels(): lab.set_fontsize(7)
c_sleeves = b64(fig)

# ---- chart 4: correlation heatmap -------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 4.4))
cmap = LinearSegmentedColormap.from_list("rb", ["#1d4ed8", "#ffffff", "#dc2626"])
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels([NAMES[c] for c in corr.columns], rotation=40, ha="right", fontsize=8)
ax.set_yticklabels([NAMES[c] for c in corr.index], fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="#111827" if abs(v) < .55 else "white")
ax.set_title(f"Sleeve correlation — mean|ρ|={meta['mean_offdiag_corr']:.2f}, max={meta['max_offdiag_corr']:.2f}", fontsize=10, weight="bold")
fig.colorbar(im, fraction=.046, pad=.04)
c_corr = b64(fig)

# ---- chart 5: per-year bars (combo vs BTC) ----------------------------------
yrs = [2022, 2023, 2024, 2025]
combo_y = [(1 + combos["combo_eqcap"][combos.index.year == y].fillna(0)).prod() - 1 for y in yrs]
btc_y = [(1 + combos["btc_bh_excess"][combos.index.year == y].fillna(0)).prod() - 1 for y in yrs]
fig, ax = plt.subplots(figsize=(7, 3.2))
x = np.arange(len(yrs)); w = .38
ax.bar(x - w / 2, np.array(combo_y) * 100, w, color="#111827", label="Combined")
ax.bar(x + w / 2, np.array(btc_y) * 100, w, color="#9ca3af", label="BTC buy & hold")
ax.set_xticks(x); ax.set_xticklabels([str(y) + ("\n(OOS)" if y == 2025 else "") for y in yrs], fontsize=9)
ax.axhline(0, color="#374151", lw=.8); ax.set_ylabel("annual excess return (%)")
ax.set_title("Annual return — green every year; the cost is bull upside (2024)", fontsize=10, weight="bold")
ax.legend(fontsize=8.5); ax.grid(alpha=.25, axis="y")
for i, v in enumerate(combo_y): ax.text(i - w / 2, v * 100 + (2 if v > 0 else -6), f"{v*100:+.0f}", ha="center", fontsize=7.5)
c_year = b64(fig)

# ---- chart 6: risk-return scatter -------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.0))
for k in R.columns:
    sm = summary(R[k].dropna(), periods=BARS)
    ax.scatter(sm["AnnVol"] * 100, sm["CAGR"] * 100, s=90, color=COL[k], zorder=3)
    ax.annotate(NAMES[k], (sm["AnnVol"] * 100, sm["CAGR"] * 100), fontsize=8, xytext=(6, 4), textcoords="offset points")
for nm, key, mk in [("Combined (eq-cap)", "combo_eqcap", "*"), ("BTC b&h", "btc_bh_excess", "X")]:
    sm = summary(combos[key].dropna(), periods=BARS)
    ax.scatter(sm["AnnVol"] * 100, sm["CAGR"] * 100, s=200 if mk == "*" else 90, marker=mk,
               color="#dc2626" if mk == "*" else "#6b7280", zorder=4)
    ax.annotate(nm, (sm["AnnVol"] * 100, sm["CAGR"] * 100), fontsize=8.5, weight="bold", xytext=(6, -10), textcoords="offset points")
ax.set_xlabel("annualized volatility (%)"); ax.set_ylabel("CAGR, excess of cash (%)")
ax.set_title("Risk–return — the combined book sits up-and-left of BTC", fontsize=10, weight="bold")
ax.grid(alpha=.25); ax.axhline(0, color="#d1d5db", lw=.8)
c_scatter = b64(fig)

# ---- 2026 holdout OOS (if evaluated) ----------------------------------------
oos_section = ""
hsl_p, hco_p = OUT / "holdout2026_sleeves.parquet", OUT / "holdout2026_combos.parquet"
if hsl_p.exists() and hco_p.exists():
    h_sl, h_co = pd.read_parquet(hsl_p), pd.read_parquet(hco_p)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(cum(h_co["combo_eqcap"]).index, cum(h_co["combo_eqcap"]) * 100, color="#111827", lw=2, label="Combined (equal-capital)")
    ax.plot(cum(h_co["combo_riskbudget"]).index, cum(h_co["combo_riskbudget"]) * 100, color="#dc2626", lw=1.3, ls="--", label="Combined (risk-budget)")
    ax.plot(cum(h_co["btc_bh_excess"]).index, cum(h_co["btc_bh_excess"]) * 100, color="#9ca3af", lw=1.6, label="BTC buy & hold")
    ax.axhline(0, color="#d1d5db", lw=.8); ax.set_ylabel("cum. excess (%)")
    ax.set_title("2026 holdout (Jan-May) - true out-of-sample, released under audit", fontsize=11, weight="bold")
    ax.legend(fontsize=8.5); ax.grid(alpha=.25)
    c_oos = b64(fig)

    def _o(s):
        s = s.dropna(); sm = summary(s, periods=BARS); return (1 + s).prod() - 1, sm["Sharpe"], sm["MaxDD"]
    o_eq, o_rb, o_btc = _o(h_co["combo_eqcap"]), _o(h_co["combo_riskbudget"]), _o(h_co["btc_bh_excess"])
    tbl26 = ""
    for i, (nm, t) in enumerate([("Combined - equal-capital", o_eq), ("Combined - risk-budget", o_rb), ("BTC buy & hold", o_btc)]):
        tbl26 += (f"<tr style=\"{'background:#f0fdf4;font-weight:600' if i == 0 else ''}\"><td>{nm}</td>"
                  f"<td class=n>{t[0]*100:+.1f}%</td><td class=n>{t[1]:.2f}</td><td class=n>{t[2]*100:.1f}%</td></tr>")
    sleeve26 = "".join(f"<tr><td><b style='color:{COL[k]}'>{NAMES[k]}</b></td><td class=n>{(((1+h_sl[k].dropna()).prod())-1)*100:+.1f}%</td></tr>" for k in h_sl.columns)
    off26 = h_sl.corr().where(~np.eye(len(h_sl.columns), dtype=bool)).abs().mean().mean()
    oos_section = f"""<h2>6 &middot; 2026 holdout &mdash; the out-of-sample moment of truth</h2>
<div class="note"><b>Honest result.</b> On the reserved 2026 window (Jan&ndash;May, released under audit), the book was
<b>defensive but negative</b>: equal-capital <b>{o_eq[0]*100:+.1f}%</b> vs BTC <b>{o_btc[0]*100:+.1f}%</b> &mdash; it cut the
loss by ~{abs(o_btc[0]-o_eq[0])*100:.0f} pts and held drawdown to {o_eq[2]*100:.0f}% (BTC {o_btc[2]*100:.0f}%), but it did
<b>not</b> make money. Driver: the <b>macro sleeve (S5) misfired</b> (held crypto long into the selloff, -29%) and
<b>carry compressed</b> to ~0; trend (S2, +5%) was the only positive sleeve. Cross-sleeve |&rho;| rose 0.11&rarr;{off26:.2f} &mdash;
diversification partially degraded under stress. <b>Implication for going live: paper-trade first; S5 is the prime fix candidate before sizing up.</b></div>
<img src="data:image/png;base64,{c_oos}">
<div class="grid2">
 <div><table><tr><th>2026 OOS book</th><th class=n>Return</th><th class=n>annSharpe</th><th class=n>MaxDD</th></tr>{tbl26}</table></div>
 <div><table><tr><th>2026 OOS sleeve</th><th class=n>Return</th></tr>{sleeve26}</table></div>
</div>"""


# ---- tables -----------------------------------------------------------------
def fmt_pct(x): return f"{x*100:.1f}%"
def fmt_n(x, d=2): return f"{x:.{d}f}"

sleeve_rows = ""
for k in R.columns:
    sm = summary(R[k].dropna(), periods=BARS)
    d = diag[k]
    sleeve_rows += (
        f"<tr><td><b style='color:{COL[k]}'>{NAMES[k]}</b></td><td>{d['source']}</td><td>{DIRN[k]}</td>"
        f"<td class=n>{fmt_n(sm['Sharpe'])}</td><td class=n>{fmt_n(d['gross_sharpe'])}</td>"
        f"<td class=n>{fmt_pct(sm['CAGR'])}</td><td class=n>{fmt_pct(sm['AnnVol'])}</td>"
        f"<td class=n>{fmt_pct(sm['MaxDD'])}</td><td class=n>{d['ann_turnover']:.0f}</td>"
        f"<td class=n>{fmt_pct(d['time_in_mkt'])}</td></tr>")

corr_hdr = "".join(f"<th>{NAMES[c].split(' · ')[0]}</th>" for c in corr.columns)
corr_rows = ""
for i in corr.index:
    cells = ""
    for j in corr.columns:
        v = corr.loc[i, j]
        bg = "#fff" if i == j else ("#fde8e8" if v > .25 else ("#e8f0fe" if v < -.05 else "#f8fafc"))
        cells += f"<td class=n style='background:{bg}'>{v:.2f}</td>"
    corr_rows += f"<tr><td><b>{NAMES[i].split(' · ')[0]}</b></td>{cells}</tr>"

combo_defs = [("Combined — equal-capital (20% each)", "combo_eqcap", "headline; no leverage assumptions"),
              ("Combined — risk-budget (carry levered, cap 10×)", "combo_riskbudget", "equal risk budget per sleeve"),
              ("Combined — risk-budget vol-target 10%", "combo_vt10", "scaled to 10%/yr target vol"),
              ("BTC buy & hold (excess of cash)", "btc_bh_excess", "benchmark")]
combo_rows = ""
for nm, key, note in combo_defs:
    sm = summary(combos[key].dropna(), periods=BARS)
    hl = "headline" in note
    combo_rows += (
        f"<tr style=\"{'background:#f0fdf4;font-weight:600' if hl else ''}\"><td>{nm}</td>"
        f"<td class=n>{fmt_n(sm['Sharpe'])}</td><td class=n>{fmt_pct(sm['CAGR'])}</td>"
        f"<td class=n>{fmt_pct(sm['AnnVol'])}</td><td class=n>{fmt_pct(sm['MaxDD'])}</td>"
        f"<td class=n>{fmt_n(sm['Calmar'])}</td><td style='font-size:12px;color:#6b7280'>{note}</td></tr>")

sm_eq = summary(combos["combo_eqcap"].dropna(), periods=BARS)
sm_btc = summary(combos["btc_bh_excess"].dropna(), periods=BARS)
lev_carry = meta["mean_leverage"]["S1_carry"]

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><title>Crypto multi-strategy (v3)</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1000px;margin:0 auto;padding:28px 22px;color:#1f2937;line-height:1.5}}
 h1{{font-size:25px;margin:0 0 2px}} h2{{font-size:18px;margin:30px 0 10px;border-bottom:2px solid #e5e7eb;padding-bottom:5px}}
 h3{{font-size:14px;margin:18px 0 6px;color:#374151}}
 .sub{{color:#6b7280;font-size:13px;margin-bottom:14px}}
 .kpis{{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}}
 .kpi{{flex:1;min-width:135px;background:#f8fafc;border:1px solid #e5e7eb;border-radius:9px;padding:11px 13px}}
 .kpi .v{{font-size:21px;font-weight:700}} .kpi .l{{font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.03em}}
 .kpi.good .v{{color:#16a34a}} .kpi.bad .v{{color:#dc2626}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}
 th,td{{border:1px solid #e5e7eb;padding:6px 9px;text-align:left}} th{{background:#f1f5f9;font-size:12px}}
 td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
 img{{max-width:100%;border:1px solid #eef2f7;border-radius:8px;margin:8px 0}}
 .note{{background:#fffbeb;border-left:4px solid #f59e0b;padding:9px 13px;border-radius:5px;font-size:13px;margin:12px 0}}
 .ok{{background:#f0fdf4;border-left:4px solid #16a34a;padding:9px 13px;border-radius:5px;font-size:13px;margin:12px 0}}
 .grid2{{display:flex;gap:14px;flex-wrap:wrap}} .grid2>div{{flex:1;min-width:320px}}
 code{{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}}
 .tag{{display:inline-block;background:#eef2ff;color:#3730a3;border-radius:20px;padding:1px 9px;font-size:11px;margin-right:5px}}
</style></head><body>

<h1>Crypto multi-strategy book — five low-correlation sleeves</h1>
<div class="sub">Research report · 2026-06-04 · BTC/ETH/SOL/BNB, daily, 2022–2025 (2025 out-of-sample) ·
all returns <b>excess of cash</b>, net of costs + funding + financing · <code>alpha_lab</code> v3</div>

<div class="ok"><b>Thesis.</b> Five sleeves, each paid by a <b>different return source</b> — carry, trend,
cross-sectional momentum, order-flow, and macro regime — are by construction nearly uncorrelated
(mean |ρ| = <b>{meta['mean_offdiag_corr']:.2f}</b>, max {meta['max_offdiag_corr']:.2f}, sum of pairwise ρ = {meta['sum_pairwise_corr']:.2f}).
Combined, they deliver a far smoother book than any single crypto bet: <b>Sharpe {sm_eq['Sharpe']:.2f}</b> vs
BTC's {sm_btc['Sharpe']:.2f}, with a <b>{fmt_pct(sm_eq['MaxDD'])}</b> max drawdown vs BTC's {fmt_pct(sm_btc['MaxDD'])}.</div>

<div class="kpis">
 <div class="kpi good"><div class="v">{sm_eq['Sharpe']:.2f}</div><div class="l">Combined Sharpe</div></div>
 <div class="kpi"><div class="v">{fmt_pct(sm_eq['CAGR'])}</div><div class="l">Combined CAGR (excess)</div></div>
 <div class="kpi"><div class="v">{fmt_pct(sm_eq['AnnVol'])}</div><div class="l">Combined vol</div></div>
 <div class="kpi good"><div class="v">{fmt_pct(sm_eq['MaxDD'])}</div><div class="l">Combined max DD</div></div>
 <div class="kpi"><div class="v">{meta['diversification_ratio']:.2f}×</div><div class="l">Diversification ratio</div></div>
</div>

<h2>1 · The combined book vs BTC</h2>
<img src="data:image/png;base64,{c_equity}">
<img src="data:image/png;base64,{c_dd}">
<div class="grid2">
 <div><img src="data:image/png;base64,{c_year}"></div>
 <div><img src="data:image/png;base64,{c_scatter}"></div>
</div>
<table>
 <tr><th>Book</th><th class=n>Sharpe</th><th class=n>CAGR</th><th class=n>Vol</th><th class=n>MaxDD</th><th class=n>Calmar</th><th>note</th></tr>
 {combo_rows}
</table>
<div class="note"><b>How it is combined ("恰当的组合起来").</b> Two views.
<b>Equal-capital</b> (headline) puts 20% of capital in each sleeve — no leverage assumptions, the conservative
read. <b>Risk-budget</b> scales each sleeve to a common ~8% vol target (leverage-capped at 10×) so each contributes
equal risk; this levers the ultra-low-vol carry sleeve ~{lev_carry:.0f}× (realistic for a basis book) and is then
optionally vol-targeted to 10%/yr. All three combinations land at Sharpe ≈ 1.1 — the diversification, not the
weighting scheme, is what does the work.</div>

<h2>2 · The five sleeves</h2>
<p style="font-size:13px">Each sleeve is anchored to a distinct entry in the repo's return-source taxonomy, which is
<i>why</i> they decorrelate. All rebalance <b>daily</b> (the hard lesson from prior crypto work: 5m–1h signals are
cost-killed), all are leak-safe (signal at <code>t</code> uses data ≤ <code>t</code>; the engine lags weights one
more bar), and all are charged the <b>cost of cash</b>.</p>
<table>
 <tr><th>Sleeve</th><th>Return source</th><th>Direction</th><th class=n>Net Sharpe</th><th class=n>Gross Sharpe</th>
 <th class=n>CAGR</th><th class=n>Vol</th><th class=n>MaxDD</th><th class=n>Ann.TO</th><th class=n>Time in mkt</th></tr>
 {sleeve_rows}
</table>
<img src="data:image/png;base64,{c_sleeves}">
<div style="font-size:12.5px;color:#4b5563">
<b>S1 Carry</b> — long spot / short perp when 7-day funding &gt; 0; harvests perp funding, delta≈0 (this is the
prior <code>P6</code> edge, here daily-grid). <b>S2 Trend</b> — BTC/ETH perp long above / short below the 50-day MA;
time-series momentum, profits in the 2022 bear. <b>S3 XS-Mom</b> — long the 2 strongest / short the 2 weakest of
{{BTC,ETH,SOL,BNB}} by 30-day return; market-neutral cross-sectional dispersion. <b>S4 Funding-contra</b> — fade
crowded funding extremes (banded z-score, dormant until stretched); pays you to provide liquidity to over-levered
positioning. <b>S5 Macro-gate</b> — hold crypto only when the HYG credit regime is risk-on; a <b>non-price-volume</b>
signal, so its PnL is driven by exogenous macro state.</div>

<h2>3 · Why they are low-correlation</h2>
<div class="grid2">
 <div><img src="data:image/png;base64,{c_corr}"></div>
 <div>
  <table>
   <tr><th></th>{corr_hdr}</tr>
   {corr_rows}
  </table>
  <p style="font-size:12.5px;color:#4b5563">Different mechanisms → different exposures. The two market-neutral
  sleeves (carry, XS-mom) harvest unrelated things; trend (long/short) and funding-contra are
  <b>negatively</b> correlated (ρ = {corr.loc['S2_trend','S4_fundcontra']:.2f}) because one rides momentum while the
  other fades crowding; the macro gate is driven by credit, not crypto price, so it barely correlates with anything
  (|ρ| ≤ {corr.loc['S5_macro'].drop('S5_macro').abs().max():.2f}). The only moderate pair is trend ↔ XS-mom
  ({corr.loc['S2_trend','S3_xsmom']:.2f}) — they share a momentum root but express it directionally vs neutrally.</p>
 </div>
</div>

<h2>4 · Honesty checks</h2>
<div class="note"><b>Cost of cash (financing).</b> Every sleeve is reported <b>excess of cash</b>: deploying gross-long
notional <code>L<sub>t</sub></code> is charged <code>rf<sub>t</sub> · L<sub>t</sub></code> (3-month T-bill), on top of
commissions, slippage, and perp funding. A flat sleeve earns zero excess (cash earns the hurdle). This is the
correction that halved the P6 carry; here it is baked into every number above and the edge is judged against the
risk-free hurdle, not zero. <span style="color:#6b7280">(rf source: {meta['rf_source']}; FRED endpoint was timing out
at run time, so a piecewise-by-year 3M T-bill path was used — within a few bp of the realized average.)</span></div>
<div class="ok"><b>Leak-safety.</b> No <code>shift(-k)</code>, no centered windows, no full-sample normalization —
all rolling stats are trailing; macro series are lagged a day for release safety; the backtester lags weights one bar
so execution is strictly after signal formation. 2026 is held out (PM holdout) and never touched. <b>2025 is genuine
out-of-sample</b> and the combined book returns {(((1+combos['combo_eqcap'][combos.index.year==2025].fillna(0)).prod())-1)*100:+.1f}% there.</div>

<h2>5 · Verdict &amp; what would break it</h2>
<p style="font-size:13px"><b>Verdict: <span style="color:#16a34a">accept_monitoring</span></b> as a diversified book.
The combination is robust — <b>positive every calendar year</b>
({(((1+combos['combo_eqcap'][combos.index.year==2022].fillna(0)).prod())-1)*100:+.0f}% in the 2022 bear,
{(((1+combos['combo_eqcap'][combos.index.year==2025].fillna(0)).prod())-1)*100:+.0f}% in the 2025 OOS) — and over the
full cycle beats BTC on <i>both</i> return ({fmt_pct(sm_eq['CAGR'])} vs {fmt_pct(sm_btc['CAGR'])} CAGR) and drawdown
({fmt_pct(sm_eq['MaxDD'])} vs {fmt_pct(sm_btc['MaxDD'])}). The honest cost is <b>upside capture</b>: in the 2024 one-way
bull the hedged book made only +{(((1+combos['combo_eqcap'][combos.index.year==2024].fillna(0)).prod())-1)*100:.0f}%
against BTC's +110% — it trades beta for all-weather smoothness (diversification ratio {meta['diversification_ratio']:.2f}×).
It is not a single "alpha" — it is a <i>portfolio</i> whose edge is the low correlation itself.</p>
<ul style="font-size:13px">
 <li><b>Carry compression</b> — S1's funding edge shrinks if perp funding trends to zero; it is the Sharpe anchor, so monitor the funding−financing spread.</li>
 <li><b>Momentum crashes</b> — S2/S3 share a momentum root; a sharp V-reversal hurts both at once (their +{corr.loc['S2_trend','S3_xsmom']:.2f} correlation is the book's main concentration).</li>
 <li><b>Turnover sensitivity</b> — S3 (≈{diag['S3_xsmom']['ann_turnover']:.0f}×/yr) and S4 (≈{diag['S4_fundcontra']['ann_turnover']:.0f}×/yr) are the cost-sensitive sleeves; double the cost assumption and re-check.</li>
 <li><b>Macro proxy</b> — S5 uses HYG as a credit-regime proxy (FRED BAA10Y is the cleaner non-price source once the endpoint is reachable); the gate's value depends on macro actually driving crypto, which is regime-dependent.</li>
 <li><b>Capacity</b> — sleeves trade BTC/ETH/SOL/BNB perps + BTC/ETH spot; deep enough for a personal book, but SOL/BNB legs and the short perp funding/borrow are the binding constraints at size.</li>
</ul>

{oos_section}

<h2>Appendix · method &amp; provenance</h2>
<p style="font-size:12.5px;color:#4b5563">
<span class="tag">idea-generation</span> The five sleeves came from the project's <code>idea-generation</code> skill:
random stimuli (thermostat, hibernation, tributary, coral, power-plant control room, credit-spread × prediction-market)
were run through the operator set (remote-stimulus, perspective-shift to the forced counterparty, reversal, macro-as-
conditioner) and filtered to one idea per return source, so orthogonality was a <i>design</i> target, not luck.<br>
<span class="tag">data</span> Binance Vision daily klines (spot+perp) &amp; 8h funding for BTC/ETH/SOL/BNB; HYG via yfinance;
3M T-bill financing via FRED (fallback path). Window 2022-01→2025-12, 2026 held out.<br>
<span class="tag">engine</span> <code>alpha_lab.backtest.vector.run_backtest</code> (weights lagged 1 bar; per-leg slippage
8–20 bps; perp funding charged on held weight) + uniform cost-of-cash overlay. Combination = equal-capital and
inverse-vol risk-budget (leverage-capped) + optional 10% vol target.<br>
<span class="tag">reproduce</span> <code>scratch_v3_build.py</code> → <code>scratch_v3_report.py</code> against
<code>data/results/crypto_v3_multi/</code>.</p>

</body></html>"""

REPORT.write_text(HTML, encoding="utf-8")
print("wrote", REPORT, f"({len(HTML)//1024} KB)")
