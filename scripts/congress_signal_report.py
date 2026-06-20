"""Generate the comprehensive Congressional-Trading-Signal research report (HTML).

Runs the whole study end-to-end and writes a single self-contained report to
``reports/congress_signal_report.html``:

  data sourcing & official-filing audit → 4-angle analysis (sector tilt, party tilt,
  single-name event study) → benchmarks (SPY / NANC / KRUZ / sector-EW) → robustness
  (regime, sensitivity, IC) → multiple-testing (bootstrap CI, Deflated Sharpe) →
  go / no-go verdict.

Run:  PYTHONPATH=src D:/conda/envs/py313/python.exe scripts/congress_signal_report.py
"""

from __future__ import annotations

import logging
import sys
import warnings

import numpy as np
import pandas as pd

from alpha_lab.analytics.event_study import event_car
from alpha_lab.backtest.congress_book import (
    backtest_weights,
    benchmark_returns,
    load_congress_book_data,
    risk_on_tilt,
    sector_tilt,
)
from alpha_lab.backtest.congress_signal import (
    sector_flow_zscore,
    sector_net_flow,
    sector_tilt_weights,
)
from alpha_lab.backtest.metrics import summary
from alpha_lab.data.congress_universe import sector_etf_map
from alpha_lab.data.loaders.congress import (
    audit_coverage,
    fetch_house_filing_index,
    load_congress_trades,
)
from alpha_lab.data.loaders.yfinance import load_prices
from alpha_lab.reporting.charts import drawdown_chart, equity_curve
from alpha_lab.stats.tests import (
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    newey_west_tstat,
)
from alpha_lab.utils.cache import cached_parquet
from alpha_lab.utils.paths import REPORTS_DIR, ensure_dir

# Quiet request/library loggers before any analysis runs.
warnings.filterwarnings("ignore")
for _n in ("httpx", "yfinance", "urllib3", "peewee"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

ASOF = "2026-06-19"
EVENT_TOP_N = 180


# ======================================================================================
# Analysis
# ======================================================================================
def build_payload() -> dict:
    bd = load_congress_book_data()
    p: dict = {"eval_start": bd.eval_start, "rf_source": bd.rf_source, "coverage": bd.coverage}

    # --- core Angle A book + benchmarks ---
    w = sector_tilt(bd)                      # dollar-neutral top3/bot3, 63d flow, 252d z, W-FRI
    res, excess = backtest_weights(bd, w, bd.sector_prices)
    bench = benchmark_returns(bd)
    p["excess"] = excess
    p["net"] = res.returns
    p["gross"] = res.gross_returns
    p["turnover_yr"] = float(res.turnover.sum() / (len(res.turnover) / 252))
    p["bench"] = bench
    p["strat_summary"] = summary(excess)
    p["net_summary"] = summary(res.returns)
    p["bench_summary"] = {c: summary(bench[c].dropna()) for c in bench.columns}

    # leg decomposition (the "it's just beta" point)
    z = sector_flow_zscore(sector_net_flow(bd.trades, bd.sector_of, bd.trading_index))
    wlong = sector_tilt_weights(z, top_n=3, bottom_n=3, long_gross=1.0, short_gross=0.0)
    wshort = sector_tilt_weights(z, top_n=3, bottom_n=3, long_gross=0.0, short_gross=1.0)
    rl, _ = backtest_weights(bd, wlong.reindex(columns=bd.sector_prices.columns).fillna(0), bd.sector_prices)
    rs, _ = backtest_weights(bd, wshort.reindex(columns=bd.sector_prices.columns).fillna(0), bd.sector_prices)
    p["long_only_summary"] = summary(rl.returns)
    p["short_only_summary"] = summary(rs.returns)

    # --- Angle C ---
    wc = risk_on_tilt(bd)
    _, exc = backtest_weights(bd, wc, bd.macro_prices)
    p["angle_c_summary"] = summary(exc)

    # --- IC by horizon (sector-flow z vs forward sector-ETF returns) ---
    px = bd.sector_prices.loc[bd.eval_start:]
    zret = z.loc[bd.eval_start:].rename(columns=sector_etf_map())
    ic_rows = []
    for h in (5, 10, 21, 42, 63):
        fwd = px.pct_change(h).shift(-h)
        cols = [c for c in zret.columns if c in fwd.columns]
        ics = []
        for dt in zret.index:
            d = pd.concat([zret.loc[dt, cols], fwd.loc[dt, cols]], axis=1).dropna()
            if len(d) >= 5:
                ics.append(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman"))
        ics = pd.Series(ics)
        t = ics.mean() / (ics.std() / np.sqrt(len(ics))) if len(ics) > 1 and ics.std() > 0 else np.nan
        ic_rows.append({"h": h, "ic": float(ics.mean()), "t": float(t), "n": len(ics)})
    p["ic"] = pd.DataFrame(ic_rows)

    # --- regime split ---
    spy = bd.bench_prices["SPY"].loc[bd.eval_start:].pct_change()
    reg = {}
    for lab, a, b in [("2018-06 – 2021-12", "2018-06-19", "2021-12-31"),
                      ("2022-01 – 2026-06", "2022-01-01", None)]:
        sl = slice(a, b)
        reg[lab] = {"LS": summary(excess.loc[sl]).get("Sharpe", np.nan),
                    "LongTop3": summary(rl.returns.loc[sl]).get("Sharpe", np.nan),
                    "SPY": summary(spy.loc[sl]).get("Sharpe", np.nan)}
    p["regime"] = reg

    # --- sensitivity grid (robustness of the null + DSR trial set) ---
    grid, trial_sharpes = [], []
    for win in (21, 63, 126):
        line = []
        for tn in (2, 3, 4):
            ww = sector_tilt(bd, window=win, top_n=tn, bottom_n=tn)
            _, exg = backtest_weights(bd, ww, bd.sector_prices)
            sh = summary(exg).get("Sharpe", np.nan)
            line.append(sh)
            trial_sharpes.append(sh)
        grid.append((win, line))
    p["grid"] = grid

    # --- multiple-testing & robustness stats on the headline ---
    p["bootstrap"] = bootstrap_sharpe_ci(excess)
    p["nw_t"] = newey_west_tstat(excess)
    p["dsr"] = deflated_sharpe_ratio(max(trial_sharpes), n_obs=len(excess), trial_sharpes=trial_sharpes)

    # --- official-filing audit (year-aligned coverage cross-check) ---
    full = load_congress_trades(asset_types=("ST",), chambers=("house", "senate"))
    audit = {}
    for yr in (2022, 2023, 2024):
        a = audit_coverage(full, fetch_house_filing_index(yr))
        audit[yr] = a
    p["audit"] = audit
    p["full_trades"] = full
    p["days_to_file"] = (full["filing_date"] - full["transaction_date"]).dt.days
    p["sector_flow"] = (
        full.assign(g=full["ticker"].map(bd.sector_of))
        .groupby("g")["amount_logmid"].apply(lambda s: s.abs().sum())
        .drop(index="Unknown", errors="ignore").sort_values(ascending=False)
    )

    # --- event study (Angle D) ---
    top = full["ticker"].value_counts().head(EVENT_TOP_N).index.tolist()
    eprices = cached_parquet(
        "congress_event_prices",
        lambda: load_prices(top + ["SPY"], start="2014-01-01"),
    )
    have = [t for t in top if t in eprices.columns]
    ev = full[full["ticker"].isin(have)]
    p["event"] = {
        "n": len(ev),
        "filing_all": event_car(ev, eprices, date_col="filing_date", pre=5, post=42),
        "filing_buys": event_car(ev[ev.sign > 0], eprices, date_col="filing_date", sign_col=None, pre=5, post=42),
        "filing_sells": event_car(ev[ev.sign < 0], eprices, date_col="filing_date", sign_col=None, pre=5, post=42),
        "txn_buys": event_car(ev[ev.sign > 0], eprices, date_col="transaction_date", sign_col=None, pre=5, post=42),
    }
    return p


# ======================================================================================
# HTML rendering
# ======================================================================================
def _fig(fig, **kw) -> str:
    fig.update_layout(margin=dict(l=50, r=30, t=50, b=40), **kw)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def _perf_table(rows: dict[str, dict]) -> str:
    cols = ["Sharpe", "CAGR", "AnnVol", "MaxDD", "Calmar", "HitRate", "NPeriods"]
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for name, d in rows.items():
        cells = ""
        for c in cols:
            v = d.get(c, float("nan"))
            if c in ("CAGR", "AnnVol", "MaxDD", "HitRate"):
                cells += f"<td>{v*100:.1f}%</td>"
            elif c == "NPeriods":
                cells += f"<td>{int(v) if v == v else 0}</td>"
            else:
                cells += f"<td>{v:.2f}</td>"
        body += f"<tr><td class='nm'>{name}</td>{cells}</tr>"
    return f"<table class='perf'><thead><tr><th>Stream</th>{head}</tr></thead><tbody>{body}</tbody></table>"


def _event_fig(event: dict):
    import plotly.graph_objects as go
    fig = go.Figure()
    series = [("filing_buys", "Filing date · BUYS", "#1f77b4"),
              ("filing_sells", "Filing date · SELLS", "#aaaaaa"),
              ("txn_buys", "Transaction date · BUYS", "#d62728")]
    for key, label, color in series:
        r = event[key]
        if r.n_events:
            fig.add_scatter(x=list(r.car.index), y=(r.car.values * 100), mode="lines",
                            name=f"{label} (n={r.n_events})", line=dict(color=color))
    fig.add_vline(x=0, line_dash="dot", line_color="#888")
    fig.add_hline(y=0, line_color="#ddd")
    fig.update_layout(title="Market-adjusted CAR around the event (single names)",
                      xaxis_title="trading days from event", yaxis_title="cumulative abnormal return (%)",
                      template="plotly_white", height=460, width=900)
    return fig


def _ic_fig(ic: pd.DataFrame):
    import plotly.graph_objects as go
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in ic["ic"]]
    fig = go.Figure(go.Bar(x=[f"{h}d" for h in ic["h"]], y=ic["ic"], marker_color=colors,
                           text=[f"t={t:+.1f}" for t in ic["t"]], textposition="outside"))
    fig.add_hline(y=0, line_color="#ddd")
    fig.update_layout(title="Sector-flow rank-IC vs forward sector-ETF return (decays & reverses)",
                      xaxis_title="forward horizon", yaxis_title="mean rank IC",
                      template="plotly_white", height=380, width=900)
    return fig


def _grid_fig(grid):
    import plotly.graph_objects as go
    wins = [w for w, _ in grid]
    z = [line for _, line in grid]
    fig = go.Figure(go.Heatmap(
        z=z, x=["top2", "top3", "top4"], y=[f"{w}d flow" for w in wins],
        colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:+.2f}" for v in line] for line in z], texttemplate="%{text}",
        colorbar=dict(title="Sharpe")))
    fig.update_layout(title="Sensitivity: dollar-neutral L/S Sharpe across configs (no tradable edge anywhere)",
                      template="plotly_white", height=360, width=760)
    return fig


def _flow_fig(flow: pd.Series):
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(x=flow.values / 1e6, y=flow.index, orientation="h", marker_color="#4c78a8"))
    fig.update_layout(title="|Net log-mid $ flow| by GICS sector, 2014–2026 ($M)",
                      xaxis_title="$M", template="plotly_white", height=420, width=820,
                      yaxis=dict(autorange="reversed"))
    return fig


def _dtf_fig(dtf: pd.Series):
    import plotly.graph_objects as go
    d = dtf[(dtf >= 0) & (dtf <= 365)]
    fig = go.Figure(go.Histogram(x=d, nbinsx=60, marker_color="#72b7b2"))
    fig.add_vline(x=45, line_dash="dash", line_color="#d62728",
                  annotation_text="45-day deadline", annotation_position="top")
    fig.update_layout(title=f"Days from transaction → public filing (median {dtf.median():.0f}d, "
                            f"{(dtf>45).mean()*100:.0f}% late)",
                      xaxis_title="days to file", yaxis_title="count",
                      template="plotly_white", height=360, width=900)
    return fig


def render_html(p: dict) -> str:
    sm = p["strat_summary"]
    spy_sh = p["bench_summary"]["SPY"]["Sharpe"]
    nanc_sh = p["bench_summary"].get("NANC", {}).get("Sharpe", float("nan"))
    bt, dsr = p["bootstrap"], p["dsr"]
    fb = p["event"]["filing_buys"].drift(0, 42)
    tb = p["event"]["txn_buys"].drift(0, 42)

    # equity curves
    full_eq = pd.DataFrame({"Congress sector L/S (excess of cash)": p["excess"],
                            "SPY": p["bench"]["SPY"], "Sector equal-weight": p["bench"]["SectorEW"]}).dropna()
    com = pd.concat([p["excess"].rename("Congress sector L/S"), p["bench"]["SPY"].rename("SPY"),
                     p["bench"]["NANC"].rename("NANC"), p["bench"]["KRUZ"].rename("KRUZ")], axis=1).dropna()

    fig_eq_full = _fig(equity_curve(full_eq), title="Equity (excess of cash) vs SPY & Sector-EW, 2018-06→2026",
                       height=440, width=900)
    fig_eq_com = _fig(equity_curve(com), title=f"Common window with NANC/KRUZ ({com.index.min().date()}→{com.index.max().date()})",
                      height=420, width=900)
    fig_dd = _fig(drawdown_chart(p["excess"]), height=320, width=900)
    fig_event = _fig(_event_fig(p["event"]))
    fig_ic = _fig(_ic_fig(p["ic"]))
    fig_grid = _fig(_grid_fig(p["grid"]))
    fig_flow = _fig(_flow_fig(p["sector_flow"]))
    fig_dtf = _fig(_dtf_fig(p["days_to_file"]))

    perf = _perf_table({
        "Congress sector L/S — excess of cash": sm,
        "Congress sector L/S — net (pre cost-of-cash)": p["net_summary"],
        "  ↳ Long top-3 only (≈ market beta)": p["long_only_summary"],
        "  ↳ Short bottom-3 only": p["short_only_summary"],
        "SPY (buy & hold)": p["bench_summary"]["SPY"],
        "NANC (Dem copy ETF)": p["bench_summary"].get("NANC", {}),
        "KRUZ (Rep copy ETF)": p["bench_summary"].get("KRUZ", {}),
        "Sector equal-weight": p["bench_summary"]["SectorEW"],
    })

    audit_rows = "".join(
        f"<tr><td>{yr}</td><td>{a['official_ptr_docs']}</td><td>{a['matched_docs']}</td>"
        f"<td>{a['coverage_pct']:.0f}%</td><td>{a['latest_official_filing']}</td></tr>"
        for yr, a in p["audit"].items())
    reg_rows = "".join(
        f"<tr><td>{lab}</td><td>{d['LS']:+.2f}</td><td>{d['LongTop3']:+.2f}</td><td>{d['SPY']:+.2f}</td></tr>"
        for lab, d in p["regime"].items())
    cov = p["coverage"]

    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1000px;margin:0 auto;
    padding:28px 22px 80px;color:#1a1a1a;line-height:1.55;background:#fff}
    h1{font-size:27px;margin:0 0 4px} h2{font-size:21px;margin:34px 0 10px;border-bottom:2px solid #eee;padding-bottom:5px}
    h3{font-size:16px;margin:20px 0 6px;color:#333} .sub{color:#666;margin:0 0 18px}
    .verdict{border-left:6px solid #d62728;background:#fff5f5;padding:14px 18px;border-radius:6px;margin:18px 0}
    .verdict.go{border-color:#2ca02c;background:#f3fbf3}
    .kpis{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
    .kpi{flex:1;min-width:150px;background:#f7f8fa;border:1px solid #eaecef;border-radius:8px;padding:10px 12px}
    .kpi .v{font-size:21px;font-weight:650} .kpi .l{font-size:12px;color:#666}
    table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}
    th,td{border:1px solid #e6e6e6;padding:6px 9px;text-align:right} th{background:#f3f4f6}
    td.nm,table.perf td:first-child{text-align:left} caption{caption-side:top;text-align:left;color:#666;font-size:12px;margin-bottom:6px}
    .note{background:#fbfbfd;border:1px solid #eee;border-radius:6px;padding:10px 14px;font-size:13px;color:#444}
    .tag{display:inline-block;background:#eef;border-radius:4px;padding:1px 7px;font-size:12px;color:#335;margin-right:5px}
    code{background:#f3f3f5;padding:1px 5px;border-radius:4px;font-size:12px} ul{margin:6px 0 6px 2px}
    footer{margin-top:40px;color:#999;font-size:12px;border-top:1px solid #eee;padding-top:12px}
    """

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Congressional Trading Signal — Research Report</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>{css}</style></head><body>

<h1>Congressional Disclosures (STOCK Act / PTR) as a Cross-Asset Signal</h1>
<p class="sub">Research report · traded only via ETFs/futures/options (never single stocks) · as of {ASOF}
· data {p['full_trades']['filing_date'].min().date()}→{p['full_trades']['filing_date'].max().date()}
· eval {p['eval_start'].date()}→2026 · rf source: {p['rf_source']}</p>

<div class="verdict">
<b>结论 / Verdict — NO-GO (reject as a tradable ETF strategy).</b><br>
<b>中文：</b>国会个股披露中确有可测信息，但<b>仅在买入侧、且集中在成交后数日到数周</b>（申报日 t→+42 买入 CAR
+{fb['mean_car']*100:.2f}%，t={fb['tstat']:.1f}）。一旦聚合到板块 ETF、做市场中性（去 beta）并扣费+扣资金成本后，
<b>超额 alpha 消失</b>：板块多空 Sharpe={sm['Sharpe']:.2f}，跑输 SPY({spy_sh:.2f})、NANC({nanc_sh:.2f})、KRUZ。
NANC/KRUZ 的领先来自科技 beta，而非选股。按计划的 Phase-2 硬门槛 → <b>不进入纸面交易</b>。
<br><br>
<b>EN:</b> A small, real, <b>buy-side-only, short-horizon</b> information signal exists, but it does <b>not survive
translation to sector-ETF expression + beta-neutralization + costs</b>. The strategy loses to every benchmark,
especially the naive copy ETF NANC. This validates the plan's skeptical prior and the post-STOCK-Act literature.
Value delivered: a clean, reusable data+backtest toolkit and a documented null — saving capital and effort.
</div>

<div class="kpis">
  <div class="kpi"><div class="v">{sm['Sharpe']:.2f}</div><div class="l">Strategy Sharpe (excess of cash)</div></div>
  <div class="kpi"><div class="v">{spy_sh:.2f} / {nanc_sh:.2f}</div><div class="l">SPY / NANC Sharpe (must beat)</div></div>
  <div class="kpi"><div class="v">[{bt['lo']:.2f}, {bt['hi']:.2f}]</div><div class="l">Bootstrap 95% Sharpe CI (spans 0)</div></div>
  <div class="kpi"><div class="v">{dsr['dsr']:.2f}</div><div class="l">Deflated Sharpe (need &gt;0.95)</div></div>
</div>

<h2>1. What this is, and the honest prior</h2>
<p>Under the STOCK Act (2012), members of Congress must disclose securities trades &gt;$1,000 within 45 days
(Periodic Transaction Reports). The question: is there an extractable, <i>tradable</i> signal once aggregated to
the sector / macro level that fits an ETF-only mandate? The prior is skeptical — early studies (Ziobrowski 2004/2011)
found senator out-performance, but post-STOCK-Act work (Karadas 2021, Belmont 2022) finds the edge largely gone, and
the live copy-ETFs NANC/KRUZ owe their returns to tech beta, not selection. This study tries to <b>falsify</b>, not confirm.</p>

<h2>2. Data — can we go straight to the filings?</h2>
<p>We tested this empirically. Verdict: scraping raw filings end-to-end is impractical, so (as the plan recommends)
we use a <b>pre-parsed source for the backtest</b> and the <b>official portals for a ground-truth audit</b>.</p>
<table><caption>Data-source feasibility (tested {ASOF})</caption>
<thead><tr><th style="text-align:left">Source</th><th style="text-align:left">Direct from filings?</th><th style="text-align:left">Role</th></tr></thead><tbody>
<tr><td style="text-align:left">House Clerk official</td><td style="text-align:left">✅ XML index parses; transactions are scanned PDFs (OCR)</td><td style="text-align:left">Audit / freshness</td></tr>
<tr><td style="text-align:left">Senate eFD official</td><td style="text-align:left">⚠️ Akamai-walled, session-gated</td><td style="text-align:left">Best-effort audit</td></tr>
<tr><td style="text-align:left"><b>kadoa public/data</b></td><td style="text-align:left">Pre-parsed all 3 portals, has filing_date + doc_url</td><td style="text-align:left"><b>Primary backtest</b></td></tr>
<tr><td style="text-align:left">Senate Stock Watcher</td><td style="text-align:left">Pre-parsed Senate only</td><td style="text-align:left">Cross-check</td></tr>
</tbody></table>
<p>We aggregate kadoa's per-filer files into <b>{len(p['full_trades']):,} single-stock congressional disclosures</b>
(2014→2026, 207 members). Point-in-time discipline: the <b>signal date is <code>filing_date</code></b>, never the
transaction date — essential, because filings are slow:</p>
{fig_dtf}
<h3>Official-filing audit (cross-check vs House Clerk index, by DocID)</h3>
<table><thead><tr><th>Year</th><th>Official House PTR docs</th><th>Matched in parsed data</th><th>Coverage</th><th>Latest official filing</th></tr></thead>
<tbody>{audit_rows}</tbody></table>
<p class="note">Coverage is partial because the official count includes PTRs holding only bonds/options/funds (we keep
single stocks), plus scanned PDFs no parser ingests. The match confirms the pre-parsed data genuinely tracks the
filings. Sector mapping covers <b>{cov['pct_flow_mapped']:.0f}% of |flow|</b> ({cov['pct_trades_mapped']:.0f}% of trades);
the unmapped tail is mostly mislabeled ETFs and foreign ADRs.</p>

<h2>3. Where the money goes (sector aggregation, Angle A)</h2>
{fig_flow}
<p>Congressional single-stock flow is dominated by Technology, then Financials / Energy / Communication Services —
the same mega-cap tech concentration that drives NANC. The signal: rolling net log-mid $ flow per GICS sector →
trailing z-score → long the top-3 / short the bottom-3 sector ETFs, dollar-neutral (to strip market beta).</p>

<h2>4. The single-name event study (Angle D — the research gate)</h2>
{fig_event}
<p>This is the crux. After a member <b>buys</b>, the stock drifts up ~{tb['mean_car']*100:.2f}% (market-adjusted) over
the next ~6 weeks around the <i>transaction</i> date (t≈{tb['tstat']:.1f}) — information exists. By the <i>filing</i>
date (what we could act on) the buy-side drift is ~{fb['mean_car']*100:.2f}% (t≈{fb['tstat']:.1f}): about half survives
the disclosure lag. <b>Sells carry no signal</b> (members sell for liquidity). So the edge is faint, buy-side-only, and
decays — consistent with the IC below.</p>
{fig_ic}
<p>Rank-IC of sector flow vs forward returns is weakly positive at 1–4 weeks and <b>reverses by ~3 months</b> — the
sweet spot is short and shallow, and our 63-day construction sits right where it fades.</p>

<h2>5. Backtest vs the three benchmarks (the Phase-2 gate)</h2>
{perf}
<p class="note">Net includes the cost of cash (3M T-bill financing on deployed capital), per the repo's
research-artifact contract. The strategy must beat SPY, the NANC/KRUZ copy-ETFs, and a sector equal-weight book —
it beats none. The decomposition is the story: <b>long-top-3 ≈ market beta</b> (Sharpe {p['long_only_summary']['Sharpe']:.2f}),
the short leg loses (Sharpe {p['short_only_summary']['Sharpe']:.2f}, shorting a rising market), and dollar-neutral
(beta removed) leaves <b>~no alpha</b>.</p>
{fig_eq_full}
{fig_eq_com}
{fig_dd}

<h2>6. Robustness & multiple testing</h2>
<h3>Regime split (excess-of-cash Sharpe)</h3>
<table><thead><tr><th>Window</th><th>Sector L/S</th><th>Long-top3</th><th>SPY</th></tr></thead><tbody>{reg_rows}</tbody></table>
<h3>Parameter sensitivity — is the null robust?</h3>
{fig_grid}
<p>Across flow windows × top-N the dollar-neutral Sharpe is centered near/below zero (range
{min(min(ln) for _, ln in p['grid']):+.2f} to {max(max(ln) for _, ln in p['grid']):+.2f}) — no configuration has a tradable
edge, so this is a robust null, not one bad parameter pick.</p>
<p><span class="tag">Bootstrap</span> 95% Sharpe CI = [{bt['lo']:.2f}, {bt['hi']:.2f}], P(Sharpe&gt;0)={bt['p_gt_0']*100:.0f}%.
<span class="tag">Newey-West</span> mean-return t = {p['nw_t']:.2f}.
<span class="tag">Deflated Sharpe</span> = {dsr['dsr']:.2f} (expected max Sharpe under the null over {dsr['n_trials']} trials
= {dsr['expected_max_sharpe_ann']:.2f}; would need &gt;0.95 to claim significance). All three agree: no significant edge.</p>

<h2>7. Risks & failure modes</h2>
<ul>
<li><b>Legislative risk (highest).</b> Bills to ban congressional single-stock trading are advancing (H.R.7008 on the
Union Calendar 2026-02; S.1498 reported by Senate HSGAC 2025-12). If enacted, members divest into blind trusts and the
PTR single-stock signal dries up — pivot path: lobbying / government-contract alt-data.</li>
<li><b>Alpha decay / already-priced.</b> The 45-day lag eats the sharpest drift; NANC/KRUZ already exist, so part of
any signal is arbitraged.</li>
<li><b>Single-name → ETF dilution.</b> The faint per-name buy drift washes out at the sector level (quantified above).</li>
<li><b>Data:</b> ~16% of |flow| unmapped; sector labels are current-not-PIT (mild for a sector signal); kadoa parsing
gaps vs scanned PDFs.</li>
</ul>

<h2>8. Verdict, gates & next steps</h2>
<div class="verdict"><b>Phase-1 gate (event study):</b> PASS — measurable buy-side drift exists.<br>
<b>Phase-2 gate (beat SPY &amp; NANC, net, risk-adjusted):</b> <b>FAIL</b> — Sharpe {sm['Sharpe']:.2f} vs SPY {spy_sh:.2f} /
NANC {nanc_sh:.2f}; CI spans 0; DSR {dsr['dsr']:.2f}. → <b>Do not advance to paper trading.</b> Status: <code>reject</code>
(with a documented, reusable toolkit).</div>
<p><b>What could still be worth a narrow look (out of current mandate / next experiments):</b></p>
<ul>
<li><b>Angle B (committee overlap)</b> — not yet testable: needs point-in-time committee rosters (GovTrack/Senate.gov).
Scaffolded in <code>congress_book.committee_weighted_flow</code>. This is the most plausible place a concentrated subset edge hides.</li>
<li><b>Single-name long-only buy-copy</b> of high-conviction filers — retains the faint buy drift, but violates the
ETF-only mandate and ignores costs/capacity; research-only.</li>
<li><b>Use as a low-correlation idea-flag</b> (not a standalone strategy) inside the existing daily idea pipeline.</li>
</ul>

<footer>
Reusable artifacts: <code>data/loaders/congress.py</code>, <code>data/congress_universe.py</code> +
<code>configs/congress_ticker_sector.csv</code>, <code>backtest/congress_signal.py</code>,
<code>backtest/congress_book.py</code>, <code>analytics/event_study.py</code>, <code>stats/tests.py</code>.
Regenerate: <code>PYTHONPATH=src python scripts/congress_signal_report.py</code>.
Decision record: <code>docs/research_decisions/2026-06-19_congressional_trading_signal.md</code>.
Not investment advice. Past performance does not predict future results.
</footer>
</body></html>"""


def main() -> None:
    print("Building congressional-signal report payload (this fetches prices)…", file=sys.stderr)
    p = build_payload()
    html = render_html(p)
    out = ensure_dir(REPORTS_DIR) / "congress_signal_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}  ({len(html)/1024:.0f} KB)")
    print(f"  Strategy Sharpe (excess) = {p['strat_summary']['Sharpe']:.2f} | "
          f"SPY = {p['bench_summary']['SPY']['Sharpe']:.2f} | "
          f"NANC = {p['bench_summary'].get('NANC',{}).get('Sharpe',float('nan')):.2f} | "
          f"DSR = {p['dsr']['dsr']:.2f}")


if __name__ == "__main__":
    main()
