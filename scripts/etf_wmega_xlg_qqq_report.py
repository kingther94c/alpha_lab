"""Build the WMEGA vs XLG vs QQQ ETF comparison report."""

from __future__ import annotations

import base64
import io
from datetime import date
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from alpha_lab.analytics.returns import annualized_vol, drawdown, simple_returns
from alpha_lab.data.loaders.yfinance import load_prices
from alpha_lab.stats.regression import rolling_ols

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "etf_wmega_xlg_qqq_2026-07-14.html"
TICKERS = ["WMEGA.SW", "XLG", "QQQ", "SPY"]
DISPLAY = {"WMEGA.SW": "WMEGA", "XLG": "XLG", "QQQ": "QQQ", "SPY": "SPY"}
COLORS = {
    "WMEGA.SW": "#7c3aed",
    "XLG": "#0891b2",
    "QQQ": "#ea580c",
    "SPY": "#64748b",
}

MSCI_ANNUAL = {
    2012: 14.12,
    2013: 24.82,
    2014: 5.31,
    2015: 0.98,
    2016: 7.39,
    2017: 21.69,
    2018: -5.47,
    2019: 29.99,
    2020: 20.12,
    2021: 27.48,
    2022: -23.33,
    2023: 34.05,
    2024: 31.70,
    2025: 19.33,
}


def pct(value: float, digits: int = 1) -> str:
    """Format a decimal as a signed percentage."""
    if pd.isna(value):
        return "—"
    return f"{value * 100:+.{digits}f}%"


def num(value: float, digits: int = 2) -> str:
    """Format a numeric value for display."""
    if pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def image_data(fig: plt.Figure) -> str:
    """Encode a Matplotlib figure as an inline PNG data URL."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight", facecolor="#f8fafc")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def style_axis(ax: plt.Axes, *, percent_axis: bool = False) -> None:
    """Apply the report's chart styling."""
    ax.set_facecolor("#f8fafc")
    ax.grid(axis="y", color="#dbe4ee", linewidth=0.7, alpha=0.9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(colors="#475569", labelsize=8)
    if percent_axis:
        ax.yaxis.set_major_formatter(lambda x, _: f"{x * 100:.0f}%")


def summary_stats(prices: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Compute total return and risk statistics from each ticker's own observations."""
    rows: list[dict[str, object]] = []
    for ticker in prices.columns:
        series = prices.loc[prices.index >= start, ticker].dropna()
        returns = simple_returns(series).dropna()
        years = (series.index[-1] - series.index[0]).days / 365.25
        total = series.iloc[-1] / series.iloc[0] - 1
        cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1 if years >= 1 else np.nan
        rows.append(
            {
                "ticker": ticker,
                "start": series.index[0],
                "end": series.index[-1],
                "observations": len(returns),
                "total_return": total,
                "cagr": cagr,
                "vol": float(annualized_vol(returns)),
                "max_dd": float(drawdown(returns).min()),
                "best_day": float(returns.max()),
                "worst_day": float(returns.min()),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def regression_stats(prices: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Estimate SPY-relative OLS statistics over a fixed sample."""
    returns = simple_returns(prices.loc[prices.index >= start]).dropna(how="all")
    rows: list[dict[str, object]] = []
    for ticker in [column for column in prices.columns if column != "SPY"]:
        pair = returns[[ticker, "SPY"]].dropna()
        x = pair["SPY"].to_numpy()
        y = pair[ticker].to_numpy()
        beta, intercept = np.polyfit(x, y, 1)
        fitted = intercept + beta * x
        ss_res = np.square(y - fitted).sum()
        ss_tot = np.square(y - y.mean()).sum()
        rows.append(
            {
                "ticker": ticker,
                "alpha": intercept * 252,
                "beta": beta,
                "r2": 1 - ss_res / ss_tot,
                "corr": pair.corr().iloc[0, 1],
                "n": len(pair),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def rolling_metrics(
    prices: pd.DataFrame, tickers: list[str], window: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute rolling beta, annualized alpha, and annualized volatility."""
    returns = simple_returns(prices).dropna(how="all")
    beta: dict[str, pd.Series] = {}
    alpha: dict[str, pd.Series] = {}
    vol: dict[str, pd.Series] = {}
    for ticker in tickers:
        pair = returns[[ticker, "SPY"]].dropna()
        coeff = rolling_ols(pair[ticker], pair["SPY"].rename("SPY"), window=window)
        beta[ticker] = coeff["SPY"]
        alpha[ticker] = coeff["const"] * 252
        vol[ticker] = pair[ticker].rolling(window).std() * np.sqrt(252)
    return pd.DataFrame(beta), pd.DataFrame(alpha), pd.DataFrame(vol)


def top_holdings() -> dict[str, pd.DataFrame]:
    """Fetch each fund's current top-ten holdings from Yahoo Finance."""
    result: dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        frame = yf.Ticker(ticker).funds_data.top_holdings.copy()
        frame = frame.rename(columns={"Name": "name", "Holding Percent": "weight"})
        frame.index.name = "symbol"
        result[ticker] = frame.reset_index()[["symbol", "name", "weight"]]
    return result


def wealth_chart(prices: pd.DataFrame, start: pd.Timestamp, tickers: list[str]) -> str:
    """Plot normalized wealth from a common start."""
    fig, ax = plt.subplots(figsize=(10.6, 4.2))
    for ticker in tickers:
        series = prices.loc[prices.index >= start, ticker].dropna()
        wealth = series / series.iloc[0] * 100
        ax.plot(wealth.index, wealth, label=DISPLAY[ticker], color=COLORS[ticker], linewidth=2)
    style_axis(ax)
    ax.set_ylabel("Growth of 100", color="#475569")
    ax.legend(frameon=False, ncol=len(tickers), loc="upper left")
    return image_data(fig)


def metric_chart(frame: pd.DataFrame, title: str, *, percent_axis: bool = False) -> str:
    """Plot rolling metrics."""
    fig, ax = plt.subplots(figsize=(10.6, 3.55))
    for ticker in frame.columns:
        ax.plot(
            frame.index,
            frame[ticker],
            label=DISPLAY[ticker],
            color=COLORS[ticker],
            linewidth=1.8,
        )
    style_axis(ax, percent_axis=percent_axis)
    ax.set_title(title, loc="left", fontsize=11, color="#0f172a", fontweight="bold")
    ax.legend(frameon=False, ncol=len(frame.columns), loc="upper left")
    return image_data(fig)


def drawdown_chart(prices: pd.DataFrame, start: pd.Timestamp, tickers: list[str]) -> str:
    """Plot drawdown histories."""
    fig, ax = plt.subplots(figsize=(10.6, 3.55))
    for ticker in tickers:
        series = prices.loc[prices.index >= start, ticker].dropna()
        dd = drawdown(simple_returns(series).dropna())
        ax.plot(dd.index, dd, label=DISPLAY[ticker], color=COLORS[ticker], linewidth=1.7)
    style_axis(ax, percent_axis=True)
    ax.set_title("Drawdown from prior peak", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, ncol=len(tickers), loc="lower left")
    return image_data(fig)


def holdings_chart(holdings: dict[str, pd.DataFrame]) -> str:
    """Plot current top-ten weights by fund."""
    names = []
    for ticker in ["WMEGA.SW", "XLG", "QQQ", "SPY"]:
        names.extend(holdings[ticker]["symbol"].tolist())
    order = list(dict.fromkeys(names))
    data = pd.DataFrame(index=order)
    for ticker, frame in holdings.items():
        data[ticker] = frame.set_index("symbol")["weight"]
    data = data.fillna(0)
    data = data.loc[data.max(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(10.6, 6.1))
    y = np.arange(len(data))
    width = 0.19
    for index, ticker in enumerate(TICKERS):
        ax.barh(
            y + (index - 1.5) * width,
            data[ticker],
            height=width,
            label=DISPLAY[ticker],
            color=COLORS[ticker],
        )
    ax.set_yticks(y, data.index)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x * 100:.0f}%")
    ax.grid(axis="x", color="#dbe4ee", linewidth=0.7)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(colors="#475569", labelsize=8)
    ax.legend(frameon=False, ncol=4, loc="lower right")
    return image_data(fig)


def html_table(headers: list[str], rows: list[list[str]], classes: str = "") -> str:
    """Render a compact HTML table."""
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f'<div class="table-wrap"><table class="{classes}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_report(
    prices: pd.DataFrame,
    holdings: dict[str, pd.DataFrame],
    common_start: pd.Timestamp,
    short_summary: pd.DataFrame,
    short_reg: pd.DataFrame,
    long_summary: pd.DataFrame,
    long_reg: pd.DataFrame,
    beta_short: pd.DataFrame,
    alpha_short: pd.DataFrame,
    vol_short: pd.DataFrame,
    beta_long: pd.DataFrame,
    alpha_long: pd.DataFrame,
) -> str:
    """Assemble the standalone Chinese HTML report."""
    short_rows = []
    for ticker in TICKERS:
        stats = short_summary.loc[ticker]
        reg = short_reg.loc[ticker] if ticker != "SPY" else None
        short_rows.append(
            [
                f'<span class="ticker {DISPLAY[ticker].lower()}">{DISPLAY[ticker]}</span>',
                stats["start"].strftime("%Y-%m-%d"),
                pct(stats["total_return"]),
                pct(stats["vol"]),
                pct(stats["max_dd"]),
                num(reg["beta"]) if reg is not None else "1.00",
                pct(reg["alpha"]) if reg is not None else "0.0%",
                num(reg["r2"]) if reg is not None else "1.00",
            ]
        )

    long_rows = []
    for ticker in ["XLG", "QQQ", "SPY"]:
        stats = long_summary.loc[ticker]
        reg = long_reg.loc[ticker] if ticker != "SPY" else None
        long_rows.append(
            [
                f'<span class="ticker {ticker.lower()}">{ticker}</span>',
                pct(stats["cagr"]),
                pct(stats["vol"]),
                pct(stats["max_dd"]),
                num(reg["beta"]) if reg is not None else "1.00",
                pct(reg["alpha"]) if reg is not None else "0.0%",
                num(reg["r2"]) if reg is not None else "1.00",
            ]
        )

    concentration_rows = []
    holding_counts = {"WMEGA.SW": 33, "XLG": 51, "QQQ": 100, "SPY": 503}
    for ticker in TICKERS:
        frame = holdings[ticker]
        top10 = float(frame["weight"].sum())
        top5 = float(frame["weight"].head(5).sum())
        concentration_rows.append(
            [
                f'<span class="ticker {DISPLAY[ticker].lower()}">{DISPLAY[ticker]}</span>',
                str(holding_counts[ticker]),
                pct(float(frame["weight"].iloc[0])),
                pct(top5),
                pct(top10),
                pct(1 - top10),
            ]
        )

    union = sorted(
        set().union(*(set(frame["symbol"]) for frame in holdings.values()))
    )
    holding_rows = []
    for symbol in union:
        name = next(
            (
                frame.loc[frame["symbol"] == symbol, "name"].iloc[0]
                for frame in holdings.values()
                if symbol in set(frame["symbol"])
            ),
            symbol,
        )
        values = []
        for ticker in TICKERS:
            match = holdings[ticker].loc[holdings[ticker]["symbol"] == symbol, "weight"]
            values.append(pct(float(match.iloc[0]), 2) if not match.empty else "—")
        holding_rows.append([f"<b>{symbol}</b><small>{escape(name)}</small>", *values])

    annual_prices = prices[["XLG", "QQQ", "SPY"]].resample("YE").last().pct_change()
    annual_rows = []
    for year in range(2012, 2026):
        row = [str(year), f"{MSCI_ANNUAL[year]:+.2f}%"]
        for ticker in ["XLG", "QQQ", "SPY"]:
            try:
                value = float(annual_prices.loc[str(year), ticker].iloc[0])
            except (KeyError, IndexError):
                value = np.nan
            row.append(pct(value, 2))
        annual_rows.append(row)

    wealth_short = wealth_chart(prices, common_start, TICKERS)
    wealth_long = wealth_chart(prices, pd.Timestamp("2016-07-13"), ["XLG", "QQQ", "SPY"])
    dd_short = drawdown_chart(prices, common_start, TICKERS)
    hold_chart = holdings_chart(holdings)
    beta_short_img = metric_chart(beta_short, "60-day rolling beta vs SPY")
    alpha_short_img = metric_chart(
        alpha_short, "60-day rolling regression alpha (annualized)", percent_axis=True
    )
    vol_short_img = metric_chart(vol_short, "60-day rolling annualized volatility", percent_axis=True)
    beta_long_img = metric_chart(beta_long, "252-day rolling beta vs SPY")
    alpha_long_img = metric_chart(
        alpha_long, "252-day rolling regression alpha (annualized)", percent_axis=True
    )

    latest_beta = beta_short.ffill().iloc[-1]
    latest_alpha = alpha_short.ffill().iloc[-1]
    short_table = html_table(
        ["ETF", "样本起点", "累计回报", "年化波动", "最大回撤", "Beta", "年化 alpha", "R²"],
        short_rows,
    )
    long_table = html_table(
        ["ETF", "CAGR", "年化波动", "最大回撤", "Beta", "年化 alpha", "R²"],
        long_rows,
    )
    concentration_table = html_table(
        ["ETF", "持仓数", "最大单股", "Top 5", "Top 10", "Top 10 之外"],
        concentration_rows,
    )
    detail_table = html_table(["公司", "WMEGA", "XLG", "QQQ", "SPY"], holding_rows, "holdings")
    annual_table = html_table(["年份", "WMEGA 标的指数", "XLG", "QQQ", "SPY"], annual_rows)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WMEGA vs XLG vs QQQ ETF 研究报告</title>
<style>
:root{{--ink:#10233d;--muted:#5f7085;--line:#d9e3ef;--paper:#f8fafc;--card:#fff;--purple:#7c3aed;--cyan:#0891b2;--orange:#ea580c;--slate:#64748b;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Noto Sans SC","Microsoft YaHei",system-ui,sans-serif;line-height:1.65}}
.hero{{background:radial-gradient(circle at 90% 10%,#6d28d9 0,transparent 36%),linear-gradient(135deg,#071629,#102a43 65%,#153e5c);color:white;padding:68px 24px 56px}}
.wrap{{max-width:1160px;margin:auto}} h1{{font-size:clamp(2.2rem,5vw,4.9rem);line-height:1.02;margin:0 0 20px;letter-spacing:-.04em}} h2{{font-size:1.7rem;margin:54px 0 18px}} h3{{font-size:1.05rem;margin:0 0 8px}} .eyebrow{{text-transform:uppercase;letter-spacing:.2em;color:#a5f3fc;font-weight:800;font-size:.78rem}} .lede{{max-width:850px;color:#d9e7f4;font-size:1.08rem}} .meta{{display:flex;flex-wrap:wrap;gap:10px;margin-top:26px}} .pill{{padding:7px 12px;border:1px solid #ffffff3b;border-radius:999px;color:#e7eef7;font-size:.82rem}}
main{{padding:0 24px 72px}} .verdict{{margin-top:-24px;background:white;border:1px solid var(--line);border-radius:20px;padding:24px;box-shadow:0 14px 40px #0f27451a;display:grid;grid-template-columns:1.3fr 1fr 1fr;gap:16px}} .verdict strong{{font-size:1.15rem}} .accent{{border-left:4px solid var(--purple);padding-left:14px}} .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px}} .card p{{margin:.35rem 0;color:var(--muted)}} .kpi{{font-size:2rem;font-weight:850;letter-spacing:-.04em;line-height:1.1}} .chart{{background:white;border:1px solid var(--line);border-radius:16px;padding:12px;margin:16px 0}} .chart img{{width:100%;height:auto;display:block}} .note{{background:#eef6ff;border-left:4px solid #2563eb;padding:14px 16px;border-radius:0 12px 12px 0;color:#29405d}} .warn{{background:#fff7ed;border-left-color:#f97316}} .table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:14px;background:white}} table{{border-collapse:collapse;width:100%;font-size:.86rem}} th{{background:#edf3f8;color:#52657a;text-align:left;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em}} th,td{{padding:11px 12px;border-bottom:1px solid #e8eef5;white-space:nowrap}} tr:last-child td{{border-bottom:0}} td:not(:first-child),th:not(:first-child){{text-align:right}} small{{display:block;color:#7b8a9b;font-weight:400}} .ticker{{display:inline-block;padding:3px 8px;border-radius:999px;color:#fff;font-weight:800;letter-spacing:.04em}} .wmega{{background:var(--purple)}} .xlg{{background:var(--cyan)}} .qqq{{background:var(--orange)}} .spy{{background:var(--slate)}} .two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .section-lead{{max-width:850px;color:var(--muted)}} a{{color:#0369a1}} footer{{background:#071629;color:#a8bbce;padding:30px 24px}} code{{background:#eaf0f6;padding:2px 5px;border-radius:5px}}
@media(max-width:820px){{.verdict,.grid,.two{{grid-template-columns:1fr}} h1{{font-size:2.6rem}} .hero{{padding-top:46px}}}}
@media print{{.hero{{background:#102a43!important}} .chart,.card,.table-wrap{{break-inside:avoid}} a{{color:inherit;text-decoration:none}}}}
</style>
</head>
<body>
<header class="hero"><div class="wrap">
<div class="eyebrow">ETF concentration study · 2026-07-14</div>
<h1>同是 mega-cap，<br>风险来源并不相同</h1>
<p class="lede">WMEGA、XLG、QQQ 都把资金推向美国超大市值公司，但它们分别代表“全球发达市场最大公司”“S&amp;P 500 前 50”“Nasdaq-100 非金融龙头”。本报告用 SPY 作为共同市场基准，拆解真实 ETF 历史、滚动 beta/alpha、波动与回撤，以及当前 Top-10 集中度。</p>
<div class="meta"><span class="pill">复权日收盘价</span><span class="pill">SPY-relative OLS</span><span class="pill">60D / 252D rolling</span><span class="pill">当前持仓快照</span><span class="pill">非投资建议</span></div>
</div></header>
<main><div class="wrap">
<section class="verdict">
<div class="accent"><strong>核心结论：WMEGA 更像“更浓的 XLG”，不是国际分散化工具。</strong><p>按 MSCI 2026-06-30 权重，WMEGA 有 97.8% 美国敞口，Top-10 高达 66.3%；它与 XLG 的前十大名单完全相同，只是权重更重。</p></div>
<div><strong>若看成熟历史</strong><p>QQQ 与 XLG 都有完整牛熊周期；WMEGA ETF 只有约十个月实盘历史，不能据此宣称长期 alpha 稳定。</p></div>
<div><strong>若看结构差异</strong><p>QQQ 的风险来自 Nasdaq 行业与上市地筛选；XLG/WMEGA 的风险更直接来自超大市值集中度。</p></div>
</section>

<h2>01 · 产品到底买了什么</h2>
<div class="grid">
<article class="card"><span class="ticker wmega">WMEGA</span><h3>UBS MSCI World Mega Cap UCITS ETF</h3><p>爱尔兰 UCITS、累积型，TER 0.12%，2025-09-03 成立。指数目标覆盖 MSCI World 自由流通市值最大的约 40%，成分仅 33 只。</p></article>
<article class="card"><span class="ticker xlg">XLG</span><h3>Invesco S&amp;P 500 Top 50 ETF</h3><p>从 S&amp;P 500 选出最大 50 家公司，年度重构。2005-05-04 成立，费率 0.20%，是“美国 mega-cap 纯度”较高的成熟载体。</p></article>
<article class="card"><span class="ticker qqq">QQQ</span><h3>Invesco QQQ ETF</h3><p>跟踪 Nasdaq-100：Nasdaq 上市、剔除金融股、规模最大的 100 家非金融公司。1999-03-10 成立，当前总费率 0.18%。</p></article>
</div>
<p class="note">WMEGA 名称中的 “World” 容易造成错觉。当前指数国家权重为美国 97.79%、荷兰 2.21%；因此它在组合里主要是美国超大盘加码，而不是补足美国以外的发达市场。</p>

<h2>02 · 共同实盘窗口：2025-09-08 起</h2>
<p class="section-lead">这是唯一包含 WMEGA ETF 自身价格的可比窗口。收益、波动和回撤基于每只 ETF 自己的有效交易日；相对 SPY 回归只使用两者都交易的日期。短样本尚未经历完整周期。</p>
{short_table}
<div class="chart"><img src="{wealth_short}" alt="共同窗口累计净值"></div>
<div class="chart"><img src="{dd_short}" alt="共同窗口回撤"></div>
<div class="two"><div class="card"><h3>最新 60 日 beta</h3><div class="kpi">WMEGA {latest_beta['WMEGA.SW']:.2f}</div><p>XLG {latest_beta['XLG']:.2f} · QQQ {latest_beta['QQQ']:.2f}</p></div><div class="card"><h3>最新 60 日年化回归 alpha</h3><div class="kpi">WMEGA {pct(latest_alpha['WMEGA.SW'])}</div><p>XLG {pct(latest_alpha['XLG'])} · QQQ {pct(latest_alpha['QQQ'])}</p></div></div>
<div class="chart"><img src="{beta_short_img}" alt="60日滚动beta"></div>
<div class="chart"><img src="{alpha_short_img}" alt="60日滚动alpha"></div>
<div class="chart"><img src="{vol_short_img}" alt="60日滚动波动率"></div>
<p class="note warn"><b>非同步收盘偏差：</b>WMEGA.SW 在瑞士交易，收盘早于美股。相同日收益回归会把部分美国午后行情错配到下一交易日，可能压低日频 beta / R² 并放大 alpha 噪声。因此 WMEGA 的滚动回归仅作早期诊断，不应作为资产配置参数直接外推。</p>

<h2>03 · 十年压力测试：XLG vs QQQ vs SPY</h2>
<p class="section-lead">2016-07-13 至 2026-07-13，WMEGA ETF 不存在，因此不做伪历史拼接。该区间用于观察成熟产品在 2018、2020、2022 等不同冲击中的相对表现。</p>
{long_table}
<div class="chart"><img src="{wealth_long}" alt="十年累计净值"></div>
<div class="chart"><img src="{beta_long_img}" alt="252日滚动beta"></div>
<div class="chart"><img src="{alpha_long_img}" alt="252日滚动alpha"></div>

<h2>04 · WMEGA 标的指数的官方回测背景</h2>
<div class="grid"><div class="card"><h3>10 年年化回报</h3><div class="kpi">15.44%</div><p>截至 2026-06-30，MSCI 净回报指数。</p></div><div class="card"><h3>10 年年化波动</h3><div class="kpi">15.73%</div><p>基于月度净回报；与本报告日频口径不同。</p></div><div class="card"><h3>历史最大回撤</h3><div class="kpi">−55.67%</div><p>2007-10-31 至 2009-03-09。</p></div></div>
<p class="note warn"><b>关键限制：</b>该 MSCI 指数于 2025-06-06 才发布；此前历史由 MSCI 依照当时可用规则和数据回溯计算。它避免了“拿今天持仓倒推”的简单生存者偏差，但仍属于指数商 backtest，不等同于可交易 ETF 的真实费用、跟踪误差和市场冲击。</p>
{annual_table}

<h2>05 · Top-10 concentration decomposition</h2>
<p class="section-lead">权重快照抓取于 2026-07-14；WMEGA 权重与 MSCI 2026-06-30 官方指数页交叉核验。Top-10 只回答“当前押注在哪”，不能解释历史回报，也不能倒用于历史回测。</p>
{concentration_table}
<div class="chart"><img src="{hold_chart}" alt="前十大持仓权重对比"></div>
{detail_table}
<div class="grid" style="margin-top:16px"><div class="card"><h3>WMEGA ≈ XLG 的放大版</h3><p>两者 Top-10 名单相同；WMEGA 对每个龙头的权重几乎都更高。差异主要来自 WMEGA 只有 33 个成分，而 XLG 约 50 个。</p></div><div class="card"><h3>QQQ 的半导体偏离更明显</h3><p>当前 MU、AMD、INTC 权重显著高于 XLG/SPY；QQQ 不持金融股，并受 Nasdaq 上市资格影响，不等同于“美国最大 100 家”。</p></div><div class="card"><h3>Alphabet 双股应合并看</h3><p>GOOGL 与 GOOG 是同一经济实体的不同股类。单行看会低估 Alphabet 对组合的真实公司级集中度。</p></div></div>

<h2>06 · 如何选择：按组合任务，而不是追逐最高回报</h2>
<div class="grid"><div class="card"><h3>选择 WMEGA</h3><p>适合明确希望用低费率 UCITS 累积型工具，重仓全球最大公司，且接受极高 Top-10 与美国集中度的投资者。需特别关注上市地流动性、买卖价差和跨市场交易时段。</p></div><div class="card"><h3>选择 XLG</h3><p>适合希望在 S&amp;P 500 体系内把敞口压缩到最头部 50 家，并需要更长 ETF 实盘历史的人。它的成分规则更直观，但集中风险仍高。</p></div><div class="card"><h3>选择 QQQ</h3><p>适合有意识地押注 Nasdaq 非金融龙头与创新/科技结构的人。它不是纯市值前 100 筛选，行业与交易所选择会形成独立风格偏离。</p></div></div>
<p class="note"><b>组合层面的判断：</b>如果已有 SPY，三者都不是“新增分散化”，而是对现有 mega-cap 权重的再加码。WMEGA/XLG 更像规模因子加码，QQQ 更像规模 + 行业/上市地风格加码。真正的分散化需要看组合整体持仓穿透，而不是 ETF 名称数量。</p>

<h2>07 · 方法、审计与可复现性</h2>
<div class="two"><div class="card"><h3>计算定义</h3><p>价格：Yahoo Finance 复权 Close。波动率：日收益标准差 × √252。最大回撤：复权财富指数相对历史峰值的最深跌幅。滚动 beta/alpha：<code>r_etf = α + β·r_SPY + ε</code>，60 或 252 个共同交易日窗口，alpha × 252 年化；这是 SPY-relative regression alpha，不是扣除无风险利率后的 Jensen alpha。</p></div><div class="card"><h3>泄漏审计结论</h3><p>本研究没有交易信号、权重优化或执行回测，不存在同收盘信号赚同收盘收益。滚动统计仅使用窗口内历史数据。当前持仓只用于当前集中度快照，未用于历史收益重建。主要残余风险是 WMEGA 历史短、跨市场非同步收盘、Yahoo 数据修订，以及 MSCI 发布前指数历史属于回溯计算。</p></div></div>
<p><b>数据有效期：</b>ETF 价格截至 {prices.dropna(how='all').index.max().strftime('%Y-%m-%d')}；WMEGA.SW 最后有效价 {prices['WMEGA.SW'].last_valid_index().strftime('%Y-%m-%d')}。报告生成日 {date.today().isoformat()}。</p>
<ul>
<li><a href="https://www.ubs.com/global/en/assetmanagement/about/news/2025-news-articles/launches-mega-cap-and-ex-mega-cap-etfs.html">UBS：Mega Cap / ex Mega Cap ETF 发布公告</a></li>
<li><a href="https://www.msci.com/indexes/index/761936/msci-world-mega-cap-18-capped-specified-index">MSCI：World Mega Cap 18% Capped Specified Index</a></li>
<li><a href="https://www.msci.com/documents/10199/255599/msci-world-mega-cap-18-capped-specified-index-usd-net.pdf">MSCI：指数事实表（2026-06-30）</a></li>
<li><a href="https://www.invesco.com/us/en/financial-products/etfs/invesco-sp-500-top-50-etf.html">Invesco：XLG 官方产品页</a></li>
<li><a href="https://www.invesco.com/qqq-etf/en/home.html">Invesco：QQQ 官方产品页</a></li>
<li><a href="https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy">State Street：SPY 官方产品页</a></li>
<li><a href="https://finance.yahoo.com/quote/WMEGA.SW/">Yahoo Finance：WMEGA.SW 行情</a>；同源抓取 XLG、QQQ、SPY 复权价格和 Top-10 持仓。</li>
</ul>
</div></main>
<footer><div class="wrap">alpha_lab · ETF research note · 仅供研究，不构成投资、税务或交易建议</div></footer>
</body></html>"""


def main() -> None:
    """Download data, compute diagnostics, and write the report."""
    prices = load_prices(TICKERS, start="2005-01-01", end="2026-07-14")
    if prices.empty or prices["WMEGA.SW"].dropna().empty:
        raise RuntimeError("Price download failed or WMEGA.SW is unavailable")
    holdings = top_holdings()

    common_start = max(prices[ticker].first_valid_index() for ticker in TICKERS)
    short_summary = summary_stats(prices, common_start)
    short_reg = regression_stats(prices, common_start)
    long_start = pd.Timestamp("2016-07-13")
    long_prices = prices[["XLG", "QQQ", "SPY"]]
    long_summary = summary_stats(long_prices, long_start)
    long_reg = regression_stats(long_prices, long_start)

    beta_short, alpha_short, vol_short = rolling_metrics(
        prices.loc[prices.index >= common_start], ["WMEGA.SW", "XLG", "QQQ"], 60
    )
    beta_long, alpha_long, _ = rolling_metrics(
        prices.loc[prices.index >= long_start], ["XLG", "QQQ"], 252
    )

    report = render_report(
        prices,
        holdings,
        common_start,
        short_summary,
        short_reg,
        long_summary,
        long_reg,
        beta_short,
        alpha_short,
        vol_short,
        beta_long,
        alpha_long,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(short_summary[["total_return", "vol", "max_dd"]].to_string())
    print(short_reg.to_string())
    print(long_summary[["cagr", "vol", "max_dd"]].to_string())
    print(long_reg.to_string())


if __name__ == "__main__":
    main()
