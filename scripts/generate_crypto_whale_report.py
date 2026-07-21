"""Generate the self-contained Chinese HTML report for the whale/ETF study."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import font_manager

matplotlib.use("Agg")
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(
        fname=FONT_PATH
    ).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BTC_DIR = ROOT / "data" / "results" / "btc_onchain_exchange_flow"
ETF_DIR = ROOT / "data" / "results" / "btc_etf_flow_study"
ETH_DIR = ROOT / "data" / "results" / "crypto_onchain_replication"
REPORT_PATH = ROOT / "reports" / "crypto_whale_etf_strategy_2026-07-20.html"

COLORS = {
    "scaled": "#006d77",
    "trend": "#e29578",
    "hold": "#6c757d",
    "loss": "#b23a48",
    "cash": "#83c5be",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_uri(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _style_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c9d1d9")
    if grid:
        ax.grid(axis="y", color="#e9ecef", linewidth=0.8)
    ax.tick_params(colors="#495057", labelsize=9)


def _wealth_chart(returns: pd.DataFrame, *, asset: str, start: str) -> str:
    columns = {
        "scaled_onchain_ma200__base_lag1__total": "趋势 + 链上分级仓位",
        "price_trend_ma200__control__total": "纯 MA200",
        "buy_hold__control__total": "买入持有",
    }
    wealth = (1.0 + returns[list(columns)].loc[start:]).cumprod()
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    palette = [COLORS["scaled"], COLORS["trend"], COLORS["hold"]]
    for (column, label), color in zip(columns.items(), palette, strict=True):
        ax.plot(wealth.index, wealth[column], label=label, color=color, linewidth=2.0)
    ax.set_yscale("log")
    ax.set_title(f"{asset} 累计财富（对数刻度，起点=1）", loc="left", weight="bold")
    ax.set_ylabel("累计财富")
    _style_axis(ax)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    return _image_uri(fig)


def _recent_chart(returns: pd.DataFrame) -> str:
    columns = {
        "scaled_onchain_ma200__base_lag1__total": "分级仓位",
        "price_trend_ma200__control__total": "纯 MA200",
        "buy_hold__control__total": "买入持有",
    }
    recent = returns[list(columns)].loc["2024-01-01":]
    wealth = (1.0 + recent).cumprod() - 1.0
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    palette = [COLORS["scaled"], COLORS["trend"], COLORS["hold"]]
    for (column, label), color in zip(columns.items(), palette, strict=True):
        ax.plot(wealth.index, wealth[column] * 100, label=label, color=color, linewidth=2.0)
    ax.axhline(0, color="#adb5bd", linewidth=0.8)
    ax.set_title("BTC 最近期累计收益：链上层降低暴露，但明显落后纯趋势", loc="left", weight="bold")
    ax.set_ylabel("累计收益（%）")
    _style_axis(ax)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    return _image_uri(fig)


def _drawdown_chart(returns: pd.DataFrame) -> str:
    columns = {
        "scaled_onchain_ma200__base_lag1__total": "趋势 + 链上分级仓位",
        "price_trend_ma200__control__total": "纯 MA200",
        "buy_hold__control__total": "买入持有",
    }
    wealth = (1.0 + returns[list(columns)].loc["2015-01-01":]).cumprod()
    drawdown = wealth.div(wealth.cummax()).sub(1.0) * 100
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    palette = [COLORS["scaled"], COLORS["trend"], COLORS["hold"]]
    for (column, label), color in zip(columns.items(), palette, strict=True):
        ax.plot(drawdown.index, drawdown[column], label=label, color=color, linewidth=1.6)
    ax.set_title("BTC 回撤：分级仓位改善全历史尾部风险", loc="left", weight="bold")
    ax.set_ylabel("回撤（%）")
    _style_axis(ax)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    return _image_uri(fig)


def _annual_chart(metrics: pd.DataFrame) -> str:
    annual = metrics.loc[
        (metrics["candidate"] == "scaled_onchain_ma200")
        & (metrics["scenario"] == "base_lag1")
        & metrics["period"].astype(str).str.fullmatch(r"20\d{2}"),
        ["period", "TotalReturn"],
    ].copy()
    annual["TotalReturn"] *= 100
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    colors = np.where(annual["TotalReturn"] >= 0, COLORS["scaled"], COLORS["loss"])
    ax.bar(annual["period"].astype(str), annual["TotalReturn"], color=colors)
    ax.axhline(0, color="#495057", linewidth=0.8)
    ax.set_title("BTC 固定分级策略年度收益（2026截至7月19日）", loc="left", weight="bold")
    ax.set_ylabel("收益（%）")
    _style_axis(ax)
    return _image_uri(fig)


def _rejection_chart() -> str:
    labels = ["ETF吸收规则\n2026冻结期", "链上二元规则\n2024–2026", "最终分级规则\n2024–2026"]
    values = [-7.766, -13.419, 17.695]
    colors = [COLORS["loss"], COLORS["loss"], COLORS["scaled"]]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.axhline(0, color="#495057", linewidth=0.8)
    for bar, value in zip(bars, values, strict=True):
        y = value + (1.2 if value >= 0 else -2.0)
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.1f}%", ha="center", weight="bold")
    ax.set_title("严格冻结暴露了两个伪强策略；最终规则只作分级而非开关", loc="left", weight="bold")
    ax.set_ylabel("冻结/最近期总收益")
    _style_axis(ax)
    return _image_uri(fig)


def _permutation_chart() -> str:
    family = pd.read_csv(BTC_DIR / "permutation_familywise.csv")
    observed = json.loads((BTC_DIR / "robustness_meta.json").read_text("utf-8"))[
        "observed_robust_score"
    ]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.hist(family["best_min_sharpe"].dropna(), bins=24, color="#d8e2dc", edgecolor="white")
    ax.axvline(observed, color=COLORS["loss"], linewidth=2, label=f"观察值 {observed:.2f}")
    ax.set_title("为何样本内显著仍不够：二元链上规则随后外部失败", loc="left", weight="bold")
    ax.set_xlabel("循环置换后，全家族最佳的最小Sharpe")
    ax.set_ylabel("次数")
    _style_axis(ax)
    ax.legend(frameon=False)
    return _image_uri(fig)


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _num(value: float) -> str:
    return f"{value:.2f}"


def _summary_table(metrics: pd.DataFrame, *, period: str) -> str:
    wanted = metrics.loc[
        (metrics["period"] == period)
        & (
            ((metrics["candidate"] == "scaled_onchain_ma200") & (metrics["scenario"] == "base_lag1"))
            | ((metrics["candidate"] == "price_trend_ma200") & (metrics["scenario"] == "control"))
            | ((metrics["candidate"] == "buy_hold") & (metrics["scenario"] == "control"))
        )
    ].copy()
    names = {
        "scaled_onchain_ma200": "趋势 + 链上分级",
        "price_trend_ma200": "纯 MA200",
        "buy_hold": "买入持有",
    }
    wanted["策略"] = wanted["candidate"].map(names)
    wanted["总收益"] = wanted["TotalReturn"].map(_pct)
    wanted["年化"] = wanted["TotalCAGR"].map(_pct)
    wanted["超额Sharpe"] = wanted["ExcessSharpe"].map(_num)
    wanted["最大回撤"] = wanted["MaxDD"].map(_pct)
    wanted["年化换手"] = wanted["AnnTurnover"].map(lambda x: f"{x:.1f}×")
    return wanted[["策略", "总收益", "年化", "超额Sharpe", "最大回撤", "年化换手"]].to_html(
        index=False, classes="metric-table", border=0
    )


def main() -> None:
    btc_returns = pd.read_parquet(BTC_DIR / "scaled_overlay_returns.parquet")
    btc_metrics = pd.read_csv(BTC_DIR / "scaled_overlay_metrics.csv", dtype={"period": str})
    eth_returns = pd.read_parquet(ETH_DIR / "eth_returns.parquet")
    eth_metrics = pd.read_csv(ETH_DIR / "eth_metrics.csv", dtype={"period": str})
    btc_data = pd.read_csv(ROOT / "data" / "interim" / "btc_coinmetrics_onchain.csv", parse_dates=["date"])
    eth_data = pd.read_csv(ROOT / "data" / "interim" / "eth_coinmetrics_onchain.csv", parse_dates=["date"])

    btc_latest = btc_data.iloc[-1]
    eth_latest = eth_data.iloc[-1]
    btc_ma200 = btc_data["PriceUSD"].rolling(200).mean().iloc[-1]
    eth_ma200 = eth_data["PriceUSD"].rolling(200).mean().iloc[-1]

    charts = {
        "btc_wealth": _wealth_chart(btc_returns, asset="BTC", start="2015-01-01"),
        "btc_recent": _recent_chart(btc_returns),
        "btc_dd": _drawdown_chart(btc_returns),
        "btc_annual": _annual_chart(btc_metrics),
        "eth_wealth": _wealth_chart(eth_returns, asset="ETH（BTC规则原样复现）", start="2017-01-01"),
        "rejections": _rejection_chart(),
        "permutation": _permutation_chart(),
    }
    generated_at = "2026-07-20 22:00 SGT"
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Crypto 大额持仓、ETF资金流与可盈利策略研究</title>
<style>
:root{{--ink:#17252a;--muted:#5f6b6d;--teal:#006d77;--mint:#83c5be;--cream:#fffaf4;--sand:#edf6f5;--coral:#e29578;--red:#b23a48;--line:#dfe7e7;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--cream);color:var(--ink);font-family:"Segoe UI","Microsoft YaHei",Arial,sans-serif;line-height:1.65}}
.wrap{{max-width:1180px;margin:auto;padding:28px 32px 72px}} .hero{{background:linear-gradient(135deg,#073b4c,#006d77);color:white;border-radius:24px;padding:46px 52px;box-shadow:0 18px 45px #006d7730}}
.eyebrow{{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:#bde0d9;font-weight:700}} h1{{font-size:42px;line-height:1.16;margin:10px 0 16px;max-width:900px}} .subtitle{{font-size:18px;color:#e6f3f1;max-width:920px}}
.verdict{{margin-top:26px;background:#ffffff14;border:1px solid #ffffff36;border-radius:14px;padding:18px 22px;font-size:17px}} .verdict b{{color:#ffd6a5}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}} .kpi{{background:white;border:1px solid var(--line);border-radius:16px;padding:19px;box-shadow:0 6px 18px #173b3f0b}} .kpi .v{{font-size:27px;font-weight:800;color:var(--teal)}} .kpi .l{{font-size:13px;color:var(--muted)}}
section{{margin-top:40px}} h2{{font-size:27px;margin:0 0 14px}} h3{{font-size:19px;margin:24px 0 10px}} p{{margin:8px 0 12px}} .lede{{font-size:17px;color:#344b4e}} .card{{background:white;border:1px solid var(--line);border-radius:18px;padding:24px;margin:14px 0;box-shadow:0 6px 18px #173b3f0a}} .warn{{border-left:5px solid var(--coral);background:#fff8f3}} .danger{{border-left:5px solid var(--red);background:#fff6f6}} .ok{{border-left:5px solid var(--teal);background:#f4fbfa}}
.chart{{width:100%;display:block;border-radius:10px}} .two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .three{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
.metric-table{{border-collapse:collapse;width:100%;font-size:14px}} .metric-table th{{background:#edf6f5;color:#234;padding:10px;text-align:right}} .metric-table th:first-child,.metric-table td:first-child{{text-align:left}} .metric-table td{{padding:10px;border-bottom:1px solid #edf1f1;text-align:right}} .metric-table tr:first-child td{{font-weight:700;color:var(--teal)}}
.rule{{font-family:Consolas,"SFMono-Regular",monospace;background:#122b32;color:#e8f5f2;border-radius:14px;padding:20px;white-space:pre-wrap;line-height:1.75}} .pill{{display:inline-block;border-radius:999px;padding:4px 10px;background:#dbeeee;color:#075d64;font-size:12px;font-weight:700;margin:2px}}
.flow{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;align-items:stretch}} .step{{background:#edf6f5;border-radius:13px;padding:16px;position:relative}} .step b{{display:block;color:var(--teal);margin-bottom:4px}} .step:not(:last-child)::after{{content:"→";position:absolute;right:-12px;top:37%;font-size:22px;color:var(--coral);z-index:2}}
.source-list a{{color:var(--teal);text-decoration:none}} .source-list li{{margin:7px 0}} .small{{font-size:13px;color:var(--muted)}} .red{{color:var(--red);font-weight:700}} .green{{color:var(--teal);font-weight:700}}
footer{{margin-top:48px;border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:13px}}
@media(max-width:800px){{.wrap{{padding:18px}}.hero{{padding:28px}}h1{{font-size:32px}}.kpis,.two,.three,.flow{{grid-template-columns:1fr}}.step::after{{display:none}}}}
@media print{{body{{background:white}}.wrap{{max-width:none}}.hero,.card{{box-shadow:none}}section{{break-inside:avoid}}}}
</style>
</head>
<body><main class="wrap">
<header class="hero">
  <div class="eyebrow">ALPHA LAB · RESEARCH REPORT · 20 JUL 2026</div>
  <h1>Crypto 大额持仓、ETF 资金流与可盈利策略</h1>
  <div class="subtitle">从 BTC 开始，延伸到 ETH 与 SOL；先追踪谁在持有，再检验这些信息能否真正转化成扣除成本与现金机会成本后的收益。</div>
  <div class="verdict"><b>结论：</b>找到一个历史盈利、跨 BTC/ETH 可复制的纸面策略，但盈利主因是 MA200 趋势；链上大额持仓只适合调节仓位和降低回撤。ETF流量与二元“鲸鱼开关”都未通过冻结外部期，不能称为 alpha。</div>
</header>

<div class="kpis">
  <div class="kpi"><div class="v">53.8%</div><div class="l">BTC 2015–2026 年化（分级策略）</div></div>
  <div class="kpi"><div class="v">1.35</div><div class="l">BTC 全期超额 Sharpe</div></div>
  <div class="kpi"><div class="v">+17.7%</div><div class="l">BTC 2024–2026 最近期总收益</div></div>
  <div class="kpi"><div class="v">+29.6%</div><div class="l">ETH 同规则最近期总收益</div></div>
</div>

<section>
<h2>1. 现在该做什么</h2>
<div class="two">
 <div class="card danger"><h3>最新 BTC 目标：0%</h3><p>数据至 {btc_latest['date'].date()}：价格 ${btc_latest['PriceUSD']:,.0f}，MA200 ${btc_ma200:,.0f}。价格低于趋势线，因此风险开关关闭。</p><p class="small">这是收盘后研究信号，不是订单。最早只能在下一执行时点实施。</p></div>
 <div class="card danger"><h3>最新 ETH 目标：0%</h3><p>数据至 {eth_latest['date'].date()}：价格 ${eth_latest['PriceUSD']:,.0f}，MA200 ${eth_ma200:,.0f}。同一规则下风险开关关闭。</p><p class="small">SOL缺少同口径免费历史，不生成虚假的目标仓位。</p></div>
</div>
<div class="card ok"><h3>固定规则</h3><div class="rule">每日 UTC 收盘 d：
1) 若 Price[d] ≤ SMA200(Price)[d]：目标 BTC/ETH = 0%
2) 若 Price[d] &gt; SMA200，且已知的 ExchangeSupply[d−1]
   &lt; SMA365(ExchangeSupply，截止 d−2)：目标 = 100%
3) 其余 Price &gt; SMA200：目标 = 50%
4) 下一收盘执行；链上日 t 最早赚取 t+2 的收益
5) BTC 成本 15bp/单位权重变化（压力 30bp）；现金赚取 13周T-bill</div></div>
</section>

<section>
<h2>2. 结果：盈利，但不要错认来源</h2>
<p class="lede">分级策略牺牲部分上涨，换取较低的风险暴露。最近期它远落后于纯MA200，因此“鲸鱼数据创造收益”的说法不成立。</p>
<div class="card"><h3>BTC 全期 2015–2026</h3>{_summary_table(btc_metrics, period='full')}</div>
<div class="card"><img class="chart" src="{charts['btc_wealth']}" alt="BTC累计财富"></div>
<div class="card"><img class="chart" src="{charts['btc_dd']}" alt="BTC回撤"></div>
<div class="two"><div class="card"><img class="chart" src="{charts['btc_recent']}" alt="BTC最近期"></div><div class="card"><img class="chart" src="{charts['btc_annual']}" alt="BTC年度收益"></div></div>
<div class="card warn"><b>关键归因：</b>最近期分级策略 +17.7%，纯MA200 +51.7%，买入持有 +53.1%。全历史分级策略的优势是Sharpe 1.40、最大回撤 -40.8%，相对纯MA200的Sharpe 1.17、最大回撤 -68.0%。链上层是风险预算，不是近期收益增强器。</div>
</section>

<section>
<h2>3. ETH 原样复制，SOL 暂不回测</h2>
<p class="lede">BTC规则、窗口与仓位完全不改，只把资产替换为ETH；交易成本提高至20bp，压力40bp。</p>
<div class="card"><h3>ETH 最近期 2024–2026</h3>{_summary_table(eth_metrics, period='recent')}</div>
<div class="card"><img class="chart" src="{charts['eth_wealth']}" alt="ETH累计财富"></div>
<div class="card ok"><b>复制结论：</b>ETH分级策略最近期 +29.6%，而买入持有 -18.0%；但纯MA200 +47.1%。说明趋势风险开关跨资产有效，链上分级仍主要降低暴露。SOL在Coin Metrics Community目录没有相同交易所流量/储备字段，且美国ETF历史很短，因此只建监控，不给回测数字。</div>
</section>

<section>
<h2>4. 为什么必须展示失败</h2>
<div class="card"><img class="chart" src="{charts['rejections']}" alt="策略淘汰"></div>
<div class="three">
 <div class="card"><h3>ETF极端流入</h3><p>2024/2025盈利，但家族级置换 p≈0.24，块自助区间含大额亏损。<span class="red">未通过。</span></p></div>
 <div class="card"><h3>ETF弱势吸收</h3><p>2024/2025超额Sharpe 1.13/1.27；冻结2026后 -7.8%，再延迟一天 -16.0%。<span class="red">淘汰。</span></p></div>
 <div class="card"><h3>二元链上稀缺</h3><p>前2024家族级 p=0.008；冻结最近期却 -13.4%。地址标签/市场结构变化击穿历史关系。<span class="red">淘汰。</span></p></div>
</div>
<div class="card"><img class="chart" src="{charts['permutation']}" alt="置换分布"><p class="small">即使静态统计检验非常漂亮，真正冻结的未来区间仍可失败。显著性不能替代时点数据与外部验证。</p></div>
</section>

<section>
<h2>5. 怎么追踪链上大额持仓</h2>
<div class="flow">
 <div class="step"><b>① 实体层</b>ETF托管、CEX、矿工、基金、项目金库、跨链桥、质押账户</div>
 <div class="step"><b>② 去噪层</b>UTXO找零、内部归集、冷钱包迁移、桥接与质押转移</div>
 <div class="step"><b>③ 特征层</b>余额变化、净流、交易所储备、大额转账、休眠币激活</div>
 <div class="step"><b>④ 决策层</b>趋势开关、仓位置信度、告警；不直接追单</div>
</div>
<div class="three">
 <div class="card"><h3>BTC</h3><p><span class="pill">UTXO实体聚类</span><span class="pill">交易所净流</span><span class="pill">ETF托管</span></p><p>必须剔除找零与交易所内部迁移。建议同时看7/30日净流、交易所总储备、沉睡币移动和价格响应。</p></div>
 <div class="card"><h3>ETH</h3><p><span class="pill">CEX</span><span class="pill">质押队列</span><span class="pill">L2桥</span><span class="pill">ETF钱包</span></p><p>大额转出交易所可能进入质押或L2，并不等于长期冷藏；需拆解目的地址与验证者流。</p></div>
 <div class="card"><h3>SOL</h3><p><span class="pill">Stake accounts</span><span class="pill">Validators</span><span class="pill">CEX</span><span class="pill">ETF custodian</span></p><p>跟踪解质押、验证者集中度与交易所实体。免费历史覆盖不足，先每日快照累积自己的时点数据库。</p></div>
</div>
<div class="card warn"><h3>推荐告警，不推荐自动交易</h3><p>告警条件可设为：实体调整后净流超过自身730日90%分位；单笔≥1亿美元；ETF 5日流量/AUM异常；或“资金流方向与价格5日方向相反”的吸收/分发事件。告警只触发复核，至少用两个独立标签源确认。</p></div>
</section>

<section>
<h2>6. ETF / ETP 追踪矩阵</h2>
<div class="card">
<table class="metric-table"><thead><tr><th>资产</th><th>Farside 当前代码</th><th>可用起点/长度</th><th>建议用途</th><th>结论</th></tr></thead><tbody>
<tr><td>BTC</td><td>IBIT, FBTC, BITB, ARKB, BTCO, EZBC, BRRR, HODL, BTCW, MSBT, GBTC, BTC</td><td>2024-01-11；644个完整交易日</td><td>每日总流、发行商广度、流量/AUM、托管链上变动</td><td>单独择时未通过2026冻结期</td></tr>
<tr><td>ETH</td><td>ETHA, ETHB, FETH, ETHW, TETH, ETHV, QETH, EZET, ETHE, ETH</td><td>2024-07-23；约2年</td><td>ETF流 + 交易所储备 + 质押/解质押拆分</td><td>同规则现货复现盈利；ETF流不单独择时</td></tr>
<tr><td>SOL</td><td>BSOL, VSOL, FSOL, TSOL, SOEZ, GSOL</td><td>历史短，且产品含质押费差异</td><td>ETF流、托管钱包、stake account与验证者净变动</td><td>仅监控；不声称统计回测</td></tr>
</tbody></table>
</div>
<p class="small">ETF流量通常在美国交易日结束后才逐步完整。策略研究统一按“交易日t的流量在t+1收盘才可用”处理；不要让同日流量赚同日收益。</p>
</section>

<section>
<h2>7. 数据与泄漏审计</h2>
<div class="two">
 <div class="card ok"><h3>已做</h3><ul><li>目标权重再滞后一天；无同收盘信号收益</li><li>阈值仅用滚动历史并 shift(1)</li><li>现金利率计入；交易成本按权重变化计</li><li>开发/验证/冻结外部期分开</li><li>家族级循环置换、块自助、成本/延迟压力</li><li>静态扫描：0 blocker；滚动quantile为人工确认后的误报</li></ul></div>
 <div class="card danger"><h3>仍未解决</h3><ul><li>Coin Metrics当前地址标签可能回填历史，缺少标签vintage</li><li>Farside历史表可能事后修订，缺少逐日快照版本</li><li>最终分级规则在两次冻结失败后提出，已无纯净holdout</li><li>BTC/ETH结果使用单一主价格序列，仍需独立成交源复核</li><li>SOL免费同口径数据不足</li></ul></div>
</div>
</section>

<section>
<h2>8. 上线前的最低门槛</h2>
<div class="card"><ol>
<li><b>只做纸面：</b>从现在起每日保存原始API响应、地址标签版本和最终目标，至少6个月。</li>
<li><b>双源复核：</b>Coin Metrics + Nansen/自建节点；价格用Coin Metrics + Binance/Coinbase独立对账。</li>
<li><b>先验证增量：</b>比较分级策略与纯MA200；若链上层滚动12个月不改善回撤或Sharpe，停用链上加权。</li>
<li><b>不自动跟鲸鱼：</b>大额转账先分类为内部迁移、托管、桥、质押或真实外部流。</li>
<li><b>硬风控：</b>最大100%现货、不加杠杆、无真实资金自动执行；任何实盘需重新授权。</li>
</ol></div>
</section>

<section>
<h2>9. 来源与复现</h2>
<div class="card source-list"><ul>
<li><a href="https://docs.coinmetrics.io/api/v4/">Coin Metrics API v4</a> — 官方时间序列接口与目录。</li>
<li><a href="https://docs.coinmetrics.io/resources/faqs">Coin Metrics FAQ</a> — 日度UpperCamelCase为区间起点时间戳；交易所流用地址聚类估算。</li>
<li><a href="https://docs.coinmetrics.io/asset-metrics/exchange/flowinexusd">Coin Metrics Exchange Deposits</a> — 交易所地址识别与UTXO找零处理说明。</li>
<li><a href="https://farside.co.uk/bitcoin-etf-flow-all-data/">Farside BTC ETF All Data</a>、<a href="https://farside.co.uk/ethereum-etf-flow-all-data/">ETH All Data</a>、<a href="https://farside.co.uk/sol/">SOL ETF Flow</a>。</li>
<li><a href="https://docs.nansen.ai/api/token-god-mode/holders">Nansen Holders API</a> — whale/exchange标签、余额变化与实体聚合。</li>
<li><a href="https://docs.dune.com/web-app/alerts">Dune Alerts</a> — 定时查询后的邮件/Webhook告警；官方说明不适合关键实时系统。</li>
</ul>
<p class="small">核心研究文件：<code>scripts/btc_etf_flow_study.py</code>、<code>scripts/btc_etf_flow_revision.py</code>、<code>scripts/btc_onchain_exchange_flow_study.py</code>、<code>scripts/btc_scaled_onchain_overlay.py</code>、<code>scripts/eth_onchain_replication.py</code>。</p>
<p class="small">BTC链上数据 SHA-256：<code>{_sha256(ROOT / 'data' / 'interim' / 'btc_coinmetrics_onchain.csv')}</code></p>
</div>
</section>

<footer>生成时间：{generated_at}。本报告为研究与纸面监控建议，不是投资建议，也不授权真实资金交易。Crypto资产可能发生大幅损失；历史回测尤其受数据标签回填、选择偏差与结构变化影响。</footer>
</main></body></html>"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(json.dumps({"report": str(REPORT_PATH), "bytes": REPORT_PATH.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
