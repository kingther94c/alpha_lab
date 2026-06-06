"""Streamlit cockpit for quant_bot_manager — monitor + control trading bots.

Pick a bot in the sidebar (one per ``configs/bots/*.yaml``); monitor its equity / positions and
control it (start/stop/pause, re-tune, manual rebalance, kill-switch). Reads the bot's SQLite
store and drives the process via the package CLI — never imports the trading code directly.

Run:  streamlit run src/quant_bot_manager/ui/app.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # put src/ on path
import pandas as pd
import streamlit as st

from quant_bot_manager.core import config, registry, state

st.set_page_config(page_title="Quant Bot Manager", layout="wide")
st.title("⚙️ Quant Bot Manager")
st.caption("alpha_lab finds the edge · quant_bot_manager runs it. Environment: Binance **demo** (mock funds).")

# ---------------- bot selector ----------------
bots = registry.list_bots() or [config.DEFAULT_BOT]
default_idx = bots.index(config.DEFAULT_BOT) if config.DEFAULT_BOT in bots else 0
bot = st.sidebar.selectbox("Bot", bots, index=default_idx)
st.sidebar.caption(f"{len(bots)} bot(s) defined in configs/bots/")

status = state.read_status(bot)
running = state.is_running(bot)
cfg = status.get("config") or state.read_config(bot)
df_eq = state.read_equity(bot)

# ---------------- status banner ----------------
if not running:
    badge = "🔴 STOPPED"
elif status.get("halted"):
    badge = "⛔ HALTED"
elif cfg.get("paused"):
    badge = "🟡 PAUSED"
else:
    badge = "🟢 RUNNING"
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Bot", badge)
eq = status.get("equity")
c2.metric("Equity (demo)", f"${eq:,.0f}" if eq else "—")
if len(df_eq) > 1 and eq:
    base = df_eq["total_equity"].iloc[0]
    pnl = df_eq["total_equity"].iloc[-1] - base
    c3.metric("PnL since start", f"${pnl:,.0f}", delta=f"{pnl / base * 100:+.2f}%")
else:
    c3.metric("PnL since start", "—")
dd = status.get("drawdown")
c4.metric("Drawdown", f"{dd:.1%}" if dd is not None else "—")
c5.metric("Last rebalance", str(status.get("last_rebal_date") or "—"))

left, right = st.columns([1, 4])
if left.button("🔄 Refresh"):
    st.rerun()
hb = status.get("last_heartbeat")
hb_txt = "—"
if hb:
    try:
        age = (dt.datetime.now(dt.UTC) - dt.datetime.fromisoformat(hb)).total_seconds()
        hb_txt = f"{age / 60:.0f} min ago"
    except Exception:
        pass
right.caption(f"heartbeat {hb_txt} · cadence {cfg.get('interval_min', '?')} min · weighting {cfg.get('method', '?')} · "
              f"capital ${float(cfg.get('capital', 0)):,.0f} · max gross {cfg.get('max_gross', '?')}× · "
              f"max DD {float(cfg.get('max_drawdown_pct', 0)) * 100:.0f}% · cycle {status.get('cycle', '—')}")

reasons = status.get("risk_reasons") or []
if status.get("halted"):
    st.error("⛔ Trading HALTED — " + ("; ".join(reasons) or "kill-switch engaged"))
elif reasons:
    st.warning("⚠️ " + "; ".join(reasons))
if status.get("error"):
    st.error(f"last cycle error: {status['error']}")

tab_mon, tab_ctrl = st.tabs(["📊 Monitor", "🎛️ Control"])

with tab_mon:
    st.subheader("Equity curve — mark-to-market (demo)")
    if len(df_eq):
        st.line_chart(df_eq.set_index("ts")[["total_equity", "fut_equity", "spot_equity"]])
        st.caption("Note: includes ~constant demo-faucet USDC; track the change, not the absolute.")
    else:
        st.info("No equity data yet — start the bot in the Control tab.")
    p = status.get("positions") or {}
    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("Perp positions")
        perps = p.get("perp", [])
        st.dataframe(pd.DataFrame(perps) if perps else pd.DataFrame({"info": ["flat / none"]}),
                     width="stretch", hide_index=True)
    with cc2:
        st.subheader("Spot balances")
        sp = p.get("spot", {})
        st.dataframe(pd.DataFrame([sp]) if sp else pd.DataFrame({"info": ["—"]}),
                     width="stretch", hide_index=True)
    st.subheader("Recent rebalances")
    reb = state.read_rebalances(bot)
    st.dataframe(reb.tail(15)[::-1] if len(reb) else pd.DataFrame({"info": ["none yet"]}),
                 width="stretch", hide_index=True)

with tab_ctrl:
    cur = state.read_config(bot)
    st.subheader("Parameters")
    with st.form("params"):
        capital = st.number_input("Capital (USDT)", value=float(cur["capital"]), step=1000.0, min_value=100.0)
        method = st.selectbox("Weighting", ["equal_capital", "risk_budget"],
                              index=0 if cur["method"] == "equal_capital" else 1)
        max_gross = st.slider("Max gross (× capital)", 0.5, 5.0, float(cur["max_gross"]), 0.1)
        max_dd = st.slider("Max drawdown kill-switch (%)", 1, 90, int(float(cur["max_drawdown_pct"]) * 100), 1)
        interval = st.number_input("Mark / rebalance interval (min)", value=float(cur["interval_min"]),
                                   step=5.0, min_value=1.0)
        if st.form_submit_button("💾 Save (applies next cycle)"):
            state.write_config({"capital": capital, "method": method, "max_gross": max_gross,
                                "max_drawdown_pct": max_dd / 100.0, "interval_min": interval}, bot)
            st.success("Config saved — the running bot picks it up on its next cycle.")

    st.subheader("Run control")
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("▶ Start", disabled=running, width="stretch"):
        st.toast(state.start_bot(state.read_config(bot), bot))
        st.rerun()
    if b2.button("⏹ Stop", disabled=not running, width="stretch"):
        st.toast(state.stop_bot(bot))
        st.rerun()
    pause_lbl = "⏵ Resume" if cfg.get("paused") else "⏸ Pause"
    if b3.button(pause_lbl, disabled=not running, width="stretch"):
        state.set_paused(not cfg.get("paused"), bot)
        st.rerun()
    if b4.button("🔁 Rebalance now", width="stretch"):
        with st.spinner("placing demo orders…"):
            out = state.manual_rebalance(cur["capital"], cur["method"], bot)
        st.code(out[-1800:] or "(no output)")

    st.subheader("🛑 Kill-switch")
    k1, k2 = st.columns(2)
    if cfg.get("halt"):
        if k1.button("✅ Release manual halt", width="stretch"):
            state.set_halt(False, bot)
            st.rerun()
    elif k1.button("🛑 HALT trading now", width="stretch"):
        state.set_halt(True, bot)
        st.rerun()
    if status.get("auto_halted"):
        if k2.button("♻ Clear drawdown auto-halt", width="stretch"):
            state.clear_auto_halt(bot)
            st.rerun()
    st.caption("HALT stops new orders (keeps marking-to-market). Auto-halt latches when drawdown "
               "breaches the limit above — clear it to resume.")

    st.divider()
    st.subheader("⛔ Live trading")
    st.warning(
        "Live = **real money** and is intentionally **not** wired into this UI. To go live, at the shell: "
        "add `BINANCE_*` live keys to `.env`, set `CONFIRM_LIVE=YES`, and run "
        "`python -m quant_bot_manager.cli run --mode live --i-understand-live --max-gross 1`. "
        "The 2026 out-of-sample was −5% — validate on demo first.")
