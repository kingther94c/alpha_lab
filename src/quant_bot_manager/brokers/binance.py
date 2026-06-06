"""Binance broker (ccxt) for the execution leg — spot + USD-M futures.

Environments: ``demo`` (demo-*.binance.com via ccxt enableDemoTrading — current mock venue),
``testnet`` (legacy spot testnet + futures-testnet via manual URL), ``live`` (real money, gated).
Spot legs (``.s``) are long-only on the spot account; perp legs (``.p``) trade USD-M futures.
"""
from __future__ import annotations

import os

import ccxt
import pandas as pd

from quant_bot_manager.brokers.base import Broker
from quant_bot_manager.core.config import EPS

SPOT_SYM = {"BTC.s": "BTC/USDT", "ETH.s": "ETH/USDT"}
PERP_SYM = {"BTC.p": "BTC/USDT:USDT", "ETH.p": "ETH/USDT:USDT",
            "SOL.p": "SOL/USDT:USDT", "BNB.p": "BNB/USDT:USDT"}


class BinanceBroker(Broker):
    name = "binance"

    def __init__(self, mode: str = "demo"):
        if mode not in ("demo", "testnet", "live"):
            raise ValueError(f"mode must be demo/testnet/live, got {mode!r}")
        self.mode = mode
        self.spot = None
        self.fut = None
        self.have_spot = self.have_fut = False

    # -- connection ---------------------------------------------------------
    def connect(self) -> None:
        if self.mode == "demo":
            dk, ds = os.environ.get("BINANCE_DEMO_KEY"), os.environ.get("BINANCE_DEMO_SECRET")
            if not (dk and ds):
                raise RuntimeError("no BINANCE_DEMO_KEY/SECRET in .env")
            self.spot = ccxt.binance({"apiKey": dk, "secret": ds, "enableRateLimit": True})
            self.spot.enableDemoTrading(True)
            self.fut = ccxt.binanceusdm({"apiKey": dk, "secret": ds, "enableRateLimit": True})
            self.fut.enableDemoTrading(True)
        else:
            live = self.mode == "live"
            if live and os.environ.get("CONFIRM_LIVE") != "YES":
                raise RuntimeError("LIVE refused: set CONFIRM_LIVE=YES to trade real money")
            pfx = "BINANCE_" + ("" if live else "TESTNET_")
            sk, ss = os.environ.get(pfx + "SPOT_KEY"), os.environ.get(pfx + "SPOT_SECRET")
            fk, fs = os.environ.get(pfx + "FUT_KEY"), os.environ.get(pfx + "FUT_SECRET")
            if not (sk or fk):
                raise RuntimeError(f"no keys in .env ({pfx}SPOT_* / {pfx}FUT_*)")
            self.spot = ccxt.binance({"apiKey": sk, "secret": ss, "enableRateLimit": True}) if (sk and ss) else None
            self.fut = ccxt.binanceusdm({"apiKey": fk, "secret": fs, "enableRateLimit": True}) if (fk and fs) else None
            if not live:  # ccxt deprecated binanceusdm sandbox; point its URLs at the futures testnet.
                if self.spot:
                    self.spot.set_sandbox_mode(True)
                if self.fut:
                    for kk, uu in list(self.fut.urls["api"].items()):
                        if isinstance(uu, str) and "fapi.binance.com" in uu:
                            self.fut.urls["api"][kk] = uu.replace("fapi.binance.com", "testnet.binancefuture.com")
        self.have_spot, self.have_fut = self.spot is not None, self.fut is not None
        if self.fut:
            self.fut.load_markets()
        if self.spot:
            self.spot.load_markets()

    # -- pricing / planning -------------------------------------------------
    def public_prices(self, legs: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        try:
            sp = ccxt.binance({"enableRateLimit": True})
            for lg in [x for x in legs if x in SPOT_SYM]:
                out[lg] = float(sp.fetch_ticker(SPOT_SYM[lg])["last"])
            fu = ccxt.binanceusdm({"enableRateLimit": True})
            for lg in [x for x in legs if x in PERP_SYM]:
                out[lg] = float(fu.fetch_ticker(PERP_SYM[lg])["last"])
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] public price fetch failed ({type(e).__name__}: {e}); using last close.")
        return out

    def build_plan(self, targets: pd.Series, capital: float, prices: dict[str, float]) -> pd.DataFrame:
        rows = []
        for leg, w in targets.items():
            if abs(w) < EPS or leg not in prices or not prices[leg]:
                continue
            notional = w * capital
            rows.append({"leg": leg, "venue": "spot" if leg.endswith(".s") else "perp",
                         "symbol": SPOT_SYM.get(leg) or PERP_SYM.get(leg), "weight": w,
                         "side": "BUY" if w > 0 else "SELL", "price": prices[leg],
                         "notional_usdt": notional, "qty": notional / prices[leg]})
        return pd.DataFrame(rows)

    # -- execution ----------------------------------------------------------
    MIN_NOTIONAL = 20.0   # Binance USD-M minimum order notional (USDT); also a sane dust floor for spot

    def rebalance_to_target(self, plan: pd.DataFrame, *, from_flat: bool = False) -> list[tuple]:
        placed: list[tuple] = []
        if plan.empty:
            return placed

        def _order(ex, venue, sym, delta, price):
            if abs(delta * price) < self.MIN_NOTIONAL:   # below exchange min / dust -> skip
                return
            amt = float(ex.amount_to_precision(sym, abs(delta)))
            if amt <= 0:
                return
            side = "buy" if delta > 0 else "sell"
            try:                                          # one bad leg must not abort the rebalance
                ex.create_order(sym, "market", side, amt)
                placed.append((venue, side, sym, amt))
            except Exception as e:   # noqa: BLE001
                print(f"  [warn] {venue} {side} {sym} {amt} failed: {type(e).__name__}: {str(e)[:90]}")

        if self.fut is not None:
            cur = {} if from_flat else {
                p["symbol"]: float(p.get("contracts") or 0) * (1 if p.get("side") == "long" else -1)
                for p in self.fut.fetch_positions()}
            for _, r in plan[plan.venue == "perp"].iterrows():
                _order(self.fut, "perp", r.symbol, r.qty - cur.get(r.symbol, 0.0), r.price)
        if self.spot is not None:
            bal = {} if from_flat else self.spot.fetch_balance()
            for _, r in plan[plan.venue == "spot"].iterrows():
                base = r.symbol.split("/")[0]
                have = 0.0 if from_flat else float(bal.get(base, {}).get("free", 0) or 0)
                _order(self.spot, "spot", r.symbol, r.qty - have, r.price)
        return placed

    # -- monitoring ---------------------------------------------------------
    def mark_to_market(self) -> tuple[float, float, float]:
        fb = self.fut.fetch_balance()
        fe = float(fb.get("info", {}).get("totalMarginBalance") or fb["total"].get("USDT", 0))
        sb = self.spot.fetch_balance()["total"]
        tk = self.spot.fetch_tickers(["BTC/USDT", "ETH/USDT"])
        se = (sb.get("USDT", 0) or 0) + (sb.get("USDC", 0) or 0) \
            + (sb.get("BTC", 0) or 0) * tk["BTC/USDT"]["last"] + (sb.get("ETH", 0) or 0) * tk["ETH/USDT"]["last"]
        return fe + se, fe, se

    def positions_snapshot(self) -> dict:
        out = {"perp": [], "spot": {}}
        try:
            for p in self.fut.fetch_positions():
                c = float(p.get("contracts") or 0)
                if c:
                    out["perp"].append({"symbol": p["symbol"], "side": p.get("side"), "contracts": c,
                                        "notional": float(p.get("notional") or 0),
                                        "uPnl": float(p.get("unrealizedPnl") or 0), "entry": p.get("entryPrice")})
            b = self.spot.fetch_balance()["total"]
            out["spot"] = {a: round(float(v), 6) for a, v in b.items() if v and a in ("BTC", "ETH", "USDT", "USDC")}
        except Exception as e:  # noqa: BLE001
            out["error"] = str(e)[:100]
        return out
