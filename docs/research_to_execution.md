# Research → Execution — operator's guide

How a tradable edge flows through `alpha_lab` end-to-end: **find it** (research leg), **hand it off**
(a target-weight function), **run it** (execution leg, mock for now). Mock = Binance **demo** funds;
going live is hard-gated (last section).

> **Interpreter.** Every `python` below means the project env. Either `conda activate py313` first, or
> substitute the full path `D:\conda\envs\py313\python.exe` for `python`. Run from the repo root, with
> the package importable (`python -m pip install -e .`, or prefix commands with `PYTHONPATH=src`).

---

## 0. The mental model

```
  RESEARCH  (alpha_lab/)                 HANDOFF                 EXECUTION  (quant_bot_manager/)
  notebooks + src/alpha_lab/   ──►  latest_target_weights()  ──►  strategy → broker → bot → runner
  finds edges                       a per-leg weight Series        runs edges as a mock/live bot
```

- **One-way dependency:** execution imports research, **never the reverse**.
- **One source of truth per strategy:** the same function computes weights for the backtest, the
  holdout eval, *and* live execution — so they can't drift. (P7 example: `alpha_lab.backtest.crypto_book`.)
- **The handoff is a function**, not a file dump: research produces `latest_target_weights(...) -> Series`.

---

## 1. Setup (once)

```bash
conda activate py313
python -m pip install -e .          # editable install so `import alpha_lab` / `quant_bot_manager` work
cp .env.example .env                # then add keys (see below); .env is gitignored
```

`.env` for the mock venue (one Binance **demo** key trades both spot + USD-M futures):
```
BINANCE_DEMO_KEY=...                # from https://demo.binance.com → API Management
BINANCE_DEMO_SECRET=...
```

---

## 2. Research leg — find / improve an edge

Notebook-first; lift reusable logic into `src/alpha_lab/`. The loop:

1. **Ideate** — use the `idea-generation` skill for new strategies/factors anchored to a *return source*
   (carry, trend, XS-mom, flow, macro, …). Orthogonality is engineered by return-source diversity.
2. **Prototype in a notebook** under `notebooks/`. Keep notebooks thin — import from the package.
3. **Backtest** with `alpha_lab.backtest.vector.run_backtest` (weights lagged 1 bar; per-leg slippage;
   funding on held weight). Vectorized pandas/numpy is the default.
4. **Leak-audit** — the cardinal sins. Run the scanner, then the manual checklist:
   ```bash
   python .claude/skills/auditing-for-leakage/scripts/scan_leakage.py <notebook_or_.py>
   ```
   No future info in signals, no same-bar signal→execution, no full-sample normalization, trailing
   windows only. (Or invoke the `auditing-for-leakage` skill.)
5. **Cost of cash** — report **excess of cash**: charge `rf_t · gross_long_notional` (financing) on top
   of commissions/slippage/funding. A flat book earns the risk-free hurdle, not zero.
6. **Out-of-sample / holdout** — the moment of truth. **Design and select on in-sample only**; look at
   the PM holdout **once**. Pattern: `notebooks/90_crypto_intraday/eval_holdout_2026.py` (releases the
   lock under audit via `PMHoldout(..., allow=True)`).
7. **Write a decision record** in `docs/research_decisions/…` — research question, mechanism, results,
   robustness, failure modes, verdict (`accept_monitoring` / etc.). See the P7 record for the template.
8. **Lift to the package** — once the logic is real, move it to `src/alpha_lab/backtest/<name>.py` as the
   single source of truth, exposing a `latest_target_weights(data, method=...) -> Series`.

**Worked example this repo carries — P7 crypto book** (`src/alpha_lab/backtest/crypto_book.py`): 5
low-correlation sleeves (carry / trend / XS-mom / funding-contra / macro), combined equal-capital. Improve
it with the same loop — e.g. the S5 vol-target fix: `notebooks/90_crypto_intraday/fix_s5.py` selects a
variant **in-sample**, `rebuild_v3_artifacts.py` + `render_multi_strategy_report.py` regenerate the HTML
report from the source of truth, `eval_holdout_2026.py` reports the OOS once.

---

## 3. The handoff — a target-weight function

Research exposes one function; execution consumes it. For P7:

```python
from alpha_lab.backtest import crypto_book as cb
bd  = cb.load_book_data("2022-01-01", "2025-12-31")   # or the live feed (below)
tgt = cb.latest_target_weights(bd, method="equal_capital")   # Series: BTC.s, ETH.p, … -> weight
```

Legs are `<COIN>.s` (spot) / `<COIN>.p` (USD-M perp). Positive = long, negative = short. That `Series`
is the entire contract between the legs.

---

## 4. Execution leg — run it as a mock bot

### 4a. Anatomy of a bot (declarative)
A bot = **strategy** (→ weights) + **feed** (market data) + **broker** (venue) + **default config**,
wired in a YAML file the registry reads:

```yaml
# configs/bots/p7_crypto_book.yaml
name: p7_crypto_book
strategy: p7_crypto_book     # -> registry STRATEGIES (live adapter wrapping alpha_lab)
feed: live_binance           # -> registry FEEDS (live ccxt klines/funding, leak-safe through yesterday)
broker: binance              # -> registry BROKERS (spot + USD-M futures)
default_config:
  capital: 10000.0
  method: equal_capital      # equal_capital | risk_budget
  max_gross: 2.0             # hard gross cap (× capital)
  interval_min: 15.0         # mark / rebalance cadence
  max_drawdown_pct: 0.20     # kill-switch: latch-halt if equity falls 20% below peak
  paused: false
  halt: false
```

The live strategy adapter (`src/quant_bot_manager/strategies/p7_crypto_book.py`) calls the research
function on live data — that's the only glue.

### 4b. CLI quickstart
```bash
python -m quant_bot_manager.cli bots                                  # list defined bots
python -m quant_bot_manager.cli plan --bot p7_crypto_book --capital 10000   # dry order plan (no auth)
python -m quant_bot_manager.cli rebalance --mode demo --capital 10000        # one-shot rebalance (demo)
python -m quant_bot_manager.cli run --mode demo --interval-min 15 --capital 10000   # continuous process
```

`plan` is the safe first look — it prints the target weights and the orders it *would* place, no keys
needed. `run` is the continuous mock-trading process: each cycle it marks-to-market, checks the
kill-switch, and rebalances once per UTC day toward the live signal.

### 4c. UI cockpit (monitor + control)
```bash
streamlit run src/quant_bot_manager/ui/app.py          # -> http://localhost:8501
```
- **Sidebar:** pick the bot (one per `configs/bots/*.yaml`).
- **Monitor tab:** equity curve (mark-to-market), perp positions, spot balances, recent rebalances, drawdown.
- **Control tab:** Start / Stop / Pause / Rebalance-now; edit capital / method / max-gross / max-drawdown /
  interval (applies next cycle); **kill-switch** (HALT / release, clear drawdown auto-halt).

The UI never imports trading code — it reads the bot's SQLite store and drives the process via the CLI.

### 4d. Risk & kill-switch (every cycle)
| control | effect |
|---|---|
| `max_gross` | a rebalance whose gross exposure exceeds the cap is **skipped** |
| `max_drawdown_pct` | equity falls this far below peak → **auto-halt latches** (must be cleared manually) |
| `halt` | hard manual kill-switch — stops new orders, keeps marking |
| `paused` | soft pause — keeps marking, places no orders |

### 4e. State & monitoring
All runtime state is one SQLite DB per bot: `data/results/bots/<bot>/bot.db` (equity, rebalances,
status, config — WAL mode, safe for concurrent UI reads). Pre-SQLite CSV/JSON is auto-imported once.

### 4f. Keep it running 24/7
Register a Windows Task Scheduler job at startup (see `notebooks/90_crypto_intraday/PAPER_TRADING.md`),
or just leave `cli run` / the UI **Start** button running.

---

## 5. Add a NEW strategy as a bot (the pluggable path)

1. **Research** it (section 2) → expose `latest_target_weights(...)` in `src/alpha_lab/…`.
2. **Live adapter:** add `src/quant_bot_manager/strategies/<name>.py` with
   `latest_targets(method="...", *, feed=...) -> (weights, asof, last_px)` (wrap the research function on
   live data; keep it leak-safe — signal through *yesterday's* close).
3. **Register** in `src/quant_bot_manager/core/registry.py`: add to `_strategies()` (and `_feeds()` /
   `_brokers()` if new).
4. **Declare** the bot: `configs/bots/<name>.yaml` (strategy / feed / broker / default_config).
5. **Verify & run:**
   ```bash
   python -m quant_bot_manager.cli bots                 # <name> shows up
   python -m quant_bot_manager.cli plan --bot <name>    # sane targets?
   python -m quant_bot_manager.cli run  --bot <name> --mode demo
   ```
   It appears in the UI's bot selector automatically.

A new **broker/venue** is the same idea: implement the `brokers/base.Broker` interface, register it, point
a bot's `broker:` at it.

---

## 6. Going LIVE (real money — hard-gated, never automatic)

Live is intentionally **not** wired into the UI and is **never** started autonomously. To go live, at the shell:
```bash
# 1) add live keys to .env:  BINANCE_FUT_KEY/SECRET (+ spot)   2) export CONFIRM_LIVE=YES
python -m quant_bot_manager.cli run --mode live --i-understand-live --max-gross 1 --capital <small>
```
Both the CLI **and** the broker refuse live unless `--i-understand-live` **and** `CONFIRM_LIVE=YES` are
present. Validate on demo first; size small; the P7 2026 OOS was negative (it defends, it doesn't profit
in a crypto bear).

---

## 7. Daily operating checklist (mock)

- [ ] Bot **running**? (UI badge 🟢, heartbeat recent) — else Start.
- [ ] **Equity / drawdown** sane? Kill-switch not latched?
- [ ] **Last rebalance** happened today (UTC) with `status ok`?
- [ ] Positions match intent (perp legs + spot)?
- [ ] Re-tune via the Control tab if needed (applies next cycle).

---

## 8. Where things live

| Thing | Path |
|---|---|
| Research strategies (source of truth) | `src/alpha_lab/backtest/` |
| Backtest engine | `alpha_lab.backtest.vector.run_backtest` |
| Leak scanner | `.claude/skills/auditing-for-leakage/scripts/scan_leakage.py` |
| Decision records | `docs/research_decisions/` |
| Live strategy adapters | `src/quant_bot_manager/strategies/` |
| Brokers / venues | `src/quant_bot_manager/brokers/` |
| Bot definitions (YAML) | `configs/bots/` |
| Registry / config / runner / risk / store | `src/quant_bot_manager/core/` |
| CLI | `python -m quant_bot_manager.cli` |
| UI cockpit | `src/quant_bot_manager/ui/app.py` |
| Per-bot runtime state | `data/results/bots/<bot>/bot.db` |
| Execution runbook | `notebooks/90_crypto_intraday/PAPER_TRADING.md` |

---

## 9. Guardrails (non-negotiable)

1. **Leak-safety** — signal at `t` uses data ≤ `t`; execution strictly after. Audit before trusting a Sharpe.
2. **Cost of cash** — always report excess-of-cash; beat the risk-free hurdle, not zero.
3. **Holdout discipline** — select on in-sample; touch the holdout once; never tune on it.
4. **Never start real-money trading autonomously** — live is triple-gated and human-initiated only.
5. **Secrets & data** — `.env`, `data/raw/`, `data/private/` are never committed; `data/raw/` is read-only.
