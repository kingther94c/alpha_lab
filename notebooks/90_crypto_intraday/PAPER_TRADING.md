# Paper- / live-trading the P7 book on Binance

The deployable strategy is the **daily-rebalanced** P7 multi-strategy book (5 low-correlation sleeves).
**Not intraday** — intraday signals were cost-killed in prior research; this book trades **once per day**.
Execution is via **ccxt**; the signal comes from `src/alpha_lab/backtest/crypto_book.py` (the same code as
the backtest), so live == research.

| File | Role |
|---|---|
| `crypto_book.py` (in `src/`) | the strategy: target weights (single source of truth) |
| `binance_paper_trade.py` | one-shot rebalance to target (dry / demo / testnet / live) |
| `mock_trader_loop.py` | **continuous mock-trading process** (mark-to-market + daily rebalance) |
| `backtrader_book.py` | backtrader replication of the book (backtest cross-check) |

## Environments (all mock = no real money except `live`)
- **`demo`** ✅ — Binance Demo Trading (`demo.binance.com`). **One key** trades spot + USD-M futures via
  ccxt `enableDemoTrading`. This is the current, working mock environment — use it.
- `testnet` — legacy spot testnet (`testnet.binance.vision`) + futures testnet (`testnet.binancefuture.com`).
  ccxt deprecated the futures-testnet sandbox, so prefer `demo`.
- `live` — **real money**. Hard-gated (see bottom).

## 1. Get a DEMO key
Go to **https://demo.binance.com** → API Management → create an HMAC key (Reading + Spot + Futures).
It comes with mock USDT on both spot and futures. Add to `.env` (gitignored):
```
BINANCE_DEMO_KEY=...
BINANCE_DEMO_SECRET=...
```

## 2. Run the continuous mock-trading process  ← the "process持续进行"
```bash
python notebooks/90_crypto_intraday/mock_trader_loop.py --interval-min 15 --capital 10000
```
What it does, forever:
- **every 15 min** — mark-to-market, append equity to `data/results/crypto_v3_multi/mock_equity_log.csv`
- **once per UTC day** — pull fresh daily klines + funding from the Binance API (ccxt, leak-safe through
  *yesterday's* close), recompute targets via `crypto_book`, and **reconcile** the demo positions
  (logged to `mock_rebalance_log.csv`)

State (`mock_state.json`) persists the last rebalance date, so a restart resumes without double-trading.

One-shot (no loop), e.g. to establish/inspect once:
```bash
python notebooks/90_crypto_intraday/binance_paper_trade.py --mode dry  --method equal_capital --capital 10000   # plan only
python notebooks/90_crypto_intraday/binance_paper_trade.py --mode demo --method equal_capital --capital 10000   # place once
```

## 3. Monitor / stop
- **Equity curve:** `data/results/crypto_v3_multi/mock_equity_log.csv`
- **Rebalances:** `data/results/crypto_v3_multi/mock_rebalance_log.csv`
- **Stop:** `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? {$_.CommandLine -like '*mock_trader_loop*'} | % {Stop-Process -Id $_.ProcessId -Force}`

## 4. Run it persistently (survives logoff / reboot)
A bare process dies with its shell/session. For 24/7, register a **Windows Task Scheduler** job at startup:
```powershell
$action  = New-ScheduledTaskAction -Execute "D:\conda\envs\py313\python.exe" `
  -Argument "notebooks\90_crypto_intraday\mock_trader_loop.py --interval-min 15 --capital 10000" `
  -WorkingDirectory "D:\projects\git_projects\alpha_lab\.claude\worktrees\gifted-elgamal-ddd50f"
$trigger = New-ScheduledTaskTrigger -AtStartup
Register-ScheduledTask -TaskName "p7_mock_trader" -Action $action -Trigger $trigger -RunLevel Limited
```

## Read before sizing — 2026 holdout reality
On the released 2026 holdout the book was **defensive but negative**: equal-capital −5% vs BTC −17%
(a third of the drawdown), but it did not make money. S5 macro misfired (−29%); carry compressed. Treat
the demo run as forward validation and S5 as the fix candidate before any real capital.

## Going LIVE (real money — hard-gated)
Only after the demo book looks right (and ideally after fixing S5). Add `BINANCE_FUT_KEY/SECRET` (+ spot)
from binance.com, set `CONFIRM_LIVE=YES` in `.env`, then:
```bash
python notebooks/90_crypto_intraday/binance_paper_trade.py --mode live --i-understand-live --capital <small> --max-gross 1
```
The executor **refuses** live unless all three guards (live keys + `CONFIRM_LIVE=YES` + `--i-understand-live`) are present.
