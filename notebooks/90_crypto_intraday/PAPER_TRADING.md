# Running the P7 book — quant_bot_manager (execution leg)

The strategy (research) lives in `alpha_lab`; the **execution engine now lives in the
`quant_bot_manager` package** (`src/quant_bot_manager/`). The P7 book is **daily-rebalanced**
(not intraday — intraday was cost-killed), executed via **ccxt**.

```
src/quant_bot_manager/
  strategies/p7_crypto_book.py  # live target weights (wraps alpha_lab.backtest.crypto_book)
  brokers/binance.py            # spot + USD-M futures: demo / testnet / live
  core/{bot,runner,config}.py   # Bot, the loop/one-shot, paths
  cli.py                        # entrypoint
  ui/app.py                     # Streamlit cockpit
```
Per-bot runtime artifacts: `data/results/bots/p7_crypto_book/` (`equity_log.csv`,
`rebalance_log.csv`, `status.json`, `config.json`, `state.json`).

> Run from the repo root with the package importable (`pip install -e .`, or `PYTHONPATH=src`).

## Setup — a Binance DEMO key (mock funds)
`https://demo.binance.com` → API Management → HMAC key (Reading + Spot + Futures). One key trades
both venues. Put in `.env` (gitignored):
```
BINANCE_DEMO_KEY=...
BINANCE_DEMO_SECRET=...
```

## Commands
```bash
# dry order plan (no auth):
python -m quant_bot_manager.cli plan --capital 10000
# one-shot rebalance on demo:
python -m quant_bot_manager.cli rebalance --mode demo --capital 10000
# continuous mock-trading process (mark-to-market + daily rebalance on live signals):
python -m quant_bot_manager.cli run --mode demo --interval-min 15 --capital 10000
# UI cockpit (monitor + control, incl. start/stop/pause/rebalance):
streamlit run src/quant_bot_manager/ui/app.py     # -> http://localhost:8501
```

## Monitor / stop
- Equity / trades / status: `data/results/bots/p7_crypto_book/`
- Stop: UI **Stop** button, or kill the `quant_bot_manager.cli run` process.

## 24/7 persistence (survives logoff/reboot)
Register a Task Scheduler job at startup:
```powershell
$action  = New-ScheduledTaskAction -Execute "D:\conda\envs\py313\python.exe" `
  -Argument "-m quant_bot_manager.cli run --mode demo --interval-min 15 --capital 10000" `
  -WorkingDirectory "D:\projects\git_projects\alpha_lab\.claude\worktrees\gifted-elgamal-ddd50f"
Register-ScheduledTask -TaskName "p7_bot" -Action $action -Trigger (New-ScheduledTaskTrigger -AtStartup) -RunLevel Limited
```

## Read before sizing — 2026 holdout reality
On the released 2026 holdout the book was **defensive but negative** (equal-capital −5% vs BTC −17%,
a third of the drawdown). S5 macro misfired; carry compressed. Validate on demo first; S5 is the fix candidate.

## Going LIVE (real money — hard-gated)
Add `BINANCE_FUT_KEY/SECRET` (+ spot) from binance.com, set `CONFIRM_LIVE=YES`, then:
```bash
python -m quant_bot_manager.cli rebalance --mode live --i-understand-live --capital <small> --max-gross 1
```
The CLI **and** the broker refuse live unless `--i-understand-live` + `CONFIRM_LIVE=YES` are both present.
