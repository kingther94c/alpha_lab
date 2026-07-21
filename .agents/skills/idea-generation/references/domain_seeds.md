# Domain seeds — alpha_lab primitives

Raw material for the combinatorial operators (morphological recombination, SCAMPER, Geneplore).
These are the things alpha_lab can *actually* test today, so ideas built from them are testable by
construction. `scripts/random_stimulus.py` samples the morphological matrix at the bottom. Extend
this file as the repo grows — it's meant to be living.

## Universes available

| Universe | Where | Notes |
|---|---|---|
| US sector SPDR ETFs | `configs/us_sector_etf.csv` | XS-momentum / active-MV work (`02`, `03`) |
| Country ETFs | `configs/country_etf_universe.csv` | XS country momentum (`05`) |
| Crypto perps, intraday | `configs/crypto_intraday_universe.csv` | BTC/ETH 5m via Binance Vision (`90_*`) |
| Broad ETFs / 60-40 sleeves | notebooks | dual-momentum (`10`), risk parity (`11`) |
| Managed futures (DBMF + proxy) | notebooks | overlay & sizing (`06`, `07`) |
| Commodities (BCOM) | `50_risk/01` | rolling-vol risk work |
| FX, rates, macro | via FRED / yfinance | mostly untapped as *signal* universes |

## Data loaders — and their native signals

The highest-novelty move is to signal on a source you haven't signalled on. (`src/alpha_lab/data/loaders/`)

- **yfinance** — daily OHLCV / adjusted close. Native signals: price trend, XS momentum, realised
  vol, value proxies, gaps, dividends.
- **fred** — macro & rates series. Native signals: yield-curve slope, credit spreads, real rates,
  growth/inflation surprises, financial-conditions indices → regime gates, carry, macro tilts.
- **binance_vision** — crypto intraday klines (and funding / open-interest where available). Native
  signals: intraday trend/reversal, **funding-rate carry**, **open-interest / positioning**,
  time-of-day & day-of-week seasonality, realised-vol microstructure.
- **polymarket** — prediction-market probabilities. *No signal built yet.* Native signals:
  market-implied event probability, forward-looking macro-event regime gates, probability drift.
- **local** — cached / private panels.

## Signal primitives (the menu)

- **Trend / time-series momentum** — multi-horizon; single-asset filter or cross-asset book.
- **Cross-sectional momentum / relative strength** — rank within a universe, hold top vs bottom.
- **Short-term reversal** — buy recent losers at a short horizon.
- **Carry** — crypto funding rate; futures roll yield; FX rate differential; equity dividend/earnings yield.
- **Value / mean-reversion** — distance from a fair-value anchor or moving average.
- **Volatility / risk** — realised vol, vol-of-vol, vol risk premium, vol-targeting.
- **Seasonality / calendar** — month, turn-of-month, day-of-week, time-of-day.
- **Cross-asset / lead-lag** — rates → equities, BTC → alts, USD → commodities, credit → equity vol.
- **Microstructure / intraday** — open interest, funding, gap behaviour, order-flow proxies.
- **Macro / regime** — yield curve, credit spreads, financial conditions, growth/inflation nowcasts.
- **Event / flow** — index/ETF rebalances, Fed decisions (`80_event_study`), prediction-market events.

## Conditioners / regimes (the most under-explored column)

Holding the signal fixed and varying the *conditioner* is where regime edge hides. Options: vol
regime (high/low), trend vs chop, bull/bear, rates up/down, risk-on/off, cross-sectional dispersion
(high/low), funding extreme, credit-spread regime, time-of-day, day-of-week, pre/post a scheduled event.

## Portfolio-construction primitives

(`src/alpha_lab/portfolio/`, `src/alpha_lab/backtest/`)

- Long-only (`long_only.py`); dollar-neutral long/short.
- Top/bottom-N selection; equal vs inverse-vol vs mean-variance weighting (`active_mv.py`).
- Vol-targeting (`vol_target.py`); risk parity; overlay / sleeve sizing.

## Frictions & reality constraints (inputs to the "appropriate" filter)

Every idea is graded against these — they are why *tradability* is a scored axis.

- **Costs** — commission_bps + slippage_bps (`configs/backtest.yaml`); be willing to double them.
- **Turnover** — high-frequency signals die on cost; check break-even cost vs assumed cost.
- **Capacity** — implied trade size vs ADV, especially for intraday and small-universe ideas.
- **Shorting** — borrow availability/cost; many "long/short" ideas are really long-only after frictions.
- **Leverage** — cap and ruin risk for vol-targeted / levered-proxy books.
- **Crypto-specific** — funding cost paid to hold, 24/7 calendar, exchange/microstructure quirks.
- **Data** — availability, vendor field changes, splice misalignment, survivorship.
- **Lookahead** — the silent killer; signals must use only data ≤ t (see `AGENTS.md`).

## Strategy morphological matrix

Columns of the box for operator 3. Pick one value per column for a candidate strategy; the script
samples this with `--matrix`. The interesting cards usually come from an *odd but feasible* row.

| Universe | Data source | Signal | Horizon | Conditioner | Portfolio | Timing |
|---|---|---|---|---|---|---|
| US sector ETFs | yfinance | TS momentum | intraday (5m) | vol high/low | long-only | calendar (ME/WE) |
| Country ETFs | fred | XS momentum | 1 day | trend vs chop | dollar-neutral L/S | vol-triggered |
| Crypto perps | binance_vision | short-term reversal | 1 week | dispersion high/low | top/bottom-N | event-triggered |
| 60-40 sleeves | polymarket | carry (funding/roll) | 1–3 months | rates up/down | inverse-vol | breakout-triggered |
| Managed futures | local | value / mean-reversion | 6–12 months | risk-on/off | mean-variance | always-on (daily) |
| Commodities (BCOM) | | volatility / vol-target | | funding extreme | risk parity | turn-of-month |
| FX / rates | | seasonality (ToD/DoW) | | time-of-day | vol-target | day-of-week |
| | | cross-asset lead-lag | | credit-spread regime | overlay / sizing | |
| | | open-interest / positioning | | pre/post scheduled event | | |
