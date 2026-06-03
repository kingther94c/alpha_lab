# Return sources — the map ideas should move along

An idea in this repo only counts if it harvests a *real* return source. This is the taxonomy of
where returns actually come from. Use it two ways during a session:

- **Anchor generation.** Locate the target on this map, then generate by *moving along it* —
  to an adjacent premium (same family, new asset), a transplanted premium (one that works
  elsewhere, via analogy), or a blended premium (two sources → an emergent edge).
- **Run the orthogonality check** (the investment-specific half of the filter): is the idea
  just market beta or a known factor (momentum / value / low-vol / carry) repackaged? If so,
  it must state the *incremental* edge over the cheap version, or be downgraded.

Each entry: **mechanism · why it persists or decays · capacity · where alpha_lab can test it ·
crowding watch.**

## 1. Risk premia — paid for bearing a risk someone else won't

- **Equity premium** — paid to hold equity drawdown risk. Persists structurally; near-infinite
  capacity. Test: yfinance ETFs. Crowding: n/a (it's the beta you must beat).
- **Term / duration premium** — paid to hold rate risk; varies with the cycle. Test: FRED rate
  series, rate ETFs. Crowding: low. *The conditioner for risk-on/off ideas lives here.*
- **Credit premium** — paid for default/illiquidity risk. Test: FRED spreads (HY/IG OAS), credit
  ETFs. Crowding: medium; correlates with equity in stress.
- **Volatility risk premium (VRP)** — sell insurance, collect implied-minus-realised. High Sharpe,
  fat left tail. Test: **needs an options-IV source (Deribit/CBOE) — current gap.** Crowding: high.
- **Carry** — hold the high-yielder, fund with the low. FX rate differential (yfinance/FRED),
  commodity roll yield (needs a futures-curve source — partial gap), **crypto perp funding
  (binance_vision — fully available)**. Crowding: high; "carry crashes" in unwinds.
- **Liquidity premium** — paid to hold the harder-to-trade asset / provide immediacy. Test: small
  caps, off-the-run, intraday provision. Crowding: self-correcting (capacity-limited).

## 2. Style factors / anomalies — part risk, part behavioural

- **Momentum** (time-series trend & cross-sectional relative strength) — under-reaction + delayed
  diffusion. Persistent but crowded; crashes on sharp reversals. Test: all universes (this repo's
  core: `02`,`05`,`08`,`09`,`10`). Crowding: **high** — the *default* version is arbitraged; edge
  is in the conditioner/timing/horizon, not the raw factor.
- **Value / mean-reversion** — overreaction correction. Long, painful drawdowns. Test: ETFs, FX
  (PPP), crypto (distance from anchor). Crowding: medium.
- **Low-volatility / low-beta / betting-against-beta** — leverage-constrained buyers overpay for
  high-beta. Test: ETF/sector vol-sorted books. Crowding: medium-high.
- **Quality / profitability** — durable cash flows under-priced. Mostly single-name; harder here.
- **Seasonality / calendar** — turn-of-month, day-of-week, time-of-day, holiday, FOMC drift. Flow
  & settlement driven. Test: any with enough samples; crypto intraday (`40`). Crowding: low but
  *unstable* — easy to overfit on few cycles.
- **Short-term reversal** — liquidity provision to impatient takers; transient impact decays.
  Test: intraday crypto (`binance_vision` aggTrades). Crowding: high; cost-sensitive.

## 3. Liquidity provision & flow — paid to absorb forced / price-insensitive trades

This is the richest, least-crowded vein for a small book, because the counterparty isn't optimising
for price. (Pairs with the **perspective-shift** operator.)

- **Order-flow / taker imbalance** — provide immediacy to aggressive takers. Test: aggTrades.
- **Index & ETF rebalance effects** — front-run / provide to mechanical rebalancers. Test:
  sector/country ETFs near reconstitution.
- **Forced de-leveraging** — vol-target & risk-parity funds cut on vol spikes; margin liquidations.
  Test: vol-spike event studies (`11`,`12`); crypto OI/funding (`binance_vision`).
- **Options-dealer gamma hedging** — dealers buy dips / sell rips (or the reverse) near big strikes.
  Test: needs dealer-gamma/OI positioning (Deribit) — **partial gap**.

## 4. Behavioural — exploit a predictable cognitive bias

Under/over-reaction, anchoring, disposition effect (hold losers, sell winners), lottery/skew
preference (overpay for cheap upside). Usually *explanations* for the factors above — name the bias,
then point to which factor/flow expresses it tradeably.

## 5. Microstructure (intraday) — short-horizon, capacity-limited

Order-flow imbalance, open-interest & funding positioning, gap behaviour, liquidation clusters.
Highest novelty per idea in this repo right now (the `90_crypto_intraday` thread), lowest capacity —
always score Tradability against cost + capacity hard.

## 6. Macro / regime — a conditioner, not a standalone edge

Yield-curve slope, credit spreads, financial-conditions indices, growth/inflation nowcasts
(all FRED). These rarely *are* the alpha; they **gate** the factors above (when to size up/down).
Treat a macro series as the **Conditioner column** of the morphological matrix, not the Signal.

## Loader → source coverage (what's testable today)

| Return source | Testable now? | Via |
|---|---|---|
| Equity / term / credit premia, value, low-vol | ✅ | yfinance, fred |
| Momentum (TS/XS), seasonality | ✅ | yfinance, binance_vision |
| Crypto funding carry, OI/funding positioning, intraday reversal | ✅ | binance_vision |
| Order-flow / taker imbalance | ✅ (verify aggTrades) | binance_vision |
| Rebalance / index flow | ◑ build event logic | yfinance + calendars |
| Forced de-leveraging | ✅ event-study | yfinance, binance_vision |
| Event-driven / macro-event regime | ◑ underused | fred, **polymarket** |
| Volatility risk premium, dealer gamma | ✗ data gap | needs options-IV (Deribit/CBOE) |
| Commodity roll carry | ◑ partial | needs a futures-curve source |

The two biggest *untapped* veins in this repo: **flow / forced-trade provision** (section 3, mostly
buildable from data on hand) and **`polymarket` event-probability regimes** (section 6, loader
exists, no signal yet). The biggest *gap* is anything VRP/options-based — flag it, don't fake it.
