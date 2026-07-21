# US sector target-8 study — 2022 holdout release

## Authorization and boundary

- User authorized opening calendar year 2022 on 2026-07-18.
- Evaluation window: 2022-01-01 through 2022-12-31.
- Data may include pre-2022 history only for trailing signals and portfolio state.
- No observation dated 2023-01-01 or later may be loaded. Calendar year 2023+ remains sealed.

## Frozen objects

The five candidates, their decision frequencies, risk rules, sector caps, transaction costs,
rebalance threshold, and the synthetic collar assumptions remain exactly as registered in
`2026-07-17_sector-target8-low-drawdown-protocol.md`. No candidate may be removed, altered, or
added after seeing 2022.

The synthetic benchmark remains long SPY total return, long a quarterly 5% OTM put, and short a
monthly 10% OTM call. Its VIX/realised-volatility proxy, skew buffers, and entry haircuts are frozen.

## Pre-release metric correction

Before loading 2022, drawdown measurement was corrected to treat initial capital of 1.0 as the
first high-water mark. Without that floor, an evaluation window that begins with losses understates
drawdown. The correction is applied symmetrically to development and holdout metrics and does not
change any signal, weight, trade, or return.

## 2022 scorecard

For each candidate and benchmark report:

- 2022 total return and 20 bp cost-stress return;
- annualized volatility, maximum drawdown, Ulcer Index, downside deviation, and daily 5% CVaR;
- excess-cash Sharpe, turnover, average equity exposure, and minimum/maximum equity exposure;
- monthly returns and lagged SPY-trend × fixed-VIX-20 regime diagnostics.

A sector candidate is a strict 2022 pass only if all of the following hold:

1. total return is at least 7.5%, and the 20 bp stress return is at least 7.0%;
2. annualized volatility is at most 10%;
3. maximum drawdown is no worse than -15%;
4. maximum drawdown and Ulcer Index both beat the synthetic collar;
5. return is no more than one percentage point below the synthetic collar.

Because this is a single calendar year, rolling five-year target attainment, subperiod gates,
bootstrap confidence bounds, and Deflated Sharpe are not re-estimated as holdout gates.

## Decision rule

- Report every frozen candidate, including failures.
- Do not tune from 2022 and do not open 2023+ in this research cycle.
- A strict pass supports further paper research; it does not authorize deployment.
- If no candidate passes, retain the development conclusion and treat any hybrid sector/options
  overlay as a new preregistered study.
