# ETF ensemble and 5% return-floor addendum — pre-2022

## Frozen question

Can monthly combinations of the previously frozen ETF strategy sleeves materially
shorten recovery time, either under the original 9%-10% return objective or after
lowering the historical CAGR floor to 5%?

The data boundary remains 2022-01-01 exclusive. Every price, sleeve return, signal and
ensemble return must end no later than 2021-12-31.

## Frozen sleeve pool

The full-GFC pool uses eight previously tested, mechanism-labelled sleeves:

1. QQQ 12% volatility target;
2. six-month top-3 cross-asset momentum;
3. low-Ulcer positive-trend sectors;
4. SPY 150-day trend to SHY;
5. fixed retail all-weather;
6. 200-day trend-gated all-weather sleeves;
7. synthetic SPY 95/105 collar;
8. monthly 65% cyclical / 63-day momentum / 100-day trend / top-2 recovery barbell.

The second evidence tier adds the HYG/IEF credit-canary sleeve. Because that signal
finishes warm-up only in 2008, credit-tier ensembles are not described as full-GFC
tests.

## Frozen ensemble sweep

- Enumerate every equal-weight subset of two through five full-GFC sleeves.
- Separately enumerate every credit-canary combination with one through four of the
  full-GFC sleeves.
- Scale each risky ensemble to 25%, 50%, 75% and 100%; allocate the residual to SHY.
- Rebalance strategy sleeves monthly, decide at the observed month-end close, and trade
  at the next available close.
- Charge the already embedded sleeve costs plus an additional 5 bp one-way meta-level
  trading cost; repeat at 10 bp meta cost for stress.
- Add two fixed all-pool inverse-volatility diagnostics using trailing 63 and 126
  sessions. They are diagnostics, not extra independent candidate families.

This creates at least 1,480 equal-weight trials before the inverse-volatility
diagnostics. Parameter neighbours count as trials rather than independent discoveries.

## Frozen evaluation

Every ensemble reports CAGR, 10 bp stress CAGR, volatility, MaxDD, total underwater
duration, trough-to-recovery duration for drawdowns reaching -5%, median recovery and
the share recovered within 20 sessions.

Two return gates are evaluated without retroactive changes:

- **Original:** CAGR >=9%, stress CAGR >=8.5%, volatility <=15%, MaxDD >=-25%.
- **Relaxed:** CAGR >=5%, stress CAGR >=4.5%, volatility <=15%, MaxDD >=-25%.

Recovery outcomes are classified separately:

- no material drawdown occurred;
- at least one material drawdown occurred and every recovery leg was <=20 sessions;
- median material recovery <=20 sessions;
- at least half of material episodes recovered within 20 sessions.

An ensemble with no -5% drawdown is an avoidance result, not evidence that it can repair
a -5% loss in 20 days. Open episodes fail the recovery test. No high-water reset is
allowed.

