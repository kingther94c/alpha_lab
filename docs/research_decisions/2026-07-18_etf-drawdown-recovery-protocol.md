# ETF drawdown-recovery addendum — pre-2022

## Frozen question

Can a weekly or monthly ETF-only rule retain approximately 9%-10% long-run CAGR,
annualized volatility at or below 15%, and maximum drawdown no worse than -25%, while
recovering every material drawdown within 20 observed trading sessions?

The market-data boundary remains unchanged: downloads end at 2022-01-01 exclusive and
the final permitted observation is 2021-12-31. No post-2021 data may enter this addendum.

## Duration definitions

- **Underwater duration:** first session below the previous equity high through the last
  underwater session before regaining that high.
- **Recovery leg:** trough through the session that regains the previous equity high.
- **Material episode:** an underwater episode whose trough reaches -5% or worse.
- **Open episode:** censored at the sample end. Its displayed recovery duration is a lower
  bound and it fails the 20-session test.
- **20-day success share:** material episodes completed within 20 trading sessions of the
  trough divided by all material episodes, including open episodes.

The duration labels are ex-post evaluation fields only. The eventual recovery date, future
trough, and duration never enter a signal or a weight.

## Frozen recovery-oriented sweep

The first sweep already contains 104 rows. This addendum adds two causal families:

1. Weekly SPY/QQQ trend controls with 20/50/75/100/150-day moving averages, plus
   8%/10%/12% 63-day volatility targets combined with 20/50/100-day trends.
2. Defensive/cyclical sector barbells using weekly or monthly decisions, 35%/50%/65%
   cyclical budgets, 63/126-day cyclical momentum, 50/100/200-day absolute trends, and
   top-2/top-3 cyclical selection. The defensive sleeve is inverse downside deviation;
   unallocated capital goes to SHY.

Every target forms from information available at the decision close and trades at the next
available close. Primary trading cost is 5 bp one-way and stress cost is 10 bp.

## Frozen decision order

1. Test the hard requirement: maximum 5%-episode trough-to-recovery duration <=20 days.
2. Among long-history rules, require CAGR >=9%, stress CAGR >=8.5%, volatility <=15%,
   MaxDD >=-25%, and 2013-2021 CAGR >=8%.
3. If no rule passes the hard duration requirement, report failure without resetting the
   high-water mark or redefining recovery. Rank the honest near-misses by maximum material
   recovery duration, then median material recovery duration and 20-day success share.
4. Parameter neighbours count as trials, not independent strategies. A selected row must
   be checked against adjacent lookbacks/budgets and against the slower 10 bp cost case.

