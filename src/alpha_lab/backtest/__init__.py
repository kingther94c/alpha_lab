"""Backtesting placeholder.

Intent: keep the architecture framework-agnostic. A minimal vectorized
backtest engine (pandas/numpy) should live here first. Backtrader integration
is optional — add a separate submodule if/when needed.

TODO:
- ``vector.py``: rebalance-by-schedule engine taking signals + prices + costs
- ``metrics.py``: summary stats (CAGR, vol, Sharpe, max DD, hit rate, turnover)
"""
