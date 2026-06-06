"""quant_bot_manager — turn researched strategies into running, monitored trading bots.

Companion to ``alpha_lab`` (research). alpha_lab *finds* edges (data, features, backtest, the P7
``crypto_book`` strategy); quant_bot_manager *runs* them: execution (brokers), scheduling (runner),
state, and a Streamlit monitoring/control UI. The handoff is a strategy's target-weight function.

Incremental by design — research code in ``alpha_lab`` is untouched.
"""
