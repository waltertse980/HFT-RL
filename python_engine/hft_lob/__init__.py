"""
hft_lob — LOB-HFT v2 engine package.

Parallel track to the Bar-RL v1 system.  All code here operates on
Databento MBO (Market-by-Order / Level 3) data rather than OHLCV bars.

Import graph
-----------
databento_pipeline   ← downloads raw .dbn.zst files from Databento
lob_reconstructor    ← replays MBO events, builds LOB snapshots (parquet)
lob_features         ← feature engineering on LOB snapshots
lob_environment      ← Gymnasium env for RL on LOB features
queue_simulator      ← FIFO fill-probability estimator
lob_backtester       ← event-driven backtest with execution modelling
live_quote_adapter   ← real-time feature builder from Alpaca NBBO stream
risk_controls        ← circuit-breaker / position-limit manager
paper_trader_lob     ← live paper trading loop (Alpaca TradingClient)
"""

__version__ = "2.0.0"
