# HFT Python Engine

Reinforcement Learning trading engine for Hong Kong (HKEX) and US markets. Uses PPO/TD3 via Stable-Baselines3, ONNX inference for <20ms latency, and vectorbt for backtesting.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 8 cores | 12+ cores (Legion 5) |
| RAM | 16 GB | 32 GB |
| GPU | GTX 1650 | RTX 3070+ (Legion 5) |
| VRAM | 4 GB | 8 GB |
| Disk | 20 GB free | 50 GB SSD |

Training uses CUDA automatically if available. On Legion 5 (RTX 3070), expect ~3-6 hours for 1M timesteps at 1m timescale.

---

## Installation

```bash
cd hft-trader/python_engine
pip install -r requirements.txt
```

For CUDA-accelerated PyTorch (RTX 3070 on Legion):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 1: Download Data

```bash
# Download all markets and timescales (recommended — takes ~5 minutes)
python data_pipeline.py

# Download specific market/timescale
python data_pipeline.py --market us --timescale 1m
python data_pipeline.py --market hk --timescale 5m

# Available timescales: 10s, 1m, 5m, 1h
# Note: 10s bars are synthesized from 1m bars via resampling
```

Data is saved to `data/` as pickle files: `us_1m.pkl`, `hk_5m.pkl`, etc.

---

## Step 2: Train Models

```bash
# Train PPO on US market, 1-minute bars, 1M steps
python trainer.py --market us --timescale 1m --algo PPO --timesteps 1000000

# Train TD3 on HK market, 5-minute bars
python trainer.py --market hk --timescale 5m --algo TD3 --timesteps 1000000

# Train all combinations (runs sequentially)
python trainer.py --market both --timescale all --algo both --timesteps 500000

# Expected training times on Legion 5 (RTX 3070):
#   10s bars:  ~45 min / 1M steps
#   1m bars:   ~25 min / 1M steps
#   5m bars:   ~15 min / 1M steps
#   1h bars:   ~8 min  / 1M steps
```

Models saved to `models/{market}_{timescale}_{algo}/`.

### View Training Progress (TensorBoard)
```bash
tensorboard --logdir logs/
# Open http://localhost:6006
```

---

## Step 3: Export to ONNX

```bash
# Export a trained model for fast inference
python trainer.py --export models/us_1m_PPO/final_model.zip --output models/us_1m_PPO/model.onnx

# Verify: test inference latency
python trainer.py --benchmark models/us_1m_PPO/model.onnx
```

Target inference: <20ms per bar. ONNX FP16 typically achieves 5-12ms on CPU.

---

## Step 4: Backtest

```bash
# Backtest a model on historical data
python backtester.py --model models/us_1m_PPO/final_model.zip --market us --timescale 1m

# With date range
python backtester.py --model models/us_1m_PPO/final_model.zip --market us --timescale 1m \
  --start 2023-01-01 --end 2023-12-31

# Include red team tests
python backtester.py --model models/us_1m_PPO/final_model.zip --market us --timescale 1m --red-team
```

---

## Step 5: Red Team Tests

```bash
# Run all adversarial scenarios
python backtester.py --model models/us_1m_PPO/final_model.zip --market us --red-team

# Scenarios:
#   flash_crash        — sudden -20% price drop over 3 bars
#   liquidity_drought  — near-zero volume for extended period
#   adverse_selection  — worst fill price on every trade
#   regime_change      — bull→bear reversal injection
#   overfitting        — in-sample vs out-of-sample Sharpe comparison
```

---

## Step 6: Paper Trading

### US Market (Alpaca)
Get free Alpaca paper trading API keys at https://alpaca.markets

```bash
python paper_trader.py --market us --ticker AAPL --model models/us_1m_PPO/model.onnx \
  --api-key YOUR_KEY --api-secret YOUR_SECRET
```

### HK Market (Local Replay)
No broker API required — replays test data at 10x real speed.

```bash
python paper_trader.py --market hk --ticker 0700.HK --model models/hk_5m_TD3/model.onnx
```

---

## Step 7: Start the API Server

The dashboard connects to this server for all ML operations.

```bash
uvicorn api_server:app --port 8000

# With auto-reload during development
uvicorn api_server:app --port 8000 --reload

# API docs: http://localhost:8000/docs
```

---

## Hyperparameter Tuning (LLM Prompts)

Use these prompts with Claude or GPT-4 to tune hyperparameters for your specific market:

**For PPO tuning:**
```
I'm training a PPO agent for HFT on {market} {timescale} bars. 
Current Sharpe: {sharpe}, Max Drawdown: {mdd}%, Win Rate: {wr}%.
Learning rate: {lr}, n_steps: {n_steps}, entropy coef: {ent}.
How should I adjust hyperparameters to improve Sharpe while limiting drawdown?
```

**For reward function tuning:**
```
My HFT RL agent has reward = realized_pnl - {tc}*|position_change| - {dd_pen}*max(0,-drawdown).
The agent is overtrading (avg {n_trades} trades/episode) and has low Sharpe ({sharpe}).
Suggest reward function modifications to encourage larger, more selective trades.
```

---

## Architecture

```
data_pipeline.py   → Download + preprocess OHLCV data (yfinance)
rl_environment.py  → Custom Gymnasium env (60-bar window, Discrete(3) actions)
trainer.py         → PPO/TD3 training, ONNX export, evaluation
backtester.py      → vectorbt backtesting, red team scenarios
paper_trader.py    → Alpaca WebSocket / HK data replay live loop
api_server.py      → FastAPI server (port 8000) for dashboard integration
```

## Files Structure

```
python_engine/
├── data/               # Downloaded datasets (pickle)
├── logs/               # TensorBoard training logs
├── models/             # Saved model checkpoints + ONNX exports
├── paper_trades.jsonl  # Live trade log (append-only)
├── requirements.txt
├── data_pipeline.py
├── rl_environment.py
├── trainer.py
├── backtester.py
├── paper_trader.py
├── api_server.py
└── README.md
```
