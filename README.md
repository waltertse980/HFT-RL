# HFT Trader — Full Setup & Operations Guide

> **Platform:** Windows 11 (Legion 5 w/ RTX 3070)  
> **Markets:** US (Alpaca) · Hong Kong (HKEX replay)  
> **Stack:** Python 3.11 · PyTorch · Stable-Baselines3 · ONNX · FastAPI · React dashboard

---

## Table of Contents

1. [Windows Setup](#1-windows-setup)
2. [Alpaca API Integration](#2-alpaca-api-integration)
3. [Data Pre-processing & 60/30/10 Split](#3-data-pre-processing--603010-split)
4. [Reinforcement Learning Training, Validation & Test](#4-reinforcement-learning-training-validation--test)
5. [Selecting a Model for Live Paper Trading](#5-selecting-a-model-for-live-paper-trading)
6. [Migrating to Real Algorithmic HFT](#6-migrating-to-real-algorithmic-hft)

---

## 1. Windows Setup

### 1.1 Prerequisites

Install these in order before touching the project code.

| Tool | Download | Notes |
|---|---|---|
| **Python 3.11** | [python.org](https://www.python.org/downloads/) | ✅ Check "Add Python to PATH" during install |
| **Git** | [git-scm.com](https://git-scm.com/download/win) | Use default settings |
| **Node.js 20 LTS** | [nodejs.org](https://nodejs.org/) | For the dashboard UI |
| **CUDA Toolkit 12.1** | [nvidia.com/cuda](https://developer.nvidia.com/cuda-12-1-0-download-archive) | RTX 3070 — match to your driver version |
| **cuDNN 8.9** | [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) | Required for PyTorch GPU |

> **Verify GPU drivers:** Run `nvidia-smi` in PowerShell. You should see your RTX 3070 and driver version ≥ 535.

---

### 1.2 Clone the Project

Open **PowerShell** (not CMD) and run:

```powershell
git clone <your-repo-url> hft-trader
cd hft-trader
```

If you don't have a git repo, copy the project folder and open PowerShell inside it.

---

### 1.3 Python Environment

Always use a virtual environment — it keeps dependencies isolated.

```powershell
# Create the virtual environment
python -m venv .venv

# Activate it (you must do this every new terminal session)
.\.venv\Scripts\Activate.ps1

# If you get an execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Your prompt should now show `(.venv)` at the start.

---

### 1.4 Install PyTorch (CUDA 12.1)

Install PyTorch with GPU support **before** the other requirements — the requirements.txt installs the CPU version by default.

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU is detected:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 3070
```

---

### 1.5 Install Python Dependencies

```powershell
cd python_engine
pip install -r requirements.txt
```

This installs: Gymnasium, Stable-Baselines3, yfinance, vectorbt, ONNX, alpaca-py, FastAPI, and all other engine dependencies. Takes ~3–5 minutes.

---

### 1.6 Install Dashboard Dependencies

```powershell
# Go back to project root
cd ..
npm install
```

---

### 1.7 Create the `.env` File

```powershell
# Inside python_engine/
copy .env.example .env   # if it exists, else create it:
New-Item python_engine\.env
```

Open `python_engine\.env` in any text editor and add:

```env
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

> Leave `ALPACA_BASE_URL` as the paper URL until you are ready for live trading (Section 6).

---

### 1.8 Verify Everything Works

```powershell
# From python_engine/
python -c "
import gymnasium, stable_baselines3, torch, yfinance, alpaca
print('gymnasium:', gymnasium.__version__)
print('sb3:', stable_baselines3.__version__)
print('torch:', torch.__version__, '| GPU:', torch.cuda.is_available())
print('yfinance:', yfinance.__version__)
print('All imports OK')
"
```

---

### 1.9 Folder Structure

```
hft-trader/
├── python_engine/          ← All ML/trading logic (Python)
│   ├── data/               ← Downloaded datasets (auto-created)
│   ├── models/             ← Saved model checkpoints (auto-created)
│   ├── logs/               ← TensorBoard training logs (auto-created)
│   ├── data_pipeline.py    ← Download + feature engineering
│   ├── rl_environment.py   ← Gymnasium environment
│   ├── trainer.py          ← PPO/TD3 training loop + ONNX export
│   ├── backtester.py       ← Historical backtest + red team tests
│   ├── paper_trader.py     ← Live paper trading (Alpaca / HK replay)
│   ├── api_server.py       ← FastAPI bridge for the dashboard
│   └── requirements.txt
├── client/                 ← React dashboard (frontend)
├── server/                 ← Express backend (API bridge)
└── SETUP_GUIDE.md          ← This file
```

---

## 2. Alpaca API Integration

### 2.1 Get Your Alpaca API Keys

1. Sign up at [alpaca.markets](https://alpaca.markets/) — free, no minimum deposit for paper trading
2. In the dashboard, switch to **Paper Trading** mode (top-right toggle)
3. Go to **API Keys** → **Generate New Key**
4. Copy both the **API Key ID** (starts with `PK...`) and the **Secret Key** (shown only once)

---

### 2.2 Set Keys in `.env`

```env
# python_engine/.env
ALPACA_API_KEY=PKXXXXXXXXXXXXXXXXXXXXXXXX
ALPACA_API_SECRET=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

The engine reads these automatically via `python-dotenv`. You never need to pass them as CLI arguments unless you want to override the `.env` values.

---

### 2.3 Verify the Connection

```powershell
python -c "
from alpaca.trading.client import TradingClient
import os; from dotenv import load_dotenv; load_dotenv()
client = TradingClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_API_SECRET'], paper=True)
account = client.get_account()
print('Account status:', account.status)
print('Buying power:', account.buying_power)
print('Connection: OK')
"
```

Expected output:
```
Account status: ACTIVE
Buying power: 100000.00
Connection: OK
```

---

### 2.4 What Alpaca Provides

| Feature | Free Tier | Notes |
|---|---|---|
| Paper trading | ✅ Full access | No deposit required |
| Historical bars (1m, 5m, 1h) | ✅ Unlimited | Up to 5+ years |
| Real-time WebSocket feed | ✅ IEX source | Delayed ~15 min for free |
| Real-time SIP feed | 💰 Paid plan | True real-time quotes |
| Live brokerage trading | ✅ | Requires funded account |

> For paper trading and backtesting, the free IEX feed is sufficient. For live HFT with edge, you'll want the SIP feed (see Section 6).

---

### 2.5 Use Alpaca as the Data Source

The `data_pipeline.py` uses **yfinance** by default for historical downloads. To use Alpaca's historical API instead (more reliable, matches live feed format):

```python
# Example: Fetch NVDA 1-minute bars via Alpaca (run interactively)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

client = StockHistoricalDataClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_API_SECRET"]
)

request = StockBarsRequest(
    symbol_or_symbols=["NVDA"],
    timeframe=TimeFrame.Minute,
    start=datetime(2023, 1, 1),
    end=datetime(2024, 12, 31),
)

bars = client.get_stock_bars(request)
df = bars.df  # MultiIndex DataFrame: (symbol, timestamp)
df = df.xs("NVDA", level="symbol").reset_index()
print(df.head())
```

---

## 3. Data Pre-processing & 60/30/10 Split

### 3.1 Concept

The training pipeline uses a **temporal** split — never random shuffle — because random shuffling leaks future data into the training set, which causes severe overfitting.

```
Full dataset (chronological order)
├── TRAIN  (60%) ── Agent learns here with PPO/TD3
├── VAL    (30%) ── Hyperparameter tuning; early stopping decision
└── TEST   (10%) ── Never touched until final evaluation
```

The split is done **per ticker** so each stock gets its own correctly proportioned segments.

---

### 3.2 Download Data for a Specific Stock

```powershell
# Download NVDA 1-minute data (recommended starting point)
python data_pipeline.py --market us --timescale 1m

# For multiple specific tickers, edit the US_TICKERS list in data_pipeline.py
# or pass tickers via the pipeline's download_us_data() function
```

To download a custom ticker list without editing the file, use this one-liner:

```powershell
python -c "
from data_pipeline import download_us_data, compute_features, aggregate_to_timescale, save_dataset

# --- Change tickers here ---
TICKERS = ['NVDA', 'AAPL', 'META']
TIMESCALE = '1m'  # '10s', '1m', '5m', or '1h'

# Download
raw = download_us_data(tickers=TICKERS, period='60d', interval='1m')

# Add features
featured = {t: compute_features(df) for t, df in raw.items()}

# Resample if needed (e.g., to 10s synthetic bars)
if TIMESCALE != '1m':
    featured = {t: aggregate_to_timescale(df, TIMESCALE) for t, df in featured.items()}

# Save to data/us_1m.pkl (or data/us_10s.pkl, etc.)
save_dataset(featured, market='us', timescale=TIMESCALE)
print('Done. Tickers saved:', list(featured.keys()))
"
```

> **yfinance limit:** 1-minute bars are available for the last 60 days only. For longer history, use Alpaca's historical API (Section 2.5) or switch to 5m/1h intervals which allow up to 2 years.

---

### 3.3 Perform the 60/30/10 Split

Save this as `python_engine/split_data.py` and run it once per ticker:

```python
"""
split_data.py — Temporal 60/30/10 train/val/test split for RL training.

Usage:
    python split_data.py --ticker NVDA --market us --timescale 1m
    python split_data.py --ticker AAPL --market us --timescale 5m
    python split_data.py --ticker META --market us --timescale 1h
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
SPLIT_DIR = Path(__file__).parent / "data" / "splits"
SPLIT_DIR.mkdir(exist_ok=True)


def split_ticker(ticker: str, market: str, timescale: str,
                 train_pct: float = 0.60, val_pct: float = 0.30,
                 seed: int = 42) -> dict:
    """
    Temporal 60/30/10 split with optional block-shuffle within each segment.

    The split is strictly forward-in-time:
        [=====TRAIN=====|===VAL===|=TEST=]
    No data from the future leaks into the past segments.

    A block-shuffle is applied WITHIN the train segment only
    (shuffles contiguous 5-day blocks, not individual bars) to improve
    sample diversity without look-ahead bias.
    """
    pkl_path = DATA_DIR / f"{market}_{timescale}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {pkl_path}\n"
            f"Run: python data_pipeline.py --market {market} --timescale {timescale}"
        )

    with open(pkl_path, "rb") as f:
        dataset: dict[str, pd.DataFrame] = pickle.load(f)

    if ticker not in dataset:
        available = list(dataset.keys())
        raise KeyError(f"Ticker '{ticker}' not in dataset. Available: {available}")

    df = dataset[ticker].copy()
    n = len(df)

    if n < 500:
        raise ValueError(f"Too few rows ({n}) for ticker {ticker}. Download more data.")

    # --- Compute cut indices ---
    train_end = int(n * train_pct)
    val_end   = int(n * (train_pct + val_pct))
    # test_end  = n  (remainder)

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    # --- Block-shuffle train segment (5-day blocks) ---
    # Estimate bars per day from the timescale
    bars_per_day = {
        "10s": 2340,  # 6.5h * 360 bars
        "1m":   390,   # 6.5h * 60 bars
        "5m":    78,   # 6.5h * 12 bars
        "1h":     7,   # ~7 trading hours
    }.get(timescale, 390)

    block_size = bars_per_day * 5  # 5-day block
    rng = np.random.default_rng(seed)

    n_train = len(train_df)
    n_blocks = n_train // block_size
    remainder = n_train % block_size

    if n_blocks > 1:
        blocks = [train_df.iloc[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
        rng.shuffle(blocks)
        if remainder:
            blocks.append(train_df.iloc[n_blocks * block_size:])
        train_df = pd.concat(blocks).reset_index(drop=False)

    # --- Save splits ---
    out_path = SPLIT_DIR / f"{ticker}_{market}_{timescale}.pkl"
    splits = {
        "ticker":    ticker,
        "market":    market,
        "timescale": timescale,
        "train":     train_df,
        "val":       val_df,
        "test":      test_df,
        "n_total":   n,
        "n_train":   len(train_df),
        "n_val":     len(val_df),
        "n_test":    len(test_df),
        "seed":      seed,
    }
    with open(out_path, "wb") as f:
        pickle.dump(splits, f)

    print(f"\n{'='*55}")
    print(f"  Ticker   : {ticker}")
    print(f"  Market   : {market}  |  Timescale: {timescale}")
    print(f"  Total    : {n:,} bars")
    print(f"  Train    : {len(train_df):,} bars  ({len(train_df)/n*100:.1f}%)")
    print(f"  Val      : {len(val_df):,} bars  ({len(val_df)/n*100:.1f}%)")
    print(f"  Test     : {len(test_df):,} bars  ({len(test_df)/n*100:.1f}%)")
    print(f"  Saved    : {out_path}")
    print(f"{'='*55}\n")

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="60/30/10 temporal data split")
    parser.add_argument("--ticker",    required=True, help="e.g. NVDA, AAPL, META")
    parser.add_argument("--market",    default="us",  choices=["us", "hk"])
    parser.add_argument("--timescale", default="1m",  choices=["10s", "1m", "5m", "1h"])
    parser.add_argument("--seed",      type=int, default=42, help="RNG seed for block shuffle")
    args = parser.parse_args()

    split_ticker(
        ticker=args.ticker,
        market=args.market,
        timescale=args.timescale,
        seed=args.seed,
    )
```

Run it:

```powershell
# Split NVDA data into train/val/test
python split_data.py --ticker NVDA --market us --timescale 1m

# Output:
# =======================================================
#   Ticker   : NVDA
#   Market   : us  |  Timescale: 1m
#   Total    : 15,600 bars
#   Train    :  9,360 bars  (60.0%)
#   Val      :  4,680 bars  (30.0%)
#   Test     :  1,560 bars  (10.0%)
#   Saved    : data/splits/NVDA_us_1m.pkl
# =======================================================
```

Repeat for other tickers:

```powershell
python split_data.py --ticker AAPL --market us --timescale 1m
python split_data.py --ticker META --market us --timescale 1m
python split_data.py --ticker NVDA --market us --timescale 5m   # slower timescale
```

---

### 3.4 Inspect the Split

```powershell
python -c "
import pickle
from pathlib import Path

splits = pickle.load(open('data/splits/NVDA_us_1m.pkl', 'rb'))
print('Train date range:', splits['train'].index[0], '->', splits['train'].index[-1])
print('Val   date range:', splits['val'].index[0],   '->', splits['val'].index[-1])
print('Test  date range:', splits['test'].index[0],  '->', splits['test'].index[-1])
"
```

---

## 4. Reinforcement Learning Training, Validation & Test

### 4.1 Overview of the Pipeline

```
split data  →  train PPO/TD3 on TRAIN set  →  monitor on VAL set
     ↓                                               ↓
   ONNX                                      pick best checkpoint
   export       ←──────── eval on TEST ─────────────┘
     ↓
  deploy to paper trader
```

---

### 4.2 Phase 1 — Training on the TRAIN Set

The `trainer.py` accepts a `--splits-path` flag to use your pre-split data instead of the full dataset. Update `trainer.py` to load splits by adding this helper (or run the full trainer and it handles the split internally):

```powershell
# Standard training on a single ticker's train split
python trainer.py ^
  --market us ^
  --timescale 1m ^
  --algo PPO ^
  --timesteps 2000000 ^
  --n-envs 4
```

> **For stock-specific training**, add a `--ticker` flag. The trainer will filter the loaded dataset to that ticker's split. If the flag isn't present yet, pass the ticker interactively via this wrapper:

```powershell
python -c "
import pickle, sys
from pathlib import Path
from data_pipeline import compute_features
from rl_environment import make_envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

TICKER    = 'NVDA'      # Change this
TIMESCALE = '1m'
ALGO      = 'PPO'       # or 'TD3'
TIMESTEPS = 2_000_000
N_ENVS    = 4

# Load splits
splits = pickle.load(open(f'data/splits/{TICKER}_us_{TIMESCALE}.pkl', 'rb'))
train_df = splits['train']
val_df   = splits['val']

# Create vectorized training envs
train_envs = make_envs({'train': train_df}, n_envs=N_ENVS)
val_envs   = make_envs({'val': val_df},   n_envs=1)

# Callbacks: save checkpoint every 100k steps + eval on val every 50k steps
checkpoint_cb = CheckpointCallback(
    save_freq=100_000 // N_ENVS,
    save_path=f'models/{TICKER}_{TIMESCALE}_{ALGO}/',
    name_prefix='ckpt',
)
eval_cb = EvalCallback(
    val_envs,
    best_model_save_path=f'models/{TICKER}_{TIMESCALE}_{ALGO}/best/',
    log_path=f'logs/{TICKER}_{TIMESCALE}_{ALGO}/',
    eval_freq=50_000 // N_ENVS,
    n_eval_episodes=5,
    deterministic=True,
)

# Build model
model = PPO(
    'MlpPolicy', train_envs,
    learning_rate=3e-4, n_steps=2048, batch_size=64,
    n_epochs=10, gamma=0.99, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.01,
    tensorboard_log=f'logs/{TICKER}_{TIMESCALE}_{ALGO}/',
    verbose=1, device='cuda',
)

# Train
model.learn(total_timesteps=TIMESTEPS, callback=[checkpoint_cb, eval_cb])
model.save(f'models/{TICKER}_{TIMESCALE}_{ALGO}/final_model')
print('Training complete.')
"
```

**Expected training time on Legion 5 (RTX 3070):**

| Timescale | 1M steps | 2M steps |
|---|---|---|
| 10s | ~55 min | ~110 min |
| 1m | ~28 min | ~55 min |
| 5m | ~18 min | ~35 min |
| 1h | ~10 min | ~20 min |

---

### 4.3 Monitor Training (TensorBoard)

Open a **second** PowerShell window:

```powershell
.\.venv\Scripts\Activate.ps1
cd python_engine
tensorboard --logdir logs/
```

Open [http://localhost:6006](http://localhost:6006) in Chrome. Watch:

| Metric | Healthy sign | Concern |
|---|---|---|
| `rollout/ep_rew_mean` | Rising, then plateauing | Sharp drops = env crash |
| `rollout/ep_len_mean` | Stable ~500–2000 steps | Too short = early termination |
| `train/policy_gradient_loss` | Decreasing | Stays flat = learning stalled |
| `train/approx_kl` | < 0.02 | > 0.05 = learning rate too high |

---

### 4.4 Phase 2 — Validation (Choose Best Checkpoint)

After training, the `EvalCallback` saves the best checkpoint to `models/NVDA_1m_PPO/best/best_model.zip`. To manually evaluate all checkpoints against the VAL set:

```powershell
python -c "
import pickle, glob
from pathlib import Path
from stable_baselines3 import PPO
from rl_environment import HFTradingEnv

TICKER    = 'NVDA'
TIMESCALE = '1m'
ALGO      = 'PPO'

splits  = pickle.load(open(f'data/splits/{TICKER}_us_{TIMESCALE}.pkl', 'rb'))
val_df  = splits['val']

checkpoints = sorted(glob.glob(f'models/{TICKER}_{TIMESCALE}_{ALGO}/ckpt_*.zip'))
results = []

for ckpt_path in checkpoints:
    model = PPO.load(ckpt_path)
    env   = HFTradingEnv(val_df)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    results.append({
        'checkpoint': Path(ckpt_path).stem,
        'val_return': info.get('total_return', 0),
        'val_sharpe': info.get('sharpe_ratio', 0),
        'n_trades':   info.get('n_trades', 0),
    })
    print(f\"{Path(ckpt_path).stem:30s}  return={info.get('total_return',0):+.2%}  sharpe={info.get('sharpe_ratio',0):.3f}\")

best = max(results, key=lambda r: r['val_sharpe'])
print(f\"\nBest checkpoint: {best['checkpoint']}  Sharpe={best['val_sharpe']:.3f}\")
"
```

**Pick the checkpoint with the highest VAL Sharpe ratio** that also has a drawdown < 15%. Do not pick purely by return — a high-return model with 30% drawdown is too risky for live trading.

---

### 4.5 Phase 3 — Test Set Evaluation (Final Gate)

Test evaluation happens **exactly once** on the held-out 10% — never use test results to make training decisions.

```powershell
# Using the backtester with your chosen model
python backtester.py ^
  --model models/NVDA_1m_PPO/best/best_model.zip ^
  --market us ^
  --timescale 1m ^
  --ticker NVDA ^
  --window-size 60 ^
  --initial-capital 100000

# Add --red-team to run adversarial scenarios simultaneously:
python backtester.py ^
  --model models/NVDA_1m_PPO/best/best_model.zip ^
  --market us ^
  --timescale 1m ^
  --red-team ^
  --output results/NVDA_1m_PPO_test_report.json
```

**Minimum thresholds to pass the test gate:**

| Metric | Minimum | Target |
|---|---|---|
| Sharpe Ratio | > 1.0 | > 1.5 |
| Max Drawdown | < 20% | < 10% |
| Win Rate | > 48% | > 55% |
| Profit Factor | > 1.1 | > 1.4 |
| Red Team Flash Crash | PASS | PASS |
| Red Team Overfitting | IS/OOS Sharpe < 2.0 | < 1.5 |

If the model fails the test gate: **go back to training**, not to the test data. Adjust hyperparameters based on VAL results only.

---

### 4.6 Export to ONNX (Required for Paper & Live Trading)

The paper trader and live engine use ONNX format, not the raw `.zip` SB3 file. ONNX runs in pure C++ without Python overhead — this is what gives sub-20ms inference.

```powershell
python trainer.py ^
  --export models/NVDA_1m_PPO/best/best_model.zip ^
  --output  models/NVDA_1m_PPO/best/model.onnx

# Verify inference latency:
python trainer.py --benchmark models/NVDA_1m_PPO/best/model.onnx
# Expected: ~8–15 ms on CPU, ~3–6 ms on GPU
```

---

## 5. Selecting a Model for Live Paper Trading

### 5.1 Model Selection Checklist

Before starting paper trading, verify:

- [ ] Test Sharpe Ratio > 1.0
- [ ] Max Drawdown < 20% on test set
- [ ] All red team tests run (Flash Crash and Overfitting at minimum)
- [ ] ONNX export successful (file size > 100 KB)
- [ ] ONNX inference < 20ms on your machine

---

### 5.2 Start the API Server (Dashboard Bridge)

```powershell
# Terminal 1 — Python engine
cd python_engine
.\.venv\Scripts\Activate.ps1
uvicorn api_server:app --port 8000 --reload
```

```powershell
# Terminal 2 — Dashboard
cd ..   # project root
npm run dev
# Open http://localhost:5000
```

The dashboard's sidebar will show **Python Engine: Connected** (green dot) once the API is reachable.

---

### 5.3 Start Paper Trading via Dashboard

1. Open the dashboard → **Live Trading** page
2. Set **Market** → US
3. Set **Ticker** → NVDA (or whichever ticker your model was trained on)
4. Set **Model** → select `NVDA_1m_PPO` from the dropdown
5. Set **Timescale** → 1m
6. Click **Start** — the engine connects to Alpaca WebSocket and begins trading

Or via CLI:

```powershell
# Using .env for credentials (recommended)
python paper_trader.py ^
  --market us ^
  --ticker NVDA ^
  --model models/NVDA_1m_PPO/best/model.onnx ^
  --timescale 1m ^
  --capital 100000

# Explicitly passing credentials (if not using .env)
python paper_trader.py ^
  --market us ^
  --ticker NVDA ^
  --model models/NVDA_1m_PPO/best/model.onnx ^
  --api-key  PKXXXXXXXXXXXXXXXXXXXXXXXX ^
  --api-secret XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

---

### 5.4 Monitor Paper Trading

**Real-time dashboard:** Watch the Live Trading page for BUY/SELL signals, P&L, and inference latency.

**Log file:**
```powershell
# Tail the live trade log
Get-Content python_engine\paper_trades.jsonl -Wait -Tail 20
```

Each line is a JSON object:
```json
{
  "timestamp": "2025-03-15T09:31:42Z",
  "ticker": "NVDA",
  "action": "BUY",
  "price": 875.20,
  "shares": 11,
  "portfolio_value": 100823.45,
  "realized_pnl": 823.45,
  "inference_ms": 11.3
}
```

---

### 5.5 Paper Trading Duration & Acceptance Criteria

Run paper trading for a minimum of **20 trading days** (4 calendar weeks) before considering live migration. Collect:

| Metric | After 20 days | Action |
|---|---|---|
| Sharpe (annualized) | > 1.0 | Continue to 30 days |
| Max intraday drawdown | < 5% | Good |
| Average inference latency | < 20 ms | Acceptable for HFT |
| Order fill rate | > 95% | Alpaca paper fills are near 100% |
| Daily P&L consistency | Positive > 60% of days | Proceed to live |

If Sharpe < 0.8 after 20 days: retrain with more data or adjusted hyperparameters.

---

### 5.6 HK Market Paper Trading (Replay Mode)

No Alpaca account is needed for HK — it replays held-out test data at 10× real speed:

```powershell
python paper_trader.py ^
  --market hk ^
  --ticker 0700.HK ^
  --model models/0700HK_1m_PPO/best/model.onnx ^
  --timescale 1m
```

---

## 6. Migrating to Real Algorithmic HFT

> ⚠️ **Risk Warning:** Live algorithmic trading involves real financial risk. Never deploy capital you cannot afford to lose entirely. Ensure you understand the regulatory requirements in your jurisdiction (Hong Kong SFC, US FINRA/SEC) before trading with real money.

---

### 6.1 Migration Checklist

Complete every item before going live:

- [ ] Paper trading Sharpe > 1.0 for ≥ 20 trading days
- [ ] No single-day loss > 3% during paper trading
- [ ] All 5 red team scenarios passed
- [ ] Inference latency < 15 ms (measured live, not just benchmark)
- [ ] Risk limits coded and tested (stop-loss, daily loss limit, position cap)
- [ ] Alpaca live brokerage account funded (minimum $2,000 for Pattern Day Trader exemption not needed for HFT with overnight close)
- [ ] Regulatory review: confirm you are not required to register as an investment adviser

---

### 6.2 Switch from Paper to Live API

**Step 1:** Change `.env`:

```env
# python_engine/.env — LIVE MODE
ALPACA_API_KEY=your_live_key_here        # Generate a NEW key in Live mode
ALPACA_API_SECRET=your_live_secret_here
ALPACA_BASE_URL=https://api.alpaca.markets   # ← Changed from paper URL
```

**Step 2:** Update `paper_trader.py` — change `paper=True` to `paper=False`:

```python
# In paper_trader.py, line ~215
# BEFORE:
self._trading_client = TradingClient(api_key, api_secret, paper=True)

# AFTER (live trading):
self._trading_client = TradingClient(api_key, api_secret, paper=False)
```

**Step 3:** Start with minimal capital:

```powershell
python paper_trader.py ^
  --market us ^
  --ticker NVDA ^
  --model models/NVDA_1m_PPO/best/model.onnx ^
  --timescale 1m ^
  --capital 5000   # Start with $5,000, not your full account
```

---

### 6.3 Risk Controls for Live Trading

The engine has built-in risk controls in `paper_trader.py`. Verify these are set appropriately for live:

```python
# In paper_trader.py → PortfolioState defaults
max_position_pct = 0.50    # Live: reduce from 0.95 to 0.50 (50% max)
stop_loss_pct    = 0.015   # Live: tighten from 2% to 1.5%
daily_loss_limit = 0.03    # Live: 3% daily max loss → kill switch
```

Or override via CLI:

```powershell
python paper_trader.py ^
  --market us ^
  --ticker NVDA ^
  --model models/NVDA_1m_PPO/best/model.onnx ^
  --capital 10000 ^
  --max-position 0.50 ^
  --stop-loss 0.015 ^
  --daily-loss-limit 0.03
```

---

### 6.4 Upgrade Data Feed for True HFT

The free Alpaca IEX feed has ~15-minute delays on some quotes. For real HFT edge you need real-time data:

| Feed | Cost | Latency | Suitable for |
|---|---|---|---|
| Alpaca IEX (free) | $0/mo | ~15 min delayed | Paper trading / testing |
| Alpaca SIP (Unlimited) | $99/mo | Real-time | Live HFT |
| Polygon.io Starter | $29/mo | Real-time | Alternative to Alpaca |
| Interactive Brokers TWS API | ~$10/mo | Real-time + colocation | Serious HFT |

Upgrade in Alpaca dashboard: **Account → Subscription → Unlimited Plan**.

Then update the data stream in `paper_trader.py`:

```python
# Line ~209 — the WebSocket feed is already real-time once you have SIP
# No code change needed; the latency improvement is automatic
```

---

### 6.5 Co-location & Latency Optimization (Advanced)

For true microsecond HFT (below 1ms round-trip), consider:

1. **Co-location:** Run the engine on a VPS in the same data centre as Alpaca/NYSE (NY4, Equinix). DigitalOcean NYC3 gives ~0.5ms to Alpaca.

2. **ONNX TensorRT:** Convert your ONNX model to TensorRT FP16 for GPU inference < 1ms:
```powershell
pip install tensorrt
python -c "
import tensorrt as trt
# Convert model.onnx → model.trt
# (requires TensorRT 8.6+ and matching CUDA)
"
```

3. **C++ inference:** For sub-1ms, wrap the ONNX model in a C++ process using `onnxruntime-cxx` and call it from Python via ctypes. This eliminates Python GIL overhead.

4. **Order routing:** Alpaca uses smart order routing. For tighter fills, consider Interactive Brokers with direct market access (DMA) routing to specific exchanges (ARCA, NASDAQ).

---

### 6.6 Regulatory Notes

| Region | Rule | Impact |
|---|---|---|
| **US** | Pattern Day Trader (PDT) | If account < $25,000, limited to 3 day trades per rolling 5 days. HFT requires ≥ $25,000. |
| **US** | Wash Sale Rule | Losses on repurchased positions within 30 days are disallowed for tax. Consult a CPA. |
| **HK** | SFC licensing | Algorithmic trading for personal accounts is generally unregulated. Trading on behalf of others requires a Type 1 SFC licence. |
| **General** | Best execution | Ensure your strategy doesn't constitute market manipulation (e.g., spoofing). |

---

### 6.7 Production Monitoring Setup

Once live, you need alerting. Add this to your workflow:

```powershell
# Scheduled task: check daily P&L every 30 minutes during market hours
# Save as monitor.py and run as a Windows Scheduled Task

import json, time
from pathlib import Path

THRESHOLD_DAILY_LOSS = -0.03  # -3% kills the bot

while True:
    trades = [json.loads(l) for l in Path('paper_trades.jsonl').read_text().splitlines()[-100:]]
    if trades:
        today_pnl = sum(t['realized_pnl'] for t in trades if t['timestamp'][:10] == time.strftime('%Y-%m-%d'))
        if today_pnl < THRESHOLD_DAILY_LOSS * 100_000:
            print("KILL SWITCH: Daily loss limit breached. Stopping.")
            # Send Telegram/email alert
            break
    time.sleep(1800)
```

---

## Quick Reference — Common Commands

```powershell
# --- Setup ---
.\.venv\Scripts\Activate.ps1                    # Activate venv every session
cd python_engine

# --- Download data ---
python data_pipeline.py --market us --timescale 1m

# --- Split a ticker ---
python split_data.py --ticker NVDA --market us --timescale 1m

# --- Train ---
python trainer.py --market us --timescale 1m --algo PPO --timesteps 2000000

# --- TensorBoard ---
tensorboard --logdir logs/

# --- Backtest + Red Team ---
python backtester.py --model models/NVDA_1m_PPO/best/best_model.zip --market us --timescale 1m --red-team

# --- Export ONNX ---
python trainer.py --export models/NVDA_1m_PPO/best/best_model.zip --output models/NVDA_1m_PPO/best/model.onnx

# --- Paper trade ---
python paper_trader.py --market us --ticker NVDA --model models/NVDA_1m_PPO/best/model.onnx --timescale 1m

# --- Start API server (for dashboard) ---
uvicorn api_server:app --port 8000

# --- Start dashboard (separate terminal, project root) ---
npm run dev
```

---

*Built with Stable-Baselines3 · ONNX Runtime · Alpaca Markets · FastAPI · React*
