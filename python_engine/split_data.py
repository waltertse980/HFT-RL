"""
split_data.py — Temporal 60/30/10 train/val/test split for RL training.

Usage:
    python split_data.py --ticker NVDA --market us --timescale 5m
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
