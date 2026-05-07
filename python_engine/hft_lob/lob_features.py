"""
lob_features.py
===============
Phase 3 — Feature engineering & data-loading helpers for the LOB stack.

Public API
----------
FEATURE_COLS : list[str]
    Canonical ordered list of feature columns used by both the XGBoost
    baseline and the LOB PPO environment.
load_feature_df(symbols: list[str]) -> pd.DataFrame
    Load + concatenate per-symbol parquet files written by
    lob_reconstructor.reconstruct_and_save.
add_rolling_features(df: pd.DataFrame) -> pd.DataFrame
    Append rolling-window engineered features. Returns a new DataFrame.
build_xy(df: pd.DataFrame, horizon: int = 5) -> tuple[np.ndarray, np.ndarray]
    Produce (X, y) for supervised classification of mid-price direction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
FEAT_DIR = BASE_DIR / "data_lob" / "features"

# ---------------------------------------------------------------------------
# Canonical feature list — must remain stable across XGB / PPO / live trader.
# ---------------------------------------------------------------------------
FEATURE_COLS: list[str] = [
    # Top-of-book microstructure
    "spread",
    "imbalance_l1",
    "imbalance_l5",
    "micro_price_dev",        # (micro_price - mid_price) / mid_price
    # Depth (5 levels each side, sizes only — prices are relative to mid)
    "bid_sz_1", "bid_sz_2", "bid_sz_3", "bid_sz_4", "bid_sz_5",
    "ask_sz_1", "ask_sz_2", "ask_sz_3", "ask_sz_4", "ask_sz_5",
    # Trade flow (1-second window)
    "trades_count_1s",
    "volume_1s",
    "signed_vol_1s",
    # Rolling features (added by add_rolling_features)
    "ret_1s", "ret_5s", "ret_30s",
    "vol_5s", "vol_30s",
    "imb_ema_5s", "imb_ema_30s",
    "signed_vol_5s", "signed_vol_30s",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_feature_df(symbols: list[str]) -> pd.DataFrame:
    """
    Load all parquet feature files matching the given symbols and concat them.

    Files are produced by lob_reconstructor.reconstruct_and_save with the
    naming convention `<SYMBOL>_<...>.parquet`.

    Returns
    -------
    pd.DataFrame
        Sorted by (symbol, timestamp). Empty if no files match.
    """
    if not FEAT_DIR.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for sym in symbols:
        for fp in sorted(FEAT_DIR.glob(f"{sym}_*.parquet")):
            try:
                df = pd.read_parquet(str(fp))
                if "symbol" not in df.columns:
                    df["symbol"] = sym
                frames.append(df)
            except Exception as exc:
                logger.warning("[load_feature_df] failed to read %s: %s", fp, exc)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append rolling features per symbol. Input must contain at least:
      mid_price, micro_price, imbalance_l1, signed_vol_1s
    """
    if df.empty:
        return df

    df = df.copy()
    # Microprice deviation
    if "micro_price" in df.columns and "mid_price" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["micro_price_dev"] = np.where(
                df["mid_price"] > 0,
                (df["micro_price"] - df["mid_price"]) / df["mid_price"],
                0.0,
            )
    else:
        df["micro_price_dev"] = 0.0

    parts: list[pd.DataFrame] = []
    for _, g in df.groupby("symbol", sort=False):
        g = g.sort_values("timestamp").copy()
        mid = g["mid_price"].astype(float)
        log_mid = np.log(mid.where(mid > 0))

        g["ret_1s"]  = log_mid.diff(1).fillna(0.0)
        g["ret_5s"]  = log_mid.diff(5).fillna(0.0)
        g["ret_30s"] = log_mid.diff(30).fillna(0.0)

        g["vol_5s"]  = g["ret_1s"].rolling(5,  min_periods=1).std().fillna(0.0)
        g["vol_30s"] = g["ret_1s"].rolling(30, min_periods=1).std().fillna(0.0)

        g["imb_ema_5s"]  = g["imbalance_l1"].ewm(span=5,  adjust=False).mean()
        g["imb_ema_30s"] = g["imbalance_l1"].ewm(span=30, adjust=False).mean()

        g["signed_vol_5s"]  = g["signed_vol_1s"].rolling(5,  min_periods=1).sum()
        g["signed_vol_30s"] = g["signed_vol_1s"].rolling(30, min_periods=1).sum()

        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    # Replace inf/nan in the feature columns with zeros
    for col in FEATURE_COLS:
        if col in out.columns:
            out[col] = out[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Supervised target (Phase 3)
# ---------------------------------------------------------------------------
def build_xy(df: pd.DataFrame, horizon: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for supervised classification of mid-price direction.

    Label encoding:
      0 -> down  (future_mid - mid_price < -threshold)
      1 -> flat  (|future_mid - mid_price| <= threshold)
      2 -> up    (future_mid - mid_price >  threshold)

    The threshold is set per symbol as 0.5 × the median spread, capped at
    a tiny floor to avoid degenerate labels in flat books.
    """
    if df.empty:
        return np.empty((0, len(FEATURE_COLS)), dtype=np.float32), \
               np.empty((0,), dtype=np.int64)

    parts_X: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []
    for _, g in df.groupby("symbol", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        future = g["mid_price"].shift(-horizon)
        delta = (future - g["mid_price"]).astype(float)

        med_spread = float(np.nanmedian(g.get("spread", pd.Series([0.0])).values))
        thr = max(0.5 * med_spread, 1e-4)

        y = np.full(len(g), 1, dtype=np.int64)  # default flat
        y[delta >  thr] = 2
        y[delta < -thr] = 0

        # Drop the last `horizon` rows where future is NaN
        valid = future.notna().values
        X = g[FEATURE_COLS].astype(np.float32).values[valid]
        y = y[valid]
        parts_X.append(X)
        parts_y.append(y)

    if not parts_X:
        return np.empty((0, len(FEATURE_COLS)), dtype=np.float32), \
               np.empty((0,), dtype=np.int64)

    return np.concatenate(parts_X, axis=0), np.concatenate(parts_y, axis=0)
