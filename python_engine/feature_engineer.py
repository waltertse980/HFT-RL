"""
feature_engineer.py — Compute OHLCV-only features for the 5-Model Council.

Reads raw bar parquet files from ``data/bars/{SYMBOL}_{timeframe}.parquet``,
computes log returns, volatility, momentum (RSI / MACD / Stochastic),
microstructure proxies (range, body, wicks), and — for the primary 1m
timeframe — fuses multi-timeframe context from 5m and 10m featured frames
using ``merge_asof`` with backward direction (no look-ahead).

Saves enriched parquet to ``data/bars/{SYMBOL}_{timeframe}_featured.parquet``.

All implementations are pure pandas/numpy — no TA-Lib dependency.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DEFAULT_BARS_DIR = BASE_DIR / "data" / "bars"

# Raw OHLCV columns — excluded from the engineered feature list
_RAW_COLS = ["open", "high", "low", "close", "volume", "vwap", "trade_count"]


# ─────────────────────────────────────────────────────────────────────────────
# Indicator helpers (pure pandas, vectorised)
# ─────────────────────────────────────────────────────────────────────────────
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Classic Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing == EWMA with alpha = 1/period (adjust=False)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _macd(close: pd.Series,
          fast: int = 12, slow: int = 26, signal: int = 9
          ) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = _ema(close, fast) - _ema(close, slow)
    macd_signal = _ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 14, d_period: int = 3
                ) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k_period, min_periods=k_period).min()
    highest_high = high.rolling(k_period, min_periods=k_period).max()
    denom = (highest_high - lowest_low).replace(0.0, np.nan)
    stoch_k = 100.0 * (close - lowest_low) / (denom + 1e-9)
    stoch_d = stoch_k.rolling(d_period, min_periods=d_period).mean()
    return stoch_k, stoch_d


# ─────────────────────────────────────────────────────────────────────────────
# Single-frame feature computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_features(df: pd.DataFrame, timeframe: str = "1m") -> pd.DataFrame:
    """
    Compute the council's OHLCV feature set on a single DataFrame.

    Parameters
    ----------
    df         : DataFrame with at least open/high/low/close/volume columns
                 and a DatetimeIndex.
    timeframe  : Free-form label (e.g. "1m", "5m") — currently unused beyond
                 logging but retained for API parity.

    Returns
    -------
    DataFrame with engineered features APPENDED to the original OHLCV columns.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    needed = {"open", "high", "low", "close", "volume"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Input DataFrame missing required columns: {missing}")

    out = df.copy()
    out = out.sort_index()

    open_ = out["open"].astype("float64")
    high = out["high"].astype("float64")
    low = out["low"].astype("float64")
    close = out["close"].astype("float64")
    volume = out["volume"].astype("float64")

    # 1. Log returns
    for k in (1, 5, 10, 20):
        ratio = close / close.shift(k)
        out[f"log_ret_{k}"] = np.log(ratio.replace(0.0, np.nan))

    # 2. Volatility (rolling std of 1-bar log returns, window=20)
    out["volatility_20"] = out["log_ret_1"].rolling(20, min_periods=20).std()

    # 3. Volume profile
    rolling_mean_vol = volume.rolling(20, min_periods=20).mean()
    out["vol_ratio"] = volume / (rolling_mean_vol + 1e-9)
    out["vol_delta"] = volume - volume.shift(1)

    # 4. Momentum indicators
    rsi_raw = _rsi(close, 14)
    # Normalise to [-0.5, 0.5] for nicer NN inputs
    out["rsi_14"] = rsi_raw / 100.0 - 0.5

    macd_line, macd_signal, macd_hist = _macd(close, 12, 26, 9)
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    stoch_k, stoch_d = _stochastic(high, low, close, 14, 3)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d

    # 5. Microstructure proxies (bar-level)
    rng = (high - low).replace(0.0, np.nan)
    out["bar_range"] = (high - low) / (close + 1e-9)
    out["body_ratio"] = (close - open_).abs() / (rng + 1e-9)

    upper_anchor = pd.concat([open_, close], axis=1).max(axis=1)
    lower_anchor = pd.concat([open_, close], axis=1).min(axis=1)
    out["upper_wick"] = (high - upper_anchor) / (rng + 1e-9)
    out["lower_wick"] = (lower_anchor - low) / (rng + 1e-9)

    # ── NaN handling: forward-fill then drop remaining (early-window NaNs) ──
    feature_cols = [c for c in out.columns if c not in _RAW_COLS]
    out[feature_cols] = out[feature_cols].ffill()
    out = out.dropna(subset=feature_cols).copy()

    # Final cast to float32 for engineered features (memory + SB3-friendly)
    for c in feature_cols:
        out[c] = out[c].astype("float32")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Multi-timeframe fusion
# ─────────────────────────────────────────────────────────────────────────────
def add_multi_timeframe_context(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame | None,
    df_10m: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Add tf5_* and tf10_* columns to a 1-minute featured DataFrame, using
    ``merge_asof`` with backward direction so a 5-minute bar with close
    timestamp ``T`` becomes visible at the first 1-minute bar with
    timestamp ``>= T`` — no look-ahead bias.

    Both inputs (df_5m / df_10m) may be None or empty; the fusion will still
    succeed and emit the relevant ``tf*_`` columns filled with zeros so that
    downstream observation shapes remain stable.
    """
    if df_1m is None or df_1m.empty:
        return df_1m.copy() if df_1m is not None else pd.DataFrame()

    base = df_1m.copy().sort_index()

    def _merge(base_df: pd.DataFrame, other: pd.DataFrame | None, prefix: str) -> pd.DataFrame:
        # Determine which columns we'd like to carry over (engineered only)
        if other is None or other.empty:
            # We cannot know the schema in advance — return base unchanged;
            # callers will pad missing tf*_ columns later if needed.
            return base_df
        eng_cols = [c for c in other.columns if c not in _RAW_COLS]
        if not eng_cols:
            return base_df
        right = other[eng_cols].copy().sort_index()
        right.index = pd.to_datetime(right.index, utc=True)
        right = right.add_prefix(prefix)
        right = right.reset_index().rename(columns={right.index.name or "index": "timestamp"})
        # Some indexes have name=None — ensure column is called 'timestamp'
        if "timestamp" not in right.columns:
            # Fallback — first col after reset_index is the old index
            right = right.rename(columns={right.columns[0]: "timestamp"})

        left = base_df.reset_index()
        left_ts_col = left.columns[0] if "timestamp" not in left.columns else "timestamp"
        if left_ts_col != "timestamp":
            left = left.rename(columns={left_ts_col: "timestamp"})
        left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True)
        right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True)

        merged = pd.merge_asof(
            left.sort_values("timestamp"),
            right.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
        merged = merged.set_index("timestamp")
        return merged

    base = _merge(base, df_5m, "tf5_")
    base = _merge(base, df_10m, "tf10_")

    # Forward-fill any gaps introduced by merge_asof at the very start
    tf_cols = [c for c in base.columns if c.startswith("tf5_") or c.startswith("tf10_")]
    if tf_cols:
        base[tf_cols] = base[tf_cols].ffill()
        # Any still-NaN values mean the higher timeframe had no preceding bar
        # for that 1m timestamp → fill with 0 to preserve a stable obs shape.
        base[tf_cols] = base[tf_cols].fillna(0.0)
        for c in tf_cols:
            base[c] = base[c].astype("float32")

    return base


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────────────────────────────────────
def _load_bars(symbol: str, timeframe: str, bars_dir: Path) -> pd.DataFrame | None:
    fpath = bars_dir / f"{symbol}_{timeframe}.parquet"
    if not fpath.exists():
        log.warning("Raw bar file missing: %s", fpath)
        return None
    df = pd.read_parquet(fpath)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    return df


def engineer_all(
    symbols: Iterable[str],
    bars_dir: str | Path = DEFAULT_BARS_DIR,
    timeframes: Iterable[str] = ("1m", "5m", "10m"),
) -> list[Path]:
    """
    Batch-process every (symbol, timeframe) combo present in ``bars_dir``:
        1. Compute single-frame features for 5m and 10m first.
        2. Compute single-frame features for 1m.
        3. Merge 5m / 10m context into the 1m frame.
        4. Save *_featured.parquet next to the raw files.

    Returns the list of paths written.
    """
    bars_dir = Path(bars_dir)
    bars_dir.mkdir(parents=True, exist_ok=True)
    tf_list = list(timeframes)
    written: list[Path] = []

    for sym in symbols:
        log.info("─── Engineering features for %s ───", sym)

        # Featured higher timeframes first (so 1m can fuse them)
        featured_by_tf: dict[str, pd.DataFrame] = {}
        for tf in tf_list:
            if tf == "1m":
                continue
            raw = _load_bars(sym, tf, bars_dir)
            if raw is None or raw.empty:
                continue
            feat = compute_features(raw, timeframe=tf)
            featured_by_tf[tf] = feat
            out_path = bars_dir / f"{sym}_{tf}_featured.parquet"
            feat.to_parquet(out_path, engine="pyarrow", index=True)
            written.append(out_path)
            log.info("Saved %s rows=%d cols=%d", out_path.name, len(feat), feat.shape[1])

        # 1m frame with multi-timeframe context
        if "1m" in tf_list:
            raw_1m = _load_bars(sym, "1m", bars_dir)
            if raw_1m is None or raw_1m.empty:
                log.warning("Skipping %s 1m — raw data missing", sym)
                continue
            feat_1m = compute_features(raw_1m, timeframe="1m")
            feat_1m = add_multi_timeframe_context(
                feat_1m,
                featured_by_tf.get("5m"),
                featured_by_tf.get("10m"),
            )
            out_path = bars_dir / f"{sym}_1m_featured.parquet"
            feat_1m.to_parquet(out_path, engine="pyarrow", index=True)
            written.append(out_path)
            log.info("Saved %s rows=%d cols=%d (with tf5_/tf10_ context)",
                     out_path.name, len(feat_1m), feat_1m.shape[1])

    return written


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute council features from raw bars")
    p.add_argument("--symbols", nargs="+", default=["TSLA", "NVDA", "AAPL"])
    p.add_argument("--timeframes", nargs="+", default=["1m", "5m", "10m"])
    p.add_argument("--bars-dir", default=str(DEFAULT_BARS_DIR))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    written = engineer_all(args.symbols, args.bars_dir, args.timeframes)
    print(f"\nWrote {len(written)} featured parquet files.")
    for p in written:
        print(f"  • {p}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
