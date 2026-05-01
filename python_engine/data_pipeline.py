"""
data_pipeline.py — HFT Data Pipeline

Downloads OHLCV data for US and HK markets via yfinance, resamples to target
timescales (including synthetic 10-second bars), computes technical features,
and persists datasets to disk.

NOTE: yfinance does not natively support 10-second bar data. The smallest
granularity is 1-minute bars. We produce synthetic 10s bars by linear
interpolation of price series and proportional volume distribution across
the 6 sub-bars per minute. These bars are approximations and should be
treated as training proxies, not real tick data.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

US_TICKERS: list[str] = ["AAPL", "NVDA", "TSLA", "META", "GOOG", "OXY"]
HK_TICKERS: list[str] = ["0700.HK", "9988.HK", "0005.HK", "2318.HK", "1299.HK"]

SUPPORTED_INTERVALS: list[str] = ["1m", "5m", "1h"]
SUPPORTED_TIMESCALES: list[str] = ["10s", "1m", "5m", "1h"]

YFINANCE_TIMEOUT_S: int = 30
DOWNLOAD_RETRIES: int = 3


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_single(
    ticker: str,
    period: str,
    interval: str,
    retries: int = DOWNLOAD_RETRIES,
) -> Optional[pd.DataFrame]:
    """Download OHLCV data for a single ticker with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                timeout=YFINANCE_TIMEOUT_S,
            )
            if df.empty:
                logger.warning("Empty data for %s (attempt %d/%d)", ticker, attempt, retries)
                continue
            # Flatten MultiIndex columns if present (happens with single tickers)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index)
            df.dropna(inplace=True)
            logger.info("Downloaded %d rows for %s @ %s", len(df), ticker, interval)
            return df
        except Exception as exc:  # noqa: BLE001
            logger.error("Error downloading %s (attempt %d/%d): %s", ticker, attempt, retries, exc)
    return None


def download_us_data(
    tickers: list[str] = US_TICKERS,
    period: str = "60d",
    interval: str = "1m",
) -> dict[str, pd.DataFrame]:
    """
    Download US market OHLCV data via yfinance.

    Parameters
    ----------
    tickers:  List of US ticker symbols (default: AAPL, NVDA, TSLA, META, GOOG, OXY).
    period:   yfinance period string, e.g. '7d', '60d'.
              Note: 1m data is limited to the last 7 days by yfinance.
    interval: One of '1m', '5m', '1h'. For 10s bars call
              aggregate_to_timescale(df, '10s') afterwards.

    Returns
    -------
    Dict mapping ticker → OHLCV DataFrame (DatetimeIndex).
    """
    result: dict[str, pd.DataFrame] = {}
    for ticker in tqdm(tickers, desc="US tickers"):
        df = _download_single(ticker, period, interval)
        if df is not None:
            result[ticker] = df
    return result


def download_hk_data(
    tickers: list[str] = HK_TICKERS,
    period: str = "60d",
    interval: str = "1m",
) -> dict[str, pd.DataFrame]:
    """
    Download Hong Kong (HKEX) market OHLCV data via yfinance.

    Tickers (default):
        0700.HK — Tencent
        9988.HK — Alibaba
        0005.HK — HSBC
        2318.HK — Ping An Insurance
        1299.HK — AIA Group

    Parameters and return value identical to download_us_data().
    """
    result: dict[str, pd.DataFrame] = {}
    for ticker in tqdm(tickers, desc="HK tickers"):
        df = _download_single(ticker, period, interval)
        if df is not None:
            result[ticker] = df
    return result


# ---------------------------------------------------------------------------
# Resampling / aggregation
# ---------------------------------------------------------------------------


def _resample_to_10s(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce synthetic 10-second bars from 1-minute OHLCV data.

    Method
    ------
    For each 1-minute bar we create 6 synthetic 10s sub-bars:
      - Close prices are linearly interpolated between successive bar closes.
      - Open = previous synthetic close (first sub-bar open = parent open).
      - High and Low are computed per sub-bar from the interpolated path.
      - Volume is distributed proportionally (uniform split for simplicity).

    NOTE: This is an approximation. Real 10s microstructure data is not
    available via yfinance. Use for training only.
    """
    if df.empty:
        return df

    synthetic_rows: list[dict] = []

    closes = df["Close"].values
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    volumes = df["Volume"].values
    timestamps = df.index

    for i in range(len(df)):
        bar_open = opens[i]
        bar_close = closes[i]
        bar_high = highs[i]
        bar_low = lows[i]
        bar_vol = volumes[i]
        base_ts = timestamps[i]

        # Create 6 sub-bar close prices via linear interpolation
        sub_closes = np.linspace(bar_open, bar_close, 7)[1:]  # 6 end-points
        sub_opens = np.linspace(bar_open, bar_close, 7)[:-1]  # 6 start-points

        vol_per_sub = bar_vol / 6.0

        for j in range(6):
            sub_open = sub_opens[j]
            sub_close = sub_closes[j]
            # High/low bounded by parent bar high/low
            sub_high = min(max(sub_open, sub_close) * (1 + 0.0001), bar_high)
            sub_low = max(min(sub_open, sub_close) * (1 - 0.0001), bar_low)
            ts = base_ts + pd.Timedelta(seconds=j * 10)
            synthetic_rows.append(
                {
                    "Open": sub_open,
                    "High": sub_high,
                    "Low": sub_low,
                    "Close": sub_close,
                    "Volume": vol_per_sub,
                    "timestamp": ts,
                }
            )

    result = pd.DataFrame(synthetic_rows).set_index("timestamp")
    result.index = pd.DatetimeIndex(result.index)
    return result


def aggregate_to_timescale(df: pd.DataFrame, timescale: str) -> pd.DataFrame:
    """
    Resample an OHLCV DataFrame to the requested timescale.

    Parameters
    ----------
    df:        OHLCV DataFrame with a DatetimeIndex.
    timescale: One of '10s', '1m', '5m', '1h'.

    Returns
    -------
    Resampled OHLCV DataFrame.
    """
    if timescale not in SUPPORTED_TIMESCALES:
        raise ValueError(f"Unsupported timescale '{timescale}'. Choose from {SUPPORTED_TIMESCALES}.")

    if timescale == "10s":
        # Build from 1m resolution — caller should pass 1m data
        return _resample_to_10s(df)

    rule_map = {"1m": "1min", "5m": "5min", "1h": "1h"}
    rule = rule_map[timescale]

    agg_funcs = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }

    # Keep only OHLCV columns
    ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    resampled = df[ohlcv_cols].resample(rule).agg(agg_funcs).dropna()
    return resampled


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(com=period - 1, min_periods=period).mean()


def _rolling_zscore(series: pd.Series, window: int = 100) -> pd.Series:
    mean = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std().replace(0, np.nan)
    return (series - mean) / std


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators and rolling z-score normalised feature columns.

    Features added
    --------------
    rsi_14           — RSI with 14-period lookback
    macd             — MACD line (EMA12 - EMA26)
    macd_signal      — Signal line (EMA9 of MACD)
    macd_hist        — MACD histogram
    bb_upper         — Bollinger Band upper (20-period, 2σ)
    bb_lower         — Bollinger Band lower
    bb_width         — Band width normalised by mid
    atr_14           — Average True Range (14-period)
    volume_ma_20     — 20-bar volume moving average
    mom_1            — 1-bar price momentum (Close % change)
    mom_5            — 5-bar price momentum
    mom_10           — 10-bar price momentum

    All features are also stored as z-score normalised versions with the
    prefix 'z_', using a rolling window of 100 bars.

    Returns
    -------
    Original df with feature columns appended (in-place copy).
    """
    out = df.copy()
    close = out["Close"]

    # RSI
    out["rsi_14"] = _rsi(close, 14)

    # MACD
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    out["macd"] = ema12 - ema26
    out["macd_signal"] = _ema(out["macd"], 9)
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # Bollinger Bands
    bb_mid = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std()
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / bb_mid.replace(0, np.nan)

    # ATR
    out["atr_14"] = _atr(out, 14)

    # Volume MA
    out["volume_ma_20"] = out["Volume"].rolling(20, min_periods=1).mean()

    # Momentum
    out["mom_1"] = close.pct_change(1)
    out["mom_5"] = close.pct_change(5)
    out["mom_10"] = close.pct_change(10)

    # Rolling z-score normalisation (window=100) for all feature columns
    feature_cols = [
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_width",
        "atr_14", "volume_ma_20",
        "mom_1", "mom_5", "mom_10",
        "Open", "High", "Low", "Close", "Volume",
    ]
    for col in feature_cols:
        if col in out.columns:
            out[f"z_{col}"] = _rolling_zscore(out[col], window=100)

    out.dropna(inplace=True)
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_dataset(data: dict[str, pd.DataFrame], market: str, timescale: str) -> Path:
    """
    Persist a market dataset dict to disk.

    Parameters
    ----------
    data:      Dict mapping ticker → DataFrame.
    market:    'us' or 'hk'.
    timescale: e.g. '10s', '1m', '5m', '1h'.

    Returns
    -------
    Path to the saved file.
    """
    path = DATA_DIR / f"{market}_{timescale}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved dataset → %s (%d tickers)", path, len(data))
    return path


def load_dataset(market: str, timescale: str) -> dict[str, pd.DataFrame]:
    """
    Load a previously saved market dataset.

    Parameters
    ----------
    market:    'us' or 'hk'.
    timescale: e.g. '10s', '1m', '5m', '1h'.

    Returns
    -------
    Dict mapping ticker → DataFrame.

    Raises
    ------
    FileNotFoundError if the dataset has not been downloaded yet.
    """
    path = DATA_DIR / f"{market}_{timescale}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run data_pipeline.py main() first."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info("Loaded dataset from %s (%d tickers)", path, len(data))
    return data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Download all markets and timescales, compute features, and save.

    Usage
    -----
        python data_pipeline.py

    Timescale note: 10s bars are derived from 1m downloads and are synthetic.
    """
    import argparse

    parser = argparse.ArgumentParser(description="HFT Data Pipeline")
    parser.add_argument(
        "--market", choices=["us", "hk", "both"], default="both",
        help="Which market(s) to download",
    )
    parser.add_argument(
        "--timescale", choices=["10s", "1m", "5m", "1h", "all"], default="all",
        help="Target timescale(s)",
    )
    parser.add_argument("--period", default="7d", help="yfinance period string (e.g. 7d, 60d)")
    args = parser.parse_args()

    markets = ["us", "hk"] if args.market == "both" else [args.market]
    timescales = ["10s", "1m", "5m", "1h"] if args.timescale == "all" else [args.timescale]

    # For 10s bars we need 1m source data — always fetch 1m
    fetch_intervals = set()
    for ts in timescales:
        fetch_intervals.add("1m" if ts == "10s" else ts)

    for market in markets:
        download_fn = download_us_data if market == "us" else download_hk_data
        tickers = US_TICKERS if market == "us" else HK_TICKERS

        raw_cache: dict[str, dict[str, pd.DataFrame]] = {}
        for interval in tqdm(fetch_intervals, desc=f"{market.upper()} intervals"):
            logger.info("Downloading %s @ %s (period=%s)", market.upper(), interval, args.period)
            raw = download_fn(tickers=tickers, period=args.period, interval=interval)
            raw_cache[interval] = raw

        for timescale in timescales:
            logger.info("Processing timescale=%s for market=%s", timescale, market)
            source_interval = "1m" if timescale == "10s" else timescale
            source_data = raw_cache.get(source_interval, {})

            processed: dict[str, pd.DataFrame] = {}
            for ticker, df in tqdm(source_data.items(), desc=f"{market}@{timescale}"):
                try:
                    resampled = aggregate_to_timescale(df, timescale)
                    featured = compute_features(resampled)
                    processed[ticker] = featured
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to process %s@%s: %s", ticker, timescale, exc)

            if processed:
                save_dataset(processed, market, timescale)
            else:
                logger.warning("No data processed for %s@%s — skipping save.", market, timescale)

    logger.info("Data pipeline complete.")


if __name__ == "__main__":
    main()
