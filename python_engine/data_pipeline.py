"""
data_pipeline.py — HFT Data Pipeline

Downloads OHLCV data for US and HK markets via Alpaca API (for US) 
and yfinance (for HK), resamples to target timescales (including synthetic 
10-second bars), computes technical features, and persists datasets to disk.
"""

from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from tqdm import tqdm

# Alpaca imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed  # <-- Added DataFeed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

US_TICKERS: list[str] = ["AAPL", "NVDA", "TSLA", "META", "GOOG", "OXY"]
HK_TICKERS: list[str] = ["0700.HK", "9988.HK", "0005.HK", "2318.HK", "1299.HK"]

SUPPORTED_INTERVALS: list[str] = ["1m", "5m", "1h", "1d", "1w"]
SUPPORTED_TIMESCALES: list[str] = ["10s", "1m", "5m", "1h", "1d", "1w"]

YFINANCE_TIMEOUT_S: int = 30
DOWNLOAD_RETRIES: int = 3


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_us_data(
    tickers: list[str] = US_TICKERS,
    period: str = "60d",
    interval: str = "1m",
) -> dict[str, pd.DataFrame]:
    """
    Download US market OHLCV data via Alpaca Historical API.
    """
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET must be set in .env")

    client = StockHistoricalDataClient(api_key, api_secret)
    
    if interval == "1m":
        tf = TimeFrame(1, TimeFrameUnit.Minute)
    elif interval == "1d":
        tf = TimeFrame(1, TimeFrameUnit.Day)
    elif interval == "1w":
        tf = TimeFrame(1, TimeFrameUnit.Week)
    else:
        tf = TimeFrame(1, TimeFrameUnit.Minute)

    days = int(period.replace("d", ""))
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    result: dict[str, pd.DataFrame] = {}
    
    try:
        # Added feed=DataFeed.IEX to prevent the 15-minute delayed SIP error on free tiers
        request_params = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=tf,
            start=start_dt,
            end=end_dt,
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(request_params)
        
        if bars.df.empty:
            logger.warning("Alpaca returned empty dataframe.")
            return result
            
        for ticker in tickers:
            if ticker in bars.df.index.get_level_values('symbol'):
                df = bars.df.xs(ticker, level='symbol').copy()
                df.rename(columns={
                    'open': 'Open', 'high': 'High', 'low': 'Low', 
                    'close': 'Close', 'volume': 'Volume'
                }, inplace=True)
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                logger.info("Downloaded %d rows for %s @ %s", len(df), ticker, interval)
                result[ticker] = df
            else:
                logger.warning("No data returned by Alpaca for %s", ticker)

    except Exception as exc: # noqa: BLE001
        logger.error("Error fetching data from Alpaca: %s", exc)

    return result

def download_hk_data(
    tickers: list[str] = HK_TICKERS,
    period: str = "60d",
    interval: str = "1m",
) -> dict[str, pd.DataFrame]:
    """Download Hong Kong (HKEX) market OHLCV data via yfinance."""
    result: dict[str, pd.DataFrame] = {}
    for ticker in tqdm(tickers, desc="HK tickers"):
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
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
                    logger.warning("Empty data for %s (attempt %d/%d)", ticker, attempt, DOWNLOAD_RETRIES)
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index)
                df.dropna(inplace=True)
                logger.info("Downloaded %d rows for %s @ %s", len(df), ticker, interval)
                result[ticker] = df
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("Error downloading %s (attempt %d/%d): %s", ticker, attempt, DOWNLOAD_RETRIES, exc)
    return result


# ---------------------------------------------------------------------------
# Resampling / aggregation
# ---------------------------------------------------------------------------

def _resample_to_10s(df: pd.DataFrame) -> pd.DataFrame:
    """Produce synthetic 10-second bars from 1-minute OHLCV data."""
    if df.empty:
        return df

    synthetic_rows: list[dict] = []
    closes, opens, highs, lows, volumes = df["Close"].values, df["Open"].values, df["High"].values, df["Low"].values, df["Volume"].values
    timestamps = df.index

    for i in range(len(df)):
        bar_open, bar_close, bar_high, bar_low, bar_vol = opens[i], closes[i], highs[i], lows[i], volumes[i]
        base_ts = timestamps[i]

        sub_closes = np.linspace(bar_open, bar_close, 7)[1:]  
        sub_opens = np.linspace(bar_open, bar_close, 7)[:-1]  
        vol_per_sub = bar_vol / 6.0

        for j in range(6):
            sub_open, sub_close = sub_opens[j], sub_closes[j]
            sub_high = min(max(sub_open, sub_close) * (1 + 0.0001), bar_high)
            sub_low = max(min(sub_open, sub_close) * (1 - 0.0001), bar_low)
            ts = base_ts + pd.Timedelta(seconds=j * 10)
            synthetic_rows.append(
                {"Open": sub_open, "High": sub_high, "Low": sub_low, "Close": sub_close, "Volume": vol_per_sub, "timestamp": ts}
            )

    result = pd.DataFrame(synthetic_rows).set_index("timestamp")
    result.index = pd.DatetimeIndex(result.index)
    return result

def aggregate_to_timescale(df: pd.DataFrame, timescale: str) -> pd.DataFrame:
    """Resample an OHLCV DataFrame to the requested timescale."""
    if timescale not in SUPPORTED_TIMESCALES:
        raise ValueError(f"Unsupported timescale '{timescale}'. Choose from {SUPPORTED_TIMESCALES}.")

    if timescale in ["1d", "1w"] and df.index.freq == "1D" and timescale == "1d":
        return df

    if timescale == "10s":
        return _resample_to_10s(df)

    rule_map = {"1m": "1min", "5m": "5min", "1h": "1h", "1d": "1D", "1w": "1W"}
    rule = rule_map[timescale]

    agg_funcs = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
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
    """Add technical indicators and rolling z-score normalised feature columns."""
    out = df.copy()
    close = out["Close"]

    out["rsi_14"] = _rsi(close, 14)
    
    ema12, ema26 = _ema(close, 12), _ema(close, 26)
    out["macd"] = ema12 - ema26
    out["macd_signal"] = _ema(out["macd"], 9)
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    bb_mid = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std()
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / bb_mid.replace(0, np.nan)

    out["atr_14"] = _atr(out, 14)
    out["volume_ma_20"] = out["Volume"].rolling(20, min_periods=1).mean()

    out["mom_1"], out["mom_5"], out["mom_10"] = close.pct_change(1), close.pct_change(5), close.pct_change(10)

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
    path = DATA_DIR / f"{market}_{timescale}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved dataset → %s (%d tickers)", path, len(data))
    return path

def load_dataset(market: str, timescale: str) -> dict[str, pd.DataFrame]:
    path = DATA_DIR / f"{market}_{timescale}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run data_pipeline.py main() first.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="HFT Data Pipeline")
    parser.add_argument(
        "--market", choices=["us", "hk", "both"], default="both",
        help="Which market(s) to download",
    )
    parser.add_argument(
        "--timescale", choices=["10s", "1m", "5m", "1h", "1d", "1w", "all"], default="all",
        help="Target timescale(s). Default 'all' fetches every supported interval.",
    )
    parser.add_argument("--period", default="60d", help="Data lookback period (e.g. 7d, 60d, 365d)")
    args = parser.parse_args()

    markets = ["us", "hk"] if args.market == "both" else [args.market]
    timescales = SUPPORTED_TIMESCALES if args.timescale == "all" else [args.timescale]

    # Explicitly demand every root timeframe needed to construct the complete target architecture
    fetch_intervals = set()
    for ts in timescales:
        if ts in ["1d", "1w"]:
            fetch_intervals.add(ts)
        else:
            fetch_intervals.add("1m")

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
            
            source_interval = timescale if timescale in ["1d", "1w"] else "1m"
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