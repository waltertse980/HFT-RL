"""
data_pipeline.py — HFT Data Pipeline  (Optimised v2)
=====================================================
Changes from v1
---------------
* Synthetic 10s bars now add Gaussian micro-noise so the network cannot
  learn the deterministic np.linspace artefact.
* New ``compute_microstructure_features()`` adds Order Book Imbalance (OBI),
  micro-price, Trade Flow Imbalance (TFI), realised volatility, spread proxy,
  and queue-depth ratio — all strictly stationary (z-scored log-returns /
  ratios, no raw price levels).
* ``compute_features()`` is enhanced: raw OHLCV prices are dropped from the
  feature set; only stationary transforms (log-returns, z-scored ratios) are
  kept.  This is the single biggest fix for the out-of-sample collapse.
* Fractional-differencing helper added (Lopez de Prado, AFML §5).
* ``load_dataset`` / ``save_dataset`` accept an optional ``ticker`` kwarg so
  callers can load a single ticker without deserialising the whole file.
* All import names fixed (``load_dataset`` singular, ``SUPPORTED_TIMESCALES``).
* Alpaca IEX feed replaces yfinance for US data (no more free-tier 15-min lag).
* yfinance retained for HK data (Futu OpenAPI is the upgrade path).

Level 2 data note
-----------------
Full L2 (bid/ask ladder, tick-by-tick) requires:
  - Alpaca Algo Trader Plus ($99/mo) → StockQuotesRequest + StockTradesRequest
  - Polygon.io flat-file tick data
  - Futu OpenAPI for HKEX
When L2 columns (bid_price_1, ask_price_1, bid_size_1, ask_size_1) are
present in the DataFrame, ``compute_microstructure_features()`` computes full
OBI/micro-price/TFI.  With OHLCV-only data it falls back to proxy versions.
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
from tqdm import tqdm

# yfinance is an optional fallback — Alpaca is the primary source for US data.
# If yfinance is not installed, HK data will be unavailable until Futu OpenAPI
# is integrated.  Install with: pip install yfinance
try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional Alpaca import (US L1 data)
# ---------------------------------------------------------------------------
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

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

US_TICKERS: list[str] = ["AAPL", "NVDA", "TSLA", "META", "GOOG", "MSFT", "AMZN"]
HK_TICKERS: list[str] = ["0700.HK", "9988.HK", "0005.HK", "2318.HK", "1299.HK"]

SUPPORTED_TIMESCALES: list[str] = ["10s", "1m", "5m", "1h"]
YFINANCE_TIMEOUT_S: int = 30
DOWNLOAD_RETRIES: int = 3

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_single_yf(
    ticker: str,
    period: str,
    interval: str,
    retries: int = DOWNLOAD_RETRIES,
) -> Optional[pd.DataFrame]:
    """Download OHLCV data for a single ticker via yfinance with retry."""
    if not _YFINANCE_AVAILABLE:
        raise ImportError(
            "yfinance is not installed. For US stocks, configure Alpaca API keys in Settings. "
            "For HK stocks, install yfinance: pip install yfinance"
        )
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
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index)
            df.dropna(inplace=True)
            logger.info("Downloaded %d rows for %s @ %s (yfinance)", len(df), ticker, interval)
            return df
        except Exception as exc:  # noqa: BLE001
            logger.error("yfinance error %s attempt %d: %s", ticker, attempt, exc)
    return None


def _parse_date_range(
    start: Optional[str],
    end: Optional[str],
    default_days: int = 7,
) -> tuple[datetime, datetime]:
    """
    Resolve an explicit start/end date string pair into datetime objects.

    If start is None, defaults to ``default_days`` days before end.
    If end is None, defaults to now (UTC).
    Accepts ISO-8601 strings: '2026-01-01', '2026-01-01T00:00:00', etc.
    """
    end_dt: datetime = (
        datetime.fromisoformat(end.replace("Z", "")) if end else datetime.utcnow()
    )
    start_dt: datetime = (
        datetime.fromisoformat(start.replace("Z", ""))
        if start
        else end_dt - timedelta(days=default_days)
    )
    return start_dt, end_dt


def download_us_data(
    tickers: list[str] = US_TICKERS,
    period: str = "60d",
    interval: str = "1m",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Download US OHLCV data.

    Tries Alpaca IEX feed first (requires ALPACA_API_KEY / ALPACA_API_SECRET
    env vars).  Falls back to yfinance automatically.

    Parameters
    ----------
    period:   Relative window used ONLY when start/end are both None.
              '7d', '60d', '365d'.  Alpaca supports up to 2 years for 1m.
    interval: '1m', '5m', '1h'
    start:    ISO-8601 date string e.g. '2026-01-01' (takes priority over period)
    end:      ISO-8601 date string e.g. '2026-05-04' (defaults to today)
    """
    result: dict[str, pd.DataFrame] = {}

    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")

    # Resolve date range: explicit start/end take priority over period
    if start or end:
        default_days = int(period.replace("d", "")) if period.endswith("d") else 7
        start_dt, end_dt = _parse_date_range(start, end, default_days=default_days)
    else:
        days = int(period.replace("d", "")) if period.endswith("d") else 7
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=days)

    logger.info(
        "US download window: %s → %s (interval=%s)",
        start_dt.date(), end_dt.date(), interval,
    )

    if _ALPACA_AVAILABLE and api_key and api_secret:
        logger.info("Using Alpaca IEX feed for US data.")
        try:
            client = StockHistoricalDataClient(api_key, api_secret)
            tf_map = {
                "1m":  TimeFrame(1,  TimeFrameUnit.Minute),
                "5m":  TimeFrame(5,  TimeFrameUnit.Minute),
                "1h":  TimeFrame(1,  TimeFrameUnit.Hour),
            }
            tf = tf_map.get(interval, TimeFrame(1, TimeFrameUnit.Minute))
            req = StockBarsRequest(
                symbol_or_symbols=tickers,
                timeframe=tf,
                start=start_dt,
                end=end_dt,
                feed=DataFeed.IEX,
            )
            bars = client.get_stock_bars(req)
            if not bars.df.empty:
                for ticker in tickers:
                    try:
                        df = bars.df.xs(ticker, level="symbol").copy()
                        df.rename(columns={
                            "open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume",
                        }, inplace=True)
                        df = df[["Open", "High", "Low", "Close", "Volume"]]
                        logger.info("Alpaca: %d rows for %s", len(df), ticker)
                        result[ticker] = df
                    except KeyError:
                        logger.warning("No Alpaca data for %s", ticker)
                if result:
                    return result
        except Exception as exc:
            logger.warning("Alpaca download failed (%s), falling back to yfinance.", exc)

    # yfinance fallback — period string only (no explicit date range support)
    yf_period = period if not (start or end) else (
        f"{max(1, (end_dt - start_dt).days)}d"
    )
    for ticker in tqdm(tickers, desc="US tickers (yfinance)"):
        df = _download_single_yf(ticker, yf_period, interval)
        if df is not None:
            result[ticker] = df
    return result


def download_hk_data(
    tickers: list[str] = HK_TICKERS,
    period: str = "60d",
    interval: str = "1m",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Download HKEX OHLCV data via yfinance.

    Upgrade path: replace with Futu OpenAPI for true L2 data.

    Parameters
    ----------
    start/end: ISO-8601 date strings.  When provided, the period is computed
               from the date delta and passed to yfinance (which only accepts
               relative periods for intraday intervals).
    """
    result: dict[str, pd.DataFrame] = {}

    if start or end:
        default_days = int(period.replace("d", "")) if period.endswith("d") else 60
        start_dt, end_dt = _parse_date_range(start, end, default_days=default_days)
        yf_period = f"{max(1, (end_dt - start_dt).days)}d"
        logger.info(
            "HK download window: %s → %s → yfinance period=%s (interval=%s)",
            start_dt.date(), end_dt.date(), yf_period, interval,
        )
    else:
        yf_period = period

    for ticker in tqdm(tickers, desc="HK tickers"):
        df = _download_single_yf(ticker, yf_period, interval)
        if df is not None:
            result[ticker] = df
    return result


# ---------------------------------------------------------------------------
# Resampling / aggregation
# ---------------------------------------------------------------------------


def _resample_to_10s(df: pd.DataFrame, noise_sigma: float = 0.0002) -> pd.DataFrame:
    """
    Produce synthetic 10-second bars from 1-minute OHLCV data.

    v2 improvement: Gaussian micro-noise is added to each sub-bar close so
    the neural network cannot learn the deterministic linspace interpolation
    artefact that caused the TSLA overfitting.

    Parameters
    ----------
    noise_sigma: Std-dev of the Gaussian noise added to each sub-close as a
                 fraction of that bar's close.  Default 0.02% (2 bps).
    """
    if df.empty:
        return df

    rng = np.random.default_rng(seed=42)
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

        # Base interpolated path
        path = np.linspace(bar_open, bar_close, 7)
        # Add noise to interior points (not first or last — anchored to bar O/C)
        noise = rng.normal(0.0, noise_sigma * abs(bar_close), size=7)
        noise[0] = 0.0
        noise[-1] = 0.0
        path = path + noise

        sub_closes = path[1:]
        sub_opens = path[:-1]
        vol_per_sub = bar_vol / 6.0

        for j in range(6):
            sub_open = float(sub_opens[j])
            sub_close = float(sub_closes[j])
            sub_high = min(max(sub_open, sub_close) * (1 + 0.0001), bar_high)
            sub_low = max(min(sub_open, sub_close) * (1 - 0.0001), bar_low)
            ts = base_ts + pd.Timedelta(seconds=j * 10)
            synthetic_rows.append({
                "Open": sub_open, "High": sub_high, "Low": sub_low,
                "Close": sub_close, "Volume": vol_per_sub, "timestamp": ts,
            })

    result = pd.DataFrame(synthetic_rows).set_index("timestamp")
    result.index = pd.DatetimeIndex(result.index)
    return result


def aggregate_to_timescale(df: pd.DataFrame, timescale: str) -> pd.DataFrame:
    """Resample an OHLCV DataFrame to the requested timescale."""
    if timescale not in SUPPORTED_TIMESCALES:
        raise ValueError(f"Unsupported timescale '{timescale}'. Choose from {SUPPORTED_TIMESCALES}.")

    if timescale == "10s":
        return _resample_to_10s(df)

    rule_map = {"1m": "1min", "5m": "5min", "1h": "1h"}
    rule = rule_map[timescale]
    agg_funcs = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[ohlcv_cols].resample(rule).agg(agg_funcs).dropna()


# ---------------------------------------------------------------------------
# Core feature helpers (stationary transforms)
# ---------------------------------------------------------------------------


def _log_returns(series: pd.Series) -> pd.Series:
    """Log returns: log(P_t / P_{t-1}).  Stationary by construction."""
    return np.log(series / series.shift(1))


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
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _rolling_zscore(series: pd.Series, window: int = 100) -> pd.Series:
    mean = series.rolling(window=window, min_periods=max(1, window // 4)).mean()
    std = series.rolling(window=window, min_periods=max(1, window // 4)).std().replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)


def _fractional_diff(series: pd.Series, d: float = 0.4, threshold: float = 1e-4) -> pd.Series:
    """
    Fractional differencing (Lopez de Prado, AFML §5).

    Preserves memory while achieving stationarity.
    d=0.4 is a common starting point; increase toward 1.0 if ADF still fails.
    """
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    weights = np.array(weights[::-1])

    result = pd.Series(np.nan, index=series.index, dtype=float)
    T = len(series)
    L = len(weights)
    arr = series.values
    for t in range(L - 1, T):
        result.iloc[t] = float(np.dot(weights, arr[t - L + 1: t + 1]))
    return result


# ---------------------------------------------------------------------------
# Microstructure feature engineering  (L2-ready)
# ---------------------------------------------------------------------------


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute microstructure features.

    If the DataFrame contains L2 columns (bid_price_1, ask_price_1,
    bid_size_1, ask_size_1) full OBI/micro-price/spread are computed.
    Otherwise, proxy versions derived from OHLCV are used — these are weaker
    but still better than raw price levels.

    All output features are stationary z-scored or ratio features.
    Raw price levels are NEVER passed to the network.

    Features produced (all prefixed z_)
    -------------------------------------
    z_log_ret        — log return of close (primary signal)
    z_log_ret_sq     — squared log return (volatility proxy)
    z_rsi_14         — RSI normalised
    z_macd_hist      — MACD histogram normalised
    z_bb_width       — Bollinger band width / mid-price
    z_atr_norm       — ATR / close (normalised volatility)
    z_vol_ratio      — volume / 20-bar rolling mean volume
    z_mom_3          — 3-bar momentum
    z_mom_10         — 10-bar momentum
    z_realised_vol   — rolling 20-bar realised vol (std of log-returns)
    z_obi            — Order Book Imbalance  (L2 or proxy)
    z_micro_price    — micro-price deviation from mid  (L2 or proxy)
    z_tfi            — Trade Flow Imbalance (cumulative signed volume)
    z_spread         — bid-ask spread / mid-price
    position         — current position encoding added by env, not here
    """
    out = df.copy()
    close = out["Close"]
    high  = out["High"]
    low   = out["Low"]
    vol   = out["Volume"]

    # ── Stationary price features ─────────────────────────────────────────
    log_ret = _log_returns(close).fillna(0.0)
    out["log_ret"]     = log_ret
    out["log_ret_sq"]  = log_ret ** 2

    # ── Classic indicators (normalised to be stationary) ──────────────────
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd  = ema12 - ema26
    macd_signal = _ema(macd, 9)
    out["rsi_14"]   = _rsi(close, 14)
    out["macd_hist"] = (macd - macd_signal) / (close.replace(0, np.nan))  # price-normalised

    bb_mid = close.rolling(20, min_periods=5).mean()
    bb_std = close.rolling(20, min_periods=5).std().replace(0, np.nan)
    out["bb_width"] = (4 * bb_std) / bb_mid.replace(0, np.nan)  # width/mid

    atr = _atr(out, 14)
    out["atr_norm"] = atr / close.replace(0, np.nan)  # ATR relative to price

    vol_ma20 = vol.rolling(20, min_periods=1).mean().replace(0, np.nan)
    out["vol_ratio"] = vol / vol_ma20

    out["mom_3"]  = close.pct_change(3).fillna(0.0)
    out["mom_10"] = close.pct_change(10).fillna(0.0)

    # ── Realised volatility (rolling std of log-returns) ──────────────────
    out["realised_vol"] = log_ret.rolling(20, min_periods=5).std().fillna(0.0)

    # ── Microstructure features ───────────────────────────────────────────
    has_l2 = all(c in df.columns for c in ["bid_price_1", "ask_price_1", "bid_size_1", "ask_size_1"])

    if has_l2:
        bid_p = out["bid_price_1"]
        ask_p = out["ask_price_1"]
        bid_s = out["bid_size_1"]
        ask_s = out["ask_size_1"]
        mid   = (bid_p + ask_p) / 2.0
        denom = (bid_s + ask_s).replace(0, np.nan)

        # Order Book Imbalance: range [-1, +1]
        out["obi"] = (bid_s - ask_s) / denom

        # Micro-price deviation from mid
        micro_p = (ask_p * bid_s + bid_p * ask_s) / denom
        out["micro_price"] = (micro_p - mid) / mid.replace(0, np.nan)

        # Bid-ask spread
        out["spread"] = (ask_p - bid_p) / mid.replace(0, np.nan)

        # Trade Flow Imbalance (rolling signed volume; positive = buy pressure)
        # tick rule: log_ret > 0 → buyer-initiated
        signed_vol = vol * np.sign(log_ret)
        out["tfi"] = signed_vol.rolling(50, min_periods=1).sum() / (
            vol.rolling(50, min_periods=1).sum().replace(0, np.nan)
        )
    else:
        # ── OHLCV proxies ────────────────────────────────────────────────
        # OBI proxy: (close - low) / (high - low) maps buyer pressure to [-1, 1]
        hl_range = (high - low).replace(0, np.nan)
        out["obi"] = ((close - low) / hl_range * 2 - 1).fillna(0.0)

        # Micro-price proxy: close vs. (high+low)/2 (similar to Williams %R normalised)
        mid_ohlc = (high + low) / 2.0
        out["micro_price"] = ((close - mid_ohlc) / hl_range).fillna(0.0)

        # Spread proxy: ATR / close (proportional to true spread)
        out["spread"] = out["atr_norm"]

        # TFI proxy: rolling signed volume using tick rule
        signed_vol = vol * np.sign(log_ret)
        out["tfi"] = signed_vol.rolling(50, min_periods=1).sum() / (
            vol.rolling(50, min_periods=1).sum().replace(0, np.nan)
        )

    # ── Z-score all features (rolling window=100) ─────────────────────────
    raw_feature_cols = [
        "log_ret", "log_ret_sq",
        "rsi_14", "macd_hist", "bb_width", "atr_norm",
        "vol_ratio", "mom_3", "mom_10", "realised_vol",
        "obi", "micro_price", "spread", "tfi",
    ]
    for col in raw_feature_cols:
        if col in out.columns:
            out[f"z_{col}"] = _rolling_zscore(out[col], window=100)

    out.dropna(subset=[f"z_{c}" for c in raw_feature_cols if c in out.columns], inplace=True)
    return out


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper that calls compute_microstructure_features().

    Kept for backward compatibility with existing trainer.py / backtester.py
    call sites.
    """
    return compute_microstructure_features(df)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_dataset(
    data: dict[str, pd.DataFrame],
    market: str,
    timescale: str,
) -> Path:
    path = DATA_DIR / f"{market}_{timescale}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved dataset → %s (%d tickers)", path, len(data))
    return path


def load_dataset(
    market: str,
    timescale: str,
    ticker: Optional[str] = None,
    tickers: Optional[list] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load a previously saved market dataset.

    Supports two storage formats (newest takes priority):
    1. Per-ticker parquet: ``{TICKER}_{market}_{timescale}.parquet``  ← new format
    2. Monolithic pickle:  ``{market}_{timescale}.pkl``               ← legacy format

    Parameters
    ----------
    market:    'us' or 'hk' (case-insensitive)
    timescale: '10s', '1m', '5m', '1h'
    ticker:    Single ticker filter (legacy kwarg, prefer ``tickers``)
    tickers:   List of tickers to load. If None, loads all available.
    start/end: Ignored here (used by download functions); accepted for
               API compatibility.
    """
    market = market.lower()
    filter_tickers: Optional[list] = None
    if ticker is not None:
        filter_tickers = [ticker]
    elif tickers:
        filter_tickers = list(tickers)

    # ── 1. Try per-ticker parquet files (new format) ──────────────────
    parquet_files = list(DATA_DIR.glob(f"*_{market}_{timescale}.parquet"))
    if parquet_files:
        data: dict[str, pd.DataFrame] = {}
        for f in parquet_files:
            tkr = f.stem.split("_")[0].upper()
            if filter_tickers and tkr not in [t.upper() for t in filter_tickers]:
                continue
            try:
                data[tkr] = pd.read_parquet(str(f))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", f.name, exc)
        if data:
            logger.info(
                "Loaded %d ticker(s) from parquet (%s/%s): %s",
                len(data), market, timescale, sorted(data.keys()),
            )
            return data

    # ── 2. Fall back to legacy monolithic pickle ─────────────────────
    pkl_path = DATA_DIR / f"{market}_{timescale}.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if filter_tickers:
            data = {k: v for k, v in data.items() if k.upper() in [t.upper() for t in filter_tickers]}
        logger.info("Loaded dataset from pickle %s (%d tickers)", pkl_path, len(data))
        return data

    raise FileNotFoundError(
        f"No dataset found for market='{market}' timescale='{timescale}'. "
        f"Expected parquet files matching '*_{market}_{timescale}.parquet' "
        f"or pickle '{market}_{timescale}.pkl' in {DATA_DIR}. "
        "Download data first via the Data Manager."
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="HFT Data Pipeline v2")
    parser.add_argument("--market", choices=["us", "hk", "both"], default="both")
    parser.add_argument("--timescale", choices=["10s", "1m", "5m", "1h", "all"], default="all")
    parser.add_argument("--period", default="7d", help="e.g. 7d (1m), 60d (5m), 365d (1h)")
    parser.add_argument("--ticker", default=None, help="Single ticker override")
    args = parser.parse_args()

    markets = ["us", "hk"] if args.market == "both" else [args.market]
    timescales = SUPPORTED_TIMESCALES if args.timescale == "all" else [args.timescale]

    fetch_intervals: set[str] = set()
    for ts in timescales:
        fetch_intervals.add("1m" if ts == "10s" else ts)

    for market in markets:
        download_fn = download_us_data if market == "us" else download_hk_data
        tickers = (US_TICKERS if market == "us" else HK_TICKERS) if not args.ticker else [args.ticker]

        raw_cache: dict[str, dict[str, pd.DataFrame]] = {}
        for interval in fetch_intervals:
            logger.info("Downloading %s @ %s (period=%s)", market.upper(), interval, args.period)
            raw_cache[interval] = download_fn(tickers=tickers, period=args.period, interval=interval)

        for timescale in timescales:
            source_interval = "1m" if timescale == "10s" else timescale
            source_data = raw_cache.get(source_interval, {})
            processed: dict[str, pd.DataFrame] = {}
            for tkr, df in tqdm(source_data.items(), desc=f"{market}@{timescale}"):
                try:
                    resampled = aggregate_to_timescale(df, timescale)
                    featured  = compute_features(resampled)
                    processed[tkr] = featured
                    logger.info("%s@%s: %d rows, %d features", tkr, timescale,
                                len(featured), len([c for c in featured.columns if c.startswith("z_")]))
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to process %s@%s: %s", tkr, timescale, exc)

            if processed:
                save_dataset(processed, market, timescale)
            else:
                logger.warning("No data processed for %s@%s — skipping save.", market, timescale)

    logger.info("Data pipeline complete.")


if __name__ == "__main__":
    main()
