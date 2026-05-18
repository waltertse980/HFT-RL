"""
data_downloader.py — Alpaca IEX bar downloader for the 5-Model Council.

Downloads 1-minute and 5-minute bars for the configured tickers and date
range, then resamples 1-minute bars into 10-minute bars for multi-timeframe
context. Saves parquet files to ``data/bars/{SYMBOL}_{timeframe}.parquet``.

Usage (CLI):
    python data_downloader.py --symbols TSLA NVDA AAPL \
        --start 2026-01-01 --end 2026-05-18

Environment:
    ALPACA_API_KEY     — Alpaca paper/live API key
    ALPACA_API_SECRET  — Alpaca paper/live secret
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    _ALPACA_AVAILABLE = True
except ImportError:  # pragma: no cover — defensive
    _ALPACA_AVAILABLE = False
    StockHistoricalDataClient = None  # type: ignore
    StockBarsRequest = None  # type: ignore
    TimeFrame = None  # type: ignore
    TimeFrameUnit = None  # type: ignore
    DataFeed = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DEFAULT_BARS_DIR = BASE_DIR / "data" / "bars"

# Approximate trading-day-bar counts for sanity checks (RTH only)
_BARS_PER_DAY = {"1m": 390, "5m": 78, "10m": 39}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_client(api_key: str | None, api_secret: str | None) -> "StockHistoricalDataClient":
    if not _ALPACA_AVAILABLE:
        raise RuntimeError(
            "alpaca-py is not installed. Run `pip install alpaca-py` and retry."
        )
    api_key = api_key or os.environ.get("ALPACA_API_KEY")
    api_secret = api_secret or os.environ.get("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError(
            "Alpaca credentials not found. Set ALPACA_API_KEY and "
            "ALPACA_API_SECRET environment variables (or pass to the function)."
        )
    return StockHistoricalDataClient(api_key, api_secret)


def _timeframe_for(tf: str):
    """Map a string timeframe ('1m', '5m') to an Alpaca TimeFrame object."""
    if tf == "1m":
        return TimeFrame.Minute
    if tf == "5m":
        return TimeFrame(5, TimeFrameUnit.Minute)
    if tf == "15m":
        return TimeFrame(15, TimeFrameUnit.Minute)
    if tf == "1h":
        return TimeFrame.Hour
    raise ValueError(f"Unsupported Alpaca timeframe: {tf!r}")


def _bars_to_df(bars_obj, symbol: str) -> pd.DataFrame:
    """
    Convert an Alpaca BarSet response into a tidy DataFrame indexed on
    timezone-aware UTC timestamps named 'timestamp'.
    """
    if bars_obj is None:
        return pd.DataFrame()

    # Newer alpaca-py: BarSet.df is a MultiIndex (symbol, timestamp).
    if hasattr(bars_obj, "df") and bars_obj.df is not None:
        df = bars_obj.df
        if isinstance(df.index, pd.MultiIndex):
            try:
                df = df.xs(symbol, level=0)
            except KeyError:
                # Symbol missing from index; nothing to return
                return pd.DataFrame()
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp"
        return df

    # Fallback: dict-like {symbol: [Bar, ...]}
    rows = []
    if hasattr(bars_obj, "data") and isinstance(bars_obj.data, dict):
        bar_list = bars_obj.data.get(symbol, [])
        for b in bar_list:
            rows.append({
                "timestamp": pd.to_datetime(getattr(b, "timestamp", None), utc=True),
                "open":       float(getattr(b, "open", 0.0)),
                "high":       float(getattr(b, "high", 0.0)),
                "low":        float(getattr(b, "low", 0.0)),
                "close":      float(getattr(b, "close", 0.0)),
                "volume":     float(getattr(b, "volume", 0.0)),
                "vwap":       float(getattr(b, "vwap", 0.0) or 0.0),
                "trade_count": int(getattr(b, "trade_count", 0) or 0),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure standard columns exist; add zero-fill for vwap/trade_count if missing."""
    if df.empty:
        return df
    cols_lc = {c.lower(): c for c in df.columns}
    rename = {}
    for std in ("open", "high", "low", "close", "volume", "vwap", "trade_count"):
        if std in df.columns:
            continue
        if std in cols_lc:
            rename[cols_lc[std]] = std
    if rename:
        df = df.rename(columns=rename)
    if "vwap" not in df.columns:
        df["vwap"] = df["close"].astype("float64")
    if "trade_count" not in df.columns:
        df["trade_count"] = 0
    # Order
    keep = [c for c in ("open", "high", "low", "close", "volume", "vwap", "trade_count") if c in df.columns]
    return df[keep].copy()


def _download_one(
    client: "StockHistoricalDataClient",
    symbol: str,
    start: datetime,
    end: datetime,
    tf_str: str,
) -> pd.DataFrame:
    """
    Download a single (symbol, timeframe) combo from Alpaca IEX feed.
    Alpaca-py handles pagination internally when get_stock_bars is called
    with a date range.
    """
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=_timeframe_for(tf_str),
        start=start,
        end=end,
        feed=DataFeed.IEX,
        adjustment="raw",
    )
    bars = client.get_stock_bars(request)
    df = _bars_to_df(bars, symbol)
    df = _ensure_columns(df)
    return df


def _resample_to_10m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample a 1-minute DataFrame into 10-minute bars (no look-ahead)."""
    if df_1m.empty:
        return df_1m
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "vwap" in df_1m.columns:
        agg["vwap"] = "mean"
    if "trade_count" in df_1m.columns:
        agg["trade_count"] = "sum"
    out = df_1m.resample("10min", closed="left", label="left").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"]).copy()
    out.index.name = "timestamp"
    return out


def _validate_row_count(df: pd.DataFrame, symbol: str, tf: str, start: datetime, end: datetime) -> None:
    if df.empty:
        log.warning("No bars for %s %s in %s → %s", symbol, tf, start.date(), end.date())
        return
    n_days = max(1, (end - start).days)
    # 5/7 weekday ratio, ~252 trading days / 365 calendar days ≈ 0.69
    trading_days = max(1, int(n_days * 252 / 365))
    expected = trading_days * _BARS_PER_DAY.get(tf, 0)
    if expected <= 0:
        return
    pct = len(df) / expected * 100.0
    msg = (f"[{symbol} {tf}] rows={len(df):,}  expected≈{expected:,}  "
           f"({pct:.1f}% of expected)")
    if pct < 80.0:
        log.warning("%s — below 80%% threshold", msg)
    else:
        log.info("%s", msg)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def download_bars(
    symbols: Iterable[str],
    start: str | datetime,
    end: str | datetime,
    timeframes: Iterable[str] = ("1m", "5m", "10m"),
    api_key: str | None = None,
    api_secret: str | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Download bars for ``symbols`` over [start, end] across the requested
    ``timeframes``. The 10-minute timeframe is resampled from 1-minute data
    if requested.

    Returns
    -------
    {symbol: {timeframe: DataFrame}}
    """
    client = _get_client(api_key, api_secret)
    start_dt = pd.Timestamp(start).to_pydatetime() if not isinstance(start, datetime) else start
    end_dt = pd.Timestamp(end).to_pydatetime() if not isinstance(end, datetime) else end

    out: dict[str, dict[str, pd.DataFrame]] = {}
    tf_list = list(timeframes)
    needs_1m_for_10m = ("10m" in tf_list) and ("1m" not in tf_list)

    for sym in symbols:
        out[sym] = {}
        # 1m (always download if requested or needed for 10m resampling)
        df_1m: pd.DataFrame | None = None
        if "1m" in tf_list or needs_1m_for_10m:
            log.info("Downloading %s 1m bars [%s → %s]", sym, start_dt.date(), end_dt.date())
            df_1m = _download_one(client, sym, start_dt, end_dt, "1m")
            if "1m" in tf_list:
                _validate_row_count(df_1m, sym, "1m", start_dt, end_dt)
                out[sym]["1m"] = df_1m

        # 5m (native download)
        if "5m" in tf_list:
            log.info("Downloading %s 5m bars [%s → %s]", sym, start_dt.date(), end_dt.date())
            df_5m = _download_one(client, sym, start_dt, end_dt, "5m")
            _validate_row_count(df_5m, sym, "5m", start_dt, end_dt)
            out[sym]["5m"] = df_5m

        # 10m (resampled from 1m)
        if "10m" in tf_list:
            if df_1m is None or df_1m.empty:
                log.warning("Cannot build 10m bars for %s — 1m data missing/empty", sym)
                out[sym]["10m"] = pd.DataFrame()
            else:
                df_10m = _resample_to_10m(df_1m)
                _validate_row_count(df_10m, sym, "10m", start_dt, end_dt)
                out[sym]["10m"] = df_10m

    return out


def download_and_save(
    symbols: Iterable[str],
    start: str | datetime,
    end: str | datetime,
    timeframes: Iterable[str] = ("1m", "5m", "10m"),
    api_key: str | None = None,
    api_secret: str | None = None,
    bars_dir: str | Path = DEFAULT_BARS_DIR,
) -> list[Path]:
    """
    Download bars and persist each (symbol, timeframe) as a parquet file at
    ``{bars_dir}/{SYMBOL}_{timeframe}.parquet``. Returns the list of paths
    written.
    """
    bars_dir = Path(bars_dir)
    bars_dir.mkdir(parents=True, exist_ok=True)

    data = download_bars(symbols, start, end, timeframes, api_key, api_secret)
    written: list[Path] = []
    for sym, by_tf in data.items():
        for tf, df in by_tf.items():
            if df is None or df.empty:
                log.warning("Skipping save for %s %s — empty DataFrame", sym, tf)
                continue
            df = df.copy()
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "timestamp"
            out_path = bars_dir / f"{sym}_{tf}.parquet"
            df.to_parquet(out_path, engine="pyarrow", index=True)
            written.append(out_path)
            log.info("Saved %s (%s rows)", out_path, len(df))
    return written


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Alpaca IEX bars for council training")
    p.add_argument("--symbols", nargs="+", default=["TSLA", "NVDA", "AAPL"])
    p.add_argument("--start", default="2026-01-01")
    p.add_argument("--end", default="2026-05-18")
    p.add_argument("--timeframes", nargs="+", default=["1m", "5m", "10m"])
    p.add_argument("--bars-dir", default=str(DEFAULT_BARS_DIR))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = download_and_save(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        timeframes=args.timeframes,
        bars_dir=args.bars_dir,
    )
    print(f"\nWrote {len(paths)} parquet files to {args.bars_dir}")
    for p in paths:
        print(f"  • {p}")
    return 0 if paths else 1


if __name__ == "__main__":
    raise SystemExit(main())
