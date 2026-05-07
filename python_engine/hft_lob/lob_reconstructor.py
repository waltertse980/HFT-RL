"""
lob_reconstructor.py
====================
Phase 2 — Reconstructs limit-order-book snapshots from Databento MBO files
and emits a compact 1-second feature parquet for downstream ML / RL.

Public API:
    reconstruct_and_save(dbn_path: str, symbol: str) -> str
        Returns the path to the parquet file written under
        `data_lob/features/`.

Output schema (per 1-second snapshot):
    timestamp           : pd.Timestamp (UTC)
    symbol              : str
    mid_price           : float
    spread              : float
    bid_px_1..5         : float (level-1..5 bid prices)
    bid_sz_1..5         : float (level-1..5 bid sizes)
    ask_px_1..5         : float (level-1..5 ask prices)
    ask_sz_1..5         : float (level-1..5 ask sizes)
    imbalance_l1        : float ((bid_sz_1 - ask_sz_1) / (bid_sz_1 + ask_sz_1))
    imbalance_l5        : float (same but summed across 5 levels)
    micro_price         : float (size-weighted mid)
    trades_count_1s     : int   (number of trades in the second)
    volume_1s           : float (total traded volume in the second)
    signed_vol_1s       : float (buy_vol − sell_vol)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
FEAT_DIR = BASE_DIR / "data_lob" / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

LEVELS = 5


class _LOBState:
    """Maintains a per-symbol price-keyed bid / ask book."""

    __slots__ = ("bids", "asks")

    def __init__(self) -> None:
        # price -> remaining size
        self.bids: dict[float, float] = defaultdict(float)
        self.asks: dict[float, float] = defaultdict(float)

    def apply(self, side: str, action: str, price: float, size: float) -> None:
        book = self.bids if side == "B" else self.asks
        if action in ("A", "M"):  # Add / Modify
            book[price] = book.get(price, 0.0) + size
        elif action in ("C", "F"):  # Cancel / Fill
            new = book.get(price, 0.0) - size
            if new <= 0:
                book.pop(price, None)
            else:
                book[price] = new
        elif action == "T":  # Trade — does not modify resting book
            pass

    def top_levels(self) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        bids_sorted = sorted(self.bids.items(), key=lambda x: -x[0])[:LEVELS]
        asks_sorted = sorted(self.asks.items(), key=lambda x: x[0])[:LEVELS]
        return bids_sorted, asks_sorted


def _empty_snapshot(ts: pd.Timestamp, symbol: str) -> dict:
    row: dict = {"timestamp": ts, "symbol": symbol}
    for i in range(1, LEVELS + 1):
        row[f"bid_px_{i}"] = np.nan
        row[f"bid_sz_{i}"] = 0.0
        row[f"ask_px_{i}"] = np.nan
        row[f"ask_sz_{i}"] = 0.0
    row.update(
        mid_price=np.nan, spread=np.nan,
        imbalance_l1=0.0, imbalance_l5=0.0, micro_price=np.nan,
        trades_count_1s=0, volume_1s=0.0, signed_vol_1s=0.0,
    )
    return row


def _snapshot(state: _LOBState, ts: pd.Timestamp, symbol: str,
              trades_count: int, volume: float, signed_vol: float) -> dict:
    bids, asks = state.top_levels()
    row = _empty_snapshot(ts, symbol)
    for i, (px, sz) in enumerate(bids, start=1):
        row[f"bid_px_{i}"] = px
        row[f"bid_sz_{i}"] = sz
    for i, (px, sz) in enumerate(asks, start=1):
        row[f"ask_px_{i}"] = px
        row[f"ask_sz_{i}"] = sz

    bp1 = row["bid_px_1"]; ap1 = row["ask_px_1"]
    bs1 = row["bid_sz_1"]; as1 = row["ask_sz_1"]
    if not np.isnan(bp1) and not np.isnan(ap1):
        row["mid_price"] = 0.5 * (bp1 + ap1)
        row["spread"] = ap1 - bp1
        denom = bs1 + as1
        if denom > 0:
            row["imbalance_l1"] = (bs1 - as1) / denom
            row["micro_price"] = (bp1 * as1 + ap1 * bs1) / denom

    bid_total = sum(row[f"bid_sz_{i}"] for i in range(1, LEVELS + 1))
    ask_total = sum(row[f"ask_sz_{i}"] for i in range(1, LEVELS + 1))
    if (bid_total + ask_total) > 0:
        row["imbalance_l5"] = (bid_total - ask_total) / (bid_total + ask_total)

    row["trades_count_1s"] = trades_count
    row["volume_1s"] = volume
    row["signed_vol_1s"] = signed_vol
    return row


def _iter_mbo(dbn_path: str):
    """Yield (timestamp, side, action, price, size, is_trade, aggressor_side) tuples."""
    try:
        import databento as db
    except ImportError as exc:
        raise RuntimeError("databento package not installed.") from exc

    store = db.DBNStore.from_file(dbn_path)
    df = store.to_df()
    # Databento MBO columns: ts_event, side ('B'/'A'/'N'), action ('A'/'M'/'C'/'F'/'T'),
    # price (already scaled to float), size
    if df.empty:
        return
    # Normalize index
    if "ts_event" in df.columns:
        ts_col = pd.to_datetime(df["ts_event"], utc=True)
    else:
        ts_col = df.index
        ts_col = pd.to_datetime(ts_col, utc=True)

    for ts, side, action, price, size in zip(
        ts_col,
        df["side"].astype(str).values,
        df["action"].astype(str).values,
        df["price"].astype(float).values,
        df["size"].astype(float).values,
    ):
        is_trade = (action == "T")
        yield ts, side, action, float(price), float(size), is_trade


def reconstruct_and_save(dbn_path: str, symbol: str) -> str:
    """
    Replay an MBO file and emit a 1-second snapshot parquet.

    Parameters
    ----------
    dbn_path : str
        Path to .dbn.zst file written by databento_pipeline.download_mbo.
    symbol : str
        Ticker symbol associated with the file.

    Returns
    -------
    str
        Path to parquet written under data_lob/features/.
    """
    state = _LOBState()
    rows: list[dict] = []
    cur_second: Optional[pd.Timestamp] = None
    trades_count = 0
    volume = 0.0
    signed_vol = 0.0

    for ts, side, action, price, size, is_trade in _iter_mbo(dbn_path):
        ts_sec = ts.floor("S")
        if cur_second is None:
            cur_second = ts_sec
        elif ts_sec != cur_second:
            rows.append(_snapshot(state, cur_second, symbol,
                                  trades_count, volume, signed_vol))
            # Fill any gap seconds with a forward-filled snapshot
            gap = int((ts_sec - cur_second).total_seconds()) - 1
            for k in range(1, gap + 1):
                rows.append(_snapshot(state, cur_second + pd.Timedelta(seconds=k),
                                      symbol, 0, 0.0, 0.0))
            cur_second = ts_sec
            trades_count = 0
            volume = 0.0
            signed_vol = 0.0

        if is_trade:
            trades_count += 1
            volume += size
            # In MBO trade messages, side is the resting (passive) side.
            # Aggressor is the opposite, so signed_vol uses the inverse sign.
            if side == "A":
                signed_vol += size  # buyer aggressed
            elif side == "B":
                signed_vol -= size  # seller aggressed
        else:
            state.apply(side, action, price, size)

    if cur_second is not None:
        rows.append(_snapshot(state, cur_second, symbol,
                              trades_count, volume, signed_vol))

    if not rows:
        logger.warning("[reconstruct] no rows produced for %s", dbn_path)

    df = pd.DataFrame(rows)
    out_path = FEAT_DIR / f"{symbol}_{Path(dbn_path).stem}.parquet"
    df.to_parquet(str(out_path), index=False)
    logger.info("[reconstruct] wrote %s (%d rows)", out_path.name, len(df))
    return str(out_path)
