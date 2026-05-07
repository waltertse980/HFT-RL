"""
databento_pipeline.py
=====================
Phase 2 — Databento MBO (Level 3) data downloader.

Fetches MBO order-book data via the Databento Python SDK and saves it as
DBN-zstd files in `data_lob/raw/`, one per (symbol, day).

Public API:
    download_mbo(symbols: list[str], start: str, end: str) -> list[str]

Returns the list of downloaded filepaths.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data_lob" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Default Databento dataset / schema for US equities NMS feed
DEFAULT_DATASET = "XNAS.ITCH"
DEFAULT_SCHEMA = "mbo"


def _daterange(start: str, end: str):
    """Yield each calendar date from start to end (inclusive)."""
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)


def download_mbo(
    symbols: list[str],
    start: str,
    end: str,
    dataset: str = DEFAULT_DATASET,
    schema: str = DEFAULT_SCHEMA,
) -> list[str]:
    """
    Download MBO data from Databento for each (symbol, day) and save to disk.

    Parameters
    ----------
    symbols : list[str]
        e.g. ["NVDA", "AAPL", "TSM", "META"]
    start, end : str
        ISO-format dates, e.g. "2026-04-28", "2026-05-02".
    dataset : str
        Databento dataset code. Default "XNAS.ITCH".
    schema : str
        Databento schema. Default "mbo".

    Returns
    -------
    list[str]
        Paths of all DBN-zstd files written.
    """
    api_key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "DATABENTO_API_KEY not set. POST /settings/databento first."
        )

    try:
        import databento as db
    except ImportError as exc:
        raise RuntimeError(
            "databento package not installed. pip install databento"
        ) from exc

    client = db.Historical(api_key)
    written: list[str] = []

    for sym in symbols:
        for day in _daterange(start, end):
            day_str = day.isoformat()
            out_path = RAW_DIR / f"{sym}_{day_str}.dbn.zst"
            if out_path.exists():
                logger.info("[databento] skip (exists): %s", out_path.name)
                written.append(str(out_path))
                continue

            t0 = f"{day_str}T13:30:00"  # 09:30 ET in UTC (DST-naive default)
            t1 = f"{day_str}T20:00:00"  # 16:00 ET in UTC
            logger.info("[databento] fetch %s %s..%s", sym, t0, t1)
            try:
                data = client.timeseries.get_range(
                    dataset=dataset,
                    symbols=[sym],
                    schema=schema,
                    start=t0,
                    end=t1,
                )
                data.to_file(str(out_path))
                written.append(str(out_path))
                logger.info("[databento] wrote %s (%d bytes)",
                            out_path.name, out_path.stat().st_size)
            except Exception as exc:
                logger.error("[databento] failed %s %s: %s", sym, day_str, exc)

    return written
