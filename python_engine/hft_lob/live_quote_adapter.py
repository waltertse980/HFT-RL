"""
live_quote_adapter.py
=====================
Phase 7 — Bridge between Alpaca's live NBBO quote stream and the LOB
feature vector the model was trained on.

Because Alpaca's free / paper feed only exposes NBBO (top-of-book), we
synthesise an approximate LOB observation:

  - bid_px_1, ask_px_1, bid_sz_1, ask_sz_1 come from the quote
  - bid_px_2..5 / ask_px_2..5 are filled with 0 (size = 0)
  - rolling features are maintained on a per-symbol deque

This is a deliberate simplification: the v2 PPO model relies primarily
on top-of-book microstructure + rolling stats, all of which can be
reconstructed from NBBO alone (with reduced fidelity).

Public API
----------
LiveLOBFeatureBuilder(symbols).on_quote(symbol, quote) -> np.ndarray | None
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Deque, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from lob_features import FEATURE_COLS
except ImportError:
    from hft_lob.lob_features import FEATURE_COLS


WARMUP_STEPS = 30  # need ~30 quotes per symbol before we can emit features


class LiveLOBFeatureBuilder:
    """Builds an LOB-style feature vector from successive NBBO quotes."""

    def __init__(self, symbols: list[str], history_len: int = 240) -> None:
        self.symbols = list(symbols)
        self.history_len = history_len
        # Per-symbol rolling state
        self._mid_hist: dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=history_len)
        )
        self._imb_hist: dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=history_len)
        )
        self._signed_vol_hist: dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=history_len)
        )
        self._last_bid: dict[str, float] = {}
        self._last_ask: dict[str, float] = {}
        self._tick_count: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    @staticmethod
    def _quote_attr(quote, *names, default=0.0) -> float:
        """alpaca-py Quote may use attribute or dict; tolerate both."""
        for n in names:
            if hasattr(quote, n):
                v = getattr(quote, n)
                if v is not None:
                    return float(v)
            if isinstance(quote, dict) and n in quote and quote[n] is not None:
                return float(quote[n])
        return float(default)

    # ------------------------------------------------------------------
    def on_quote(self, symbol: str, quote) -> Optional[np.ndarray]:
        bid_px = self._quote_attr(quote, "bid_price", "bp", "bidPrice")
        ask_px = self._quote_attr(quote, "ask_price", "ap", "askPrice")
        bid_sz = self._quote_attr(quote, "bid_size", "bs", "bidSize")
        ask_sz = self._quote_attr(quote, "ask_size", "as", "askSize")

        if bid_px <= 0 or ask_px <= 0 or ask_px < bid_px:
            return None

        mid = 0.5 * (bid_px + ask_px)
        spread = ask_px - bid_px
        denom_l1 = bid_sz + ask_sz
        imb_l1 = (bid_sz - ask_sz) / denom_l1 if denom_l1 > 0 else 0.0
        micro = (
            (bid_px * ask_sz + ask_px * bid_sz) / denom_l1
            if denom_l1 > 0 else mid
        )
        micro_dev = (micro - mid) / mid if mid > 0 else 0.0

        # Heuristic signed volume from inside-quote movement
        prev_bid = self._last_bid.get(symbol, bid_px)
        prev_ask = self._last_ask.get(symbol, ask_px)
        signed_vol_1s = 0.0
        if ask_px < prev_ask:
            signed_vol_1s = -ask_sz   # sellers pressing, ask went down
        elif bid_px > prev_bid:
            signed_vol_1s = +bid_sz   # buyers pressing, bid went up
        self._last_bid[symbol] = bid_px
        self._last_ask[symbol] = ask_px

        # Update history
        self._mid_hist[symbol].append(mid)
        self._imb_hist[symbol].append(imb_l1)
        self._signed_vol_hist[symbol].append(signed_vol_1s)
        self._tick_count[symbol] += 1

        if self._tick_count[symbol] < WARMUP_STEPS:
            return None

        mid_arr = np.asarray(self._mid_hist[symbol], dtype=np.float64)
        imb_arr = np.asarray(self._imb_hist[symbol], dtype=np.float64)
        sv_arr = np.asarray(self._signed_vol_hist[symbol], dtype=np.float64)

        log_mid = np.log(np.where(mid_arr > 0, mid_arr, 1.0))
        ret_1s = float(log_mid[-1] - log_mid[-2]) if len(log_mid) >= 2 else 0.0
        ret_5s = float(log_mid[-1] - log_mid[-6]) if len(log_mid) >= 6 else 0.0
        ret_30s = float(log_mid[-1] - log_mid[-31]) if len(log_mid) >= 31 else 0.0

        diffs = np.diff(log_mid)
        vol_5s  = float(diffs[-5:].std()) if len(diffs) >= 5 else 0.0
        vol_30s = float(diffs[-30:].std()) if len(diffs) >= 30 else 0.0

        # EMA imbalance
        def _ema(a: np.ndarray, span: int) -> float:
            alpha = 2 / (span + 1)
            v = a[0]
            for x in a[1:]:
                v = alpha * x + (1 - alpha) * v
            return float(v)

        imb_ema_5s  = _ema(imb_arr[-min(len(imb_arr), 5):], 5)  if len(imb_arr) else 0.0
        imb_ema_30s = _ema(imb_arr[-min(len(imb_arr), 30):], 30) if len(imb_arr) else 0.0

        sv_5s  = float(sv_arr[-5:].sum())  if len(sv_arr) else 0.0
        sv_30s = float(sv_arr[-30:].sum()) if len(sv_arr) else 0.0

        # Build a dict mirroring FEATURE_COLS exactly.
        feat = {
            "spread":            spread,
            "imbalance_l1":      imb_l1,
            "imbalance_l5":      imb_l1,           # NBBO ≈ L1 only
            "micro_price_dev":   micro_dev,
            "bid_sz_1": bid_sz, "bid_sz_2": 0.0, "bid_sz_3": 0.0,
            "bid_sz_4": 0.0, "bid_sz_5": 0.0,
            "ask_sz_1": ask_sz, "ask_sz_2": 0.0, "ask_sz_3": 0.0,
            "ask_sz_4": 0.0, "ask_sz_5": 0.0,
            "trades_count_1s":   0.0,             # not available from NBBO
            "volume_1s":         0.0,
            "signed_vol_1s":     signed_vol_1s,
            "ret_1s":  ret_1s,  "ret_5s":  ret_5s,  "ret_30s": ret_30s,
            "vol_5s":  vol_5s,  "vol_30s": vol_30s,
            "imb_ema_5s":  imb_ema_5s,  "imb_ema_30s": imb_ema_30s,
            "signed_vol_5s":  sv_5s,    "signed_vol_30s": sv_30s,
        }
        try:
            obs = np.array([feat[c] for c in FEATURE_COLS], dtype=np.float32)
        except KeyError as exc:
            logger.error("[live-features] missing feature %s", exc)
            return None
        return obs
