"""
candlestick_shaper.py — Model C's reward shaper: bar-level pattern bonuses.

The shaper inspects the most recent three observation rows (collected via
``set_prev_obs``) and rewards actions aligning with simple candlestick
patterns:

    • Morning-star proxy   → +alpha for LONG (action == 1)
        bar[-3] bearish (body_ratio>0 and close<open via lower_wick > upper_wick)
        bar[-2] small body (body_ratio < 0.3, bar_range > 0.001)
        bar[-1] bullish    (upper_wick > lower_wick)

    • Evening-star proxy   → +alpha for SHORT (action == 2)
        inverse of the above

    • Doji (current bar body_ratio < 0.1) → +alpha * 0.5 for FLAT (action == 0)

The shaper relies on four observation indices that pinpoint the
``body_ratio``, ``upper_wick``, ``lower_wick`` and ``bar_range`` features for
the *current* bar within the flattened observation vector.

Note: morning/evening star detection here uses lower_wick/upper_wick balance
as a coarse bull/bear proxy because the observation vector does not expose
raw open/close. This is a simplification — not a literal candlestick rule —
but it is consistent and produces useful shaping pressure during training.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class CandlestickShaper:
    """Three-bar candlestick-pattern reward shaper."""

    def __init__(
        self,
        body_ratio_idx: int,
        upper_wick_idx: int,
        lower_wick_idx: int,
        bar_range_idx: int,
    ) -> None:
        self.body_idx = int(body_ratio_idx)
        self.upper_idx = int(upper_wick_idx)
        self.lower_idx = int(lower_wick_idx)
        self.range_idx = int(bar_range_idx)
        # We keep the last 3 *previous* observations. Combined with the
        # current obs passed to compute_bonus, that gives 4 reference points
        # but we only need the most recent 3 (current + last 2) for the
        # 3-bar pattern, so we store size 3 to retain robustness.
        self._history: deque[np.ndarray] = deque(maxlen=3)

    def set_prev_obs(self, obs: np.ndarray | None) -> None:
        if obs is not None:
            self._history.append(np.asarray(obs))

    # ── Internal feature extractors ─────────────────────────────────────
    def _safe_get(self, obs: np.ndarray, idx: int) -> float:
        try:
            return float(obs[idx])
        except (IndexError, TypeError, ValueError):
            return 0.0

    def _is_bullish(self, obs: np.ndarray) -> bool:
        # Heuristic: lower wick larger than upper wick → buying pressure
        return self._safe_get(obs, self.lower_idx) > self._safe_get(obs, self.upper_idx)

    def _is_bearish(self, obs: np.ndarray) -> bool:
        return self._safe_get(obs, self.upper_idx) > self._safe_get(obs, self.lower_idx)

    def _is_small_body(self, obs: np.ndarray) -> bool:
        return (
            self._safe_get(obs, self.body_idx) < 0.3
            and self._safe_get(obs, self.range_idx) > 0.001
        )

    def _is_doji(self, obs: np.ndarray) -> bool:
        return self._safe_get(obs, self.body_idx) < 0.1

    # ── Public bonus computation ────────────────────────────────────────
    def compute_bonus(
        self,
        obs: np.ndarray,
        action: int,
        step: int,
        alpha: float,
    ) -> float:
        a = int(action)
        alpha = float(alpha)

        # Doji on the current bar → reward FLAT
        if self._is_doji(obs) and a == 0:
            return alpha * 0.5

        # 3-bar patterns require at least 2 historical obs + current obs
        if len(self._history) < 2:
            return 0.0

        bar_m3 = self._history[-2]  # 3rd-to-last (i.e., t-2 from current)
        bar_m2 = self._history[-1]  # last previous (i.e., t-1)
        bar_m1 = obs                # current bar

        # Morning star: bearish, small-body, bullish → reward LONG
        if (
            self._is_bearish(bar_m3)
            and self._is_small_body(bar_m2)
            and self._is_bullish(bar_m1)
            and a == 1
        ):
            return alpha

        # Evening star: bullish, small-body, bearish → reward SHORT
        if (
            self._is_bullish(bar_m3)
            and self._is_small_body(bar_m2)
            and self._is_bearish(bar_m1)
            and a == 2
        ):
            return alpha

        return 0.0
