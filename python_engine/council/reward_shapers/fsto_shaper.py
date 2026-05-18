"""
fsto_shaper.py — Model B's reward shaper: Fast Stochastic Oscillator crossover.

Bonus rules (scaled by alpha):
    • %K crosses above %D  (bullish cross) + action == LONG  → +alpha
    • %K crosses below %D  (bearish cross) + action == SHORT → +alpha
    • No crossover         + action == FLAT                 → +alpha * 0.3
    • otherwise                                              → 0.0
"""
from __future__ import annotations

import numpy as np


class FSTOShaper:
    """Stochastic-oscillator crossover shaping bonus."""

    def __init__(self, stoch_k_idx: int, stoch_d_idx: int) -> None:
        self.k_idx = int(stoch_k_idx)
        self.d_idx = int(stoch_d_idx)
        self._prev_obs: np.ndarray | None = None

    def set_prev_obs(self, obs: np.ndarray | None) -> None:
        self._prev_obs = obs

    def compute_bonus(
        self,
        obs: np.ndarray,
        action: int,
        step: int,
        alpha: float,
    ) -> float:
        if self._prev_obs is None:
            return 0.0
        try:
            pk = float(self._prev_obs[self.k_idx])
            pd_ = float(self._prev_obs[self.d_idx])
            ck = float(obs[self.k_idx])
            cd = float(obs[self.d_idx])
        except (IndexError, TypeError, ValueError):
            return 0.0

        cross_up = pk < pd_ and ck >= cd
        cross_down = pk > pd_ and ck <= cd
        a = int(action)

        if cross_up and a == 1:
            return float(alpha)
        if cross_down and a == 2:
            return float(alpha)
        if (not cross_up) and (not cross_down) and a == 0:
            return float(alpha) * 0.3
        return 0.0
