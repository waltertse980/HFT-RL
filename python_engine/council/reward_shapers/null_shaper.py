"""
null_shaper.py — Model A's shaper: no shaping bonus, pure RL.
"""
from __future__ import annotations

import numpy as np


class NullShaper:
    """No-op reward shaper used by Model A."""

    def set_prev_obs(self, obs: np.ndarray | None) -> None:  # pragma: no cover
        return None

    def compute_bonus(
        self,
        obs: np.ndarray,
        action: int,
        step: int,
        alpha: float,
    ) -> float:
        return 0.0
