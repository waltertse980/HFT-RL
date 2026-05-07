"""
lob_environment.py
==================
Phase 4 — Gymnasium environment for LOB-HFT v2 PPO training.

Observation
-----------
np.float32 vector of shape (len(FEATURE_COLS) + 1,) where the trailing
element is the agent's current position {-1, 0, +1}.

Action space (Discrete(3))
--------------------------
0 -> FLAT (target = 0)
1 -> LONG (target = +1)
2 -> SHORT (target = -1)

Reward
------
Per-step PnL = position * mid-price log-return − fee × |Δposition|

Where `fee` is a relative cost (e.g. 0.0001) charged on every position change.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

logger = logging.getLogger(__name__)

try:
    from lob_features import add_rolling_features, FEATURE_COLS
except ImportError:
    from hft_lob.lob_features import add_rolling_features, FEATURE_COLS


class LOBTradingEnv(gym.Env):
    """Single-symbol LOB environment for PPO."""

    metadata = {"render_modes": []}
    spec = None  # silence SB3 Monitor warning when no spec is registered

    def __init__(
        self,
        df: pd.DataFrame,
        fee: float = 0.0001,
        max_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        if df.empty:
            raise ValueError("LOBTradingEnv: empty dataframe")

        # Ensure all feature columns are present
        if "ret_1s" not in df.columns:
            df = add_rolling_features(df)

        # Keep only what we need + label-like target absent (used in unit tests)
        self.df = df.reset_index(drop=True).copy()
        for c in FEATURE_COLS:
            if c not in self.df.columns:
                self.df[c] = 0.0
        if "mid_price" not in self.df.columns:
            self.df["mid_price"] = 1.0

        self.fee = float(fee)
        self.n_steps = len(self.df) - 1
        self.max_steps = int(max_steps) if max_steps else self.n_steps

        # Spaces
        n_feat = len(FEATURE_COLS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_feat + 1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # State
        self.t: int = 0
        self.position: int = 0
        self.entry_price: float = 0.0
        self.cumulative_pnl: float = 0.0
        self.trade_count: int = 0

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        row = self.df.iloc[self.t]
        feats = row[FEATURE_COLS].to_numpy(dtype=np.float32)
        return np.append(feats, np.float32(self.position)).astype(np.float32)

    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.t = 0
        self.position = 0
        self.entry_price = 0.0
        self.cumulative_pnl = 0.0
        self.trade_count = 0
        return self._obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        target_map = {0: 0, 1: 1, 2: -1}
        target_position = target_map[int(action)]

        # Mid-price log-return between t and t+1
        p_now = float(self.df.iloc[self.t]["mid_price"])
        p_next = float(self.df.iloc[self.t + 1]["mid_price"])
        if p_now > 0 and p_next > 0:
            log_ret = float(np.log(p_next / p_now))
        else:
            log_ret = 0.0

        pnl = self.position * log_ret

        # Trading cost on position changes
        delta = abs(target_position - self.position)
        cost = self.fee * delta
        reward = pnl - cost
        self.cumulative_pnl += reward
        if delta > 0:
            self.trade_count += 1
            self.entry_price = p_next

        # Advance state
        self.position = target_position
        self.t += 1

        terminated = bool(self.t >= self.n_steps)
        truncated = bool(self.t >= self.max_steps)

        info: dict[str, Any] = {
            "pnl": pnl,
            "cost": cost,
            "position": self.position,
            "cumulative_pnl": self.cumulative_pnl,
            "trade_count": self.trade_count,
            "mid_price": p_next,
        }
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self) -> None:  # pragma: no cover
        pass

    def close(self) -> None:  # pragma: no cover
        pass
