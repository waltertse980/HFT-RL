"""
intraday_env.py — Gymnasium environment for council intraday bar trading.

Design
------
* One episode  = one trading day, sampled uniformly at random from the
                 days present in the supplied featured 1-minute DataFrame.
* Action space = Discrete(3)  → 0 FLAT, 1 LONG, 2 SHORT
* Observation  = flat float32 vector containing
      (a) `window_size` rows of 1m engineered features (most recent first row,
          oldest first  → standard chronological flattening)
      (b) latest engineered features for 5m (or zeros if df_5m=None)
      (c) latest engineered features for 10m (or zeros if df_10m=None)
      (d) 4-element portfolio state
              [position_encoded, unrealised_pnl_norm, cash_ratio, time_of_day]
* Reward       = pct portfolio-value change minus transaction-cost penalty,
                 clipped to [-1, 1]. An optional ``shaper`` may add a shaping
                 bonus controlled by a per-env alpha (set via ``set_shaping_alpha``).
* Termination  = day exhausted OR position == FLAT at end-of-day.
                 On the very last step of a day, any open position is forcibly
                 closed and the realised PnL is folded into the final reward.

The constructor selects feature columns by EXCLUDING raw OHLCV / vwap /
trade_count columns from the input DataFrame. All other columns are used as
features. The DataFrame's DatetimeIndex must be timezone-aware (UTC).

The environment is **single-threaded** and is designed to be wrapped in a
``DummyVecEnv`` — never ``SubprocVecEnv`` (Windows-stability requirement).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

log = logging.getLogger(__name__)

# Columns that are NOT considered features
_RAW_COLS = {"open", "high", "low", "close", "volume", "vwap", "trade_count"}

# Minutes in the RTH session — used to normalise time-of-day
_RTH_MINUTES = 390.0


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _RAW_COLS]


class IntradayEnv(gym.Env):
    """
    Gymnasium-compatible single-symbol intraday bar trading environment.
    """

    metadata = {"render_modes": []}

    # ── Construction ─────────────────────────────────────────────────────
    def __init__(
        self,
        featured_df_1m: pd.DataFrame,
        featured_df_5m: pd.DataFrame | None = None,
        featured_df_10m: pd.DataFrame | None = None,
        window_size: int = 60,
        transaction_cost: float = 0.0001,
        initial_capital: float = 100_000.0,
        shaper: Any | None = None,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()

        if featured_df_1m is None or featured_df_1m.empty:
            raise ValueError("featured_df_1m is required and cannot be empty")

        # ── Store base 1m frame (sorted, tz-aware UTC) ──────────────────
        df = featured_df_1m.copy().sort_index()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("featured_df_1m must have a DatetimeIndex")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        self._df_1m = df

        # Feature column list (engineered only, raw cols excluded)
        self._feature_cols = _feature_columns(df)
        if not self._feature_cols:
            raise ValueError("No engineered feature columns found in featured_df_1m")
        self._n_features_1m = len(self._feature_cols)

        # 5m / 10m latest-row features (engineered only)
        self._df_5m = self._prep_higher_tf(featured_df_5m, "5m")
        self._df_10m = self._prep_higher_tf(featured_df_10m, "10m")
        self._n_features_5m = len(self._df_5m_cols)
        self._n_features_10m = len(self._df_10m_cols)

        # Trading-day index → list of timestamps within each day
        ny_index = self._df_1m.index.tz_convert("America/New_York")
        day_keys = pd.DatetimeIndex(ny_index).date
        self._df_1m["_day_key"] = day_keys
        self._days: list = sorted({d for d in day_keys})
        # Map day → integer positions within self._df_1m
        self._day_to_pos: dict = {}
        positions = np.arange(len(self._df_1m))
        for d in self._days:
            mask = (day_keys == d)
            self._day_to_pos[d] = positions[mask]
        self._df_1m = self._df_1m.drop(columns=["_day_key"])

        # ── Trading parameters ─────────────────────────────────────────
        self.window_size = int(window_size)
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.transaction_cost = float(transaction_cost)
        self.initial_capital = float(initial_capital)

        # ── Reward shaper (optional) ───────────────────────────────────
        self.shaper = shaper
        self._alpha = 1.0  # multiplicative weight on shaping bonus

        # ── RNG ────────────────────────────────────────────────────────
        self._np_random: np.random.Generator
        self._seed(random_seed)

        # ── Episode state (initialised in reset) ───────────────────────
        self._day_date = None
        self._day_positions: np.ndarray = np.empty(0, dtype=np.int64)
        self._step_in_day: int = 0
        self.position: int = 0       # -1 short, 0 flat, +1 long
        self.shares: float = 0.0
        self.entry_price: float = 0.0
        self.cash: float = self.initial_capital
        self.portfolio_value: float = self.initial_capital
        self._prev_portfolio_value: float = self.initial_capital
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.n_trades: int = 0
        self._prev_obs: np.ndarray | None = None

        # ── Spaces ─────────────────────────────────────────────────────
        obs_dim = (
            self.window_size * self._n_features_1m
            + self._n_features_5m
            + self._n_features_10m
            + 4  # portfolio state
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # Cached numpy arrays for fast indexing
        self._arr_1m = self._df_1m[self._feature_cols].to_numpy(dtype=np.float32, copy=True)
        self._close_1m = self._df_1m["close"].to_numpy(dtype=np.float64, copy=True)
        self._ts_1m_ns = self._df_1m.index.asi8  # int64 nanoseconds

        # 5m / 10m latest-bar matrices (aligned to a fast asof-search)
        self._arr_5m, self._ts_5m_ns = self._materialise_higher_tf(self._df_5m, self._df_5m_cols)
        self._arr_10m, self._ts_10m_ns = self._materialise_higher_tf(self._df_10m, self._df_10m_cols)

        # A day is viable if it has at least 2 bars (so we can take at least
        # one step that isn't the forced-close last bar). When pos[0] is
        # smaller than window_size-1 the observation builder pads with the
        # first available row, which keeps obs shape stable.
        viable = [d for d in self._days if len(self._day_to_pos[d]) >= 2]
        self._days = viable
        if not self._days:
            raise ValueError(
                "No trading days with at least 2 bars found. Provide more bars "
                "or check the input DataFrame indexing."
            )

    # ── Helpers for higher-timeframe handling ────────────────────────────
    def _prep_higher_tf(self, df: pd.DataFrame | None, label: str) -> pd.DataFrame:
        """
        Prepare a higher-timeframe DataFrame for asof-lookup. Sets the
        corresponding `_df_*_cols` attribute on self.

        If ``df`` is None or empty, sets an empty cols list and returns an
        empty DataFrame so downstream code can no-op gracefully.
        """
        attr_name = f"_df_{label}_cols"
        if df is None or df.empty:
            setattr(self, attr_name, [])
            return pd.DataFrame()

        out = df.copy().sort_index()
        if not isinstance(out.index, pd.DatetimeIndex):
            raise TypeError(f"featured_df_{label} must have a DatetimeIndex")
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
        cols = _feature_columns(out)
        setattr(self, attr_name, cols)
        return out

    def _materialise_higher_tf(
        self, df: pd.DataFrame, cols: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if df is None or df.empty or not cols:
            return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int64)
        arr = df[cols].to_numpy(dtype=np.float32, copy=True)
        ts = df.index.asi8
        return arr, ts

    # ── Public API: alpha control for reward shapers ─────────────────────
    def set_shaping_alpha(self, alpha: float) -> None:
        """Update the reward-shaping multiplier (used by Models B & C)."""
        self._alpha = float(alpha)

    def get_shaping_alpha(self) -> float:
        return float(self._alpha)

    # ── Seeding ──────────────────────────────────────────────────────────
    def _seed(self, seed: int | None) -> None:
        self._np_random = np.random.default_rng(seed)

    # ── Reset ────────────────────────────────────────────────────────────
    def reset(self, seed: int | None = None, options: dict | None = None
              ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._seed(seed)

        # Pick a random trading day with enough lookback
        day_idx = int(self._np_random.integers(0, len(self._days)))
        self._day_date = self._days[day_idx]
        self._day_positions = self._day_to_pos[self._day_date]

        # ── Reset portfolio state ──────────────────────────────────────
        self._step_in_day = 0
        self.position = 0
        self.shares = 0.0
        self.entry_price = 0.0
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        self._prev_portfolio_value = self.initial_capital
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.n_trades = 0
        self._prev_obs = None

        obs = self._build_observation()
        info = self._build_info(reward=0.0, terminated=False, truncated=False, action=0)
        return obs, info

    # ── Step ─────────────────────────────────────────────────────────────
    def step(self, action: int | np.integer) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        if action not in (0, 1, 2):
            raise ValueError(f"Invalid action {action!r}; expected 0/1/2")

        prev_obs = self._prev_obs

        # Current bar position in the underlying 1m array
        cur_pos = int(self._day_positions[self._step_in_day])
        cur_price = float(self._close_1m[cur_pos])

        # ── Detect intent / execute trade transition ───────────────────
        intent_position = 0 if action == 0 else (1 if action == 1 else -1)
        position_change = abs(intent_position - self.position)

        # If forced to close at end-of-day, mark this flag
        is_last_step = (self._step_in_day + 1) >= len(self._day_positions)
        if is_last_step and intent_position != 0:
            # Force-close at last bar — override intent to FLAT
            intent_position = 0
            position_change = abs(intent_position - self.position)

        # Execute the transition (close old, open new) at current price
        self._execute_transition(intent_position, cur_price)

        # ── Mark portfolio to current price ────────────────────────────
        self._mark_to_market(cur_price)

        # ── Base reward: pct change in portfolio value minus tx cost ───
        if self._prev_portfolio_value > 0:
            base_reward = (
                (self.portfolio_value - self._prev_portfolio_value)
                / self._prev_portfolio_value
            )
        else:
            base_reward = 0.0
        base_reward -= self.transaction_cost * position_change

        # ── Reward shaping bonus ───────────────────────────────────────
        shaped_bonus = 0.0
        # Build the new obs *before* advancing — shaper compares prev to cur
        obs = self._build_observation()
        if self.shaper is not None:
            if hasattr(self.shaper, "set_prev_obs"):
                try:
                    self.shaper.set_prev_obs(prev_obs)
                except Exception:  # pragma: no cover — shaper must not crash env
                    log.exception("Shaper.set_prev_obs raised — ignoring")
            try:
                shaped_bonus = float(
                    self.shaper.compute_bonus(obs, action, self._step_in_day, self._alpha)
                )
            except Exception:  # pragma: no cover
                log.exception("Shaper.compute_bonus raised — using 0.0")
                shaped_bonus = 0.0

        reward = float(np.clip(base_reward + shaped_bonus, -1.0, 1.0))

        # ── Advance step and decide termination ────────────────────────
        self._prev_portfolio_value = self.portfolio_value
        self._step_in_day += 1
        self._prev_obs = obs

        terminated = False
        truncated = False
        if self._step_in_day >= len(self._day_positions):
            terminated = True
        elif self.position == 0 and is_last_step:
            terminated = True

        info = self._build_info(
            reward=reward, terminated=terminated, truncated=truncated, action=action
        )
        return obs, reward, terminated, truncated, info

    # ── Portfolio mechanics ──────────────────────────────────────────────
    # Cash-accounting model used here:
    #   FLAT  : shares = 0, cash = equity
    #   LONG  : shares > 0 (== equity_at_entry / entry_price), cash = 0
    #           equity = shares * price
    #   SHORT : shares < 0 (== -equity_at_entry / entry_price), cash = equity_at_entry
    #           equity = cash + shares * (price - entry_price)
    #                 = cash + |shares| * (entry_price - price)
    def _execute_transition(self, target_position: int, price: float) -> None:
        """
        Move from current position to target_position at ``price``.
        Handles the close of an existing position (realising PnL) and
        opening of a new one with sizing = 100% of current equity.
        """
        if target_position == self.position:
            return

        # ── Close existing position if any ─────────────────────────────
        if self.position != 0:
            if self.position == 1:
                # Long: realised = shares * (price - entry_price)
                #       cash_after = shares * price
                realised = self.shares * (price - self.entry_price)
                self.cash = self.shares * price  # convert position back into cash
            else:
                # Short: realised = |shares| * (entry_price - price)
                #                 = -shares * (entry_price - price)
                # Existing cash at this moment equals equity_at_entry; the new
                # equity-after-close is just equity_at_entry + realised PnL.
                realised = -self.shares * (self.entry_price - price)
                self.cash = self.cash + realised
            self.realized_pnl += float(realised)
            self.shares = 0.0
            self.entry_price = 0.0
            self.position = 0
            self.n_trades += 1

        # ── Open new position if requested ─────────────────────────────
        if target_position == 1:
            if price <= 0 or self.cash <= 0:
                return
            self.shares = self.cash / price
            self.entry_price = price
            self.cash = 0.0  # all equity is now in the long position
            self.position = 1
            self.n_trades += 1
        elif target_position == -1:
            if price <= 0 or self.cash <= 0:
                return
            # Negative shares; cash STAYS equal to current equity. The proceeds
            # of the short sale and the liability cancel out at entry: equity
            # only moves as price diverges from entry.
            self.shares = -self.cash / price
            self.entry_price = price
            # cash is unchanged on entry under this accounting model
            self.position = -1
            self.n_trades += 1

    def _mark_to_market(self, price: float) -> None:
        """Recompute portfolio_value and unrealized_pnl based on current price."""
        if self.position == 1:
            self.unrealized_pnl = self.shares * (price - self.entry_price)
            self.portfolio_value = self.shares * price + self.cash
        elif self.position == -1:
            # Equity = cash + shares*(price - entry_price), with shares < 0
            self.unrealized_pnl = self.shares * (price - self.entry_price)
            self.portfolio_value = self.cash + self.unrealized_pnl
        else:
            self.unrealized_pnl = 0.0
            self.portfolio_value = self.cash

    # ── Observation builder ──────────────────────────────────────────────
    def _build_observation(self) -> np.ndarray:
        # Slice last `window_size` 1m feature rows ending at current pos
        cur_pos = int(self._day_positions[self._step_in_day])
        start = cur_pos - self.window_size + 1
        if start < 0:
            # Pre-pad with the first available row to keep shape stable
            pad_rows = -start
            window = np.vstack([
                np.tile(self._arr_1m[0:1, :], (pad_rows, 1)),
                self._arr_1m[0:cur_pos + 1, :],
            ])
            start = 0
        else:
            window = self._arr_1m[start:cur_pos + 1, :]
        window_flat = window.reshape(-1).astype(np.float32, copy=False)

        # Higher-timeframe asof lookup
        cur_ts_ns = int(self._ts_1m_ns[cur_pos])
        tf5_row = self._asof_row(cur_ts_ns, self._ts_5m_ns, self._arr_5m, self._n_features_5m)
        tf10_row = self._asof_row(cur_ts_ns, self._ts_10m_ns, self._arr_10m, self._n_features_10m)

        # Portfolio state (4-d)
        cur_price = float(self._close_1m[cur_pos])
        pos_encoded = float(self.position)  # already -1 / 0 / 1
        unreal_norm = (
            self.unrealized_pnl / self.initial_capital
            if self.initial_capital > 0 else 0.0
        )
        cash_ratio = (
            self.cash / self.initial_capital if self.initial_capital > 0 else 0.0
        )
        # time-of-day: minutes since session open (in NY tz). Approximate using
        # step_in_day / max_steps; this stays in [0, 1] within RTH.
        day_len = max(1, len(self._day_positions))
        time_norm = float(self._step_in_day) / float(day_len)

        port = np.array(
            [pos_encoded, unreal_norm, cash_ratio, time_norm], dtype=np.float32
        )

        obs = np.concatenate([window_flat, tf5_row, tf10_row, port]).astype(np.float32)
        # Sanity: replace NaN/inf with zeros to keep SB3 happy
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    @staticmethod
    def _asof_row(
        target_ns: int, ts_array: np.ndarray, arr: np.ndarray, n_features: int
    ) -> np.ndarray:
        """Return most recent row at or before target_ns, or zeros if none / empty."""
        if n_features == 0 or arr.size == 0 or ts_array.size == 0:
            return np.zeros(n_features, dtype=np.float32)
        # searchsorted: first index with ts > target → take idx - 1
        idx = int(np.searchsorted(ts_array, target_ns, side="right")) - 1
        if idx < 0:
            return np.zeros(n_features, dtype=np.float32)
        return arr[idx].astype(np.float32, copy=False)

    # ── Info dict ───────────────────────────────────────────────────────
    def _build_info(self, reward: float, terminated: bool, truncated: bool, action: int) -> dict:
        return {
            "portfolio_value": float(self.portfolio_value),
            "position": int(self.position),
            "unrealised_pnl": float(self.unrealized_pnl),
            "realized_pnl": float(self.realized_pnl),
            "n_trades": int(self.n_trades),
            "day_date": str(self._day_date) if self._day_date is not None else None,
            "step_in_day": int(self._step_in_day),
            "action": int(action),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    # ── Rendering (no-op) ───────────────────────────────────────────────
    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
