"""
rl_environment.py — Custom Gymnasium Environments for HFT RL

Provides:
  HFTradingEnv      — Single-ticker Gymnasium environment for 10s bar trading.
  MultiMarketEnv    — Wraps multiple HFTradingEnv instances; one action per market.
  make_envs()       — Factory for SubprocVecEnv parallel training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

matplotlib.use("Agg")  # non-interactive backend for server usage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2

POSITION_FLAT = 0
POSITION_LONG = 1
POSITION_SHORT = -1


# ---------------------------------------------------------------------------
# Structured return types
# ---------------------------------------------------------------------------


@dataclass
class StepInfo:
    """Extra info dict returned by step()."""
    portfolio_value: float = 0.0
    position: int = POSITION_FLAT
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    drawdown_pct: float = 0.0
    n_trades: int = 0
    current_price: float = 0.0
    step_idx: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "drawdown_pct": self.drawdown_pct,
            "n_trades": self.n_trades,
            "current_price": self.current_price,
            "step_idx": self.step_idx,
        }


# ---------------------------------------------------------------------------
# Feature column helper
# ---------------------------------------------------------------------------

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return z-score normalised feature columns; fall back to raw OHLCV."""
    z_cols = [c for c in df.columns if c.startswith("z_")]
    if z_cols:
        return z_cols
    # Fallback: use OHLCV numeric columns
    return [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]


# ---------------------------------------------------------------------------
# HFTradingEnv
# ---------------------------------------------------------------------------


class HFTradingEnv(gym.Env):
    """
    High-Frequency Trading Environment for 10-second bar RL.

    Observation space
    -----------------
    Rolling window of ``window_size`` bars × N features (normalised OHLCV +
    technical indicators). Flattened to a 1-D float32 array.

    Action space
    ------------
    Discrete(3) — 0 = HOLD, 1 = BUY (go long), 2 = SELL (go short)

    Reward function
    ---------------
    r = realized_pnl
        - transaction_cost * |position_change|
        - 0.1 * max(0, -drawdown_pct)
        - 0.001 * (action != HOLD)    # discourages overtrading

    Episode termination
    -------------------
    - Portfolio value drops below 50 % of initial capital.
    - Data exhausted (end of DataFrame).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 60,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.0001,
        max_position: float = 0.95,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        if df.empty:
            raise ValueError("df must not be empty.")

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.render_mode = render_mode

        self._feature_cols: list[str] = _get_feature_cols(df)
        self._n_features: int = len(self._feature_cols)
        self._obs_dim: int = window_size * self._n_features

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # Episode state (initialised in reset)
        self._step_idx: int = window_size
        self._portfolio_value: float = initial_capital
        self._cash: float = initial_capital
        self._position: int = POSITION_FLAT
        self._entry_price: float = 0.0
        self._shares_held: float = 0.0
        self._max_portfolio_value: float = initial_capital
        self._n_trades: int = 0
        self._realized_pnl: float = 0.0
        self._portfolio_history: list[float] = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Random start for training diversity; leave at least 200 steps
        max_start = max(self.window_size, len(self.df) - 200)
        self._step_idx = int(self.np_random.integers(self.window_size, max(max_start, self.window_size + 1)))

        self._portfolio_value = self.initial_capital
        self._cash = self.initial_capital
        self._position = POSITION_FLAT
        self._entry_price = 0.0
        self._shares_held = 0.0
        self._max_portfolio_value = self.initial_capital
        self._n_trades = 0
        self._realized_pnl = 0.0
        self._portfolio_history = [self.initial_capital]

        obs = self._get_obs()
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._step_idx >= len(self.df):
            obs = self._get_obs()
            return obs, 0.0, True, False, {}

        current_row = self.df.iloc[self._step_idx]
        current_price = float(current_row["Close"])
        prev_portfolio = self._portfolio_value

        # --- Execute action ---
        position_change = 0
        cost = 0.0

        if action == ACTION_BUY and self._position != POSITION_LONG:
            # Close short if open
            if self._position == POSITION_SHORT:
                self._realized_pnl += (self._entry_price - current_price) * self._shares_held
                
                cost_to_buy_back = self._shares_held * current_price
                buy_cost = self.transaction_cost * cost_to_buy_back
                
                # You get back the proceeds of the initial short sale, 
                # minus what it costs to buy the shares back, minus fees.
                initial_short_proceeds = self._shares_held * self._entry_price
                self._cash += initial_short_proceeds - cost_to_buy_back - buy_cost
                
                cost += buy_cost
                self._n_trades += 1
                position_change += 1

            # Open long
            invest_amount = self._cash * self.max_position
            self._shares_held = invest_amount / current_price
            entry_cost = self.transaction_cost * invest_amount
            self._cash -= invest_amount + entry_cost
            cost += entry_cost
            self._position = POSITION_LONG
            self._entry_price = current_price
            self._n_trades += 1
            position_change += 1

        elif action == ACTION_SELL and self._position != POSITION_SHORT:
            # Close long if open
            if self._position == POSITION_LONG:
                proceeds = self._shares_held * current_price
                sell_cost = self.transaction_cost * proceeds
                self._realized_pnl += (current_price - self._entry_price) * self._shares_held
                
                # Add proceeds of selling the long position back to cash
                self._cash += proceeds - sell_cost 
                cost += sell_cost
                self._n_trades += 1
                position_change += 1

            # Open short
            invest_amount = self._cash * self.max_position
            self._shares_held = invest_amount / current_price
            short_cost = self.transaction_cost * invest_amount
            
            # Lock up 'invest_amount' of your own cash as 100% margin.
            # Net cash change is $0, minus the transaction fee.
            self._cash -= short_cost 
            
            cost += short_cost
            self._position = POSITION_SHORT
            self._entry_price = current_price
            self._n_trades += 1
            position_change += 1

        # --- Mark-to-market portfolio value & Unrealized PnL ---
        if self._position == POSITION_LONG:
            unrealized = (current_price - self._entry_price) * self._shares_held
            self._portfolio_value = self._cash + (self._shares_held * current_price)

        elif self._position == POSITION_SHORT:
            unrealized = (self._entry_price - current_price) * self._shares_held
            cost_to_buy_back = self._shares_held * current_price
            initial_short_proceeds = self._shares_held * self._entry_price
            self._portfolio_value = self._cash + initial_short_proceeds - cost_to_buy_back

        else:
            unrealized = 0.0
            self._portfolio_value = self._cash

        self._max_portfolio_value = max(self._max_portfolio_value, self._portfolio_value)
        self._portfolio_history.append(self._portfolio_value)

        # --- Log Returns Reward ---
        # We calculate the step reward as the log return of the portfolio value.
        # This provides a bounded, symmetric scale for PnL improvements.
        # Log(current / previous) = Log(current) - Log(previous)
        
        # Ensure we don't log(0) if the portfolio blew up
        safe_portfolio = max(self._portfolio_value, 1e-8)
        safe_prev_portfolio = max(prev_portfolio, 1e-8)
        
        log_return = np.log(safe_portfolio / safe_prev_portfolio)
        drawdown_pct = (self._max_portfolio_value - self._portfolio_value) / (self._max_portfolio_value + 1e-8)
        
        reward = (
            log_return 
            - self.transaction_cost * abs(position_change)  # transaction cost penalty
            - 0.1 * max(0.0, drawdown_pct)                  # actual drawdown penalty
            - 0.001 * float(action != 0)                    # overtrading penalty
        )

        # --- Termination conditions ---
        terminated = bool(
            self._portfolio_value < 0.5 * self.initial_capital
            or self._step_idx >= len(self.df) - 1
        )
        truncated = False

        info = {
            "portfolio_value": self._portfolio_value,
            "position": self._position,
            "realized_pnl": self._realized_pnl,
            "unrealized_pnl": unrealized,
            "drawdown_pct": drawdown_pct,
            "n_trades": self._n_trades,
            "current_price": current_price,
            "step_idx": self._step_idx,
        }

        self._step_idx += 1
        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Return the last window_size bars as a flat float32 array."""
        start = max(0, self._step_idx - self.window_size)
        end = self._step_idx
        window = self.df.iloc[start:end][self._feature_cols].values.astype(np.float32)

        # Pad with zeros if at the beginning of the episode
        if len(window) < self.window_size:
            pad = np.zeros((self.window_size - len(window), self._n_features), dtype=np.float32)
            window = np.vstack([pad, window])

        obs = window.flatten()
        # Clip to observation space bounds; replace NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(obs, -10.0, 10.0)

    def render(self) -> Optional[np.ndarray]:
        """Plot portfolio value history via matplotlib."""
        if not self._portfolio_history:
            return None

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self._portfolio_history, linewidth=1, color="#2196F3")
        ax.axhline(y=self.initial_capital, color="#9E9E9E", linestyle="--", linewidth=0.8)
        ax.set_title("Portfolio Value")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value ($)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return img

        plt.show()
        plt.close(fig)
        return None

    def close(self) -> None:
        plt.close("all")


# ---------------------------------------------------------------------------
# MultiMarketEnv
# ---------------------------------------------------------------------------


class MultiMarketEnv(gym.Env):
    """
    Multi-market wrapper that concatenates observations from N single-market
    HFTradingEnv instances and expects one action per market.

    Observation space: concatenation of all sub-env observations (flat float32).
    Action space:      MultiDiscrete([3] * n_markets) — one action per market.

    Useful for training a single policy that simultaneously trades across
    multiple tickers/markets.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_dict: dict[str, pd.DataFrame],
        window_size: int = 60,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.0001,
        max_position: float = 0.95,
    ) -> None:
        super().__init__()

        self._envs: dict[str, HFTradingEnv] = {
            ticker: HFTradingEnv(
                df=df,
                window_size=window_size,
                initial_capital=initial_capital / len(data_dict),
                transaction_cost=transaction_cost,
                max_position=max_position,
            )
            for ticker, df in data_dict.items()
        }
        self._tickers: list[str] = list(self._envs.keys())

        obs_dim = sum(env.observation_space.shape[0] for env in self._envs.values())  # type: ignore[index]
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([3] * len(self._envs))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        obs_parts: list[np.ndarray] = []
        for env in self._envs.values():
            obs, _ = env.reset(seed=seed, options=options)
            obs_parts.append(obs)
        return np.concatenate(obs_parts, dtype=np.float32), {}

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs_parts: list[np.ndarray] = []
        total_reward = 0.0
        terminated = False
        combined_info: dict[str, Any] = {}

        for i, (ticker, env) in enumerate(self._envs.items()):
            action = int(actions[i])
            obs, reward, term, trunc, info = env.step(action)
            obs_parts.append(obs)
            total_reward += reward
            terminated = terminated or term
            combined_info[ticker] = info

        obs = np.concatenate(obs_parts, dtype=np.float32)
        return obs, total_reward, terminated, False, combined_info

    def render(self) -> None:
        for ticker, env in self._envs.items():
            logger.info("Rendering %s...", ticker)
            env.render()

    def close(self) -> None:
        for env in self._envs.values():
            env.close()


# ---------------------------------------------------------------------------
# Vectorised environment factory
# ---------------------------------------------------------------------------


def make_envs(
    data_dict: dict[str, pd.DataFrame],
    n_envs: int = 4,
    window_size: int = 60,
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.0001,
    max_position: float = 0.95,
    multi_market: bool = False,
) -> VecEnv:
    """
    Create a SubprocVecEnv of HFTradingEnv (single-ticker) or MultiMarketEnv.

    Parameters
    ----------
    data_dict:        Dict mapping ticker → feature DataFrame (from compute_features).
    n_envs:           Number of parallel workers.
    window_size:      Observation window in bars.
    initial_capital:  Starting capital per sub-environment.
    transaction_cost: Fractional transaction cost (e.g. 0.001 = 10 bps).
    max_position:     Maximum fraction of capital in a single position.
    multi_market:     If True, create a MultiMarketEnv over all tickers.
                      If False, use the first ticker only (for single-ticker PPO).

    Returns
    -------
    SubprocVecEnv ready for Stable-Baselines3 training.
    """
    if multi_market:
        def _make_env():
            def _init():
                env = MultiMarketEnv(
                    data_dict=data_dict,
                    window_size=window_size,
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost,
                    max_position=max_position,
                )
                return env
            return _init

        env_fns = [_make_env() for _ in range(n_envs)]
        return SubprocVecEnv(env_fns)

    # Single-ticker: pick first ticker from data_dict
    ticker, df = next(iter(data_dict.items()))
    logger.info("Creating %d parallel envs for ticker=%s", n_envs, ticker)

    def _make_single(df_: pd.DataFrame):
        def _init():
            return HFTradingEnv(
                df=df_,
                window_size=window_size,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                max_position=max_position,
            )
        return _init

    env_fns = [_make_single(df) for _ in range(n_envs)]
    return SubprocVecEnv(env_fns)


# ---------------------------------------------------------------------------
# Standalone test / smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    logging.basicConfig(level=logging.DEBUG)

    # Build a tiny synthetic OHLCV + feature dataset for smoke testing
    np.random.seed(42)
    n = 500
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.002,
        "Low": close * 0.998,
        "Close": close,
        "Volume": np.random.randint(1_000, 100_000, n).astype(float),
    })
    # Add minimal z-score features
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"z_{col}"] = (df[col] - df[col].rolling(100, min_periods=1).mean()) / (
            df[col].rolling(100, min_periods=1).std().replace(0, 1)
        )

    env = HFTradingEnv(df, window_size=60)
    obs, _ = env.reset(seed=0)
    print(f"Obs shape: {obs.shape}, dtype: {obs.dtype}")
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print(f"Final portfolio: ${info['portfolio_value']:,.2f}  n_trades={info['n_trades']}")
    print("HFTradingEnv smoke test passed.")
