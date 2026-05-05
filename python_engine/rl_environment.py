"""
rl_environment.py — Custom Gymnasium Environments for HFT RL  (Optimised v2)
=============================================================================
Changes from v1
---------------
* Pure normalised capital-gain reward: reward = (portfolio_t - portfolio_{t-1})
  / initial_capital.  No penalties, no log-return proxy.  Transaction cost is
  modelled as real portfolio drag (money leaves the account) rather than a
  reward shaping term.
* Position {-1, 0, +1} is appended to the observation vector.  The MDP was
  previously non-Markov because the agent did not know its own position.
* US Pattern Day Trader (PDT) circuit breaker:
  - Tracks intraday round-trips per calendar day.
  - If day_trades_today >= 3 AND capital < PDT_THRESHOLD ($25 000), forces
    action to HOLD for the rest of the day and logs a warning.
  - PDT enforcement is disabled in HK mode.
* HK stamp duty (0.13% per side) applied automatically when market='hk'.
* Board-lot enforcement for HK: shares rounded down to nearest lot size.
* MultiMarketEnv now uses a flat Discrete(3*N) action space (not MultiDiscrete)
  so PPO can handle it natively without additional wrappers.
* _ContinuousHFTEnv moved here (was inlined in trainer.py) to avoid repetition.
* All docstrings, type hints, and the smoke-test section updated.
* Fixed: __main__ guard was `"____main__"` (4 underscores) → `"__main__"`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_HOLD = 0
ACTION_BUY  = 1
ACTION_SELL = 2

POSITION_FLAT  =  0
POSITION_LONG  =  1
POSITION_SHORT = -1

PDT_THRESHOLD: float = 25_000.0   # USD — below this, PDT rule applies
HK_STAMP_DUTY: float = 0.0013     # 0.13% per side on HKEX
DEFAULT_BOARD_LOT: int = 500      # Default HKEX board lot size (parameterisable)


# ---------------------------------------------------------------------------
# Feature column helper
# ---------------------------------------------------------------------------


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return z-score feature columns (z_* prefix).

    The observation vector is built from these columns PLUS one extra slot
    for the position scalar appended in _get_obs().  The observation_space
    shape accounts for this extra dimension.
    """
    z_cols = [c for c in df.columns if c.startswith("z_")]
    if z_cols:
        return sorted(z_cols)
    # Fallback to raw OHLCV if no z_ columns exist (smoke-test / legacy data)
    return [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]


# ---------------------------------------------------------------------------
# StepInfo
# ---------------------------------------------------------------------------


@dataclass
class StepInfo:
    """Structured info dict returned by step()."""
    portfolio_value: float = 0.0
    position: int = POSITION_FLAT
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    drawdown_pct: float = 0.0
    n_trades: int = 0
    current_price: float = 0.0
    step_idx: int = 0
    pdt_blocked: bool = False

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
            "pdt_blocked": self.pdt_blocked,
        }


# ---------------------------------------------------------------------------
# HFTradingEnv
# ---------------------------------------------------------------------------


class HFTradingEnv(gym.Env):
    """
    High-Frequency Trading Environment for 10s / 1m / 5m / 1h bar RL.

    Observation space
    -----------------
    [window_size × n_features] flattened z-score features  +  [1] position.
    Shape: (window_size * n_features + 1,)   dtype: float32
    Bounds: [-10, +10]

    Action space
    ------------
    Discrete(3): 0 = HOLD, 1 = BUY (go long), 2 = SELL (go short)

    Reward
    ------
    Pure normalised capital gain per step:
        r_t = (portfolio_t − portfolio_{t−1}) / initial_capital

    Transaction costs are deducted from the portfolio (real money out),
    not subtracted from the reward as a shaping penalty.  This means the
    agent must learn to avoid over-trading naturally.

    Market-specific rules
    ---------------------
    market='us':  PDT circuit breaker (max 3 day-trades / 5-day window if
                  portfolio < $25 000).
    market='hk':  HK stamp duty (0.13% per side), board-lot rounding.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 60,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        max_position: float = 0.95,
        render_mode: Optional[str] = None,
        market: str = "us",
        board_lot: int = DEFAULT_BOARD_LOT,
    ) -> None:
        super().__init__()

        if df.empty:
            raise ValueError("df must not be empty.")
        if len(df) < window_size + 1:
            raise ValueError(
                f"DataFrame has only {len(df)} rows; need at least {window_size + 1}."
            )

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.render_mode = render_mode
        self.market = market.lower()
        self.board_lot = board_lot

        self._feature_cols: list[str] = _get_feature_cols(df)
        self._n_features: int = len(self._feature_cols)
        # +1 for position scalar
        self._obs_dim: int = window_size * self._n_features + 1

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # Episode state — initialised in reset()
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

        # PDT tracking (US only)
        self._day_trades_today: int = 0
        self._pdt_window: list[date] = []   # rolling 5-business-day window
        self._current_day: Optional[date] = None
        self._position_opened_today: bool = False

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

        # Random start: leave at least 200 bars for a meaningful episode.
        # Guard against datasets too short to have a gap between window_size
        # and (len - 200): clamp so low < high always holds.
        low  = self.window_size
        high = max(low + 1, len(self.df) - 200)
        self._step_idx = int(self.np_random.integers(low, high))

        self._portfolio_value    = self.initial_capital
        self._cash               = self.initial_capital
        self._position           = POSITION_FLAT
        self._entry_price        = 0.0
        self._shares_held        = 0.0
        self._max_portfolio_value = self.initial_capital
        self._n_trades           = 0
        self._realized_pnl       = 0.0
        self._portfolio_history  = [self.initial_capital]

        # PDT state
        self._day_trades_today    = 0
        self._pdt_window          = []
        self._current_day         = None
        self._position_opened_today = False

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._step_idx >= len(self.df):
            return self._get_obs(), 0.0, True, False, {}

        current_row   = self.df.iloc[self._step_idx]
        current_price = float(current_row["Close"])
        prev_portfolio = self._portfolio_value
        pdt_blocked   = False

        # ── PDT Circuit Breaker (US only) ─────────────────────────────────
        if self.market == "us" and self._portfolio_value < PDT_THRESHOLD:
            action = self._apply_pdt_rules(action, current_row)
            if action == ACTION_HOLD and self._position != POSITION_FLAT:
                pdt_blocked = True

        # ── Execute action ────────────────────────────────────────────────
        position_change = 0

        if action == ACTION_BUY and self._position != POSITION_LONG:
            position_change += self._open_long(current_price)

        elif action == ACTION_SELL and self._position != POSITION_SHORT:
            position_change += self._open_short(current_price)

        # ── Mark-to-market ────────────────────────────────────────────────
        unrealized = self._mark_to_market(current_price)

        self._max_portfolio_value = max(self._max_portfolio_value, self._portfolio_value)
        self._portfolio_history.append(self._portfolio_value)

        # ── Pure capital-gain reward ──────────────────────────────────────
        #
        # r_t = (V_t - V_{t-1}) / V_0
        #
        # Transaction costs are already deducted from self._cash when trades
        # are executed, so they flow into the reward automatically.
        # No explicit penalty terms — the agent learns to avoid over-trading
        # because each trade reduces V through real transaction costs.
        reward = (self._portfolio_value - prev_portfolio) / self.initial_capital
# Guard against any residual NaN/inf from price data anomalies
        if not np.isfinite(reward):
            reward = 0.0
        reward = float(np.clip(reward, -1.0, 1.0))  # cap at ±100% per step
        
        # ── Termination ───────────────────────────────────────────────────
        terminated = bool(
            self._portfolio_value < 0.5 * self.initial_capital
            or self._step_idx >= len(self.df) - 1
        )

        drawdown_pct = (self._max_portfolio_value - self._portfolio_value) / (
            self._max_portfolio_value + 1e-8
        )

        info = StepInfo(
            portfolio_value=self._portfolio_value,
            position=self._position,
            realized_pnl=self._realized_pnl,
            unrealized_pnl=unrealized,
            drawdown_pct=drawdown_pct,
            n_trades=self._n_trades,
            current_price=current_price,
            step_idx=self._step_idx,
            pdt_blocked=pdt_blocked,
        ).to_dict()

        self._step_idx += 1
        return self._get_obs(), float(reward), terminated, False, info

    # ------------------------------------------------------------------
    # Trade execution helpers
    # ------------------------------------------------------------------

    def _cost(self, notional: float) -> float:
        """Compute round-trip cost for one side of a trade."""
        base_cost = self.transaction_cost * notional
        if self.market == "hk":
            base_cost += HK_STAMP_DUTY * notional
        return base_cost

    def _round_shares(self, shares: float) -> float:
        """For HK: round down to nearest board lot."""
        if self.market == "hk":
            return float(int(shares // self.board_lot) * self.board_lot)
        return shares

    def _open_long(self, price: float) -> int:
        """Close short (if open) and open long.  Returns position_changes count."""
        changes = 0

        if self._position == POSITION_SHORT:
            # Close short: buy back
            buyback = self._shares_held * price
            cost = self._cost(buyback)
            pnl = (self._entry_price - price) * self._shares_held - cost
            self._realized_pnl += pnl
            # Restore cash: original short proceeds + PnL
            self._cash += self._shares_held * self._entry_price + pnl
            self._n_trades += 1
            changes += 1

        invest = self._cash * self.max_position
        shares = self._round_shares(invest / price)
        if shares <= 0:
            return changes
        actual_invest = shares * price
        cost = self._cost(actual_invest)
        self._cash -= actual_invest + cost
        self._shares_held = shares
        self._position    = POSITION_LONG
        self._entry_price = price
        self._n_trades   += 1
        changes += 1
        self._position_opened_today = True
        return changes

    def _open_short(self, price: float) -> int:
        """Close long (if open) and open short.  Returns position_changes count."""
        changes = 0

        if self._position == POSITION_LONG:
            # Close long: sell
            proceeds = self._shares_held * price
            cost = self._cost(proceeds)
            pnl = (price - self._entry_price) * self._shares_held - cost
            self._realized_pnl += pnl
            self._cash += proceeds - cost
            self._n_trades += 1
            changes += 1
            # PDT: count this as a completed day-trade if opened today
            if self.market == "us" and self._position_opened_today:
                self._day_trades_today += 1
                self._position_opened_today = False

        invest = self._cash * self.max_position
        shares = self._round_shares(invest / price)
        if shares <= 0:
            return changes
        actual_invest = shares * price
        cost = self._cost(actual_invest)
        # Short sale: broker credits proceeds to cash, we pay transaction cost.
        # MTM will then deduct mark-to-market losses as price moves against us.
        self._cash += actual_invest - cost
        self._shares_held = shares
        self._position    = POSITION_SHORT
        self._entry_price = price
        self._n_trades   += 1
        changes += 1
        self._position_opened_today = True
        return changes

    def _mark_to_market(self, price: float) -> float:
        """Update portfolio value and return unrealised PnL."""
        if self._position == POSITION_LONG:
            unrealized = (price - self._entry_price) * self._shares_held
            self._portfolio_value = self._cash + self._shares_held * price
        elif self._position == POSITION_SHORT:
            unrealized = (self._entry_price - price) * self._shares_held
            self._portfolio_value = self._cash + self._entry_price * self._shares_held - price * self._shares_held
        else:
            unrealized = 0.0
            self._portfolio_value = self._cash
        return unrealized

    # ------------------------------------------------------------------
    # PDT logic
    # ------------------------------------------------------------------

    def _apply_pdt_rules(self, action: int, row: pd.Series) -> int:
        """
        Enforce US PDT rule.

        If we have already made >= 3 day-trades in the rolling 5-business-day
        window AND the portfolio is below $25 000, block any action that would
        open or close an intraday position.
        """
        # Determine current trading date from the DataFrame index if available
        try:
            current_date = pd.Timestamp(row.name).date()
        except Exception:
            return action

        # Reset day counter at day boundary
        if self._current_day != current_date:
            self._current_day = current_date
            self._position_opened_today = False
            # Maintain rolling 5-business-day window
            self._pdt_window.append(current_date)
            cutoff = current_date - timedelta(days=7)  # ~5 business days
            self._pdt_window = [d for d in self._pdt_window if d >= cutoff]
            # Count day-trades in window (each window entry = 1 day)
            # The counter self._day_trades_today tracks trades this calendar day
            self._day_trades_today = 0

        if self._day_trades_today >= 3:
            if action != ACTION_HOLD:
                logger.warning(
                    "PDT circuit breaker: blocked action=%d — "
                    "day_trades_today=%d, portfolio=$%.0f < $25,000.",
                    action, self._day_trades_today, self._portfolio_value,
                )
            return ACTION_HOLD

        return action

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        Return (window_size × n_features + 1,) float32 observation.

        The last element is the current position encoded as float:
          POSITION_SHORT=-1, POSITION_FLAT=0, POSITION_LONG=+1.
        This makes the MDP fully observable — the agent knows its own state.
        """
        start  = max(0, self._step_idx - self.window_size)
        end    = self._step_idx
        window = self.df.iloc[start:end][self._feature_cols].values.astype(np.float32)

        if len(window) < self.window_size:
            pad    = np.zeros((self.window_size - len(window), self._n_features), dtype=np.float32)
            window = np.vstack([pad, window])

        obs = window.flatten()
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, -10.0, 10.0)

        # Append position as normalised scalar [-1, 0, +1]
        position_enc = np.array([float(self._position)], dtype=np.float32)
        return np.concatenate([obs, position_enc])

    # ------------------------------------------------------------------
    # Render / close
    # ------------------------------------------------------------------

    def render(self) -> Optional[np.ndarray]:
        if not self._portfolio_history:
            return None
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self._portfolio_history, linewidth=1, color="#20D9A1")
        ax.axhline(y=self.initial_capital, color="#8A95A5", linestyle="--", linewidth=0.8)
        ax.set_title("Portfolio Value")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
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
# _ContinuousHFTEnv  (TD3 wrapper)
# ---------------------------------------------------------------------------


class _ContinuousHFTEnv(HFTradingEnv):
    """
    Thin wrapper converting Discrete(3) → Box([0, 3)) for TD3.

    TD3 requires a continuous action space.  We map the continuous output to
    the nearest discrete action at step-time.  This lives in rl_environment.py
    rather than inlined in trainer.py to avoid duplication.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(low=0.0, high=3.0, shape=(1,), dtype=np.float32)

    def step(self, action):  # type: ignore[override]
        discrete_action = int(np.clip(np.round(action[0]), 0, 2))
        return super().step(discrete_action)


# ---------------------------------------------------------------------------
# MultiMarketEnv
# ---------------------------------------------------------------------------


class MultiMarketEnv(gym.Env):
    """
    Multi-market wrapper trading N tickers simultaneously.

    Action space: Discrete(3 ** N) — encodes one Discrete(3) action per
    ticker as a single integer.  This is compatible with PPO without needing
    any extra wrappers.  Use N <= 3 to keep the action space tractable.

    Observation space: concatenation of all sub-env observations (flat).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_dict: dict[str, pd.DataFrame],
        window_size: int = 60,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        max_position: float = 0.95,
        market: str = "us",
    ) -> None:
        super().__init__()

        n = len(data_dict)
        if n > 3:
            logger.warning(
                "MultiMarketEnv: %d tickers → action space size 3^%d=%d. "
                "Consider using n<=3 for tractable learning.", n, n, 3**n
            )

        self._envs: dict[str, HFTradingEnv] = {
            ticker: HFTradingEnv(
                df=df,
                window_size=window_size,
                initial_capital=initial_capital / n,
                transaction_cost=transaction_cost,
                max_position=max_position,
                market=market,
            )
            for ticker, df in data_dict.items()
        }
        self._tickers: list[str] = list(self._envs.keys())
        self._n: int = n

        obs_dim = sum(env.observation_space.shape[0] for env in self._envs.values())  # type: ignore
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        # Flat Discrete: encode N independent Discrete(3) as single integer
        self.action_space = spaces.Discrete(3 ** n)

    def _decode_action(self, flat_action: int) -> list[int]:
        """Decode flat integer action into per-ticker Discrete(3) actions."""
        actions = []
        for _ in range(self._n):
            actions.append(flat_action % 3)
            flat_action //= 3
        return actions

    def reset(self, *, seed=None, options=None):
        obs_parts = []
        for env in self._envs.values():
            obs, _ = env.reset(seed=seed)
            obs_parts.append(obs)
        return np.concatenate(obs_parts, dtype=np.float32), {}

    def step(self, action: int):
        per_ticker_actions = self._decode_action(int(action))
        obs_parts  = []
        total_rew  = 0.0
        terminated = False
        info: dict[str, Any] = {}

        for i, (ticker, env) in enumerate(self._envs.items()):
            obs, rew, term, _, sub_info = env.step(per_ticker_actions[i])
            obs_parts.append(obs)
            total_rew  += rew
            terminated  = terminated or term
            info[ticker] = sub_info

        return np.concatenate(obs_parts, dtype=np.float32), total_rew, terminated, False, info

    def render(self) -> None:
        for ticker, env in self._envs.items():
            logger.info("Rendering %s ...", ticker)
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
    transaction_cost: float = 0.001,
    max_position: float = 0.95,
    multi_market: bool = False,
    market: str = "us",
) -> VecEnv:
    """
    Create a SubprocVecEnv of HFTradingEnv or MultiMarketEnv.

    Parameters
    ----------
    data_dict:   ticker → feature DataFrame.
    n_envs:      Parallel workers.
    market:      'us' or 'hk' — propagated to each env for PDT / stamp-duty.
    multi_market: If True, use MultiMarketEnv over all tickers.
    """
    if multi_market:
        def _mm_init():
            return MultiMarketEnv(
                data_dict=data_dict,
                window_size=window_size,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                max_position=max_position,
                market=market,
            )
        return DummyVecEnv([_mm_init] * n_envs)

    ticker, df = next(iter(data_dict.items()))
    logger.info("Creating %d parallel envs for ticker=%s market=%s", n_envs, ticker, market)

    def _make(df_: pd.DataFrame):
        def _init():
            return HFTradingEnv(
                df=df_,
                window_size=window_size,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                max_position=max_position,
                market=market,
            )
        return _init

    # DummyVecEnv runs envs in-process (no subprocess pipes).
    # On Windows, SubprocVecEnv uses 'spawn' which is slow and fragile
    # when the parent process is uvicorn with --reload.
    return DummyVecEnv([_make(df)] * n_envs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    np.random.seed(42)
    n = 600
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "Open":   close * 0.999,
        "High":   close * 1.002,
        "Low":    close * 0.998,
        "Close":  close,
        "Volume": np.random.randint(1_000, 100_000, n).astype(float),
    })
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"z_{col}"] = (
            (df[col] - df[col].rolling(100, min_periods=1).mean())
            / df[col].rolling(100, min_periods=1).std().replace(0, 1)
        )

    env = HFTradingEnv(df, window_size=60, market="us")
    obs, _ = env.reset(seed=0)
    print(f"Obs shape: {obs.shape}  (expected {env._obs_dim})")
    assert obs.shape == (env._obs_dim,), "Shape mismatch!"
    assert obs[-1] == 0.0, "Initial position should be FLAT=0"

    for step in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    print(f"Final portfolio: ${info['portfolio_value']:,.2f}  "
          f"n_trades={info['n_trades']}  "
          f"position_in_obs={obs[-1]:.0f}")
    print("HFTradingEnv smoke test PASSED.")
