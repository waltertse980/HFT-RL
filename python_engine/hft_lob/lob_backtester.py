"""
lob_backtester.py
=================
Phase 5 — Event-driven backtester for LOB PPO models.

Loads a trained LOB PPO model + VecNormalize statistics, replays the
saved feature parquet step-by-step, and reports realistic performance
metrics annualised at second-level frequency.

Public API
----------
run_lob_backtest(
    model_path: str,
    vecnorm_path: str,
    symbols: list[str],
    fee: float = 0.0001,
) -> dict
    Returns:
        {
            "sharpe": float,            # annualised
            "max_drawdown": float,
            "total_return": float,      # log-PnL
            "win_rate": float,
            "n_trades": int,
            "avg_pnl_per_trade": float,
            "position_ratio": float,    # share of steps spent in non-flat
            "n_steps": int,
            "per_symbol": {sym: {...}, ...},
        }
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from lob_features import load_feature_df, add_rolling_features, FEATURE_COLS
    from lob_environment import LOBTradingEnv
except ImportError:
    from hft_lob.lob_features import load_feature_df, add_rolling_features, FEATURE_COLS
    from hft_lob.lob_environment import LOBTradingEnv

# 23,400 seconds in a regular US trading day (6.5h × 3600)
SECONDS_PER_TRADING_YEAR = 252 * 23_400
ANNUALISATION = float(np.sqrt(SECONDS_PER_TRADING_YEAR))


def _eval_one(model, env, vecnorm) -> dict:
    """Run a single deterministic rollout and collect per-step stats."""
    obs, _ = env.reset()
    rewards: list[float] = []
    positions: list[int] = []
    trades = 0
    prev_pos = 0

    while True:
        if vecnorm is not None:
            obs_in = vecnorm.normalize_obs(obs[np.newaxis, :])
        else:
            obs_in = obs[np.newaxis, :]
        action, _ = model.predict(obs_in, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action[0]))
        rewards.append(float(reward))
        positions.append(int(info["position"]))
        if info["position"] != prev_pos:
            trades += 1
            prev_pos = info["position"]
        if terminated or truncated:
            break

    rewards_arr = np.asarray(rewards, dtype=np.float64)
    positions_arr = np.asarray(positions, dtype=np.int64)
    n = max(len(rewards_arr), 1)

    total_return = float(rewards_arr.sum())
    std = float(rewards_arr.std()) if rewards_arr.std() > 0 else 1e-9
    sharpe = float(rewards_arr.mean() / std * ANNUALISATION) if n > 1 else 0.0

    cum = np.cumsum(rewards_arr)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(dd.min()) if len(dd) else 0.0

    wins = int((rewards_arr > 0).sum())
    win_rate = float(wins / n)
    avg_pnl_per_trade = float(total_return / trades) if trades else 0.0
    position_ratio = float((positions_arr != 0).sum() / n)

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_return": total_return,
        "win_rate": win_rate,
        "n_trades": int(trades),
        "avg_pnl_per_trade": avg_pnl_per_trade,
        "position_ratio": position_ratio,
        "n_steps": int(n),
    }


def run_lob_backtest(
    model_path: str,
    vecnorm_path: str,
    symbols: list[str],
    fee: float = 0.0001,
) -> dict:
    """Backtest a saved LOB PPO model on saved feature parquet."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:
        raise RuntimeError("stable_baselines3 not installed") from exc

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model_path not found: {model_path}")

    df = load_feature_df(symbols)
    if df.empty:
        raise RuntimeError(
            f"No LOB feature data for symbols={symbols}. Download first."
        )
    df = add_rolling_features(df)

    # Build a temporary VecEnv just so VecNormalize can be loaded
    vecnorm = None
    if vecnorm_path and os.path.exists(vecnorm_path):
        sample_sym = symbols[0]
        sub = df[df["symbol"] == sample_sym].copy()
        if sub.empty:
            sub = df.copy()
        dummy = DummyVecEnv([lambda: LOBTradingEnv(sub, fee=fee)])
        vecnorm = VecNormalize.load(vecnorm_path, dummy)
        vecnorm.training = False
        vecnorm.norm_reward = False

    model = PPO.load(model_path, device="cpu")

    per_symbol: dict[str, dict] = {}
    aggregated_rewards: list[float] = []
    total_trades = 0
    total_position_steps = 0
    total_steps = 0

    for sym in symbols:
        sub = df[df["symbol"] == sym].copy()
        if sub.empty or len(sub) < 100:
            logger.warning("[lob-backtest] %s: insufficient data, skipping", sym)
            continue
        env = LOBTradingEnv(sub, fee=fee)
        stats = _eval_one(model, env, vecnorm)
        per_symbol[sym] = stats
        # Aggregate
        total_trades += stats["n_trades"]
        total_position_steps += int(stats["position_ratio"] * stats["n_steps"])
        total_steps += stats["n_steps"]
        # We don't have per-step rewards across symbols here — recompute
        # an "all-symbol" aggregate by summing returns weighted by n_steps
        aggregated_rewards.append(stats["total_return"])

    if not per_symbol:
        raise RuntimeError("No symbols produced any backtest output.")

    # Portfolio-level aggregates (equal-weight across symbols)
    sharpes = [s["sharpe"] for s in per_symbol.values()]
    returns = [s["total_return"] for s in per_symbol.values()]
    mdds = [s["max_drawdown"] for s in per_symbol.values()]
    wrs = [s["win_rate"] for s in per_symbol.values()]
    pnls = [s["avg_pnl_per_trade"] for s in per_symbol.values()]

    result = {
        "sharpe":            float(np.mean(sharpes)),
        "max_drawdown":      float(np.min(mdds)),
        "total_return":      float(np.sum(returns)),
        "win_rate":          float(np.mean(wrs)),
        "n_trades":          int(total_trades),
        "avg_pnl_per_trade": float(np.mean(pnls)) if pnls else 0.0,
        "position_ratio":    float(total_position_steps / max(total_steps, 1)),
        "n_steps":           int(total_steps),
        "per_symbol":        per_symbol,
    }
    logger.info(
        "[lob-backtest] sharpe=%.3f mdd=%.3f ret=%.3f trades=%d pos_ratio=%.3f",
        result["sharpe"], result["max_drawdown"], result["total_return"],
        result["n_trades"], result["position_ratio"],
    )
    return result
