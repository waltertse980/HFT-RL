"""
trainer.py — RL Model Training, ONNX Export, and Evaluation

Supports PPO and TD3 from Stable-Baselines3.
Exports trained policy networks to ONNX FP16 (opset 17).
Provides evaluation metrics on held-out test data.

Usage
-----
    python trainer.py --market us --timescale 1m --algo PPO --timesteps 1000000
    python trainer.py --market hk --timescale 10s --algo TD3 --timesteps 500000
"""

from __future__ import annotations

import argparse
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict

import numpy as np
import pandas as pd
import torch
import onnx
import onnxruntime as ort
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from data_pipeline import load_dataset
from rl_environment import HFTradingEnv, make_envs

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Training device: %s", DEVICE)


# ---------------------------------------------------------------------------
# Typed return dicts
# ---------------------------------------------------------------------------

class EvalMetrics(TypedDict):
    sharpe: float
    max_drawdown: float
    win_rate: float
    total_return: float
    n_trades: int
    avg_pnl_per_trade: float


# ---------------------------------------------------------------------------
# Early-stopping callback
# ---------------------------------------------------------------------------

class EarlyStoppingCallback(BaseCallback):
    """
    Stop training if mean reward has not improved by ``min_delta``
    over the last ``patience`` calls (each call = every ``check_freq`` steps).
    """

    def __init__(
        self,
        check_freq: int = 10_000,
        patience: int = 20,
        min_delta: float = 1e-4,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.min_delta = min_delta
        self._best_mean_reward = -np.inf
        self._no_improve_count = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        # ep_rew_mean is logged by SB3 Monitor wrapper
        if len(self.model.ep_info_buffer) == 0:
            return True

        mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
        if mean_reward > self._best_mean_reward + self.min_delta:
            self._best_mean_reward = mean_reward
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

        if self._no_improve_count >= self.patience:
            if self.verbose:
                logger.info(
                    "Early stopping triggered: no improvement for %d checks "
                    "(≈%d steps). Best mean reward: %.4f",
                    self.patience,
                    self.patience * self.check_freq,
                    self._best_mean_reward,
                )
            return False  # stops training

        return True


# ---------------------------------------------------------------------------
# Data splitting helper
# ---------------------------------------------------------------------------

def _train_test_split(
    data_dict: dict[str, pd.DataFrame],
    test_ratio: float = 0.2,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    train, test = {}, {}
    for ticker, df in data_dict.items():
        n = len(df)
        split = int(n * (1 - test_ratio))
        train[ticker] = df.iloc[:split].copy()
        test[ticker] = df.iloc[split:].copy()
    return train, test


# ---------------------------------------------------------------------------
# PPO / TD3 configuration builders
# ---------------------------------------------------------------------------

def _build_ppo(
    env,
    tensorboard_log: str,
    learning_rate: float = 3e-4,
) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=8192,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        target_kl=0.015,
        verbose=1,
        tensorboard_log=tensorboard_log,
        device=DEVICE,
    )

def _build_td3(
    env,
    tensorboard_log: str,
    learning_rate: float = 1e-3,
) -> TD3:
    return TD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        policy_delay=2,
        verbose=1,
        tensorboard_log=tensorboard_log,
        device=DEVICE,
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model(
    market: str,
    timescale: str,
    algorithm: str = "PPO",
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    window_size: int = 60,
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.0001,
) -> str:
    run_name = f"{market}_{timescale}_{algorithm}"
    tb_log_dir = str(LOGS_DIR / run_name)
    ckpt_dir = MODELS_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset: market=%s timescale=%s", market, timescale)
    data_dict = load_dataset(market, timescale)
    train_data, test_data = _train_test_split(data_dict, test_ratio=0.2)

    logger.info("Creating training environments (n_envs=%d)...", n_envs)

    ticker = next(iter(train_data.keys()))
    df_train = train_data[ticker]

    if algorithm == "PPO":
        def _env_fn():
            env = HFTradingEnv(
                df=df_train,
                window_size=window_size,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
            )
            return Monitor(env)

        env_fns = [_env_fn for _ in range(n_envs)]
        vec_env = SubprocVecEnv(env_fns)
        
        # Apply normalization to stabilize rewards and observations for PPO
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_reward=10.0)
        model = _build_ppo(vec_env, tb_log_dir)

    elif algorithm == "TD3":
        logger.warning(
            "TD3 requires a continuous action space. Using single-env DummyVecEnv. "
            "Consider using PPO for better parallel scaling."
        )

        class _ContinuousHFTEnv(HFTradingEnv):
            """Thin wrapper converting Discrete(3) to Box([0,3)) for TD3."""
            import gymnasium as _gym

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                import gymnasium.spaces as _spc
                self.action_space = _spc.Box(low=0.0, high=3.0, shape=(1,), dtype=np.float32)

            def step(self, action):
                discrete_action = int(np.clip(action[0], 0, 2.999))
                return super().step(discrete_action)

        def _td3_env_fn():
            env = _ContinuousHFTEnv(
                df=df_train,
                window_size=window_size,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
            )
            return Monitor(env)

        vec_env = DummyVecEnv([_td3_env_fn])
        
        # Apply normalization to TD3 as well
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_reward=10.0)
        model = _build_td3(vec_env, tb_log_dir)

    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose 'PPO' or 'TD3'.")

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix="model",
    )
    early_stop_cb = EarlyStoppingCallback(
        check_freq=10_000,
        patience=20,
        verbose=1,
    )
    callbacks = [checkpoint_cb, early_stop_cb]

    logger.info("Starting training: %s  timesteps=%d", run_name, total_timesteps)
    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=run_name,
        reset_num_timesteps=True,
        progress_bar=True,
    )
    elapsed = time.time() - t0
    logger.info("Training complete in %.1f s", elapsed)

    final_path = str(ckpt_dir / "model_final")
    model.save(final_path)
    
    # Save the VecNormalize statistics
    vec_norm_path = str(ckpt_dir / "vec_normalize.pkl")
    vec_env.save(vec_norm_path)
    
    logger.info("Model saved → %s.zip", final_path)
    logger.info("Normalization stats saved → %s", vec_norm_path)
    vec_env.close()
    
    return final_path + ".zip"


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(
    model_path: str,
    output_path: str,
    obs_dim: int,
    algorithm: str = "PPO",
) -> str:
    """
    Export a Stable-Baselines3 model's policy network to ONNX FP16 (opset 17).
    Note: The exported ONNX model expects *normalized* observations if trained 
    with VecNormalize. In production, you must scale inputs identically.
    """
    logger.info("Loading model from %s", model_path)
    ModelClass = PPO if algorithm == "PPO" else TD3
    model = ModelClass.load(model_path, device="cpu")

    policy = model.policy
    policy.eval()

    dummy_input = torch.zeros(1, obs_dim, dtype=torch.float32)
    output_path_ = Path(output_path)
    output_path_.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting to ONNX (opset 17)...")
    buffer = io.BytesIO()
    torch.onnx.export(
        policy,
        dummy_input,
        buffer,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action_logits"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action_logits": {0: "batch_size"},
        },
    )

    buffer.seek(0)
    onnx_model = onnx.load_model(buffer)

    from onnxconverter_common import float16  # type: ignore
    try:
        onnx_model_fp16 = float16.convert_float_to_float16(onnx_model)
        onnx.save(onnx_model_fp16, str(output_path_))
        logger.info("Saved FP16 ONNX model → %s", output_path_)
    except ImportError:
        logger.warning("onnxconverter_common not found; saving FP32 ONNX model.")
        onnx.save(onnx_model, str(output_path_))

    logger.info("Verifying ONNX export with onnxruntime...")
    try:
        sess = ort.InferenceSession(str(output_path_), providers=["CPUExecutionProvider"])
        dummy_np = np.zeros((1, obs_dim), dtype=np.float32)
        outputs = sess.run(None, {"observation": dummy_np})
        logger.info("ONNX verification passed. Output shape: %s", outputs[0].shape)
    except Exception as exc:  # noqa: BLE001
        logger.error("ONNX verification failed: %s", exc)

    return str(output_path_)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_path: str,
    test_data: pd.DataFrame,
    algorithm: str = "PPO",
    window_size: int = 60,
    initial_capital: float = 100_000.0,
    n_eval_episodes: int = 5,
) -> EvalMetrics:
    
    model_dir = Path(model_path).parent
    vec_norm_path = model_dir / "vec_normalize.pkl"

    ModelClass = PPO if algorithm == "PPO" else TD3
    model = ModelClass.load(model_path, device="cpu")

    all_returns: list[float] = []
    all_n_trades: list[int] = []
    all_pnls: list[float] = []
    all_equity_curves: list[list[float]] = []

    for ep in range(n_eval_episodes):
        def _eval_env_fn():
            env = HFTradingEnv(
                df=test_data,
                window_size=window_size,
                initial_capital=initial_capital,
                transaction_cost=0.001,
            )
            return Monitor(env)

        eval_env = DummyVecEnv([_eval_env_fn])
        eval_env.seed(ep * 42)

        # Load normalization statistics to ensure inputs match training conditions
        if vec_norm_path.exists():
            eval_env = VecNormalize.load(str(vec_norm_path), eval_env)
            # Disable updates and reward scaling for pure evaluation
            eval_env.training = False
            eval_env.norm_reward = False

        obs = eval_env.reset()
        done = False
        equity_curve = [initial_capital]
        final_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            
            done = dones[0]
            info = infos[0]

            if "portfolio_value" in info:
                equity_curve.append(info["portfolio_value"])
            if done:
                final_info = info

        final_value = equity_curve[-1]
        total_return = (final_value - initial_capital) / initial_capital
        all_returns.append(total_return)
        all_n_trades.append(final_info.get("n_trades", 0))
        all_pnls.append(final_info.get("realized_pnl", 0.0))
        all_equity_curves.append(equity_curve)
        
        eval_env.close()

    # Aggregate metrics
    avg_return = float(np.mean(all_returns))
    avg_n_trades = int(np.mean(all_n_trades))
    avg_pnl = float(np.mean(all_pnls))
    avg_pnl_per_trade = avg_pnl / max(avg_n_trades, 1)

    longest_curve = max(all_equity_curves, key=len)
    step_returns = np.diff(longest_curve) / (np.array(longest_curve[:-1]) + 1e-8)
    sharpe = float(
        np.mean(step_returns) / (np.std(step_returns) + 1e-8) * np.sqrt(252 * 6.5 * 360)
        if len(step_returns) > 1 else 0.0
    )

    curve_arr = np.array(longest_curve)
    running_max = np.maximum.accumulate(curve_arr)
    drawdowns = (running_max - curve_arr) / (running_max + 1e-8)
    max_drawdown = float(np.max(drawdowns))
    win_rate = float(np.mean([r > 0 for r in all_returns]))

    metrics: EvalMetrics = {
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_return": avg_return,
        "n_trades": avg_n_trades,
        "avg_pnl_per_trade": avg_pnl_per_trade,
    }
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="HFT RL Trainer")
    parser.add_argument("--market", choices=["us", "hk", "both"], default="us")
    parser.add_argument(
        "--timescale", choices=["10s", "1m", "5m", "1h", "all"], default="1m"
    )
    parser.add_argument("--algo", choices=["PPO", "TD3", "both"], default="PPO")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument(
        "--export-onnx", action="store_true", help="Export to ONNX after training"
    )
    parser.add_argument("--window-size", type=int, default=60)
    args = parser.parse_args()

    markets = ["us", "hk"] if args.market == "both" else [args.market]
    timescales = ["10s", "1m", "5m", "1h"] if args.timescale == "all" else [args.timescale]
    algos = ["PPO", "TD3"] if args.algo == "both" else [args.algo]

    for market in markets:
        for timescale in timescales:
            for algo in algos:
                logger.info("=" * 60)
                logger.info("Training: market=%s timescale=%s algo=%s", market, timescale, algo)
                logger.info("=" * 60)
                try:
                    model_path = train_model(
                        market=market,
                        timescale=timescale,
                        algorithm=algo,
                        total_timesteps=args.timesteps,
                        n_envs=args.n_envs,
                        window_size=args.window_size,
                    )
                    logger.info("Saved model: %s", model_path)

                    if args.export_onnx:
                        data_dict = load_dataset(market, timescale)
                        ticker = next(iter(data_dict.keys()))
                        df = data_dict[ticker]
                        from rl_environment import _get_feature_cols
                        n_features = len(_get_feature_cols(df))
                        obs_dim = args.window_size * n_features

                        onnx_path = model_path.replace(".zip", ".onnx")
                        export_to_onnx(
                            model_path=model_path,
                            output_path=onnx_path,
                            obs_dim=obs_dim,
                            algorithm=algo,
                        )

                    # Quick evaluation
                    data_dict = load_dataset(market, timescale)
                    ticker = next(iter(data_dict.keys()))
                    df = data_dict[ticker]
                    n = len(df)
                    test_df = df.iloc[int(n * 0.8):]
                    metrics = evaluate_model(
                        model_path=model_path,
                        test_data=test_df,
                        algorithm=algo,
                        window_size=args.window_size,
                    )
                    logger.info("Eval metrics: %s", metrics)

                except Exception as exc:  # noqa: BLE001
                    logger.error("Training failed for %s/%s/%s: %s", market, timescale, algo, exc)

if __name__ == "__main__":
    main()