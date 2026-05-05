"""
trainer.py — RL Model Training, ONNX Export, and Evaluation
============================================================
Orchestrates PPO and TD3 training for the HFT reinforcement-learning system.

Key capabilities
----------------
* 60 / 30 / 10 chronological train / val / test split
* SubprocVecEnv (PPO) and DummyVecEnv (TD3), both wrapped with VecNormalize
* EvalCallback (val set) + EarlyStoppingCallback + CheckpointCallback
* ONNX export with optional FP16 conversion via onnxconverter_common
* Annualised Sharpe computation for all supported timescales
* Optional progress_cb hook for API-server job tracking
* market='us' / 'hk' propagated to all env constructors
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Optional
from typing_extensions import TypedDict

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Stable-Baselines3 imports
# ---------------------------------------------------------------------------
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

# ---------------------------------------------------------------------------
# Project imports  — names must be exactly as exported by each module
# ---------------------------------------------------------------------------
from data_pipeline import load_dataset, SUPPORTED_TIMESCALES  # noqa: E402
from rl_environment import (  # noqa: E402
    HFTradingEnv,
    _ContinuousHFTEnv,
    make_envs,
    _get_feature_cols,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# TensorBoard logging is disabled unconditionally.
# The installed tensorboard package has a broken tensorflow.io shim that causes
# mid-training crashes.  Removing it has zero impact on training quality —
# SB3 trains identically without it.  Re-enable by installing standalone:
#   pip uninstall tensorboard && pip install tensorboard
_TENSORBOARD_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).parent
CHECKPOINT_DIR: Path = BASE_DIR / "checkpoints"
ONNX_DIR: Path = BASE_DIR / "onnx"
LOG_DIR: Path = BASE_DIR / "logs"

for _d in (CHECKPOINT_DIR, ONNX_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_WINDOW_SIZE: int = 60
DEFAULT_INITIAL_CAPITAL: float = 100_000.0
DEFAULT_TRANSACTION_COST: float = 0.001
DEFAULT_TOTAL_TIMESTEPS: int = 1_000_000
DEFAULT_N_ENVS: int = 4
DEFAULT_EVAL_FREQ: int = 10_000
DEFAULT_N_EVAL_EPISODES: int = 5
EARLY_STOP_PATIENCE: int = 5  # × eval_freq = 50 000 steps without improvement

# Market default tickers (mirrors data_pipeline.py)
US_TICKERS: list[str] = ["AAPL", "NVDA", "TSLA", "META", "GOOG", "MSFT", "AMZN"]
HK_TICKERS: list[str] = ["0700.HK", "9988.HK", "0005.HK", "2318.HK", "1299.HK"]


# ---------------------------------------------------------------------------
# EvalMetrics TypedDict
# ---------------------------------------------------------------------------


class EvalMetrics(TypedDict):
    sharpe: float
    max_drawdown: float
    win_rate: float
    total_return: float
    n_trades: int
    avg_pnl_per_trade: float


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class EarlyStoppingCallback(BaseCallback):
    """
    Stops training if the eval mean reward does not improve for
    ``patience`` evaluation rounds.

    This wraps SB3's built-in StopTrainingOnNoModelImprovement inside a
    lightweight outer callback so we can log the stopping event clearly.
    """

    def __init__(self, patience: int = EARLY_STOP_PATIENCE, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.patience = patience
        self._best_mean_reward: float = -np.inf
        self._no_improve_count: int = 0

    def _on_step(self) -> bool:
        # The parent EvalCallback sets locals_['mean_reward'] after each eval;
        # we read it from the parent via self.parent.
        if self.parent is None:
            return True
        # EvalCallback stores best_mean_reward on itself
        current_best = getattr(self.parent, "best_mean_reward", -np.inf)
        if current_best > self._best_mean_reward + 1e-6:
            self._best_mean_reward = current_best
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

        if self._no_improve_count >= self.patience:
            logger.info(
                "EarlyStoppingCallback: no improvement for %d eval rounds — stopping.",
                self.patience,
            )
            return False  # signals SB3 to stop training
        return True


class ProgressCallback(BaseCallback):
    """
    Calls an optional ``progress_cb(timestep, progress_fraction)`` hook
    so that an API server can track job progress in real time.
    """

    def __init__(
        self,
        total_timesteps: int,
        progress_cb: Optional[Callable[[int, float], None]] = None,
        report_interval: int = 10_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.total_timesteps = total_timesteps
        self.progress_cb = progress_cb
        self.report_interval = report_interval
        self._last_reported: int = 0

    def _on_step(self) -> bool:
        if self.progress_cb is None:
            return True
        step = self.num_timesteps
        if step - self._last_reported >= self.report_interval:
            fraction = min(1.0, step / max(1, self.total_timesteps))
            try:
                self.progress_cb(step, fraction)
            except Exception as exc:  # noqa: BLE001
                logger.warning("progress_cb raised: %s", exc)
            self._last_reported = step
        return True


# ---------------------------------------------------------------------------
# Data split helper
# ---------------------------------------------------------------------------


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.3,
    test_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological 3-way split: train / validation / test.

    Parameters
    ----------
    df:           Feature DataFrame (time-ordered; no shuffling applied).
    train_ratio:  Fraction of rows for training.
    val_ratio:    Fraction of rows for validation.
    test_ratio:   Fraction of rows for final held-out test.

    Returns
    -------
    (train_df, val_df, test_df) — all share the same columns as *df*.
    """
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {total:.6f}"
    )
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_df = df.iloc[:n_train].copy()
    val_df   = df.iloc[n_train : n_train + n_val].copy()
    test_df  = df.iloc[n_train + n_val :].copy()

    logger.info(
        "Data split — train: %d rows, val: %d rows, test: %d rows",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Annualisation factor helper
# ---------------------------------------------------------------------------


def _get_annualisation_factor(timescale: str) -> float:
    """
    Return the annualisation factor sqrt(T) for Sharpe computation.

    Derivations
    -----------
    10s : sqrt(252 * 6.5 hours/day * 3600 s/hour / 10 s/bar)
    1m  : sqrt(252 * 390 bars/day)
    5m  : sqrt(252 * 78 bars/day)
    1h  : sqrt(252 * 6.5 bars/day)
    """
    factors: dict[str, float] = {
        "10s": sqrt(252 * 6.5 * 3600 / 10),   # ≈ 2872
        "1m":  sqrt(252 * 390),                # ≈ 313
        "5m":  sqrt(252 * 78),                 # ≈ 140
        "1h":  sqrt(252 * 6.5),                # ≈ 40
    }
    if timescale not in factors:
        logger.warning(
            "Unknown timescale '%s' for annualisation — defaulting to 1m.", timescale
        )
        return factors["1m"]
    return factors[timescale]


# ---------------------------------------------------------------------------
# Model builder helpers
# ---------------------------------------------------------------------------


def _build_ppo(
    vec_env: VecNormalize,
    window_size: int,
    n_features: int,
    total_timesteps: int,
) -> PPO:
    """
    Instantiate a PPO model tuned for HFT environments.

    Policy architecture: two hidden layers of 256 units each.
    Hyperparameters are chosen for noisy high-frequency rewards:
    - large n_steps buffer amortises high step noise
    - moderate entropy coefficient encourages exploration
    - separate value and policy networks (pi/vf)
    """
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    )
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(LOG_DIR / "ppo_tensorboard") if _TENSORBOARD_OK else None,
        device="cpu",   # PPO+MLP is faster on CPU; GPU overhead dominates for small nets
        verbose=1,
    )
    logger.info("PPO model created — total_timesteps=%d", total_timesteps)
    return model


def _build_td3(
    vec_env: VecNormalize,
    window_size: int,
    n_features: int,
) -> TD3:
    """
    Instantiate a TD3 model for continuous HFT action spaces.

    Uses a twin-critic architecture with target policy smoothing.
    Buffer size is capped at 500 000 to stay within memory on a laptop GPU.
    """
    policy_kwargs = dict(
        net_arch=[256, 256],
        activation_fn=torch.nn.ReLU,
    )
    model = TD3(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-3,
        buffer_size=500_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=None,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(LOG_DIR / "td3_tensorboard") if _TENSORBOARD_OK else None,
        verbose=1,
    )
    logger.info("TD3 model created.")
    return model


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------


def train_model(
    market: str = "us",
    timescale: str = "10s",
    algorithm: str = "PPO",
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    n_envs: int = DEFAULT_N_ENVS,
    window_size: int = DEFAULT_WINDOW_SIZE,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    tickers: Optional[list[str]] = None,
    progress_cb: Optional[Callable[[int, float], None]] = None,
) -> str:
    """
    Train a PPO or TD3 model on HFT data.

    Parameters
    ----------
    market:          'us' or 'hk'
    timescale:       '10s', '1m', '5m', '1h'
    algorithm:       'PPO' or 'TD3'
    total_timesteps: Total environment steps to train for.
    n_envs:          Number of parallel envs (PPO only; TD3 always uses 1).
    window_size:     Observation look-back window (bars).
    initial_capital: Starting portfolio value.
    transaction_cost: Per-trade cost as fraction of notional.
    tickers:         List of tickers to include.  Uses market defaults if None.
    progress_cb:     Optional callback(timestep, fraction) for API progress.

    Returns
    -------
    Path to the saved final model (.zip).
    """
    algorithm = algorithm.upper()
    market    = market.lower()   # normalise: 'US' -> 'us', 'HK' -> 'hk'
    if algorithm not in ("PPO", "TD3"):
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose PPO or TD3.")
    if timescale not in SUPPORTED_TIMESCALES:
        raise ValueError(f"Unsupported timescale '{timescale}'. Choose from {SUPPORTED_TIMESCALES}.")

    # ── Resolve tickers ──────────────────────────────────────────────────────
    if tickers is None:
        tickers = US_TICKERS if market == "us" else HK_TICKERS

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("Loading dataset: market=%s timescale=%s tickers=%s", market, timescale, tickers)
    data_dict: dict[str, pd.DataFrame] = load_dataset(market, timescale)
    # Filter to requested tickers (use those that are actually available)
    available = {t: data_dict[t] for t in tickers if t in data_dict}
    if not available:
        raise ValueError(
            f"None of the requested tickers {tickers} found in dataset. "
            f"Available: {list(data_dict.keys())}"
        )
    logger.info("Using tickers: %s", list(available.keys()))

    # ── Use first ticker for single-ticker training ───────────────────────────
    primary_ticker = list(available.keys())[0]
    primary_df = available[primary_ticker]

    # ── 60/30/10 split ───────────────────────────────────────────────────────
    train_df, val_df, test_df = split_data(primary_df, 0.6, 0.3, 0.1)

    # ── Checkpoint directory ─────────────────────────────────────────────────
    run_id = f"{algorithm.lower()}_{market}_{timescale}_{int(time.time())}"
    ckpt_dir = CHECKPOINT_DIR / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Checkpoints → %s", ckpt_dir)

    # ── Determine n_features and obs_dim from training data ───────────────────
    n_features = len(_get_feature_cols(train_df))
    obs_dim = window_size * n_features + 1  # +1 for position scalar

    # ── Build training VecEnv ─────────────────────────────────────────────────
    env_kwargs = dict(
        window_size=window_size,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        market=market,
    )

    if algorithm == "PPO":
        train_data_dict = {primary_ticker: train_df}
        raw_vec_env = make_envs(
            data_dict=train_data_dict,
            n_envs=n_envs,
            market=market,
            window_size=window_size,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
        )
    else:  # TD3 — single env, continuous action space
        def _make_td3_env():
            return Monitor(
                _ContinuousHFTEnv(
                    df=train_df,
                    **env_kwargs,
                )
            )
        raw_vec_env = DummyVecEnv([_make_td3_env])

    vec_env = VecNormalize(
        raw_vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    # ── Build eval VecEnv (val set) ───────────────────────────────────────────
    if algorithm == "PPO":
        def _make_eval_env():
            return Monitor(
                HFTradingEnv(
                    df=val_df,
                    **env_kwargs,
                )
            )
    else:
        def _make_eval_env():
            return Monitor(
                _ContinuousHFTEnv(
                    df=val_df,
                    **env_kwargs,
                )
            )

    raw_eval_vec = DummyVecEnv([_make_eval_env])
    eval_vec_env = VecNormalize(
        raw_eval_vec,
        norm_obs=True,
        norm_reward=False,  # do not normalise reward during evaluation
        clip_obs=10.0,
        training=False,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    early_stopping_cb = EarlyStoppingCallback(
        patience=EARLY_STOP_PATIENCE,
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env=eval_vec_env,
        best_model_save_path=str(ckpt_dir),
        log_path=str(ckpt_dir / "eval_logs"),
        eval_freq=max(DEFAULT_EVAL_FREQ // n_envs, 1),
        n_eval_episodes=DEFAULT_N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        callback_after_eval=early_stopping_cb,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=str(ckpt_dir / "periodic"),
        name_prefix=algorithm.lower(),
        save_vecnormalize=True,
        verbose=1,
    )

    progress_cb_obj = ProgressCallback(
        total_timesteps=total_timesteps,
        progress_cb=progress_cb,
        report_interval=10_000,
        verbose=0,
    )

    callbacks = CallbackList([eval_cb, checkpoint_cb, progress_cb_obj])

    # ── Instantiate model ──────────────────────────────────────────────────────
    if algorithm == "PPO":
        model = _build_ppo(
            vec_env=vec_env,
            window_size=window_size,
            n_features=n_features,
            total_timesteps=total_timesteps,
        )
    else:
        model = _build_td3(
            vec_env=vec_env,
            window_size=window_size,
            n_features=n_features,
        )

    # ── Train ──────────────────────────────────────────────────────────────────
    logger.info("Starting training — %s for %d timesteps.", algorithm, total_timesteps)
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=True,
    )

    # ── Save final model + VecNormalize stats ──────────────────────────────────
    final_model_path = str(ckpt_dir / f"{algorithm.lower()}_final")
    model.save(final_model_path)
    vec_env.save(str(ckpt_dir / "vecnorm.pkl"))
    logger.info("Saved model → %s.zip", final_model_path)
    logger.info("Saved VecNormalize stats → %s/vecnorm.pkl", ckpt_dir)

    # Close environments
    vec_env.close()
    eval_vec_env.close()

    return final_model_path + ".zip"


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


class _PolicyWrapper(torch.nn.Module):
    """Thin wrapper exposing only the deterministic action head for ONNX export."""

    def __init__(self, sb3_model: Any) -> None:
        super().__init__()
        self.policy = sb3_model.policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the deterministic action (argmax for discrete, mean for continuous)."""
        with torch.no_grad():
            dist = self.policy.get_distribution(obs)
            if hasattr(dist.distribution, "probs"):
                # Discrete policy — return argmax
                return dist.distribution.probs.argmax(dim=-1)
            else:
                # Continuous policy — return mean
                return dist.distribution.mean


def export_to_onnx(
    model_path: str,
    output_path: str,
    obs_dim: int,
    algorithm: str = "PPO",
) -> str:
    """
    Export a trained SB3 model to ONNX format.

    obs_dim must already include the +1 position scalar appended by the
    environment (i.e. window_size * n_features + 1).

    Attempts FP16 conversion via onnxconverter_common; falls back to FP32
    gracefully if the package is not installed or conversion fails.

    Parameters
    ----------
    model_path:   Path to the saved .zip model (or path without extension).
    output_path:  Destination .onnx file path.
    obs_dim:      Flat observation size including position scalar.
    algorithm:    'PPO' or 'TD3'.

    Returns
    -------
    Path to the written ONNX file (may be FP16 or FP32 depending on support).
    """
    algorithm = algorithm.upper()
    market    = market.lower()   # normalise: 'US' -> 'us', 'HK' -> 'hk'
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # ── Load model on CPU ─────────────────────────────────────────────────────
    logger.info("Loading %s model from %s for ONNX export.", algorithm, model_path)
    ModelClass = PPO if algorithm == "PPO" else TD3
    model = ModelClass.load(model_path, device="cpu")
    model.policy.set_training_mode(False)

    # ── Build wrapper + dummy input ───────────────────────────────────────────
    wrapper = _PolicyWrapper(model)
    wrapper.eval()

    dummy_input = torch.zeros((1, obs_dim), dtype=torch.float32)

    fp32_path = str(output_path_obj.with_suffix(".fp32.onnx"))

    # ── Export FP32 ───────────────────────────────────────────────────────────
    logger.info("Exporting to ONNX (opset 17, FP32) → %s", fp32_path)
    torch.onnx.export(
        wrapper,
        dummy_input,
        fp32_path,
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch_size"}, "action": {0: "batch_size"}},
        do_constant_folding=True,
    )

    # ── Verify FP32 with onnxruntime ──────────────────────────────────────────
    _verify_onnx(fp32_path, obs_dim)

    # ── Attempt FP16 conversion ───────────────────────────────────────────────
    final_path = fp32_path
    try:
        from onnxconverter_common import convert_float_to_float16
        import onnx

        fp16_path = str(output_path_obj)
        logger.info("Converting to FP16 → %s", fp16_path)
        fp32_model = onnx.load(fp32_path)
        fp16_model = convert_float_to_float16(fp32_model)
        onnx.save(fp16_model, fp16_path)
        _verify_onnx(fp16_path, obs_dim, dtype=np.float16)
        logger.info("FP16 ONNX model verified → %s", fp16_path)
        final_path = fp16_path
    except ImportError:
        logger.warning(
            "onnxconverter_common not installed — keeping FP32 ONNX model at %s.", fp32_path
        )
        final_path = fp32_path
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "FP16 conversion failed (%s) — keeping FP32 ONNX model at %s.", exc, fp32_path
        )
        final_path = fp32_path

    logger.info("ONNX export complete → %s", final_path)
    return final_path


def _verify_onnx(onnx_path: str, obs_dim: int, dtype=np.float32) -> None:
    """Run a single forward pass through the ONNX model to verify it loads cleanly."""
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy = np.zeros((1, obs_dim), dtype=dtype)
        outputs = sess.run(None, {"obs": dummy})
        logger.info("ONNX verification passed — output shape: %s", [o.shape for o in outputs])
    except Exception as exc:  # noqa: BLE001
        logger.error("ONNX verification failed for %s: %s", onnx_path, exc)
        raise


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model_path: str,
    test_data: pd.DataFrame,
    algorithm: str = "PPO",
    window_size: int = DEFAULT_WINDOW_SIZE,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    n_eval_episodes: int = DEFAULT_N_EVAL_EPISODES,
    market: str = "us",
    timescale: str = "10s",
    vecnorm_path: Optional[str] = None,
) -> EvalMetrics:
    """
    Evaluate a trained model on held-out test data and return EvalMetrics.

    Parameters
    ----------
    model_path:        Path to saved SB3 model (.zip).
    test_data:         Held-out test DataFrame (output of split_data).
    algorithm:         'PPO' or 'TD3'.
    window_size:       Observation look-back window (bars).
    initial_capital:   Portfolio starting value.
    transaction_cost:  Per-trade cost as fraction of notional.
    n_eval_episodes:   Number of evaluation rollouts to average over.
    market:            'us' or 'hk' — passed to env constructor.
    timescale:         '10s', '1m', '5m', '1h' — used for Sharpe annualisation.
    vecnorm_path:      Optional path to vecnorm.pkl; loads normalisation stats.

    Returns
    -------
    EvalMetrics TypedDict with sharpe, max_drawdown, win_rate, total_return,
    n_trades, avg_pnl_per_trade.
    """
    algorithm = algorithm.upper()
    ModelClass = PPO if algorithm == "PPO" else TD3

    logger.info(
        "Evaluating %s model: %s | market=%s timescale=%s n_episodes=%d",
        algorithm, model_path, market, timescale, n_eval_episodes,
    )

    annualisation = _get_annualisation_factor(timescale)

    env_kwargs: dict[str, Any] = dict(
        window_size=window_size,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        market=market,
    )

    EnvClass = _ContinuousHFTEnv if algorithm == "TD3" else HFTradingEnv

    def _make_eval_env():
        return Monitor(EnvClass(df=test_data, **env_kwargs))

    raw_vec = DummyVecEnv([_make_eval_env])

    # Load VecNormalize stats for consistent normalisation
    if vecnorm_path and Path(vecnorm_path).exists():
        vec_env = VecNormalize.load(vecnorm_path, raw_vec)
        vec_env.training = False
        vec_env.norm_reward = False
        logger.info("Loaded VecNormalize stats from %s", vecnorm_path)
    else:
        vec_env = VecNormalize(
            raw_vec,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            training=False,
        )
        if vecnorm_path:
            logger.warning("VecNormalize path not found: %s — using fresh stats.", vecnorm_path)

    model = ModelClass.load(model_path, env=vec_env)

    # ── Run evaluation rollouts ───────────────────────────────────────────────
    all_portfolio_histories: list[list[float]] = []
    all_n_trades: list[int] = []
    all_realized_pnl: list[float] = []
    episode_total_returns: list[float] = []
    wins: list[bool] = []

    for episode in range(n_eval_episodes):
        obs = vec_env.reset()
        done = False
        portfolio_history: list[float] = [float(initial_capital)]
        ep_n_trades = 0
        ep_realized_pnl = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_arr = vec_env.step(action)
            done = bool(done_arr[0])

            # info_arr[0] may be a dict (Monitor) or a StepInfo-derived dict
            raw_info = (info_arr[0] if (info_arr is not None and len(info_arr) > 0) else {})
            # Handle both dict and object (Monitor wraps info into a dict)
            if hasattr(raw_info, "get"):
                info = raw_info
            elif hasattr(raw_info, "__dict__"):
                info = raw_info.__dict__
            else:
                info = {}

            pv = info.get("portfolio_value", None)
            if pv is not None and np.isfinite(float(pv)):
                portfolio_history.append(float(pv))
            ep_n_trades = int(info.get("n_trades", ep_n_trades))
            rpnl = info.get("realized_pnl", None)
            if rpnl is not None and np.isfinite(float(rpnl)):
                ep_realized_pnl = float(rpnl)

        # ── Per-episode metrics ──────────────────────────────────────────────
        final_pv = portfolio_history[-1] if len(portfolio_history) > 1 else initial_capital
        ep_total_return = (final_pv - initial_capital) / initial_capital
        if not np.isfinite(ep_total_return):
            ep_total_return = 0.0
        episode_total_returns.append(ep_total_return)
        all_portfolio_histories.append(portfolio_history)
        all_n_trades.append(ep_n_trades)
        all_realized_pnl.append(ep_realized_pnl)
        # Win = episode ended with positive realized PnL (not total_return which can
        # include unrealized marks that never close).
        wins.append(ep_realized_pnl > 0.0)

        logger.debug(
            "Episode %d/%d — total_return=%.4f  realized_pnl=%.4f  n_trades=%d  "
            "portfolio_start=%.2f  portfolio_end=%.2f",
            episode + 1, n_eval_episodes,
            ep_total_return, ep_realized_pnl, ep_n_trades,
            portfolio_history[0], portfolio_history[-1],
        )

    vec_env.close()

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    # Sharpe: computed from per-bar portfolio *return* (not RL reward scalars)
    # Concatenate all bar-level portfolio values, compute bar returns, then
    # annualise.  This is the correct financial definition.
    all_bar_returns: list[float] = []
    for hist in all_portfolio_histories:
        hist_arr = np.array(hist, dtype=np.float64)
        if len(hist_arr) < 2:
            continue
        bar_rets = np.diff(hist_arr) / (hist_arr[:-1] + 1e-8)  # % change per bar
        # Remove any non-finite values (price gaps, first-step anomalies)
        bar_rets = bar_rets[np.isfinite(bar_rets)]
        all_bar_returns.extend(bar_rets.tolist())

    if len(all_bar_returns) > 1:
        ret_arr  = np.array(all_bar_returns, dtype=np.float64)
        mean_ret = float(np.nanmean(ret_arr))
        std_ret  = float(np.nanstd(ret_arr))
        if std_ret < 1e-10:
            sharpe = 0.0  # zero-volatility: undefined, treat as 0
        else:
            sharpe = (mean_ret / std_ret) * annualisation
        # Clamp to a plausible range — anything > 10 is suspicious
        sharpe = float(np.clip(sharpe, -50.0, 50.0))
    else:
        sharpe = 0.0

    # Max drawdown across all episodes (worst single episode)
    max_drawdown = 0.0
    for hist in all_portfolio_histories:
        hist_arr = np.array(hist, dtype=np.float64)
        if len(hist_arr) < 2:
            continue
        running_max = np.maximum.accumulate(hist_arr)
        # Avoid divide-by-zero when running_max is 0 (degenerate)
        denom = np.where(running_max > 1e-8, running_max, 1e-8)
        drawdowns = (running_max - hist_arr) / denom
        ep_dd = float(np.nanmax(drawdowns))
        if np.isfinite(ep_dd) and ep_dd > max_drawdown:
            max_drawdown = ep_dd

    win_rate   = float(np.mean(wins)) if wins else 0.0
    ep_returns = [r for r in episode_total_returns if np.isfinite(r)]
    total_return = float(np.mean(ep_returns)) if ep_returns else 0.0
    total_trades = int(np.sum(all_n_trades))
    total_pnl    = float(np.nansum(all_realized_pnl))
    avg_pnl_per_trade = total_pnl / max(total_trades, 1)

    metrics: EvalMetrics = {
        "sharpe":            sharpe,
        "max_drawdown":      max_drawdown,
        "win_rate":          win_rate,
        "total_return":      total_return,
        "n_trades":          total_trades,
        "avg_pnl_per_trade": avg_pnl_per_trade,
    }

    logger.info(
        "Evaluation results — Sharpe: %.3f | MaxDD: %.3f | WinRate: %.2f%% | "
        "TotalReturn: %.4f | nTrades: %d | AvgPnL/Trade: %.4f",
        sharpe, max_drawdown, win_rate * 100, total_return, total_trades, avg_pnl_per_trade,
    )
    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HFT RL Trainer — train, export, and evaluate PPO/TD3 agents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--market",
        choices=["us", "hk", "both"],
        default="us",
        help="Market(s) to train on.",
    )
    parser.add_argument(
        "--timescale",
        choices=["10s", "1m", "5m", "1h", "all"],
        default="10s",
        help="Bar timescale(s) to use.",
    )
    parser.add_argument(
        "--algo",
        choices=["PPO", "TD3", "both"],
        default="PPO",
        help="RL algorithm(s) to train.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help="Total environment timesteps for training.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=DEFAULT_N_ENVS,
        help="Number of parallel environments (PPO only).",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export trained model(s) to ONNX after training.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Observation look-back window in bars.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Tickers to train on.  Uses market defaults if not specified.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=DEFAULT_INITIAL_CAPITAL,
        help="Starting portfolio value.",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=DEFAULT_TRANSACTION_COST,
        help="Per-trade transaction cost as fraction of notional.",
    )

    args = parser.parse_args()

    # Expand 'both' and 'all' to lists
    markets    = ["us", "hk"]              if args.market    == "both" else [args.market]
    timescales = SUPPORTED_TIMESCALES      if args.timescale == "all"  else [args.timescale]
    algorithms = ["PPO", "TD3"]            if args.algo      == "both" else [args.algo]

    for market in markets:
        tickers = args.tickers  # None → market defaults inside train_model
        for timescale in timescales:
            for algo in algorithms:
                logger.info(
                    "=== Training %s | market=%s | timescale=%s ===",
                    algo, market, timescale,
                )
                try:
                    model_path = train_model(
                        market=market,
                        timescale=timescale,
                        algorithm=algo,
                        total_timesteps=args.timesteps,
                        n_envs=args.n_envs,
                        window_size=args.window_size,
                        initial_capital=args.initial_capital,
                        transaction_cost=args.transaction_cost,
                        tickers=tickers,
                    )
                    logger.info("Model saved to: %s", model_path)

                    if args.export_onnx:
                        # Determine obs_dim: load data to get n_features
                        data_dict = load_dataset(market, timescale)
                        resolved_tickers = tickers or (US_TICKERS if market == "us" else HK_TICKERS)
                        available = {t: data_dict[t] for t in resolved_tickers if t in data_dict}
                        primary_df = list(available.values())[0]
                        n_features = len(_get_feature_cols(primary_df))
                        obs_dim = args.window_size * n_features + 1

                        onnx_out = str(
                            ONNX_DIR / f"{algo.lower()}_{market}_{timescale}.onnx"
                        )
                        export_path = export_to_onnx(
                            model_path=model_path,
                            output_path=onnx_out,
                            obs_dim=obs_dim,
                            algorithm=algo,
                        )
                        logger.info("ONNX model saved to: %s", export_path)

                    # ── Final evaluation on test set ───────────────────────────
                    data_dict = load_dataset(market, timescale)
                    resolved_tickers = tickers or (US_TICKERS if market == "us" else HK_TICKERS)
                    available = {t: data_dict[t] for t in resolved_tickers if t in data_dict}
                    primary_df = list(available.values())[0]
                    _, _, test_df = split_data(primary_df, 0.6, 0.3, 0.1)

                    # Locate vecnorm.pkl from checkpoint dir
                    # model_path is like ".../checkpoints/<run_id>/ppo_final.zip"
                    ckpt_dir = Path(model_path).parent
                    vecnorm_path = str(ckpt_dir / "vecnorm.pkl")

                    metrics = evaluate_model(
                        model_path=model_path,
                        test_data=test_df,
                        algorithm=algo,
                        window_size=args.window_size,
                        initial_capital=args.initial_capital,
                        transaction_cost=args.transaction_cost,
                        market=market,
                        timescale=timescale,
                        vecnorm_path=vecnorm_path,
                    )
                    logger.info(
                        "Final test metrics for %s/%s/%s:\n"
                        "  Sharpe:           %.4f\n"
                        "  Max Drawdown:     %.4f\n"
                        "  Win Rate:         %.2f%%\n"
                        "  Total Return:     %.4f\n"
                        "  Trades:           %d\n"
                        "  Avg PnL / Trade:  %.4f",
                        market, timescale, algo,
                        metrics["sharpe"],
                        metrics["max_drawdown"],
                        metrics["win_rate"] * 100,
                        metrics["total_return"],
                        metrics["n_trades"],
                        metrics["avg_pnl_per_trade"],
                    )

                except Exception as exc:
                    logger.error(
                        "Training failed for %s/%s/%s: %s",
                        market, timescale, algo, exc,
                        exc_info=True,
                    )


if __name__ == "__main__":
    main()
