"""
hpo_optuna.py — Hyperparameter Optimisation for HFT RL Trading System
======================================================================
Uses Optuna TPE sampler + MedianPruner to find optimal hyperparameters
for PPO/TD3 training on HFT environments.

Usage
-----
    python hpo_optuna.py --market us --timescale 1m --algo PPO --trials 50

For persistent storage (enables parallel workers and resume):
    python hpo_optuna.py --storage sqlite:///hpo.db --n-jobs 4
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from data_pipeline import load_dataset, SUPPORTED_TIMESCALES
from rl_environment import HFTradingEnv, _ContinuousHFTEnv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress verbose Optuna/SB3 output during trials
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
HPO_DIR = BASE_DIR / "hpo_results"
HPO_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_N_TRIALS = 50
DEFAULT_N_TIMESTEPS_PER_TRIAL = 200_000
DEFAULT_N_EVAL_EPISODES = 5

# Net architecture map — shared across PPO and TD3 helpers
_NET_ARCH_MAP: dict[str, list[int]] = {
    "small":  [64, 64],
    "medium": [256, 256],
    "large":  [512, 256, 128],
}

# ---------------------------------------------------------------------------
# TrialPruningCallback
# ---------------------------------------------------------------------------


class TrialPruningCallback(BaseCallback):
    """
    Reports intermediate mean episode reward to an Optuna trial so the
    MedianPruner can prune unpromising trials early.

    Parameters
    ----------
    trial:     The active Optuna trial object.
    eval_freq: How many training steps between each report to Optuna.
    verbose:   SB3 verbosity level.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        eval_freq: int = 10_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.trial = trial
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # ep_info_buffer is a deque of dicts with keys 'r', 'l', 't'
            ep_info_buffer = self.model.ep_info_buffer
            if ep_info_buffer:
                mean_reward = float(np.mean([ep["r"] for ep in ep_info_buffer]))
            else:
                mean_reward = -np.inf

            self.trial.report(mean_reward, step=self.n_calls)

            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return True


# ---------------------------------------------------------------------------
# Hyperparameter samplers
# ---------------------------------------------------------------------------


def _sample_ppo_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    Sample PPO hyperparameters from the Optuna trial search space.

    Returns a dict ready to be unpacked into ``PPO(...)``.  The ``net_arch``
    key holds the actual list-of-ints architecture (not the string tag), and
    a ``policy_kwargs`` key is included for convenience.
    """
    net_arch_tag: str = trial.suggest_categorical(
        "net_arch", ["small", "medium", "large"]
    )
    arch: list[int] = _NET_ARCH_MAP[net_arch_tag]
    net_arch_dict = dict(pi=arch, vf=arch)

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "n_epochs": trial.suggest_int("n_epochs", 3, 20),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
        # Stored as string for Optuna parameter logging; actual arch below
        "net_arch": net_arch_tag,
        # Resolved arch stored separately for model construction
        "_net_arch_list": net_arch_dict,
    }


def _sample_td3_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    Sample TD3 hyperparameters from the Optuna trial search space.

    Returns a dict ready to be unpacked into ``TD3(...)``.
    """
    net_arch_tag: str = trial.suggest_categorical(
        "net_arch", ["small", "medium", "large"]
    )
    arch: list[int] = _NET_ARCH_MAP[net_arch_tag]

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_categorical(
            "buffer_size", [100_000, 500_000, 1_000_000]
        ),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "tau": trial.suggest_float("tau", 0.001, 0.02),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "policy_delay": trial.suggest_int("policy_delay", 1, 4),
        "target_policy_noise": trial.suggest_float("target_policy_noise", 0.1, 0.5),
        "target_noise_clip": trial.suggest_float("target_noise_clip", 0.3, 1.0),
        # Stored as string for Optuna parameter logging; actual arch below
        "net_arch": net_arch_tag,
        # Resolved arch stored separately for model construction
        "_net_arch_list": arch,
    }


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------


def _make_objective(
    market: str,
    timescale: str,
    algorithm: str,
    n_timesteps: int,
    n_eval_episodes: int,
    device: str,
) -> Callable[[optuna.Trial], float]:
    """
    Returns a closure ``objective(trial) -> float`` compatible with
    ``study.optimize()``.

    The objective:
    1. Loads the dataset and picks the first ticker.
    2. Splits chronologically: 70% train / 15% val / 15% test.
    3. Samples hyperparameters according to ``algorithm``.
    4. Builds a vectorised, normalised environment.
    5. Trains for ``n_timesteps`` steps with pruning callback.
    6. Evaluates on the val split for ``n_eval_episodes`` episodes.
    7. Cleans up all objects to prevent memory leaks.
    8. Returns mean validation reward (to be maximised).
    """
    algo_upper = algorithm.upper()

    def objective(trial: optuna.Trial) -> float:
        train_env = None
        val_env = None
        model = None

        try:
            # ------------------------------------------------------------------
            # 1. Load dataset & pick ticker
            # ------------------------------------------------------------------
            data_dict = load_dataset(market, timescale)
            ticker = sorted(data_dict.keys())[0]
            df = data_dict[ticker].copy()

            # ------------------------------------------------------------------
            # 2. Chronological split  70 / 15 / 15
            # ------------------------------------------------------------------
            n = len(df)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)

            df_train = df.iloc[:train_end].reset_index(drop=True)
            df_val = df.iloc[train_end:val_end].reset_index(drop=True)

            # Minimum length guard (window_size default = 60, need +1)
            min_rows = 62
            if len(df_train) < min_rows or len(df_val) < min_rows:
                logger.warning(
                    "Trial %d skipped — not enough rows after split "
                    "(train=%d, val=%d, need>=%d).",
                    trial.number, len(df_train), len(df_val), min_rows,
                )
                return -np.inf

            # ------------------------------------------------------------------
            # 3. Sample hyperparameters
            # ------------------------------------------------------------------
            if algo_upper == "PPO":
                params = _sample_ppo_params(trial)
            else:
                params = _sample_td3_params(trial)

            net_arch_resolved = params.pop("_net_arch_list")
            params.pop("net_arch")  # remove string tag; it was logged to trial

            # ------------------------------------------------------------------
            # 4 & 5. Build training environment
            # ------------------------------------------------------------------
            env_cls = HFTradingEnv if algo_upper == "PPO" else _ContinuousHFTEnv
            env_kwargs = dict(df=df_train, market=market)

            def _make_train_env():
                env = env_cls(**env_kwargs)
                env = Monitor(env)
                return env

            train_env = DummyVecEnv([_make_train_env])
            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

            # ------------------------------------------------------------------
            # 6. Build model
            # ------------------------------------------------------------------
            policy_kwargs = {"net_arch": net_arch_resolved}

            if algo_upper == "PPO":
                model = PPO(
                    policy="MlpPolicy",
                    env=train_env,
                    policy_kwargs=policy_kwargs,
                    device=device,
                    verbose=0,
                    **params,
                )
            else:  # TD3
                model = TD3(
                    policy="MlpPolicy",
                    env=train_env,
                    policy_kwargs=policy_kwargs,
                    device=device,
                    verbose=0,
                    **params,
                )

            # ------------------------------------------------------------------
            # 7. Train with pruning callback
            # ------------------------------------------------------------------
            pruning_cb = TrialPruningCallback(trial, eval_freq=10_000, verbose=0)
            model.learn(total_timesteps=n_timesteps, callback=pruning_cb)

            # ------------------------------------------------------------------
            # 8. Evaluate on validation split
            # ------------------------------------------------------------------
            def _make_val_env():
                env = env_cls(df=df_val, market=market)
                env = Monitor(env)
                return env

            val_env = DummyVecEnv([_make_val_env])
            # Use training normalisation stats (no reward normalisation at eval time)
            val_env = VecNormalize(
                val_env,
                norm_obs=True,
                norm_reward=False,
                training=False,
            )
            # Copy running mean/var from training env
            val_env.obs_rms = train_env.obs_rms
            val_env.ret_rms = train_env.ret_rms

            episode_rewards: list[float] = []
            for _ in range(n_eval_episodes):
                obs = val_env.reset()
                ep_reward = 0.0
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, info = val_env.step(action)
                    ep_reward += float(reward[0])
                    done = bool(terminated[0])
                episode_rewards.append(ep_reward)

            mean_val_reward = float(np.mean(episode_rewards))
            logger.info(
                "Trial %d finished — mean_val_reward=%.4f",
                trial.number, mean_val_reward,
            )
            return mean_val_reward

        except optuna.exceptions.TrialPruned:
            # Re-raise so Optuna records this as a pruned trial
            raise

        except Exception as exc:
            logger.error(
                "Trial %d raised an unexpected exception: %s",
                trial.number, exc, exc_info=True,
            )
            return -np.inf

        finally:
            # ------------------------------------------------------------------
            # 9. Clean up to prevent memory leaks
            # ------------------------------------------------------------------
            try:
                if val_env is not None:
                    val_env.close()
            except Exception:
                pass
            try:
                if train_env is not None:
                    train_env.close()
            except Exception:
                pass
            if model is not None:
                del model

    return objective


# ---------------------------------------------------------------------------
# Main HPO runner
# ---------------------------------------------------------------------------


def run_hpo(
    market: str = "us",
    timescale: str = "1m",
    algorithm: str = "PPO",
    n_trials: int = DEFAULT_N_TRIALS,
    n_timesteps_per_trial: int = DEFAULT_N_TIMESTEPS_PER_TRIAL,
    n_eval_episodes: int = DEFAULT_N_EVAL_EPISODES,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    device: str = "auto",
    n_jobs: int = 1,
) -> optuna.Study:
    """
    Run Optuna HPO for the HFT RL trading system.

    Parameters
    ----------
    market:                  'us' or 'hk'
    timescale:               One of SUPPORTED_TIMESCALES
    algorithm:               'PPO' or 'TD3'
    n_trials:                Total number of trials to run
    n_timesteps_per_trial:   Training steps per trial
    n_eval_episodes:         Validation episodes used to compute the objective
    study_name:              Optuna study name (auto-generated if None)
    storage:                 Optuna storage URL for persistence, e.g.
                             "sqlite:///hpo.db" (required for n_jobs > 1)
    device:                  PyTorch device string or 'auto'
    n_jobs:                  Parallel trial workers (requires storage)

    Returns
    -------
    optuna.Study with completed / pruned trials.
    """
    if timescale not in SUPPORTED_TIMESCALES:
        raise ValueError(
            f"timescale must be one of {SUPPORTED_TIMESCALES}, got '{timescale}'"
        )

    algorithm = algorithm.upper()
    if algorithm not in ("PPO", "TD3"):
        raise ValueError(f"algorithm must be 'PPO' or 'TD3', got '{algorithm}'")

    # ------------------------------------------------------------------
    # 1. Study name
    # ------------------------------------------------------------------
    if study_name is None:
        study_name = f"hft_{market}_{timescale}_{algorithm}_{int(time.time())}"

    logger.info(
        "Starting HPO study '%s'  market=%s  timescale=%s  algo=%s  "
        "trials=%d  timesteps/trial=%d  n_jobs=%d",
        study_name, market, timescale, algorithm,
        n_trials, n_timesteps_per_trial, n_jobs,
    )

    # ------------------------------------------------------------------
    # 2. Create (or resume) study
    # ------------------------------------------------------------------
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(n_startup_trials=10, multivariate=True, seed=42),
        pruner=MedianPruner(
            n_startup_trials=5, n_warmup_steps=5, interval_steps=1
        ),
        load_if_exists=True,
    )

    # ------------------------------------------------------------------
    # 3. Tag study with metadata
    # ------------------------------------------------------------------
    study.set_user_attr("market", market)
    study.set_user_attr("timescale", timescale)
    study.set_user_attr("algorithm", algorithm)
    study.set_user_attr("n_timesteps_per_trial", n_timesteps_per_trial)
    study.set_user_attr("n_eval_episodes", n_eval_episodes)
    study.set_user_attr("device", device)

    # ------------------------------------------------------------------
    # 4. Build objective
    # ------------------------------------------------------------------
    objective = _make_objective(
        market=market,
        timescale=timescale,
        algorithm=algorithm,
        n_timesteps=n_timesteps_per_trial,
        n_eval_episodes=n_eval_episodes,
        device=device,
    )

    # ------------------------------------------------------------------
    # 5. Optimise
    # ------------------------------------------------------------------
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    # ------------------------------------------------------------------
    # 6. Log and persist best params
    # ------------------------------------------------------------------
    if study.best_trial is not None:
        logger.info(
            "HPO complete — best trial #%d  value=%.4f\nBest params: %s",
            study.best_trial.number,
            study.best_value,
            json.dumps(study.best_params, indent=2),
        )

        best_params_path = HPO_DIR / f"{study_name}_best_params.json"
        payload = {
            "study_name": study_name,
            "market": market,
            "timescale": timescale,
            "algorithm": algorithm,
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
        }
        with open(best_params_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Best params saved to %s", best_params_path)
    else:
        logger.warning(
            "No successful trials completed — best params not saved."
        )

    return study


# ---------------------------------------------------------------------------
# Utilities for downstream use
# ---------------------------------------------------------------------------


def get_best_params(study_name_or_path: str) -> dict[str, Any]:
    """
    Load the best hyperparameters saved by a previous ``run_hpo()`` call.

    Parameters
    ----------
    study_name_or_path:
        Either the study name (used to reconstruct the canonical path under
        ``HPO_DIR``) or an absolute/relative path to a JSON file.

    Returns
    -------
    Dict containing keys: ``study_name``, ``market``, ``timescale``,
    ``algorithm``, ``best_trial``, ``best_value``, ``best_params``.

    Raises
    ------
    FileNotFoundError if the file does not exist.
    """
    candidate = Path(study_name_or_path)
    if not candidate.is_absolute():
        candidate = HPO_DIR / f"{study_name_or_path}_best_params.json"

    if not candidate.exists():
        raise FileNotFoundError(
            f"Best params file not found: {candidate}\n"
            "Run run_hpo() first to generate it."
        )

    with open(candidate, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    logger.info(
        "Loaded best params from %s  (study=%s, value=%.4f)",
        candidate,
        payload.get("study_name", "?"),
        payload.get("best_value", float("nan")),
    )
    return payload


def apply_best_params_to_trainer(
    best_params: dict[str, Any],
    algorithm: str,
) -> dict[str, Any]:
    """
    Convert Optuna best-params dict into keyword arguments suitable for
    ``train_model()`` in ``trainer.py``.

    The ``net_arch`` string tag is mapped back to an actual architecture
    dict/list, and a ``policy_kwargs`` key is injected so callers can pass
    the result directly to SB3 model constructors.

    Parameters
    ----------
    best_params: Dict returned by ``study.best_params`` (or the ``best_params``
                 sub-key from ``get_best_params()``).
    algorithm:   'PPO' or 'TD3'

    Returns
    -------
    Dict of kwargs ready for ``train_model()``.
    """
    algo_upper = algorithm.upper()
    params = dict(best_params)  # shallow copy — do not mutate caller's dict

    net_arch_tag: str = params.pop("net_arch", "medium")
    raw_arch = _NET_ARCH_MAP.get(net_arch_tag, _NET_ARCH_MAP["medium"])

    if algo_upper == "PPO":
        net_arch_resolved: Any = dict(pi=raw_arch, vf=raw_arch)
    else:
        net_arch_resolved = raw_arch

    params["policy_kwargs"] = {"net_arch": net_arch_resolved}

    # Rename algorithm-specific keys if trainer.py uses different names
    # (currently a 1-to-1 mapping, but kept explicit for clarity)
    return params


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HFT RL Hyperparameter Optimisation (Optuna)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--market",
        choices=["us", "hk"],
        default="us",
        help="Market universe to load from data_pipeline.",
    )
    parser.add_argument(
        "--timescale",
        choices=list(SUPPORTED_TIMESCALES),
        default="1m",
        help="Bar timescale for the environment.",
    )
    parser.add_argument(
        "--algo",
        choices=["PPO", "TD3"],
        default="PPO",
        help="RL algorithm whose hyperparams are optimised.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Total number of Optuna trials.",
    )
    parser.add_argument(
        "--timesteps-per-trial",
        type=int,
        default=DEFAULT_N_TIMESTEPS_PER_TRIAL,
        help="Training steps per trial.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///hpo.db. "
             "Required for --n-jobs > 1 and for resuming interrupted studies.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel trial workers. Requires --storage.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device ('auto', 'cpu', 'cuda', 'mps').",
    )
    args = parser.parse_args()

    if args.n_jobs > 1 and args.storage is None:
        parser.error("--n-jobs > 1 requires --storage to be set.")

    study = run_hpo(
        market=args.market,
        timescale=args.timescale,
        algorithm=args.algo,
        n_trials=args.trials,
        n_timesteps_per_trial=args.timesteps_per_trial,
        storage=args.storage,
        n_jobs=args.n_jobs,
        device=args.device,
    )

    print(f"\nBest trial:  {study.best_trial.number}")
    print(f"Best value:  {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
