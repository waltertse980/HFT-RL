"""
continuous_learner.py
=====================
Champion/Shadow continuous learning manager for an HFT RL trading system.

Manages two model slots:
  - Champion : the currently live production model
  - Shadow   : a newly trained candidate being evaluated before promotion

When Shadow passes safety gates (Sharpe improvement + drawdown check +
KL-divergence check) it is automatically promoted to Champion.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from data_pipeline import load_dataset
from rl_environment import HFTradingEnv
from trainer import evaluate_model, split_data

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelSlot:
    """Metadata for a Champion or Shadow model slot."""

    path: str
    vecnorm_path: Optional[str]
    market: str
    timescale: str
    algorithm: str
    sharpe: float
    max_drawdown: float
    win_rate: float
    total_return: float
    trained_at: str  # ISO timestamp
    is_live: bool = False
    promotion_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSlot":
        return cls(**d)


@dataclass
class ContinuousLearnerState:
    """Persisted state for the continuous learner."""

    champion: Optional[ModelSlot] = None
    shadow: Optional[ModelSlot] = None
    last_promotion_at: Optional[str] = None
    n_promotions: int = 0
    n_trials: int = 0
    is_shadow_training: bool = False


# ---------------------------------------------------------------------------
# Safety-gate constants
# ---------------------------------------------------------------------------

MIN_SHARPE_IMPROVEMENT: float = 0.05   # Shadow Sharpe must exceed Champion by at least this
MAX_DRAWDOWN_REGRESSION: float = 0.02  # Shadow max_drawdown must not be more than 2pp worse
MAX_KL_DIVERGENCE: float = 0.5        # KL divergence between action distributions must be < this
MIN_EVALUATION_EPISODES: int = 10      # Minimum episodes to evaluate a candidate
STATE_FILE_NAME: str = "continuous_state.json"

# Number of random observations used for KL-divergence estimation
_KL_N_SAMPLES: int = 1_000


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ContinuousLearner:
    """
    Thread-safe Champion/Shadow continuous learning manager.

    Usage::

        cl = ContinuousLearner(base_dir=Path("python_engine"), market="us", timescale="1m")
        cl.start()   # starts background training loop
        # …
        cl.stop()
    """

    # ------------------------------------------------------------------
    # Construction / teardown
    # ------------------------------------------------------------------

    def __init__(
        self,
        base_dir: Path,
        market: str = "us",
        timescale: str = "1m",
        algorithm: str = "PPO",
        retrain_interval_hours: float = 24.0,
        n_timesteps: int = 500_000,
        n_eval_episodes: int = 10,
        auto_promote: bool = True,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.market = market
        self.timescale = timescale
        self.algorithm = algorithm.upper()
        self.retrain_interval_hours = retrain_interval_hours
        self.n_timesteps = n_timesteps
        self.n_eval_episodes = max(n_eval_episodes, MIN_EVALUATION_EPISODES)
        self.auto_promote = auto_promote

        # Threading primitives
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._background_thread: Optional[threading.Thread] = None

        # Persistent state
        self._state_file = self.base_dir / STATE_FILE_NAME
        self._state: ContinuousLearnerState = self._load_state()

        # Ensure required directories exist
        (self.base_dir / "models" / "champion").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "models" / "shadow").mkdir(parents=True, exist_ok=True)

        logger.info(
            "ContinuousLearner initialised: market=%s timescale=%s algo=%s "
            "interval=%.1fh n_timesteps=%d auto_promote=%s",
            market,
            timescale,
            algorithm,
            retrain_interval_hours,
            n_timesteps,
            auto_promote,
        )

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> ContinuousLearnerState:
        """Load persisted state from disk, or return a fresh state on error."""
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
            state = ContinuousLearnerState()
            if raw.get("champion"):
                state.champion = ModelSlot.from_dict(raw["champion"])
            if raw.get("shadow"):
                state.shadow = ModelSlot.from_dict(raw["shadow"])
            state.last_promotion_at = raw.get("last_promotion_at")
            state.n_promotions = int(raw.get("n_promotions", 0))
            state.n_trials = int(raw.get("n_trials", 0))
            state.is_shadow_training = bool(raw.get("is_shadow_training", False))
            logger.info("Loaded continuous learner state from %s", self._state_file)
            return state
        except FileNotFoundError:
            logger.info("No existing state file at %s — starting fresh.", self._state_file)
            return ContinuousLearnerState()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to parse state file %s (%s) — starting fresh.", self._state_file, exc
            )
            return ContinuousLearnerState()

    def _save_state(self) -> None:
        """Serialise state to JSON.  Must be called while holding ``self._lock``."""
        payload: dict[str, Any] = {
            "champion": self._state.champion.to_dict() if self._state.champion else None,
            "shadow": self._state.shadow.to_dict() if self._state.shadow else None,
            "last_promotion_at": self._state.last_promotion_at,
            "n_promotions": self._state.n_promotions,
            "n_trials": self._state.n_trials,
            "is_shadow_training": self._state.is_shadow_training,
        }
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        logger.debug("State saved to %s", self._state_file)

    # ------------------------------------------------------------------
    # Public thread-safe accessors
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a serialisable snapshot of the current learner state."""
        with self._lock:
            return {
                "champion": self._state.champion.to_dict() if self._state.champion else None,
                "shadow": self._state.shadow.to_dict() if self._state.shadow else None,
                "is_shadow_training": self._state.is_shadow_training,
                "n_promotions": self._state.n_promotions,
                "n_trials": self._state.n_trials,
                "last_promotion_at": self._state.last_promotion_at,
            }

    def get_champion_path(self) -> Optional[str]:
        """Thread-safe getter for the champion model path."""
        with self._lock:
            return self._state.champion.path if self._state.champion else None

    # ------------------------------------------------------------------
    # Model evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_candidate(
        self,
        model_path: str,
        vecnorm_path: Optional[str] = None,
    ) -> dict:
        """
        Evaluate a candidate model on the held-out test set.

        Loads the dataset for the configured market/timescale, uses the last
        10 % of rows as the test partition, and delegates to
        ``trainer.evaluate_model``.

        Returns
        -------
        dict
            Metrics dict containing at least ``sharpe``, ``max_drawdown``,
            ``win_rate``, and ``total_return``.
        """
        dataset = load_dataset(self.market, self.timescale)
        # ``load_dataset`` may return a dict keyed by ticker; grab the last one.
        if isinstance(dataset, dict):
            ticker = list(dataset.keys())[-1]
            df = dataset[ticker]
        else:
            df = dataset

        n_test = max(1, int(len(df) * 0.10))
        test_df = df.iloc[-n_test:].copy()

        metrics = evaluate_model(
            model_path,
            test_df,
            self.algorithm,
            n_eval_episodes=self.n_eval_episodes,
            market=self.market,
            vecnorm_path=vecnorm_path,
        )
        logger.info(
            "Evaluated %s → sharpe=%.4f max_dd=%.4f wr=%.4f ret=%.4f",
            model_path,
            metrics.get("sharpe", float("nan")),
            metrics.get("max_drawdown", float("nan")),
            metrics.get("win_rate", float("nan")),
            metrics.get("total_return", float("nan")),
        )
        return metrics

    # ------------------------------------------------------------------
    # KL divergence
    # ------------------------------------------------------------------

    def _compute_kl_divergence(self, model_path_a: str, model_path_b: str) -> float:
        """
        Estimate the KL divergence between the action distributions of two models.

        For stochastic policies (PPO) we compute KL(P‖Q) analytically over a
        sample of random observations drawn from HFTradingEnv.
        For deterministic policies (TD3) we use the MSE between mean actions as
        a surrogate.

        Returns 0.0 on any error (conservative — do not block promotion on KL
        estimation failures).
        """
        try:
            algo = self.algorithm.upper()

            if algo == "PPO":
                model_a = PPO.load(model_path_a)
                model_b = PPO.load(model_path_b)
            elif algo == "TD3":
                model_a = TD3.load(model_path_a)
                model_b = TD3.load(model_path_b)
            else:
                # Fallback: try PPO
                model_a = PPO.load(model_path_a)
                model_b = PPO.load(model_path_b)

            # Build a temporary environment to generate observations
            dataset = load_dataset(self.market, self.timescale)
            if isinstance(dataset, dict):
                ticker = list(dataset.keys())[-1]
                df = dataset[ticker]
            else:
                df = dataset

            # Use a small slice so the env resets quickly
            sample_df = df.iloc[: min(len(df), _KL_N_SAMPLES + 200)].copy()

            def _make_env():
                env = HFTradingEnv(df=sample_df)
                env = Monitor(env)
                return env

            vec_env = DummyVecEnv([_make_env])

            # Collect random observations
            obs_list: list[np.ndarray] = []
            obs = vec_env.reset()
            rng = np.random.default_rng(seed=42)
            for _ in range(_KL_N_SAMPLES):
                obs_list.append(obs.copy())
                action = np.array([vec_env.action_space.sample()])
                obs, _, done, _ = vec_env.step(action)
                if done[0]:
                    obs = vec_env.reset()

            vec_env.close()
            obs_array = np.vstack(obs_list)  # shape (N, obs_dim)

            import torch

            obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32)

            if algo == "PPO":
                # PPO has a stochastic policy — compute probability distributions
                with torch.no_grad():
                    dist_a = model_a.policy.get_distribution(obs_tensor)
                    dist_b = model_b.policy.get_distribution(obs_tensor)

                    # For discrete action spaces get full probability vectors
                    if hasattr(dist_a.distribution, "probs"):
                        probs_a = dist_a.distribution.probs  # (N, n_actions)
                        probs_b = dist_b.distribution.probs

                        eps = 1e-8
                        p = probs_a + eps
                        q = probs_b + eps
                        kl = (p * (p / q).log()).sum(dim=-1).mean().item()
                    else:
                        # Continuous / diagonal Gaussian: use built-in KL
                        kl = torch.distributions.kl_divergence(
                            dist_a.distribution, dist_b.distribution
                        ).mean().item()

                return float(kl)

            else:
                # TD3 is deterministic: use MSE of mean actions as surrogate
                with torch.no_grad():
                    actions_a = model_a.policy.actor(obs_tensor)
                    actions_b = model_b.policy.actor(obs_tensor)
                mse = float(((actions_a - actions_b) ** 2).mean().item())
                return mse

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "KL divergence computation failed (%s) — returning 0.0 (conservative).", exc
            )
            return 0.0

    # ------------------------------------------------------------------
    # Safety gates
    # ------------------------------------------------------------------

    def _passes_safety_gates(
        self,
        candidate_metrics: dict,
        champion_metrics: Optional[dict],
    ) -> tuple[bool, str]:
        """
        Evaluate whether the candidate model passes all safety gates.

        Returns
        -------
        (passed: bool, reason: str)
        """
        # Gate 0: first model ever — unconditionally promote
        if champion_metrics is None:
            logger.info("No existing champion — candidate passes by default (first model).")
            return True, "first_model"

        candidate_sharpe = float(candidate_metrics.get("sharpe", -999.0))
        champion_sharpe = float(champion_metrics.get("sharpe", -999.0))
        candidate_dd = float(candidate_metrics.get("max_drawdown", 1.0))
        champion_dd = float(champion_metrics.get("max_drawdown", 1.0))

        # Gate 1: Sharpe improvement
        sharpe_threshold = champion_sharpe + MIN_SHARPE_IMPROVEMENT
        if candidate_sharpe < sharpe_threshold:
            reason = (
                f"failed: sharpe_improvement "
                f"(candidate={candidate_sharpe:.4f} < required={sharpe_threshold:.4f})"
            )
            logger.info("Safety gate FAILED — %s", reason)
            return False, reason

        # Gate 2: Drawdown regression
        dd_limit = champion_dd + MAX_DRAWDOWN_REGRESSION
        if candidate_dd > dd_limit:
            reason = (
                f"failed: drawdown_regression "
                f"(candidate={candidate_dd:.4f} > limit={dd_limit:.4f})"
            )
            logger.info("Safety gate FAILED — %s", reason)
            return False, reason

        # Gate 3: KL divergence (only when a champion path is available)
        champion_path = self._state.champion.path if self._state.champion else None
        shadow_path = candidate_metrics.get("_model_path")
        if champion_path and shadow_path:
            kl = self._compute_kl_divergence(champion_path, shadow_path)
            logger.info("KL divergence between champion and shadow: %.4f", kl)
            if kl >= MAX_KL_DIVERGENCE:
                reason = (
                    f"failed: kl_divergence "
                    f"(kl={kl:.4f} >= max={MAX_KL_DIVERGENCE:.4f})"
                )
                logger.info("Safety gate FAILED — %s", reason)
                return False, reason
        else:
            logger.debug("Skipping KL gate — one or both model paths unavailable.")

        logger.info(
            "All safety gates PASSED — sharpe=%.4f dd=%.4f",
            candidate_sharpe,
            candidate_dd,
        )
        return True, "all_gates_passed"

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote_shadow(self, reason: str = "auto") -> bool:
        """
        Promote the current shadow model to champion.

        Copies model artefacts to ``base_dir/models/champion/``, updates
        state, and persists to disk.

        Returns True on success, False if there is no shadow to promote.
        """
        with self._lock:
            if self._state.shadow is None:
                logger.warning("promote_shadow called but no shadow is registered.")
                return False

            shadow = self._state.shadow
            champion_dir = self.base_dir / "models" / "champion"
            champion_dir.mkdir(parents=True, exist_ok=True)

            # Copy model file(s)
            shadow_path = Path(shadow.path)
            dest_model = champion_dir / shadow_path.name
            try:
                if shadow_path.is_dir():
                    if dest_model.exists():
                        shutil.rmtree(dest_model)
                    shutil.copytree(shadow_path, dest_model)
                else:
                    shutil.copy2(shadow_path, dest_model)
                logger.info("Copied model %s → %s", shadow_path, dest_model)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to copy shadow model artefacts: %s", exc)
                return False

            # Copy VecNormalize statistics if present
            dest_vecnorm: Optional[str] = None
            if shadow.vecnorm_path:
                vn_path = Path(shadow.vecnorm_path)
                dest_vn = champion_dir / vn_path.name
                try:
                    shutil.copy2(vn_path, dest_vn)
                    dest_vecnorm = str(dest_vn)
                    logger.info("Copied VecNormalize stats %s → %s", vn_path, dest_vn)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not copy vecnorm stats: %s", exc)

            # Build updated champion slot
            new_champion = ModelSlot(
                path=str(dest_model),
                vecnorm_path=dest_vecnorm,
                market=shadow.market,
                timescale=shadow.timescale,
                algorithm=shadow.algorithm,
                sharpe=shadow.sharpe,
                max_drawdown=shadow.max_drawdown,
                win_rate=shadow.win_rate,
                total_return=shadow.total_return,
                trained_at=shadow.trained_at,
                is_live=True,
                promotion_reason=reason,
            )

            self._state.champion = new_champion
            self._state.shadow = None
            self._state.n_promotions += 1
            self._state.last_promotion_at = datetime.now(timezone.utc).isoformat()
            self._save_state()

        logger.info(
            "Shadow promoted to Champion (reason=%s, n_promotions=%d, sharpe=%.4f)",
            reason,
            self._state.n_promotions,
            new_champion.sharpe,
        )
        return True

    # ------------------------------------------------------------------
    # Shadow registration
    # ------------------------------------------------------------------

    def register_shadow(
        self,
        model_path: str,
        vecnorm_path: Optional[str],
        metrics: dict,
    ) -> None:
        """
        Register a newly trained model as the Shadow candidate.

        Parameters
        ----------
        model_path:   filesystem path to the saved model.
        vecnorm_path: filesystem path to VecNormalize stats, or None.
        metrics:      dict with keys sharpe, max_drawdown, win_rate, total_return.
        """
        with self._lock:
            self._state.shadow = ModelSlot(
                path=model_path,
                vecnorm_path=vecnorm_path,
                market=self.market,
                timescale=self.timescale,
                algorithm=self.algorithm,
                sharpe=float(metrics.get("sharpe", 0.0)),
                max_drawdown=float(metrics.get("max_drawdown", 0.0)),
                win_rate=float(metrics.get("win_rate", 0.0)),
                total_return=float(metrics.get("total_return", 0.0)),
                trained_at=datetime.now(timezone.utc).isoformat(),
                is_live=False,
                promotion_reason="",
            )
            self._save_state()

        logger.info(
            "Shadow registered: path=%s sharpe=%.4f max_dd=%.4f",
            model_path,
            self._state.shadow.sharpe,
            self._state.shadow.max_drawdown,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_shadow(self) -> Optional[str]:
        """
        Train a new Shadow model and optionally promote it to Champion.

        Steps
        -----
        1. Set ``is_shadow_training = True`` and persist.
        2. Import ``trainer.train_model`` and train for ``n_timesteps`` steps.
        3. Evaluate the trained model.
        4. Check safety gates against the current Champion.
        5. If ``auto_promote`` and gates pass → promote.
        6. Otherwise register shadow without promoting.
        7. Increment ``n_trials``, clear ``is_shadow_training``, persist.

        Returns the model path on success, or None on failure.
        """
        with self._lock:
            self._state.is_shadow_training = True
            self._save_state()

        model_path: Optional[str] = None
        try:
            # ------------------------------------------------------------------
            # 1. Train
            # ------------------------------------------------------------------
            logger.info(
                "Starting shadow training: algo=%s market=%s timescale=%s n_timesteps=%d",
                self.algorithm,
                self.market,
                self.timescale,
                self.n_timesteps,
            )
            import trainer  # local import to avoid circular dependencies at module load

            shadow_dir = self.base_dir / "models" / "shadow"
            shadow_dir.mkdir(parents=True, exist_ok=True)

            train_result = trainer.train_model(
                market=self.market,
                timescale=self.timescale,
                algorithm=self.algorithm,
                n_timesteps=self.n_timesteps,
                output_dir=str(shadow_dir),
            )

            # ``train_model`` may return a path string or a dict with a "path" key
            if isinstance(train_result, dict):
                model_path = train_result.get("path") or train_result.get("model_path")
                vecnorm_path: Optional[str] = train_result.get("vecnorm_path")
            else:
                model_path = str(train_result)
                vecnorm_path = None

            if not model_path:
                raise RuntimeError("train_model returned no model path.")

            logger.info("Shadow model trained and saved to: %s", model_path)

            # ------------------------------------------------------------------
            # 2. Evaluate
            # ------------------------------------------------------------------
            candidate_metrics = self._evaluate_candidate(model_path, vecnorm_path)
            candidate_metrics["_model_path"] = model_path  # used by KL gate

            # ------------------------------------------------------------------
            # 3. Register shadow
            # ------------------------------------------------------------------
            self.register_shadow(model_path, vecnorm_path, candidate_metrics)

            # ------------------------------------------------------------------
            # 4. Safety gates
            # ------------------------------------------------------------------
            with self._lock:
                champion_metrics: Optional[dict] = None
                if self._state.champion:
                    champion_metrics = {
                        "sharpe": self._state.champion.sharpe,
                        "max_drawdown": self._state.champion.max_drawdown,
                        "win_rate": self._state.champion.win_rate,
                        "total_return": self._state.champion.total_return,
                    }

            passed, gate_reason = self._passes_safety_gates(candidate_metrics, champion_metrics)

            # ------------------------------------------------------------------
            # 5. Promote or keep as shadow
            # ------------------------------------------------------------------
            if self.auto_promote and passed:
                self.promote_shadow(f"auto_passed_safety_gates:{gate_reason}")
            else:
                if not passed:
                    logger.info(
                        "Shadow NOT promoted — safety gates: %s. "
                        "Shadow remains registered for manual inspection.",
                        gate_reason,
                    )
                else:
                    logger.info(
                        "Shadow training complete; auto_promote=False. "
                        "Shadow registered but not promoted."
                    )

        except Exception as exc:  # noqa: BLE001
            logger.error("Shadow training failed: %s", exc, exc_info=True)
            model_path = None

        finally:
            with self._lock:
                self._state.is_shadow_training = False
                self._state.n_trials += 1
                self._save_state()

        return model_path

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background training loop in a daemon thread."""
        if self._background_thread and self._background_thread.is_alive():
            logger.warning("ContinuousLearner background loop is already running.")
            return
        self._stop_event.clear()
        self._background_thread = threading.Thread(
            target=self._background_loop,
            name="ContinuousLearner-bg",
            daemon=True,
        )
        self._background_thread.start()
        logger.info(
            "ContinuousLearner background loop started (interval=%.1fh).",
            self.retrain_interval_hours,
        )

    def stop(self) -> None:
        """Signal the background loop to stop and wait for it to exit."""
        logger.info("Stopping ContinuousLearner background loop…")
        self._stop_event.set()
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=60)
            if self._background_thread.is_alive():
                logger.warning("Background thread did not exit within 60 s.")
        logger.info("ContinuousLearner stopped.")

    def _background_loop(self) -> None:
        """
        Periodically retrain the Shadow model.

        Waits ``retrain_interval_hours * 3600`` seconds between cycles.
        Errors are logged but do not terminate the loop.
        """
        interval_seconds = self.retrain_interval_hours * 3600.0
        logger.info(
            "Background loop running; will retrain every %.1f s.", interval_seconds
        )
        while not self._stop_event.is_set():
            try:
                self.train_shadow()
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Unhandled error in background training cycle: %s", exc, exc_info=True
                )
            # Wait for the interval, but wake up early if stop is requested
            self._stop_event.wait(timeout=interval_seconds)

        logger.info("Background loop exited.")

    # ------------------------------------------------------------------
    # Force retrain
    # ------------------------------------------------------------------

    def force_retrain(self) -> str:
        """
        Immediately trigger a shadow training cycle in a new thread.

        Returns
        -------
        str
            "triggered"
        """
        t = threading.Thread(
            target=self.train_shadow,
            name="ContinuousLearner-force-retrain",
            daemon=True,
        )
        t.start()
        logger.info("Force retrain triggered (thread=%s).", t.name)
        return "triggered"

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ContinuousLearner("
            f"market={self.market!r}, "
            f"timescale={self.timescale!r}, "
            f"algorithm={self.algorithm!r}, "
            f"n_promotions={self._state.n_promotions}, "
            f"n_trials={self._state.n_trials}"
            f")"
        )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Champion/Shadow Continuous Learner for HFT RL trading system."
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        type=Path,
        help="Base directory for model artefacts and state file (default: .)",
    )
    parser.add_argument(
        "--market",
        default="us",
        help="Market identifier passed to data_pipeline.load_dataset (default: us)",
    )
    parser.add_argument(
        "--timescale",
        default="1m",
        help="Bar timescale passed to data_pipeline.load_dataset (default: 1m)",
    )
    parser.add_argument(
        "--algo",
        default="PPO",
        help="RL algorithm: PPO or TD3 (default: PPO)",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=24.0,
        help="Hours between automatic retraining cycles (default: 24.0)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Training timesteps per cycle (default: 500,000)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current learner status as JSON and exit.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Trigger an immediate retraining cycle and exit.",
    )
    args = parser.parse_args()

    cl = ContinuousLearner(
        base_dir=Path(args.base_dir),
        market=args.market,
        timescale=args.timescale,
        algorithm=args.algo,
        retrain_interval_hours=args.interval_hours,
        n_timesteps=args.timesteps,
    )

    if args.status:
        print(json.dumps(cl.get_status(), indent=2, default=str))
    elif args.force_retrain:
        result = cl.force_retrain()
        print(f"Force retrain: {result}")
        # Wait for the spawned thread to finish before exiting
        time.sleep(2)
    else:
        print("Starting continuous learner (Ctrl-C to stop)…")
        cl.start()
        try:
            while True:
                time.sleep(60)
                print("Status:", json.dumps(cl.get_status(), indent=2, default=str))
        except KeyboardInterrupt:
            cl.stop()
            print("Stopped.")
