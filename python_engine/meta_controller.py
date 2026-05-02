"""
meta_controller.py
==================
Meta-Controller for an HFT Reinforcement-Learning trading system.

Responsibilities
----------------
1. Market regime detection  (Bull / Bear / Sideways / Volatile)
2. Kelly Criterion capital allocation across multiple RL models
3. Weighted signal aggregation from multiple SB3 models
4. Position sizing with risk controls

Usage
-----
    mc = MetaController()
    mc.register_model(
        model_id="ppo_us_1m",
        model_path="/models/ppo_us_1m.zip",
        vecnorm_path="/models/ppo_us_1m_vecnorm.pkl",
        weight=1.0,
        market="us",
        timescale="1m",
        algorithm="PPO",
    )
    signal = mc.get_signal(obs_dict, prices=close_prices)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelSignal:
    """Signal output from a single RL model."""

    model_id: str
    action: int       # 0=sell, 1=hold, 2=buy
    confidence: float  # 0.0 to 1.0 (softmax probability of chosen action)
    weight: float     # allocation weight for this model
    market: str
    timescale: str


@dataclass
class AggregatedSignal:
    """Final aggregated trading signal."""

    action: int                   # 0=sell, 1=hold, 2=buy
    confidence: float             # weighted average confidence
    kelly_fraction: float         # recommended position size as fraction of capital
    regime: MarketRegime
    model_signals: list[ModelSignal]
    timestamp: float
    reasoning: str

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        d = asdict(self)
        d["regime"] = self.regime.value
        return d


# ---------------------------------------------------------------------------
# Regime Detector
# ---------------------------------------------------------------------------


class RegimeDetector:
    """
    Detects market regime from recent OHLCV bar data using:
    - 20-bar EMA vs 50-bar EMA (trend direction)
    - Realised volatility (20-bar rolling std of returns)
    - Average Directional Index (ADX) proxy

    Thresholds
    ----------
    VOLATILITY_THRESHOLD : 1.5% std of returns  -> VOLATILE
    TREND_THRESHOLD      : 0.5% EMA separation  -> BULL or BEAR
    """

    VOLATILITY_THRESHOLD: float = 0.015  # 1.5% std of returns = volatile
    TREND_THRESHOLD: float = 0.005       # 0.5% EMA separation = trending

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, prices: np.ndarray) -> MarketRegime:
        """
        Detect the current market regime.

        Parameters
        ----------
        prices : np.ndarray
            1-D array of close prices, most-recent bar **last**.
            Must contain at least 50 elements.

        Returns
        -------
        MarketRegime
        """
        prices = np.asarray(prices, dtype=np.float64).ravel()

        if len(prices) < 50:
            logger.debug(
                "RegimeDetector: insufficient bars (%d < 50); returning UNKNOWN",
                len(prices),
            )
            return MarketRegime.UNKNOWN

        ema20 = self._ema(prices[-20:], 20)
        ema50 = self._ema(prices[-50:], 50)

        trend = (ema20 - ema50) / (ema50 + 1e-8)

        # Realised volatility: std of 1-bar log-returns over last 20 bars
        returns = np.diff(prices[-21:]) / (prices[-21:-1] + 1e-8)
        vol = float(np.std(returns))

        logger.debug(
            "RegimeDetector: ema20=%.5f ema50=%.5f trend=%.5f vol=%.5f",
            ema20,
            ema50,
            trend,
            vol,
        )

        if vol > self.VOLATILITY_THRESHOLD:
            return MarketRegime.VOLATILE
        if trend > self.TREND_THRESHOLD:
            return MarketRegime.BULL
        if trend < -self.TREND_THRESHOLD:
            return MarketRegime.BEAR
        return MarketRegime.SIDEWAYS

    def get_regime_multiplier(self, regime: MarketRegime) -> float:
        """
        Risk multiplier for each regime.
        Scales the Kelly fraction before position sizing.

        Returns
        -------
        float in (0, 1]
        """
        return {
            MarketRegime.BULL: 1.0,
            MarketRegime.BEAR: 0.5,
            MarketRegime.SIDEWAYS: 0.7,
            MarketRegime.VOLATILE: 0.3,
            MarketRegime.UNKNOWN: 0.5,
        }[regime]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(arr: np.ndarray, span: int) -> float:
        """Compute trailing exponential moving average."""
        alpha = 2.0 / (span + 1)
        ema = float(arr[0])
        for x in arr[1:]:
            ema = alpha * float(x) + (1.0 - alpha) * ema
        return ema


# ---------------------------------------------------------------------------
# Kelly Criterion
# ---------------------------------------------------------------------------


def compute_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    max_fraction: float = 0.25,
    safety_divisor: float = 4.0,
) -> float:
    """
    Compute fractional Kelly criterion for position sizing.

    Kelly formula
    -------------
    f* = (p * b - q) / b

    where
        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss  (win/loss ratio)

    The raw Kelly is then divided by ``safety_divisor`` (default 4 → quarter-
    Kelly) and capped at ``max_fraction``.

    Parameters
    ----------
    win_rate :      Fraction of winning trades, clipped to [0, 1].
    avg_win :       Average winning trade return (positive float).
    avg_loss :      Average losing trade magnitude (positive float).
    max_fraction :  Hard upper cap on the returned fraction.
    safety_divisor: Divide raw Kelly by this factor for conservatism.

    Returns
    -------
    float in [0, max_fraction]
    """
    if avg_loss <= 0.0 or avg_win <= 0.0:
        return 0.0

    b = avg_win / avg_loss
    p = float(np.clip(win_rate, 0.0, 1.0))
    q = 1.0 - p

    kelly = (p * b - q) / b
    fractional_kelly = kelly / safety_divisor

    return float(np.clip(fractional_kelly, 0.0, max_fraction))


# ---------------------------------------------------------------------------
# Signal Aggregation
# ---------------------------------------------------------------------------


def aggregate_signals(
    signals: list[ModelSignal],
    regime_multiplier: float = 1.0,
    min_confidence_threshold: float = 0.4,
) -> tuple[int, float]:
    """
    Aggregate multiple model signals using weighted voting.

    Algorithm
    ---------
    1. Discard signals whose confidence < ``min_confidence_threshold``.
    2. For each action a in {0, 1, 2}: score[a] = Σ weight_i * confidence_i
       for every model i that chose action a.
    3. Normalise scores by total weight of accepted signals.
    4. Apply ``regime_multiplier`` to non-HOLD actions (0 and 2) to dampen
       directional bets in unfavourable regimes.
    5. Select the action with the highest weighted score.
    6. Final confidence = winning_score / total_score (after multiplier).

    Returns
    -------
    (action, confidence) where action ∈ {0, 1, 2} and confidence ∈ [0, 1].

    If no signals pass the threshold, returns (1, 0.0) — HOLD with zero
    confidence.
    """
    # Filter by confidence threshold
    valid = [s for s in signals if s.confidence >= min_confidence_threshold]

    if not valid:
        logger.debug(
            "aggregate_signals: no signals exceeded confidence threshold %.2f; "
            "returning HOLD",
            min_confidence_threshold,
        )
        return 1, 0.0

    # Accumulate weighted scores per action
    scores: dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0}
    total_weight = 0.0

    for sig in valid:
        scores[sig.action] += sig.weight * sig.confidence
        total_weight += sig.weight

    # Normalise
    if total_weight > 0.0:
        for a in scores:
            scores[a] /= total_weight

    # Apply regime multiplier to directional actions only
    scores[0] *= regime_multiplier
    scores[2] *= regime_multiplier
    # HOLD (1) is unscaled — regime dampening reduces directional positions,
    # making HOLD relatively more attractive.

    total_score = sum(scores.values())
    best_action = int(max(scores, key=lambda a: scores[a]))

    confidence = (scores[best_action] / total_score) if total_score > 1e-9 else 0.0

    logger.debug(
        "aggregate_signals: scores=%s best_action=%d confidence=%.4f",
        {k: f"{v:.4f}" for k, v in scores.items()},
        best_action,
        confidence,
    )

    return best_action, float(np.clip(confidence, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Meta-Controller
# ---------------------------------------------------------------------------


class MetaController:
    """
    Aggregates signals from multiple SB3 RL models, detects the market
    regime, computes a Kelly position size, and produces a final trading
    signal.

    Usage
    -----
    mc = MetaController()
    mc.register_model(
        model_id="ppo_us_1m",
        model_path="/models/ppo_us_1m.zip",
        vecnorm_path="/models/ppo_us_1m_vecnorm.pkl",
        weight=1.0,
        market="us",
        timescale="1m",
        algorithm="PPO",
    )
    signal = mc.get_signal({"ppo_us_1m": obs_array}, prices=close_array)
    """

    # Supported algorithms mapped to SB3 loader classes
    _ALGO_MAP: dict[str, type] = {
        "PPO": PPO,
        "TD3": TD3,
    }

    def __init__(
        self,
        kelly_fraction_override: Optional[float] = None,
        regime_override: Optional[MarketRegime] = None,
        min_confidence_threshold: float = 0.4,
        max_kelly_fraction: float = 0.25,
    ) -> None:
        """
        Initialise the MetaController.

        Parameters
        ----------
        kelly_fraction_override :
            If set, bypass Kelly calculation and use this fixed fraction.
        regime_override :
            If set, bypass regime detection and use this fixed regime.
        min_confidence_threshold :
            Discard model signals below this confidence level.
        max_kelly_fraction :
            Hard cap on the Kelly fraction before regime scaling.
        """
        # model_id -> {"model": SB3Model, "weight": float,
        #               "market": str, "timescale": str,
        #               "vecnorm": Optional[VecNormalize],
        #               "algorithm": str}
        self._models: dict[str, dict] = {}

        self._regime_detector = RegimeDetector()
        self._regime_override = regime_override
        self._kelly_override = kelly_fraction_override
        self._min_confidence = min_confidence_threshold
        self._max_kelly = max_kelly_fraction

        # Kelly stats — prior estimates, updated via update_model_stats()
        self._win_rate: float = 0.55
        self._avg_win: float = 0.012
        self._avg_loss: float = 0.008

        self._current_regime: MarketRegime = MarketRegime.UNKNOWN

        self._state_file = Path(__file__).parent / "meta_state.json"
        self._load_state()

        logger.info(
            "MetaController initialised (kelly_override=%s, regime_override=%s, "
            "min_confidence=%.2f, max_kelly=%.2f)",
            kelly_fraction_override,
            regime_override,
            min_confidence_threshold,
            max_kelly_fraction,
        )

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def register_model(
        self,
        model_id: str,
        model_path: str,
        vecnorm_path: Optional[str],
        weight: float,
        market: str,
        timescale: str,
        algorithm: str = "PPO",
    ) -> None:
        """
        Load and register an SB3 model.

        Parameters
        ----------
        model_id :    Unique identifier for this model.
        model_path :  Path to the SB3 ``.zip`` checkpoint.
        vecnorm_path: Path to a VecNormalize stats file, or None.
        weight :      Voting weight (positive float).
        market :      Market string, e.g. ``"us"`` or ``"crypto"``.
        timescale :   Bar timescale, e.g. ``"1m"`` or ``"10s"``.
        algorithm :   ``"PPO"`` or ``"TD3"`` (case-sensitive).

        Raises
        ------
        ValueError  : Unsupported algorithm string.
        FileNotFoundError : model_path does not exist.
        """
        algo_upper = algorithm.upper()
        if algo_upper not in self._ALGO_MAP:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Choose from: {list(self._ALGO_MAP.keys())}"
            )

        loader = self._ALGO_MAP[algo_upper]
        logger.info(
            "register_model: loading %s model from '%s'", algo_upper, model_path
        )
        model = loader.load(model_path)

        vecnorm: Optional[VecNormalize] = None
        if vecnorm_path and Path(vecnorm_path).exists():
            logger.info(
                "register_model: loading VecNormalize stats from '%s'", vecnorm_path
            )
            # VecNormalize.load requires a venv; supply a dummy placeholder env.
            # The normaliser is used only for obs normalisation, not stepping.
            dummy_env = DummyVecEnv([lambda: _DummyEnv(model.observation_space)])
            vecnorm = VecNormalize.load(vecnorm_path, dummy_env)
            vecnorm.training = False
            vecnorm.norm_reward = False
        elif vecnorm_path:
            logger.warning(
                "register_model: vecnorm_path '%s' not found; skipping.", vecnorm_path
            )

        self._models[model_id] = {
            "model": model,
            "weight": float(weight),
            "market": market,
            "timescale": timescale,
            "vecnorm": vecnorm,
            "algorithm": algo_upper,
        }

        logger.info(
            "register_model: registered model '%s' (weight=%.2f, algo=%s)",
            model_id,
            weight,
            algo_upper,
        )

    def unregister_model(self, model_id: str) -> None:
        """
        Remove a registered model.

        Parameters
        ----------
        model_id : ID of the model to remove.
        """
        if model_id in self._models:
            del self._models[model_id]
            logger.info("unregister_model: removed model '%s'", model_id)
        else:
            logger.warning("unregister_model: model '%s' not found; no-op.", model_id)

    def update_model_stats(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> None:
        """
        Update the live-trading statistics used in the Kelly calculation.

        Parameters
        ----------
        win_rate : Fraction of winning trades, e.g. 0.56.
        avg_win  : Average return on winning trades (positive float).
        avg_loss : Average magnitude of loss on losing trades (positive float).
        """
        self._win_rate = float(np.clip(win_rate, 0.0, 1.0))
        self._avg_win = max(float(avg_win), 1e-8)
        self._avg_loss = max(float(avg_loss), 1e-8)
        logger.info(
            "update_model_stats: win_rate=%.3f avg_win=%.5f avg_loss=%.5f",
            self._win_rate,
            self._avg_win,
            self._avg_loss,
        )
        self._save_state()

    # ------------------------------------------------------------------
    # Core signal generation
    # ------------------------------------------------------------------

    def get_signal(
        self,
        observations: dict[str, np.ndarray],
        prices: Optional[np.ndarray] = None,
    ) -> AggregatedSignal:
        """
        Generate an aggregated trading signal.

        Parameters
        ----------
        observations :
            Mapping of model_id -> observation array.  Each array is passed
            directly to the corresponding SB3 model.
        prices :
            Optional 1-D array of close prices (most recent last) used for
            regime detection.  If omitted, the last detected regime is reused.

        Returns
        -------
        AggregatedSignal
        """
        # ---- Regime detection ----------------------------------------
        if self._regime_override is not None:
            regime = self._regime_override
        elif prices is not None:
            regime = self._regime_detector.detect(np.asarray(prices))
            self._current_regime = regime
            self._save_state()
        else:
            regime = self._current_regime

        regime_multiplier = self._regime_detector.get_regime_multiplier(regime)

        # ---- Collect model signals ------------------------------------
        model_signals: list[ModelSignal] = []

        for model_id, meta in self._models.items():
            if model_id not in observations:
                logger.debug("get_signal: no observation provided for '%s'; skipping", model_id)
                continue

            try:
                obs = observations[model_id]
                action, confidence = self._query_model(model_id, meta, obs)

                model_signals.append(
                    ModelSignal(
                        model_id=model_id,
                        action=action,
                        confidence=confidence,
                        weight=meta["weight"],
                        market=meta["market"],
                        timescale=meta["timescale"],
                    )
                )
                logger.debug(
                    "get_signal: model='%s' action=%d confidence=%.4f",
                    model_id,
                    action,
                    confidence,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "get_signal: model '%s' raised %s: %s; skipping.",
                    model_id,
                    type(exc).__name__,
                    exc,
                )

        # ---- Aggregate -----------------------------------------------
        action, confidence = aggregate_signals(
            model_signals,
            regime_multiplier=regime_multiplier,
            min_confidence_threshold=self._min_confidence,
        )

        # ---- Kelly position size -------------------------------------
        if self._kelly_override is not None:
            raw_kelly = float(np.clip(self._kelly_override, 0.0, self._max_kelly))
        else:
            raw_kelly = compute_kelly_fraction(
                win_rate=self._win_rate,
                avg_win=self._avg_win,
                avg_loss=self._avg_loss,
                max_fraction=self._max_kelly,
            )

        kelly_fraction = float(np.clip(raw_kelly * regime_multiplier, 0.0, self._max_kelly))

        # ---- Build reasoning string ----------------------------------
        action_names = {0: "SELL", 1: "HOLD", 2: "BUY"}
        reasoning = (
            f"Regime={regime.value} (multiplier={regime_multiplier:.2f}), "
            f"action={action_names[action]}, confidence={confidence:.4f}, "
            f"kelly={kelly_fraction:.4f} "
            f"(win_rate={self._win_rate:.3f}, avg_win={self._avg_win:.4f}, "
            f"avg_loss={self._avg_loss:.4f}), "
            f"models_queried={len(model_signals)}/{len(self._models)}"
        )

        signal = AggregatedSignal(
            action=action,
            confidence=confidence,
            kelly_fraction=kelly_fraction,
            regime=regime,
            model_signals=model_signals,
            timestamp=time.time(),
            reasoning=reasoning,
        )

        logger.info("get_signal: %s", reasoning)
        return signal

    # ------------------------------------------------------------------
    # Status / configuration
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """
        Return a serialisable status snapshot.

        Returns
        -------
        dict with keys:
            regime, kelly_fraction, active_models, signal_weights, n_models
        """
        kelly = (
            self._kelly_override
            if self._kelly_override is not None
            else compute_kelly_fraction(
                win_rate=self._win_rate,
                avg_win=self._avg_win,
                avg_loss=self._avg_loss,
                max_fraction=self._max_kelly,
            )
        )

        return {
            "regime": self._current_regime.value,
            "kelly_fraction": round(float(kelly), 6),
            "active_models": list(self._models.keys()),
            "signal_weights": {mid: meta["weight"] for mid, meta in self._models.items()},
            "n_models": len(self._models),
            "win_rate": self._win_rate,
            "avg_win": self._avg_win,
            "avg_loss": self._avg_loss,
            "min_confidence_threshold": self._min_confidence,
            "max_kelly_fraction": self._max_kelly,
            "kelly_override": self._kelly_override,
            "regime_override": self._regime_override.value if self._regime_override else None,
        }

    def configure(
        self,
        kelly_fraction: Optional[float] = None,
        regime_override: Optional[str] = None,
    ) -> None:
        """
        Update runtime overrides.

        Parameters
        ----------
        kelly_fraction :  Fixed Kelly fraction; pass None to re-enable auto.
        regime_override : Regime name string (e.g. ``"bull"``); None = auto.
        """
        if kelly_fraction is not None:
            self._kelly_override = float(np.clip(kelly_fraction, 0.0, self._max_kelly))
            logger.info("configure: kelly_override set to %.4f", self._kelly_override)
        else:
            self._kelly_override = None
            logger.info("configure: kelly_override cleared (auto mode)")

        if regime_override is not None:
            self._regime_override = MarketRegime(regime_override.lower())
            logger.info("configure: regime_override set to %s", self._regime_override)
        else:
            self._regime_override = None
            logger.info("configure: regime_override cleared (auto mode)")

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Persist regime and Kelly stats to a JSON file."""
        state = {
            "current_regime": self._current_regime.value,
            "win_rate": self._win_rate,
            "avg_win": self._avg_win,
            "avg_loss": self._avg_loss,
        }
        try:
            self._state_file.write_text(json.dumps(state, indent=2))
            logger.debug("_save_state: wrote %s", self._state_file)
        except OSError as exc:
            logger.warning("_save_state: could not write state file: %s", exc)

    def _load_state(self) -> None:
        """Load persisted state from JSON file, if it exists."""
        if not self._state_file.exists():
            logger.debug("_load_state: no state file found at %s", self._state_file)
            return
        try:
            state = json.loads(self._state_file.read_text())
            self._current_regime = MarketRegime(state.get("current_regime", "unknown"))
            self._win_rate = float(state.get("win_rate", self._win_rate))
            self._avg_win = float(state.get("avg_win", self._avg_win))
            self._avg_loss = float(state.get("avg_loss", self._avg_loss))
            logger.info(
                "_load_state: loaded state (regime=%s, win_rate=%.3f)",
                self._current_regime.value,
                self._win_rate,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("_load_state: failed to parse state file: %s", exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _query_model(
        self,
        model_id: str,
        meta: dict,
        obs: np.ndarray,
    ) -> tuple[int, float]:
        """
        Query a single registered model for an action and confidence.

        For PPO: uses the policy distribution to extract softmax probs.
        For TD3: deterministic action; confidence fixed at 1.0.

        Parameters
        ----------
        model_id : Human-readable name (for logging).
        meta :     Entry from self._models.
        obs :      Raw observation array for this model.

        Returns
        -------
        (action, confidence)
        """
        import torch  # local import — avoids hard dependency at module level

        model = meta["model"]
        algorithm = meta["algorithm"]
        vecnorm: Optional[VecNormalize] = meta["vecnorm"]

        # Normalise observation if a VecNormalize stats object is available
        obs_norm = obs.copy()
        if vecnorm is not None:
            # VecNormalize expects shape (n_envs, obs_dim); add batch dim
            obs_norm = vecnorm.normalize_obs(obs_norm[np.newaxis, :])[0]

        if algorithm == "PPO":
            obs_tensor = torch.as_tensor(obs_norm[np.newaxis, :]).float()
            obs_tensor = obs_tensor.to(model.policy.device)

            with torch.no_grad():
                distribution = model.policy.get_distribution(obs_tensor)
                probs = distribution.distribution.probs.cpu().numpy().ravel()

            # Ensure we have exactly 3 action probabilities
            if len(probs) != 3:
                logger.warning(
                    "_query_model: model '%s' has %d action probs (expected 3); "
                    "using argmax only.",
                    model_id,
                    len(probs),
                )
                action = int(np.argmax(probs))
                confidence = float(probs[action])
            else:
                action = int(np.argmax(probs))
                confidence = float(probs[action])

        elif algorithm == "TD3":
            # TD3 is deterministic; map continuous action to discrete buy/sell/hold
            obs_tensor = torch.as_tensor(obs_norm[np.newaxis, :]).float()
            obs_tensor = obs_tensor.to(model.policy.device)

            with torch.no_grad():
                raw_action = model.policy.actor(obs_tensor).cpu().numpy().ravel()

            # Map to discrete: raw_action is expected in [-1, 1]
            # > +0.33 -> BUY (2), < -0.33 -> SELL (0), else HOLD (1)
            scalar = float(raw_action[0]) if len(raw_action) >= 1 else 0.0
            if scalar > 0.33:
                action = 2
            elif scalar < -0.33:
                action = 0
            else:
                action = 1
            confidence = 1.0

        else:
            raise ValueError(f"Unknown algorithm '{algorithm}' for model '{model_id}'")

        return action, confidence


# ---------------------------------------------------------------------------
# Dummy environment helper (for VecNormalize loading)
# ---------------------------------------------------------------------------


class _DummyEnv:
    """Minimal Gym-compatible env for VecNormalize initialisation."""

    def __init__(self, observation_space) -> None:
        import gym  # noqa: F401  # type: ignore
        self.observation_space = observation_space
        self.action_space = gym.spaces.Discrete(3)
        self.reward_range = (-float("inf"), float("inf"))
        self.spec = None
        self.metadata: dict = {}

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        return [seed]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MetaController — HFT RL meta-signal aggregator"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print controller status as JSON and exit.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run a demo signal generation with random synthetic data "
            "(no models need to be registered)."
        ),
    )
    args = parser.parse_args()

    mc = MetaController()

    if args.status:
        print(json.dumps(mc.get_status(), indent=2))

    if args.demo:
        print("Demo: generating random signal (no models registered)")
        # Synthetic close-price series for regime detection
        rng = np.random.default_rng(42)
        synthetic_prices = 100.0 + np.cumsum(rng.normal(0.0, 0.5, 100))

        signal = mc.get_signal({}, prices=synthetic_prices)

        output = {
            "action": {0: "SELL", 1: "HOLD", 2: "BUY"}[signal.action],
            "confidence": round(signal.confidence, 4),
            "kelly_fraction": round(signal.kelly_fraction, 4),
            "regime": signal.regime.value,
            "timestamp": signal.timestamp,
            "reasoning": signal.reasoning,
            "n_model_signals": len(signal.model_signals),
        }
        print(json.dumps(output, indent=2))

        print("\nController status:")
        print(json.dumps(mc.get_status(), indent=2))
