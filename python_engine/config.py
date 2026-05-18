"""
config.py — Centralised hyperparameter configuration for the 5-Model Council.

All numeric hyperparameters live here. Import CouncilConfig anywhere that
needs training parameters. This is the single source of truth.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).parent


def _default_symbols() -> list[str]:
    return ["TSLA", "NVDA", "AAPL"]


def _default_timeframes() -> list[str]:
    return ["1m", "5m", "10m"]


def _default_policy_kwargs() -> dict[str, Any]:
    return dict(net_arch=[64, 64])


@dataclass
class CouncilConfig:
    # ── Training phases ───────────────────────────────────────────────────
    warmup_steps: int = 100_000
    council_start_steps: int = 100_000
    arbiter_start_steps: int = 500_000
    total_steps: int = 1_000_000

    # ── Evaluation cadence ────────────────────────────────────────────────
    eval_every_k_steps: int = 5_000
    eval_episodes: int = 10

    # ── Elo system ────────────────────────────────────────────────────────
    initial_elo: float = 1500.0
    k_factor: float = 32.0

    # ── Gap-gated visibility ──────────────────────────────────────────────
    gap_threshold_x: float = 0.3
    gap_adaptive: bool = True

    # ── Rewards ───────────────────────────────────────────────────────────
    lambda_gap: float = 0.1
    max_gap_clip: float = 0.5
    lambda_irl: float = 0.2
    transaction_cost: float = 0.0001

    # ── Reward-shaping decay (Models B and C) ─────────────────────────────
    shaping_initial_alpha: float = 0.5
    shaping_decay_rate: float = 0.999
    shaping_min_alpha: float = 0.01

    # ── Execution filters (Models D and E) ────────────────────────────────
    filter_d_threshold: float = -0.0005
    filter_e_threshold: float = -0.0020

    # ── PPO hyperparameters (shared across A, B, C) ───────────────────────
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    policy_kwargs: dict = field(default_factory=_default_policy_kwargs)

    # ── Data ──────────────────────────────────────────────────────────────
    symbols: list = field(default_factory=_default_symbols)
    timeframes: list = field(default_factory=_default_timeframes)
    primary_timeframe: str = "1m"
    window_size: int = 60

    # ── Paths (relative to this file's directory) ─────────────────────────
    log_dir: str = "logs/council"
    bars_dir: str = "data/bars"
    checkpoints_dir: str = "checkpoints"

    def __post_init__(self) -> None:
        # Ensure mutable defaults are not None (defensive — dataclass field()
        # already prevents the classic mutable-default bug)
        if self.symbols is None:
            self.symbols = _default_symbols()
        if self.timeframes is None:
            self.timeframes = _default_timeframes()
        if self.policy_kwargs is None:
            self.policy_kwargs = _default_policy_kwargs()

        # Create directories
        (BASE_DIR / self.log_dir).mkdir(parents=True, exist_ok=True)
        (BASE_DIR / self.bars_dir).mkdir(parents=True, exist_ok=True)
        (BASE_DIR / self.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # ── Convenience absolute-path getters ────────────────────────────────
    def log_path(self) -> Path:
        return BASE_DIR / self.log_dir

    def bars_path(self) -> Path:
        return BASE_DIR / self.bars_dir

    def checkpoints_path(self) -> Path:
        return BASE_DIR / self.checkpoints_dir
