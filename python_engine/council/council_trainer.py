"""
council_trainer.py — 5-Model Council Training Orchestrator.

Architecture
------------
Phase 1 (0 → warmup_steps):
    A, B, C train independently on IntradayEnv. No Elo competition yet.

Phase 2 (warmup_steps → arbiter_start_steps):
    Competitive council with gap-gated visibility:
        • Validation Sharpe drives Elo round-robin updates each cycle.
        • The strongest trained model becomes the "leader".
        • Inferior agents whose normalised Elo gap exceeds ``gap_threshold_x``
          receive a behavioural-cloning bonus from the leader's trade journal.
        • Shaping alpha for Models B/C decays geometrically.

Phase 3 (arbiter_start_steps → total_steps):
    Models D and E (ExecutionFilter ensembles) are added to the Elo ranking.
    Their Sharpe is computed on the validation set using current A/B/C policies.

Threading
---------
* Single-process, DummyVecEnv-only, n_envs=1 per agent.
* Training proceeds round-robin across A, B, C in each eval-cycle window.
* A ``stop_event`` (threading.Event) lets the API stop the loop between cycles.

Logging
-------
* JSON log file written under ``cfg.log_path()`` containing one record per
  eval cycle (cycle id, phase, Sharpe map, Elo map, leader, gap map,
  visibility map, shaping_alpha map, timestamp).
* The ``status_dict`` passed in is mutated in place with the latest record
  + an ``elo_history`` array for charting.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Lazy stable-baselines3 import so the rest of the file imports even if
# the SB3 install is broken; the actual train() will fail fast with a clear
# error message.
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    _SB3_AVAILABLE = True
    _SB3_IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover — defensive
    PPO = None  # type: ignore
    DummyVecEnv = None  # type: ignore
    _SB3_AVAILABLE = False
    _SB3_IMPORT_ERROR = _e

from config import CouncilConfig  # type: ignore[import-not-found]
from environments.intraday_env import IntradayEnv  # type: ignore[import-not-found]
from council.elo_tracker import EloTracker
from council.trade_journal import TradeJournal
from council.reward_shapers.null_shaper import NullShaper
from council.reward_shapers.fsto_shaper import FSTOShaper
from council.reward_shapers.candlestick_shaper import CandlestickShaper
from council.inverse_rl import compute_irl_bonus  # noqa: F401 — re-exported for tests
from council.execution_filter import ExecutionFilter
from split_data import split_by_dates  # type: ignore[import-not-found]

log = logging.getLogger(__name__)

TRAINED_IDS = ["A", "B", "C"]
ARBITER_IDS = ["D", "E"]
ALL_IDS = TRAINED_IDS + ARBITER_IDS

# Raw OHLCV columns excluded from the feature list (mirrors IntradayEnv)
_RAW_COLS = {"open", "high", "low", "close", "volume", "vwap", "trade_count"}


# ─────────────────────────────────────────────────────────────────────────────
# Feature-index resolution
# ─────────────────────────────────────────────────────────────────────────────
def _engineered_columns(df_1m: pd.DataFrame) -> list[str]:
    return [c for c in df_1m.columns if c not in _RAW_COLS]


def _feature_idx_in_obs(
    df_1m: pd.DataFrame, window_size: int, feature_name: str
) -> int:
    """
    Find the index into a flattened IntradayEnv observation vector that
    corresponds to ``feature_name`` on the most-recent 1m bar.

    The 1m window is laid out as window_size rows × n_features, flattened
    row-major. The most recent bar is at row index (window_size - 1).
    """
    eng_cols = _engineered_columns(df_1m)
    if feature_name not in eng_cols:
        # Fall back to 0 — shapers will receive a constant 0.0 feature and
        # contribute no bonus. This degrades gracefully if the column is
        # absent rather than crashing training.
        log.warning("Feature %r not found in featured df — shaper indices will be invalid", feature_name)
        return 0
    col_pos = eng_cols.index(feature_name)
    n_features = len(eng_cols)
    last_row_start = (window_size - 1) * n_features
    return last_row_start + col_pos


# ─────────────────────────────────────────────────────────────────────────────
# Env / model builders
# ─────────────────────────────────────────────────────────────────────────────
def _build_env(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame | None,
    df_10m: pd.DataFrame | None,
    cfg: CouncilConfig,
    shaper: Any | None = None,
) -> "DummyVecEnv":
    """Build a DummyVecEnv with a single IntradayEnv inside."""
    if DummyVecEnv is None:
        raise RuntimeError(f"stable-baselines3 unavailable: {_SB3_IMPORT_ERROR}")

    def _make() -> IntradayEnv:
        return IntradayEnv(
            featured_df_1m=df_1m,
            featured_df_5m=df_5m,
            featured_df_10m=df_10m,
            window_size=cfg.window_size,
            transaction_cost=cfg.transaction_cost,
            initial_capital=100_000.0,
            shaper=shaper,
        )

    # DummyVecEnv ONLY — never SubprocVecEnv (Windows crash constraint)
    return DummyVecEnv([_make])


def _build_ppo(env: "DummyVecEnv", cfg: CouncilConfig) -> "PPO":
    if PPO is None:
        raise RuntimeError(f"stable-baselines3 unavailable: {_SB3_IMPORT_ERROR}")
    return PPO(
        "MlpPolicy",
        env=env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        policy_kwargs=cfg.policy_kwargs,
        tensorboard_log=None,  # NEVER use TensorBoard (constraint)
        verbose=0,
        device="cpu",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation: Sharpe on held-out episodes
# ─────────────────────────────────────────────────────────────────────────────
def _compute_sharpe(
    model: Any,
    val_df_1m: pd.DataFrame,
    val_df_5m: pd.DataFrame | None,
    val_df_10m: pd.DataFrame | None,
    cfg: CouncilConfig,
    n_episodes: int = 10,
) -> float:
    """
    Run ``model`` deterministically over ``n_episodes`` validation days and
    return annualised Sharpe. Returns 0.0 if fewer than 5 valid days are
    available or if return-std is degenerate.
    """
    try:
        env = IntradayEnv(
            featured_df_1m=val_df_1m,
            featured_df_5m=val_df_5m,
            featured_df_10m=val_df_10m,
            window_size=cfg.window_size,
            transaction_cost=cfg.transaction_cost,
            initial_capital=100_000.0,
            shaper=None,
        )
    except Exception as exc:
        log.warning("Sharpe eval: env construction failed: %s", exc)
        return 0.0

    daily_returns: list[float] = []
    for _ in range(int(n_episodes)):
        try:
            obs, _ = env.reset()
        except Exception as exc:
            log.warning("Sharpe eval: env.reset failed: %s", exc)
            break
        done = False
        start_val = float(env.portfolio_value)
        while not done:
            try:
                action, _ = model.predict(obs, deterministic=True)
            except Exception as exc:
                log.warning("Sharpe eval: model.predict failed: %s", exc)
                done = True
                break
            try:
                a = int(np.asarray(action).flatten()[0])
            except Exception:
                a = 0
            obs, _r, terminated, truncated, _info = env.step(a)
            done = bool(terminated) or bool(truncated)
        end_val = float(env.portfolio_value)
        if start_val > 0:
            daily_returns.append((end_val - start_val) / start_val)

    if len(daily_returns) < 5:
        return 0.0
    arr = np.asarray(daily_returns, dtype=np.float64)
    mean_r = float(arr.mean())
    std_r = float(arr.std())
    if std_r < 1e-8:
        return 0.0
    return float(mean_r / std_r * np.sqrt(252.0))


# ─────────────────────────────────────────────────────────────────────────────
# Sharpe for ExecutionFilter (D, E)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_sharpe_filter(
    flt: ExecutionFilter,
    models_dict: dict[str, Any],
    val_df_1m: pd.DataFrame,
    val_df_5m: pd.DataFrame | None,
    val_df_10m: pd.DataFrame | None,
    cfg: CouncilConfig,
    n_episodes: int = 10,
) -> float:
    """
    Like ``_compute_sharpe`` but drives the env via an ExecutionFilter
    polling A/B/C. Position / entry-price feed comes from the env itself.
    """
    try:
        env = IntradayEnv(
            featured_df_1m=val_df_1m,
            featured_df_5m=val_df_5m,
            featured_df_10m=val_df_10m,
            window_size=cfg.window_size,
            transaction_cost=cfg.transaction_cost,
            initial_capital=100_000.0,
            shaper=None,
        )
    except Exception as exc:
        log.warning("Filter eval: env construction failed: %s", exc)
        return 0.0

    daily_returns: list[float] = []
    for _ in range(int(n_episodes)):
        try:
            obs, _ = env.reset()
        except Exception:
            break
        done = False
        start_val = float(env.portfolio_value)
        while not done:
            current_price = (
                float(env._close_1m[int(env._day_positions[env._step_in_day])])
                if env._step_in_day < len(env._day_positions) else 0.0
            )
            try:
                action = flt.decide(
                    obs=obs,
                    current_price=current_price,
                    entry_price=float(env.entry_price),
                    position=int(env.position),
                    models_dict=models_dict,
                )
            except Exception as exc:
                log.warning("Filter eval: decide failed: %s", exc)
                action = 0
            obs, _r, terminated, truncated, _info = env.step(int(action))
            done = bool(terminated) or bool(truncated)
        end_val = float(env.portfolio_value)
        if start_val > 0:
            daily_returns.append((end_val - start_val) / start_val)

    if len(daily_returns) < 5:
        return 0.0
    arr = np.asarray(daily_returns, dtype=np.float64)
    mean_r = float(arr.mean())
    std_r = float(arr.std())
    if std_r < 1e-8:
        return 0.0
    return float(mean_r / std_r * np.sqrt(252.0))


# ─────────────────────────────────────────────────────────────────────────────
# Journal recording: replay one validation day to capture the leader's actions
# ─────────────────────────────────────────────────────────────────────────────
def _record_journal(
    model: Any,
    journal: TradeJournal,
    val_df_1m: pd.DataFrame,
    val_df_5m: pd.DataFrame | None,
    val_df_10m: pd.DataFrame | None,
    cfg: CouncilConfig,
    global_step_offset: int,
) -> None:
    """
    Replay a single deterministic day through ``model`` and append each
    decision to ``journal`` using ``global_step_offset + step_in_day`` as
    the step key (so inferior models can look it up later).
    """
    try:
        env = IntradayEnv(
            featured_df_1m=val_df_1m,
            featured_df_5m=val_df_5m,
            featured_df_10m=val_df_10m,
            window_size=cfg.window_size,
            transaction_cost=cfg.transaction_cost,
            initial_capital=100_000.0,
            shaper=None,
        )
    except Exception as exc:
        log.warning("Journal record: env construction failed: %s", exc)
        return

    try:
        obs, _ = env.reset()
    except Exception:
        return
    done = False
    while not done:
        try:
            action, _ = model.predict(obs, deterministic=True)
            a = int(np.asarray(action).flatten()[0])
        except Exception:
            a = 0
        cur_pos = (
            int(env._day_positions[env._step_in_day])
            if env._step_in_day < len(env._day_positions) else 0
        )
        cur_price = float(env._close_1m[cur_pos]) if cur_pos < len(env._close_1m) else 0.0
        step_key = int(global_step_offset + env._step_in_day)
        journal.record_action(
            step=step_key,
            action=a,
            price=cur_price,
            portfolio_value=float(env.portfolio_value),
            reward=0.0,
            position_after=int(env.position),
            day_date=str(env._day_date) if env._day_date is not None else "",
        )
        obs, _r, terminated, truncated, _info = env.step(a)
        done = bool(terminated) or bool(truncated)


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────
def _write_log_record(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:  # pragma: no cover
        log.warning("Failed to append log record to %s: %s", log_path, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def start_council_training(
    cfg: CouncilConfig,
    featured_data: dict,
    stop_event: threading.Event,
    status_dict: dict,
) -> None:
    """
    Run the full 3-phase council training loop.

    Parameters
    ----------
    cfg            : Hyperparameter container.
    featured_data  : {symbol: {timeframe: DataFrame}} — must include the
                     primary timeframe (cfg.primary_timeframe, usually '1m')
                     for at least one symbol.
    stop_event     : Cooperative stop signal, checked between eval cycles.
    status_dict    : Mutable dict updated in place with the latest cycle's
                     summary (read by the FastAPI status endpoints).
    """
    if not _SB3_AVAILABLE:
        raise RuntimeError(f"stable-baselines3 unavailable: {_SB3_IMPORT_ERROR}")

    # ── Resolve training symbol & frames ────────────────────────────────
    if not featured_data:
        raise RuntimeError("featured_data is empty — nothing to train on")
    symbol = cfg.symbols[0] if cfg.symbols and cfg.symbols[0] in featured_data else next(iter(featured_data))
    sym_data = featured_data.get(symbol, {})
    df_1m_all = sym_data.get(cfg.primary_timeframe) or sym_data.get("1m")
    if df_1m_all is None or df_1m_all.empty:
        raise RuntimeError(f"No primary-timeframe data for symbol {symbol}")
    df_5m_all = sym_data.get("5m")
    df_10m_all = sym_data.get("10m")

    log.info("Council training symbol=%s primary_tf=%s rows_1m=%d",
             symbol, cfg.primary_timeframe, len(df_1m_all))

    # ── Date-based split ────────────────────────────────────────────────
    train_1m, val_1m, _test_1m = split_by_dates(df_1m_all)
    if df_5m_all is not None and not df_5m_all.empty:
        train_5m, val_5m, _ = split_by_dates(df_5m_all)
    else:
        train_5m, val_5m = None, None
    if df_10m_all is not None and not df_10m_all.empty:
        train_10m, val_10m, _ = split_by_dates(df_10m_all)
    else:
        train_10m, val_10m = None, None

    if train_1m.empty:
        raise RuntimeError("Training split is empty — check date boundaries")

    # ── Resolve obs-vector indices for shapers ──────────────────────────
    k_idx = _feature_idx_in_obs(train_1m, cfg.window_size, "stoch_k")
    d_idx = _feature_idx_in_obs(train_1m, cfg.window_size, "stoch_d")
    body_idx = _feature_idx_in_obs(train_1m, cfg.window_size, "body_ratio")
    upper_idx = _feature_idx_in_obs(train_1m, cfg.window_size, "upper_wick")
    lower_idx = _feature_idx_in_obs(train_1m, cfg.window_size, "lower_wick")
    range_idx = _feature_idx_in_obs(train_1m, cfg.window_size, "bar_range")

    # ── Build shapers, envs, models ─────────────────────────────────────
    shaper_a = NullShaper()
    shaper_b = FSTOShaper(stoch_k_idx=k_idx, stoch_d_idx=d_idx)
    shaper_c = CandlestickShaper(
        body_ratio_idx=body_idx,
        upper_wick_idx=upper_idx,
        lower_wick_idx=lower_idx,
        bar_range_idx=range_idx,
    )

    env_a = _build_env(train_1m, train_5m, train_10m, cfg, shaper=shaper_a)
    env_b = _build_env(train_1m, train_5m, train_10m, cfg, shaper=shaper_b)
    env_c = _build_env(train_1m, train_5m, train_10m, cfg, shaper=shaper_c)

    model_a = _build_ppo(env_a, cfg)
    model_b = _build_ppo(env_b, cfg)
    model_c = _build_ppo(env_c, cfg)

    models: dict[str, Any] = {"A": model_a, "B": model_b, "C": model_c}
    envs: dict[str, Any] = {"A": env_a, "B": env_b, "C": env_c}

    # ── Initial shaping alphas ──────────────────────────────────────────
    alphas: dict[str, float] = {
        "A": 0.0,
        "B": float(cfg.shaping_initial_alpha),
        "C": float(cfg.shaping_initial_alpha),
    }
    # Push initial alphas to envs
    for sb3_env, alpha in ((env_b, alphas["B"]), (env_c, alphas["C"])):
        try:
            sb3_env.env_method("set_shaping_alpha", alpha)
        except Exception:  # pragma: no cover
            pass

    # ── Elo / journals ──────────────────────────────────────────────────
    elo = EloTracker(model_ids=TRAINED_IDS, initial_elo=cfg.initial_elo, k_factor=cfg.k_factor)
    journals: dict[str, TradeJournal] = {m: TradeJournal(m) for m in ALL_IDS}

    # ── Execution filters D and E (built later, but instantiate up front) ─
    filter_d = ExecutionFilter(
        threshold_pct=cfg.filter_d_threshold,
        model_ids=TRAINED_IDS,
        elo_tracker=elo,
    )
    filter_e = ExecutionFilter(
        threshold_pct=cfg.filter_e_threshold,
        model_ids=TRAINED_IDS,
        elo_tracker=elo,
    )

    # ── Log paths ───────────────────────────────────────────────────────
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    log_path = cfg.log_path() / f"council_run_{ts_str}.jsonl"
    log.info("Council logging to %s", log_path)

    # ── Init shared status_dict ─────────────────────────────────────────
    status_dict.update({
        "running": True,
        "phase": "warmup",
        "cycle": 0,
        "total_steps": 0,
        "sharpe": {},
        "elo": {m: cfg.initial_elo for m in TRAINED_IDS},
        "leader": None,
        "gaps": {},
        "visibility": {},
        "shaping_alpha": dict(alphas),
        "error": None,
        "elo_history": [],
        "trade_journals": {m: [] for m in ALL_IDS},
        "symbol": symbol,
        "log_path": str(log_path),
    })

    # ── Round-robin training loop ───────────────────────────────────────
    total_steps = 0
    cycle = 0
    K = int(cfg.eval_every_k_steps)
    recent_gaps: list[float] = []
    gap_threshold = float(cfg.gap_threshold_x)

    try:
        while total_steps < cfg.total_steps:
            if stop_event.is_set():
                log.info("Stop signal received — exiting council loop")
                break

            # Determine current phase BEFORE the upcoming train chunk
            if total_steps < cfg.warmup_steps:
                phase = "warmup"
            elif total_steps < cfg.arbiter_start_steps:
                phase = "competitive"
            else:
                phase = "arbiter"

            # ── Train each trained model for K steps (round-robin) ────
            for mid in TRAINED_IDS:
                try:
                    models[mid].learn(
                        total_timesteps=K,
                        reset_num_timesteps=False,
                        progress_bar=False,
                    )
                except Exception as exc:
                    log.error("Training %s failed: %s", mid, exc)
                    status_dict["error"] = f"train {mid}: {exc}"
                total_steps += K
                if stop_event.is_set():
                    break
            if stop_event.is_set():
                break

            cycle += 1

            # ── Decay shaping alphas (B and C) ─────────────────────────
            for mid in ("B", "C"):
                new_alpha = max(cfg.shaping_min_alpha, alphas[mid] * cfg.shaping_decay_rate)
                alphas[mid] = float(new_alpha)
                try:
                    envs[mid].env_method("set_shaping_alpha", new_alpha)
                except Exception:  # pragma: no cover
                    pass

            # ── Validation Sharpe for trained models ───────────────────
            sharpe_map: dict[str, float] = {}
            if val_1m is not None and not val_1m.empty:
                for mid in TRAINED_IDS:
                    sharpe_map[mid] = _compute_sharpe(
                        models[mid], val_1m, val_5m, val_10m, cfg, n_episodes=cfg.eval_episodes
                    )
            else:
                sharpe_map = {m: 0.0 for m in TRAINED_IDS}

            # ── Elo round-robin update ────────────────────────────────
            elo.update_round_robin(sharpe_map)
            leader = max(sharpe_map.items(), key=lambda kv: kv[1])[0] if sharpe_map else "A"
            leader_rating = elo.ratings.get(leader, cfg.initial_elo)

            # ── Phase 3 arbiter Sharpe and Elo ─────────────────────────
            if phase == "arbiter":
                if "D" not in elo:
                    elo.register("D")
                if "E" not in elo:
                    elo.register("E")
                if val_1m is not None and not val_1m.empty:
                    sharpe_d = _compute_sharpe_filter(filter_d, models, val_1m, val_5m, val_10m, cfg, cfg.eval_episodes)
                    sharpe_e = _compute_sharpe_filter(filter_e, models, val_1m, val_5m, val_10m, cfg, cfg.eval_episodes)
                else:
                    sharpe_d = 0.0
                    sharpe_e = 0.0
                sharpe_map["D"] = sharpe_d
                sharpe_map["E"] = sharpe_e
                # Add D, E to Elo update vs A/B/C using the same scoring rule
                # (we already updated A/B/C above; do D,E pairwise vs everyone)
                ext_scores = {m: sharpe_map[m] for m in ALL_IDS if m in sharpe_map}
                # We re-call update_round_robin which snapshots history once
                elo.update_round_robin(ext_scores)
                # Pick global leader across all 5
                leader = max(ext_scores.items(), key=lambda kv: kv[1])[0]
                leader_rating = elo.ratings.get(leader, cfg.initial_elo)

            # ── Compute Elo gaps vs leader, visibility map ─────────────
            gaps: dict[str, float] = {}
            visibility: dict[str, bool] = {}
            max_gap = 0.0
            for mid in TRAINED_IDS:
                if mid == leader:
                    continue
                g = elo.normalised_gap(leader, mid)
                gaps[mid] = float(g)
                max_gap = max(max_gap, g)
                visibility[mid] = bool(g >= gap_threshold)
            recent_gaps.append(max_gap)

            # ── Adaptive gap threshold (after 10 cycles) ───────────────
            if cfg.gap_adaptive and len(recent_gaps) >= 10:
                gap_threshold = float(np.median(recent_gaps[-10:]))

            # ── Update leader's trade journal for IRL bonus next cycle ─
            if phase != "warmup" and leader in models and val_1m is not None and not val_1m.empty:
                _record_journal(
                    models[leader],
                    journals[leader],
                    val_1m, val_5m, val_10m,
                    cfg,
                    global_step_offset=total_steps,
                )

            # ── Build cycle log record ─────────────────────────────────
            elo_snapshot = {
                m: float(elo.ratings.get(m, cfg.initial_elo)) if m in elo else None
                for m in ALL_IDS
            }
            record = {
                "cycle": cycle,
                "total_steps": int(total_steps),
                "phase": phase,
                "sharpe": {k: float(v) for k, v in sharpe_map.items()},
                "elo": elo_snapshot,
                "leader": leader,
                "leader_rating": float(leader_rating),
                "gaps": gaps,
                "visibility": visibility,
                "gap_threshold": float(gap_threshold),
                "shaping_alpha": {k: float(v) for k, v in alphas.items()},
                "filter_weights": {
                    "D": filter_d.weights_snapshot() if phase == "arbiter" else None,
                    "E": filter_e.weights_snapshot() if phase == "arbiter" else None,
                },
                "timestamp": time.time(),
            }
            _write_log_record(log_path, record)

            # ── Mutate status_dict (read by the API) ───────────────────
            status_dict["phase"] = phase
            status_dict["cycle"] = cycle
            status_dict["total_steps"] = int(total_steps)
            status_dict["sharpe"] = record["sharpe"]
            status_dict["elo"] = elo_snapshot
            status_dict["leader"] = leader
            status_dict["gaps"] = gaps
            status_dict["visibility"] = visibility
            status_dict["shaping_alpha"] = dict(alphas)
            status_dict["gap_threshold"] = float(gap_threshold)
            status_dict["elo_history"] = elo.get_history()
            # Trade-journal snapshot (last 200 entries each)
            status_dict["trade_journals"] = {
                m: journals[m].to_dict_list(n=200) for m in ALL_IDS
            }

            log.info(
                "cycle=%d steps=%d phase=%s leader=%s sharpe=%s elo=%s",
                cycle, total_steps, phase, leader,
                {k: round(v, 3) for k, v in record["sharpe"].items()},
                {k: (round(v, 1) if v is not None else None) for k, v in elo_snapshot.items()},
            )

    finally:
        status_dict["running"] = False
        # Close envs (DummyVecEnv supports close())
        for e in envs.values():
            try:
                e.close()
            except Exception:  # pragma: no cover
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI for local debugging
# ─────────────────────────────────────────────────────────────────────────────
def _cli_main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run council training from local featured parquet files")
    parser.add_argument("--symbol", default="TSLA")
    parser.add_argument("--total-steps", type=int, default=20_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = CouncilConfig(
        symbols=[args.symbol],
        total_steps=args.total_steps,
        eval_every_k_steps=args.eval_every,
        warmup_steps=args.warmup_steps,
        arbiter_start_steps=max(args.total_steps // 2, args.warmup_steps + args.eval_every),
    )

    bars_dir = cfg.bars_path()
    featured_data: dict = {args.symbol: {}}
    for tf in cfg.timeframes:
        fp = bars_dir / f"{args.symbol}_{tf}_featured.parquet"
        if fp.exists():
            featured_data[args.symbol][tf] = pd.read_parquet(fp)

    stop_event = threading.Event()
    status: dict = {}
    start_council_training(cfg, featured_data, stop_event, status)
    print(json.dumps({k: v for k, v in status.items() if k != "elo_history"}, default=str, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli_main())
