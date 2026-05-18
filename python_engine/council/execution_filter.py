"""
execution_filter.py — Models D and E: Elo-weighted ensemble with drawdown veto.

D and E are NOT trained policies. They synthesise an execution decision from
the three trained models (A, B, C) by:

    1. Polling each trained model for its action (deterministic predict).
    2. Computing per-model weights from current Elo ratings (softmax over
       ratings divided by 100 for numerical stability).
    3. Forming a weighted vote over actions {0, 1, 2}.
    4. Vetoing the trade to FLAT if the *current* trade's unrealised
       drawdown (vs. entry price) exceeds ``threshold_pct``.

D and E differ only in ``threshold_pct``:
    • D — tight stop:     ``filter_d_threshold = -0.0005``  (-5 bps)
    • E — wide stop:      ``filter_e_threshold = -0.0020``  (-20 bps)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


class ExecutionFilter:
    """Elo-weighted ensemble + drawdown veto."""

    def __init__(
        self,
        threshold_pct: float,
        model_ids: list[str],
        elo_tracker: Any,
    ) -> None:
        self.threshold_pct = float(threshold_pct)
        self.model_ids = list(model_ids)
        self.elo_tracker = elo_tracker

    # ── Weight computation (live, from Elo) ──────────────────────────────
    def _current_weights(self) -> dict[str, float]:
        """Softmax over ratings/100 across registered model_ids."""
        if self.elo_tracker is None:
            n = max(1, len(self.model_ids))
            return {m: 1.0 / n for m in self.model_ids}
        ratings = {m: float(self.elo_tracker.ratings.get(m, 0.0)) for m in self.model_ids}
        # Numerical stability: subtract the max
        max_r = max(ratings.values()) if ratings else 0.0
        exps = {m: math.exp((r - max_r) / 100.0) for m, r in ratings.items()}
        total = sum(exps.values())
        if total <= 0:
            n = max(1, len(self.model_ids))
            return {m: 1.0 / n for m in self.model_ids}
        return {m: v / total for m, v in exps.items()}

    # ── Public entry point ──────────────────────────────────────────────
    def decide(
        self,
        obs: np.ndarray,
        current_price: float,
        entry_price: float,
        position: int,
        models_dict: dict[str, Any],
    ) -> int:
        """
        Return the filtered action (0=FLAT, 1=LONG, 2=SHORT).

        ``models_dict`` maps model_id → SB3-compatible model exposing
        ``predict(obs, deterministic=True) -> (action, _state)``.

        ``position`` is the *current* position of the executing portfolio
        (-1, 0, +1). ``entry_price`` is the price at which that position
        was opened (or 0.0 if flat).
        """
        # ── Drawdown veto ─────────────────────────────────────────────
        if position != 0 and entry_price > 0 and current_price > 0:
            if position == 1:
                # Long: drawdown if current < entry
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                # Short: drawdown if current > entry
                pnl_pct = (entry_price - current_price) / entry_price
            if pnl_pct <= self.threshold_pct:
                # Force close
                return 0

        # ── Weighted vote over actions ────────────────────────────────
        weights = self._current_weights()
        scores = np.zeros(3, dtype=np.float64)  # index = action
        for mid, model in models_dict.items():
            if mid not in self.model_ids:
                continue
            w = weights.get(mid, 0.0)
            if w <= 0 or model is None:
                continue
            try:
                action, _ = model.predict(obs, deterministic=True)
                a = int(np.asarray(action).flatten()[0])
            except Exception:  # pragma: no cover — defensive
                continue
            if a in (0, 1, 2):
                scores[a] += w
        if not np.any(scores):
            return 0
        return int(np.argmax(scores))

    # ── Introspection ───────────────────────────────────────────────────
    def weights_snapshot(self) -> dict[str, float]:
        """Return the current weight vector for logging / debugging."""
        return self._current_weights()
