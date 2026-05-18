"""
elo_tracker.py — Elo rating system for the 5-Model Council.

Uses the classical Elo update rule:
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    new_a      = rating_a + k * (score_a - expected_a)

where ``score_a`` is 1.0 for a win, 0.5 for a tie, 0.0 for a loss.

Models with no rating yet are initialised to ``initial_elo`` (default 1500).
"""
from __future__ import annotations

import time
from typing import Iterable


class EloTracker:
    """
    Tracks Elo ratings for an arbitrary set of model IDs and records the
    full ratings history after every update.
    """

    def __init__(
        self,
        model_ids: Iterable[str] = (),
        initial_elo: float = 1500.0,
        k_factor: float = 32.0,
    ) -> None:
        self.initial_elo = float(initial_elo)
        self.k_factor = float(k_factor)
        self.ratings: dict[str, float] = {m: float(initial_elo) for m in model_ids}
        self.history: list[dict] = []
        self._cycle: int = 0

    # ── Membership ───────────────────────────────────────────────────────
    def register(self, model_id: str) -> None:
        """Add a model_id with the initial rating if it is not already known."""
        if model_id not in self.ratings:
            self.ratings[model_id] = self.initial_elo

    def __contains__(self, model_id: str) -> bool:
        return model_id in self.ratings

    # ── Update mechanics ─────────────────────────────────────────────────
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Probability that A beats B under the Elo model."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(self, model_a: str, model_b: str, score_a: float) -> tuple[float, float]:
        """
        Apply a single Elo match update.

        Parameters
        ----------
        model_a, model_b : Model IDs (auto-registered if missing).
        score_a          : 1.0 = A wins, 0.0 = A loses, 0.5 = draw.

        Returns
        -------
        (new_rating_a, new_rating_b)
        """
        self.register(model_a)
        self.register(model_b)
        ra = self.ratings[model_a]
        rb = self.ratings[model_b]
        ea = self.expected_score(ra, rb)
        eb = 1.0 - ea
        score_b = 1.0 - score_a
        new_a = ra + self.k_factor * (score_a - ea)
        new_b = rb + self.k_factor * (score_b - eb)
        self.ratings[model_a] = new_a
        self.ratings[model_b] = new_b
        return new_a, new_b

    def update_round_robin(self, scores: dict[str, float]) -> None:
        """
        Convenience: pairwise-update every (a, b) combination from a dict of
        ``{model_id: scalar_score}`` (e.g. Sharpe ratios). A scores 1 vs B
        if scores[A] > scores[B], 0 if less, 0.5 if equal.
        """
        ids = list(scores.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                sa = scores[a]
                sb = scores[b]
                if sa > sb:
                    score_a = 1.0
                elif sa < sb:
                    score_a = 0.0
                else:
                    score_a = 0.5
                self.update(a, b, score_a)
        self._snapshot()

    def _snapshot(self) -> None:
        self._cycle += 1
        self.history.append({
            "cycle": self._cycle,
            "ratings": dict(self.ratings),
            "timestamp": time.time(),
        })

    # ── Queries ─────────────────────────────────────────────────────────
    def get_history(self) -> list[dict]:
        """Return a shallow copy of the full ratings history."""
        return list(self.history)

    def get_ranking(self) -> list[tuple[str, float]]:
        """Return [(model_id, rating)] sorted by rating descending."""
        return sorted(self.ratings.items(), key=lambda kv: kv[1], reverse=True)

    def get_leader(self) -> str | None:
        ranking = self.get_ranking()
        return ranking[0][0] if ranking else None

    def gap(self, leader_id: str, model_id: str) -> float:
        """Return ``leader_rating - model_rating`` (>= 0)."""
        if leader_id not in self.ratings or model_id not in self.ratings:
            return 0.0
        return max(0.0, self.ratings[leader_id] - self.ratings[model_id])

    def normalised_gap(self, leader_id: str, model_id: str) -> float:
        """
        Gap normalised by the leader's rating so it is roughly bounded.
        Returns ``0.0`` when ratings are missing.
        """
        if leader_id not in self.ratings or model_id not in self.ratings:
            return 0.0
        leader = self.ratings[leader_id]
        if leader <= 0:
            return 0.0
        return max(0.0, (leader - self.ratings[model_id]) / leader)

    # ── Misc ────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Reset all ratings back to ``initial_elo`` and clear history."""
        for k in list(self.ratings.keys()):
            self.ratings[k] = self.initial_elo
        self.history.clear()
        self._cycle = 0

    def to_dict(self) -> dict:
        """Serialisable snapshot for JSON logging."""
        return {
            "ratings": dict(self.ratings),
            "ranking": self.get_ranking(),
            "leader": self.get_leader(),
            "initial_elo": self.initial_elo,
            "k_factor": self.k_factor,
        }
