"""
trade_journal.py — Per-model trade-action log + behavioural-cloning helper.

The journal stores ``TradeEntry`` records keyed by the council step index.
Inferior models can later query the leader's journal at the same step to
receive a behavioural-cloning bonus when their action matches the leader's
(see ``inverse_rl.compute_irl_bonus``).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import Iterable


@dataclass
class TradeEntry:
    """A single (step, action) record with optional outcome metadata."""
    step: int
    action: int                # 0 FLAT, 1 LONG, 2 SHORT
    obs_hash: int = 0          # optional; can be used to detect identical states
    price: float = 0.0
    portfolio_value: float = 0.0
    reward: float = 0.0
    position_after: int = 0
    day_date: str = ""
    timestamp: float = field(default_factory=time.time)


class TradeJournal:
    """
    Bounded FIFO log of trade actions for one model.

    Lookups are O(1) by step index via an internal mapping.
    """

    def __init__(self, model_id: str, max_entries: int = 100_000) -> None:
        self.model_id = str(model_id)
        self.max_entries = int(max_entries)
        self._entries: deque[TradeEntry] = deque(maxlen=self.max_entries)
        self._by_step: dict[int, TradeEntry] = {}

    # ── Mutators ────────────────────────────────────────────────────────
    def record(self, entry: TradeEntry) -> None:
        """Add a TradeEntry. If the deque is full, evict the oldest."""
        # If we're about to evict, remove the corresponding step mapping too
        if len(self._entries) == self.max_entries:
            evicted = self._entries[0]
            self._by_step.pop(evicted.step, None)
        self._entries.append(entry)
        self._by_step[entry.step] = entry

    def record_action(
        self,
        step: int,
        action: int,
        price: float = 0.0,
        portfolio_value: float = 0.0,
        reward: float = 0.0,
        position_after: int = 0,
        day_date: str = "",
        obs_hash: int = 0,
    ) -> TradeEntry:
        """Convenience wrapper that constructs and records a TradeEntry."""
        entry = TradeEntry(
            step=int(step),
            action=int(action),
            obs_hash=int(obs_hash),
            price=float(price),
            portfolio_value=float(portfolio_value),
            reward=float(reward),
            position_after=int(position_after),
            day_date=str(day_date),
        )
        self.record(entry)
        return entry

    def clear(self) -> None:
        self._entries.clear()
        self._by_step.clear()

    # ── Lookups ─────────────────────────────────────────────────────────
    def get(self, step: int) -> TradeEntry | None:
        return self._by_step.get(int(step))

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterable[TradeEntry]:
        return iter(self._entries)

    def latest(self, n: int = 1) -> list[TradeEntry]:
        """Return the most recent ``n`` entries (chronological order)."""
        if n <= 0:
            return []
        if n >= len(self._entries):
            return list(self._entries)
        return list(self._entries)[-n:]

    # ── Behavioural-cloning helper ──────────────────────────────────────
    def to_observation_bonus(self, inferior_action: int, step: int) -> float:
        """
        Return a base bonus of ``1.0`` if the inferior model's action at
        ``step`` matches the leader's recorded action; ``-0.25`` if it
        actively disagrees (different non-zero positions); ``0.0`` otherwise.

        Designed to be scaled by ``lambda_irl * min(gap, 1.0)`` in
        :func:`council.inverse_rl.compute_irl_bonus`.
        """
        entry = self._by_step.get(int(step))
        if entry is None:
            return 0.0
        leader_action = int(entry.action)
        inf = int(inferior_action)
        if leader_action == inf:
            return 1.0
        # Opposite-direction trade is mildly penalised
        if {leader_action, inf} == {1, 2}:
            return -0.25
        return 0.0

    # ── Serialisation ───────────────────────────────────────────────────
    def to_dict_list(self, n: int | None = None) -> list[dict]:
        """Return up to ``n`` most-recent entries as plain dicts (JSON-ready)."""
        if n is None:
            items = list(self._entries)
        else:
            items = self.latest(n)
        return [asdict(e) for e in items]
