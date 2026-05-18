"""
inverse_rl.py — Behavioural-cloning bonus from the council's leader.

When the council is in its competitive phase, each inferior agent receives a
small extra reward for matching the leader's action at the same step:

    bonus = lambda_irl * base_bonus * min(gap, 1.0)

where ``base_bonus`` ∈ {-0.25, 0.0, +1.0} comes from
:meth:`TradeJournal.to_observation_bonus`, and ``gap`` is the normalised Elo
gap between the leader and the inferior.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — only for type checkers
    from .trade_journal import TradeJournal


def compute_irl_bonus(
    leader_journal: "TradeJournal",
    inferior_action: int,
    inferior_step: int,
    gap: float,
    lambda_irl: float,
) -> float:
    """
    Return the inverse-RL (behavioural-cloning) bonus for an inferior agent.

    Parameters
    ----------
    leader_journal  : Leader model's TradeJournal.
    inferior_action : Action chosen by the inferior agent at ``inferior_step``.
    inferior_step   : The council step index (same scale as the leader).
    gap             : Elo gap (normalised, e.g. ``(leader - inferior) / leader``).
                      Values are clipped to [0, 1] for stability.
    lambda_irl      : Scaling coefficient (see CouncilConfig.lambda_irl).

    Returns
    -------
    Float bonus to add to the inferior agent's reward.
    """
    if leader_journal is None:
        return 0.0
    base = float(leader_journal.to_observation_bonus(int(inferior_action), int(inferior_step)))
    gap_clipped = max(0.0, min(float(gap), 1.0))
    return float(lambda_irl) * base * gap_clipped


def batch_irl_bonus(
    leader_journal: "TradeJournal",
    actions: list[int],
    steps: list[int],
    gap: float,
    lambda_irl: float,
) -> list[float]:
    """Vectorised helper — returns one bonus per (action, step) pair."""
    out: list[float] = []
    for a, s in zip(actions, steps):
        out.append(compute_irl_bonus(leader_journal, a, s, gap, lambda_irl))
    return out
