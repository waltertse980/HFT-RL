"""
queue_simulator.py
==================
Phase 4 — Queue-position fill simulator for the event-driven LOB backtester.

A simple, dependency-free model of how a passive limit order at the top of
book would be filled, given a stream of trade events on the same side.

Usage
-----
    sim = QueuePositionSimulator(initial_queue_ahead_size=1500.0)
    # ... for each subsequent trade on the same side and price:
    filled, remaining = sim.on_trade(trade_size=200.0, our_size=100.0)

The model assumes FIFO at the price level: every trade consumes resting
size from the front of the queue. Once the cumulative trade size exceeds
`initial_queue_ahead_size`, our order starts to fill.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QueuePositionSimulator:
    """FIFO queue-position model for a single resting limit order."""

    initial_queue_ahead_size: float = 0.0
    # Mutable state
    consumed_ahead: float = 0.0
    filled_size: float = 0.0
    cancelled: bool = False

    def reset(self, initial_queue_ahead_size: float) -> None:
        self.initial_queue_ahead_size = float(initial_queue_ahead_size)
        self.consumed_ahead = 0.0
        self.filled_size = 0.0
        self.cancelled = False

    def queue_ahead_remaining(self) -> float:
        return max(0.0, self.initial_queue_ahead_size - self.consumed_ahead)

    def on_trade(self, trade_size: float, our_size: float) -> tuple[float, float]:
        """
        Process a trade on our side at our price.

        Returns
        -------
        filled_now : float
            Size filled in this single trade event.
        remaining_size : float
            Our remaining un-filled size after this event.
        """
        if self.cancelled:
            return 0.0, max(0.0, our_size - self.filled_size)

        ahead = self.queue_ahead_remaining()
        size = float(trade_size)
        filled_now = 0.0

        if ahead > 0:
            consume = min(size, ahead)
            self.consumed_ahead += consume
            size -= consume

        if size > 0:
            remaining_us = max(0.0, our_size - self.filled_size)
            fill = min(size, remaining_us)
            self.filled_size += fill
            filled_now = fill

        remaining = max(0.0, our_size - self.filled_size)
        return filled_now, remaining

    def on_cancel(self) -> None:
        self.cancelled = True

    def is_fully_filled(self, our_size: float) -> bool:
        return self.filled_size >= our_size - 1e-9
