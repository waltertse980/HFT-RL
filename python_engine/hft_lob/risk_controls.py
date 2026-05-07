"""
risk_controls.py
================
Phase 7 — Circuit-breaker risk manager for the LOB paper / live trader.

Tracks four families of safety conditions and gates every order:
  1. Daily realised + unrealised PnL floor
  2. Peak-to-trough drawdown floor
  3. Per-symbol position limit
  4. Consecutive submission errors → halt

When any limit trips, `is_halted` is set True and `check_and_approve()`
returns False until `reset_daily()` is called.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RiskManager:
    max_daily_loss_usd: float = -500.0
    max_drawdown_pct: float = -0.05
    max_position_units: int = 1
    max_consecutive_errors: int = 5

    # Live state
    starting_equity: float = 100_000.0
    peak_equity: float = field(default=0.0)
    last_equity: float = field(default=0.0)
    daily_pnl: float = 0.0
    consecutive_errors: int = 0
    is_halted: bool = False
    halted_reason: str = ""
    halted_at: float = 0.0

    # ------------------------------------------------------------------
    def reset_daily(self, starting_equity: float | None = None) -> None:
        if starting_equity is not None:
            self.starting_equity = float(starting_equity)
            self.last_equity = self.starting_equity
            self.peak_equity = self.starting_equity
        else:
            self.last_equity = self.starting_equity
            self.peak_equity = self.starting_equity
        self.daily_pnl = 0.0
        self.consecutive_errors = 0
        self.is_halted = False
        self.halted_reason = ""
        self.halted_at = 0.0
        logger.info("[risk] daily reset; equity=%.2f", self.starting_equity)

    # ------------------------------------------------------------------
    def update_equity(self, current_equity: float) -> None:
        self.last_equity = float(current_equity)
        if self.last_equity > self.peak_equity:
            self.peak_equity = self.last_equity
        self.daily_pnl = self.last_equity - self.starting_equity

    # ------------------------------------------------------------------
    def record_error(self) -> None:
        self.consecutive_errors += 1
        if self.consecutive_errors >= self.max_consecutive_errors:
            self._halt(
                f"consecutive_errors >= {self.max_consecutive_errors}"
            )

    def record_success(self) -> None:
        self.consecutive_errors = 0

    # ------------------------------------------------------------------
    def _halt(self, reason: str) -> None:
        if not self.is_halted:
            self.is_halted = True
            self.halted_reason = reason
            self.halted_at = time.time()
            logger.error("[risk] HALTED: %s", reason)

    # ------------------------------------------------------------------
    def check_and_approve(
        self,
        current_position: int,
        current_equity: float,
        proposed_action: int,
    ) -> bool:
        """
        Returns True if the proposed action is permitted.
        proposed_action: 0=FLAT, 1=LONG, 2=SHORT (matches LOBTradingEnv).
        """
        if self.is_halted:
            return False

        self.update_equity(current_equity)

        # 1. Daily PnL floor
        if self.daily_pnl <= self.max_daily_loss_usd:
            self._halt(
                f"daily_pnl {self.daily_pnl:.2f} <= max_daily_loss {self.max_daily_loss_usd:.2f}"
            )
            return False

        # 2. Drawdown floor
        if self.peak_equity > 0:
            dd = (self.last_equity - self.peak_equity) / self.peak_equity
            if dd <= self.max_drawdown_pct:
                self._halt(
                    f"drawdown {dd:.4f} <= max_drawdown {self.max_drawdown_pct:.4f}"
                )
                return False

        # 3. Position-limit check
        target_map = {0: 0, 1: 1, 2: -1}
        target = target_map.get(int(proposed_action), 0)
        if abs(target) > self.max_position_units:
            return False

        return True

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "is_halted": self.is_halted,
            "halted_reason": self.halted_reason,
            "halted_at": self.halted_at,
            "starting_equity": self.starting_equity,
            "peak_equity": self.peak_equity,
            "last_equity": self.last_equity,
            "daily_pnl": self.daily_pnl,
            "consecutive_errors": self.consecutive_errors,
            "max_daily_loss_usd": self.max_daily_loss_usd,
            "max_drawdown_pct": self.max_drawdown_pct,
        }
