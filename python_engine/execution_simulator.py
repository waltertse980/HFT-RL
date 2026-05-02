"""
execution_simulator.py
======================
Execution Simulator for the HFT RL trading system.

Simulates realistic order execution with:
- Gaussian slippage model (spread cost + market impact + noise)
- Latency simulation with exponential jitter
- Partial fill simulation
- Circuit breaker (drawdown / daily-loss / consecutive-loss)
- HK stamp duty and board-lot enforcement
- US tick-size enforcement ($0.01 minimum increment)

Used during backtesting and paper trading to produce realistic P&L estimates.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Deque

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class CircuitBreakerState(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"   # 10% drawdown threshold crossed
    HALTED = "halted"     # 20% drawdown threshold crossed


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Order:
    order_id: str
    side: OrderSide
    quantity: int                   # shares requested
    requested_price: float          # mid price at time of order
    market: str                     # 'us' or 'hk'
    ticker: str
    timestamp: float = field(default_factory=time.time)

    # Filled fields (populated after execution simulation)
    filled_quantity: int = 0
    fill_price: float = 0.0
    slippage_bps: float = 0.0       # basis points of slippage actually applied
    latency_ms: float = 0.0         # simulated latency in milliseconds
    status: OrderStatus = OrderStatus.PENDING
    reject_reason: str = ""


@dataclass
class ExecutionReport:
    order: Order
    commission: float       # total commission paid
    net_pnl: float          # P&L after commission and slippage
    market_impact: float    # estimated market impact in currency units
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Flatten the nested order fields for convenience
        d["order_id"] = self.order.order_id
        d["side"] = self.order.side.value
        d["quantity"] = self.order.filled_quantity
        d["fill_price"] = self.order.fill_price
        d["status"] = self.order.status.value
        return d


@dataclass
class CircuitBreakerStatus:
    state: CircuitBreakerState
    peak_equity: float
    current_equity: float
    drawdown_pct: float
    daily_loss_pct: float
    consecutive_losses: int
    halted_at: Optional[float] = None   # timestamp when halted
    resume_at: Optional[float] = None   # timestamp when auto-resume is allowed


# ---------------------------------------------------------------------------
# Market profile constants
# ---------------------------------------------------------------------------

# Realistic market microstructure parameters by market
MARKET_PROFILES: dict[str, dict] = {
    "us": {
        "base_spread_bps": 2.0,                      # NYSE/NASDAQ typical spread
        "base_latency_ms": 5.0,                      # co-located latency baseline
        "latency_jitter_ms": 3.0,                    # random latency jitter
        "commission_per_share": 0.005,               # $0.005 per share (Alpaca)
        "min_commission": 1.0,                       # minimum $1 per trade
        "partial_fill_prob": 0.15,                   # 15% chance of partial fill
        "min_fill_ratio": 0.5,                       # at least 50% filled if partial
        "market_impact_bps_per_1k_shares": 0.5,      # 0.5 bps per 1 000 shares
    },
    "hk": {
        "base_spread_bps": 5.0,                      # HKEX wider spreads
        "base_latency_ms": 8.0,
        "latency_jitter_ms": 5.0,
        "stamp_duty_rate": 0.0013,                   # 0.13% HK stamp duty per side
        "sfc_levy": 0.000027,                        # SFC transaction levy
        "board_lot": 500,                            # default board lot size
        "partial_fill_prob": 0.25,                   # higher partial fill prob in HK
        "min_fill_ratio": 0.4,
        "market_impact_bps_per_1k_shares": 1.2,
    },
}

# Circuit breaker thresholds
CB_WARNING_DRAWDOWN: float = 0.10       # 10% drawdown from peak → WARNING
CB_HALT_DRAWDOWN: float = 0.20          # 20% drawdown from peak → HALT
CB_DAILY_LOSS_HALT: float = 0.05        # 5% daily loss → HALT
CB_CONSECUTIVE_LOSS_HALT: int = 8       # 8 consecutive losses → WARNING
CB_HALT_DURATION_HOURS: float = 1.0     # auto-resume after 1 hour


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ExecutionSimulator:
    """
    Realistic order execution simulator with:
    - Gaussian slippage model (market impact + spread cost)
    - Latency simulation with jitter
    - Partial fill simulation
    - Circuit breaker (drawdown + daily loss + consecutive loss)
    - HK stamp duty and board-lot enforcement
    - US tick-size enforcement ($0.01 min increment)
    """

    def __init__(
        self,
        market: str = "us",
        initial_capital: float = 100_000.0,
        volume_participation_rate: float = 0.01,  # max 1% of average daily volume
        seed: Optional[int] = None,
    ) -> None:
        if market not in MARKET_PROFILES:
            raise ValueError(
                f"Unknown market '{market}'. Valid options: {list(MARKET_PROFILES.keys())}"
            )

        self._rng = np.random.default_rng(seed)
        self._profile: dict = MARKET_PROFILES[market]
        self._market: str = market
        self._initial_capital: float = initial_capital
        self._volume_participation_rate: float = volume_participation_rate

        # Equity tracking
        self._peak_equity: float = initial_capital
        self._current_equity: float = initial_capital
        self._day_start_equity: float = initial_capital

        # Fill history (ring buffer)
        self._recent_fills: Deque[ExecutionReport] = deque(maxlen=1000)

        # Circuit breaker state
        self._consecutive_losses: int = 0
        self._cb_state: CircuitBreakerState = CircuitBreakerState.NORMAL
        self._cb_halted_at: Optional[float] = None

        # Aggregate cost tracking
        self._total_commission: float = 0.0
        self._total_slippage_cost: float = 0.0

        logger.info(
            "ExecutionSimulator initialised: market=%s capital=%.2f",
            market,
            initial_capital,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_order(
        self,
        order: Order,
        mid_price: float,
        avg_daily_volume: int = 100_000,
    ) -> ExecutionReport:
        """
        Simulate the full lifecycle of a single order.

        Parameters
        ----------
        order:
            Order to execute (mutated in place with fill details).
        mid_price:
            Current mid price of the instrument.
        avg_daily_volume:
            Estimated average daily volume in shares (used for impact cap).

        Returns
        -------
        ExecutionReport with all execution details populated.
        """
        profile = self._profile

        # ----------------------------------------------------------------
        # 1. Circuit breaker check
        # ----------------------------------------------------------------
        cb_status = self.check_circuit_breaker()
        if cb_status.state == CircuitBreakerState.HALTED:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "Circuit breaker HALTED"
            logger.warning(
                "Order %s rejected — circuit breaker is HALTED", order.order_id
            )
            report = ExecutionReport(
                order=order,
                commission=0.0,
                net_pnl=0.0,
                market_impact=0.0,
            )
            self._recent_fills.append(report)
            return report

        # ----------------------------------------------------------------
        # 2. HK board-lot enforcement
        # ----------------------------------------------------------------
        quantity = order.quantity
        if self._market == "hk":
            board_lot: int = profile.get("board_lot", 500)
            quantity = max(board_lot, (quantity // board_lot) * board_lot)
            if quantity == 0:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "Quantity below minimum board lot"
                report = ExecutionReport(
                    order=order,
                    commission=0.0,
                    net_pnl=0.0,
                    market_impact=0.0,
                )
                self._recent_fills.append(report)
                return report

        # Apply volume participation cap
        max_participatable = int(avg_daily_volume * self._volume_participation_rate)
        quantity = min(quantity, max(1, max_participatable))

        # ----------------------------------------------------------------
        # 3. Latency
        # ----------------------------------------------------------------
        base_latency: float = profile["base_latency_ms"]
        jitter: float = profile["latency_jitter_ms"]
        latency_ms: float = base_latency + float(
            self._rng.exponential(jitter)
        )
        order.latency_ms = latency_ms

        # ----------------------------------------------------------------
        # 4. Partial fill simulation
        # ----------------------------------------------------------------
        partial_fill_prob: float = profile["partial_fill_prob"]
        min_fill_ratio: float = profile["min_fill_ratio"]

        if self._rng.random() < partial_fill_prob:
            fill_ratio: float = float(
                self._rng.uniform(min_fill_ratio, 1.0)
            )
            filled_qty: int = max(1, int(quantity * fill_ratio))
            order.status = OrderStatus.PARTIAL
        else:
            filled_qty = quantity
            order.status = OrderStatus.FILLED

        # ----------------------------------------------------------------
        # 5. Slippage
        # ----------------------------------------------------------------
        half_spread_bps: float = profile["base_spread_bps"] / 2.0
        impact_bps_per_1k: float = profile["market_impact_bps_per_1k_shares"]
        market_impact_bps: float = (filled_qty / 1_000.0) * impact_bps_per_1k

        total_slippage_bps: float = half_spread_bps + market_impact_bps

        # Gaussian noise proportional to total slippage
        noise_bps: float = float(
            self._rng.normal(0.0, total_slippage_bps * 0.3)
        )
        total_slippage_bps = max(0.0, total_slippage_bps + noise_bps)

        slippage_multiplier: float = total_slippage_bps / 10_000.0

        if order.side == OrderSide.BUY:
            fill_price: float = mid_price * (1.0 + slippage_multiplier)
        else:
            fill_price = mid_price * (1.0 - slippage_multiplier)

        # Tick size enforcement
        if self._market == "us":
            fill_price = round(fill_price / 0.01) * 0.01   # $0.01 tick
        else:
            fill_price = round(fill_price / 0.001) * 0.001  # HKD 0.001 tick

        # Guard against zero/negative fill price
        fill_price = max(fill_price, 0.001)

        order.filled_quantity = filled_qty
        order.fill_price = fill_price
        order.slippage_bps = total_slippage_bps

        # ----------------------------------------------------------------
        # 6. Commission
        # ----------------------------------------------------------------
        if self._market == "us":
            commission: float = max(
                filled_qty * profile["commission_per_share"],
                profile["min_commission"],
            )
        else:
            commission = fill_price * filled_qty * (
                profile["stamp_duty_rate"] + profile["sfc_levy"]
            )

        # ----------------------------------------------------------------
        # 7. Market impact (in currency units)
        # ----------------------------------------------------------------
        market_impact_dollars: float = (
            fill_price * filled_qty * slippage_multiplier
        )

        # ----------------------------------------------------------------
        # 8. Net P&L approximation (costs are negative from account view)
        # ----------------------------------------------------------------
        net_pnl: float = -(commission + market_impact_dollars)

        # ----------------------------------------------------------------
        # 9. Aggregate cost tracking
        # ----------------------------------------------------------------
        self._total_commission += commission
        self._total_slippage_cost += market_impact_dollars

        # ----------------------------------------------------------------
        # 10. Create ExecutionReport
        # ----------------------------------------------------------------
        report = ExecutionReport(
            order=order,
            commission=commission,
            net_pnl=net_pnl,
            market_impact=market_impact_dollars,
        )

        # ----------------------------------------------------------------
        # 11. Update equity and circuit breaker
        # ----------------------------------------------------------------
        self._update_circuit_breaker(report)

        # ----------------------------------------------------------------
        # 12. Append to fill history
        # ----------------------------------------------------------------
        self._recent_fills.append(report)

        logger.debug(
            "Executed %s %d@%.4f bps=%.2f commission=%.4f latency=%.1fms",
            order.side.value,
            filled_qty,
            fill_price,
            total_slippage_bps,
            commission,
            latency_ms,
        )

        return report

    # ------------------------------------------------------------------

    def _update_circuit_breaker(self, report: ExecutionReport) -> None:
        """
        Update equity tracking and circuit breaker state after a fill.
        """
        # Apply P&L to running equity
        self._current_equity += report.net_pnl

        # Update peak equity (only rises)
        self._peak_equity = max(self._peak_equity, self._current_equity)

        # Compute drawdown metrics
        drawdown_pct: float = (
            (self._peak_equity - self._current_equity) / (self._peak_equity + 1e-8)
        )
        daily_loss_pct: float = (
            (self._day_start_equity - self._current_equity)
            / (self._day_start_equity + 1e-8)
        )
        daily_loss_pct = max(daily_loss_pct, 0.0)  # only positive losses

        # Consecutive loss counter
        if report.net_pnl < 0.0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Evaluate thresholds
        if (
            drawdown_pct >= CB_HALT_DRAWDOWN
            or daily_loss_pct >= CB_DAILY_LOSS_HALT
        ):
            if self._cb_state != CircuitBreakerState.HALTED:
                self._cb_state = CircuitBreakerState.HALTED
                self._cb_halted_at = time.time()
                logger.critical(
                    "CIRCUIT BREAKER HALTED — drawdown=%.2f%% daily_loss=%.2f%% "
                    "consecutive_losses=%d equity=%.2f",
                    drawdown_pct * 100,
                    daily_loss_pct * 100,
                    self._consecutive_losses,
                    self._current_equity,
                )
        elif (
            drawdown_pct >= CB_WARNING_DRAWDOWN
            or self._consecutive_losses >= CB_CONSECUTIVE_LOSS_HALT
        ):
            if self._cb_state == CircuitBreakerState.NORMAL:
                logger.warning(
                    "CIRCUIT BREAKER WARNING — drawdown=%.2f%% consecutive_losses=%d",
                    drawdown_pct * 100,
                    self._consecutive_losses,
                )
            self._cb_state = CircuitBreakerState.WARNING
        else:
            # Reset from WARNING back to NORMAL if conditions improve
            if self._cb_state == CircuitBreakerState.WARNING:
                logger.info("Circuit breaker returning to NORMAL state.")
            if self._cb_state != CircuitBreakerState.HALTED:
                self._cb_state = CircuitBreakerState.NORMAL

    # ------------------------------------------------------------------

    def check_circuit_breaker(self) -> CircuitBreakerStatus:
        """
        Return the current circuit breaker status.

        If currently HALTED and the auto-resume window has elapsed,
        the breaker is automatically reset to NORMAL.
        """
        resume_at: Optional[float] = None

        if self._cb_state == CircuitBreakerState.HALTED and self._cb_halted_at is not None:
            resume_at = self._cb_halted_at + CB_HALT_DURATION_HOURS * 3600.0
            if time.time() >= resume_at:
                logger.info(
                    "Circuit breaker auto-resuming after %.1f hour halt.",
                    CB_HALT_DURATION_HOURS,
                )
                self._cb_state = CircuitBreakerState.NORMAL
                self._cb_halted_at = None
                resume_at = None

        drawdown_pct: float = (
            (self._peak_equity - self._current_equity) / (self._peak_equity + 1e-8)
        )
        daily_loss_pct: float = max(
            (self._day_start_equity - self._current_equity)
            / (self._day_start_equity + 1e-8),
            0.0,
        )

        return CircuitBreakerStatus(
            state=self._cb_state,
            peak_equity=self._peak_equity,
            current_equity=self._current_equity,
            drawdown_pct=drawdown_pct,
            daily_loss_pct=daily_loss_pct,
            consecutive_losses=self._consecutive_losses,
            halted_at=self._cb_halted_at,
            resume_at=resume_at,
        )

    # ------------------------------------------------------------------

    def reset_daily(self) -> None:
        """
        Reset daily tracking variables (call at the start of each trading day).
        """
        logger.info(
            "Daily reset: day_start_equity updated %.2f → %.2f",
            self._day_start_equity,
            self._current_equity,
        )
        self._day_start_equity = self._current_equity
        self._consecutive_losses = 0

    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """
        Return a summary statistics dictionary.
        """
        return {
            "total_fills": len(self._recent_fills),
            "total_commission_paid": self._total_commission,
            "total_slippage_cost": self._total_slippage_cost,
            "current_equity": self._current_equity,
            "peak_equity": self._peak_equity,
            "drawdown_pct": (
                (self._peak_equity - self._current_equity)
                / (self._peak_equity + 1e-8)
            ),
            "circuit_breaker_state": self._cb_state.value,
            "consecutive_losses": self._consecutive_losses,
        }

    # ------------------------------------------------------------------

    def replay_fills(self, n: int = 20) -> list[dict]:
        """
        Return the last *n* fill reports as plain dictionaries.

        Parameters
        ----------
        n:
            Maximum number of recent fills to return.
        """
        fills = list(self._recent_fills)
        return [f.to_dict() for f in fills[-n:]]


# ---------------------------------------------------------------------------
# Standalone backtest wrapper
# ---------------------------------------------------------------------------

def simulate_backtest(
    actions: np.ndarray,
    prices: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    market: str = "us",
    ticker: str = "AAPL",
    initial_capital: float = 100_000.0,
    seed: int = 42,
) -> dict:
    """
    Replay a sequence of RL actions through the execution simulator and
    return a comprehensive set of performance metrics.

    Parameters
    ----------
    actions:
        Integer array of RL actions: 0 = sell, 1 = hold, 2 = buy.
    prices:
        Mid-price array aligned with *actions*.
    volumes:
        Optional average daily volume array.  Falls back to 1 000 000.
    market:
        'us' or 'hk'.
    ticker:
        Instrument ticker label (for order IDs).
    initial_capital:
        Starting capital in currency units.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        equity_curve, total_return_pct, sharpe, max_drawdown,
        total_commission, total_slippage, n_trades, fill_rate,
        circuit_breaker_triggers
    """
    if len(actions) != len(prices):
        raise ValueError(
            f"actions length {len(actions)} != prices length {len(prices)}"
        )

    sim = ExecutionSimulator(
        market=market,
        initial_capital=initial_capital,
        seed=seed,
    )

    equity_curve: list[float] = [initial_capital]
    position: int = 0          # +1 = long, 0 = flat, -1 = short
    n_trades: int = 0
    n_intended: int = 0
    cb_triggers: int = 0
    order_counter: int = 0

    default_adv: int = 1_000_000

    for step_idx in range(len(actions)):
        action: int = int(actions[step_idx])
        mid_price: float = float(prices[step_idx])
        adv: int = int(volumes[step_idx]) if volumes is not None else default_adv

        order: Optional[Order] = None

        if action == 2 and position <= 0:
            # BUY signal — go long (or close short and go long)
            n_intended += 1
            order_counter += 1
            quantity = max(1, int(initial_capital * 0.01 / (mid_price + 1e-8)))
            order = Order(
                order_id=f"{ticker}_{order_counter:06d}",
                side=OrderSide.BUY,
                quantity=quantity,
                requested_price=mid_price,
                market=market,
                ticker=ticker,
            )

        elif action == 0 and position >= 0:
            # SELL signal — go short (or close long and go short)
            n_intended += 1
            order_counter += 1
            quantity = max(1, int(initial_capital * 0.01 / (mid_price + 1e-8)))
            order = Order(
                order_id=f"{ticker}_{order_counter:06d}",
                side=OrderSide.SELL,
                quantity=quantity,
                requested_price=mid_price,
                market=market,
                ticker=ticker,
            )

        if order is not None:
            report = sim.simulate_order(order, mid_price=mid_price, avg_daily_volume=adv)

            if report.order.status == OrderStatus.REJECTED:
                cb_triggers += 1
            else:
                n_trades += 1
                if order.side == OrderSide.BUY:
                    position = 1
                else:
                    position = -1

        equity_curve.append(sim._current_equity)

    # Reset CB trigger count uses the cb state check approach —
    # we approximate circuit_breaker_triggers as rejections above.
    # (A more accurate count would require hooking into _update_circuit_breaker.)

    stats = sim.get_stats()
    equity_arr = np.array(equity_curve, dtype=np.float64)

    # Total return
    total_return_pct: float = (equity_arr[-1] - equity_arr[0]) / (equity_arr[0] + 1e-8) * 100.0

    # Annualised Sharpe (assume 1-minute bars, 252 trading days, 390 min/day)
    bar_returns: np.ndarray = np.diff(equity_arr) / (equity_arr[:-1] + 1e-8)
    bars_per_year: int = 252 * 390
    if bar_returns.std() > 1e-12:
        sharpe: float = float(
            bar_returns.mean() / bar_returns.std() * np.sqrt(bars_per_year)
        )
    else:
        sharpe = 0.0

    # Max drawdown
    rolling_peak: np.ndarray = np.maximum.accumulate(equity_arr)
    drawdowns: np.ndarray = (rolling_peak - equity_arr) / (rolling_peak + 1e-8)
    max_drawdown: float = float(drawdowns.max()) * 100.0  # percent

    fill_rate: float = (n_trades / n_intended) if n_intended > 0 else 0.0

    return {
        "equity_curve": equity_curve,
        "total_return_pct": total_return_pct,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "total_commission": stats["total_commission_paid"],
        "total_slippage": stats["total_slippage_cost"],
        "n_trades": n_trades,
        "fill_rate": fill_rate,
        "circuit_breaker_triggers": cb_triggers,
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    print("ExecutionSimulator demo — generating 100 random orders...")
    sim = ExecutionSimulator(market="us", initial_capital=100_000.0, seed=42)

    rng = np.random.default_rng(42)
    for i in range(100):
        price = 150.0 + float(rng.normal(0, 2.0))
        side = OrderSide.BUY if rng.random() > 0.5 else OrderSide.SELL
        order = Order(
            order_id=f"demo_{i:04d}",
            side=side,
            quantity=int(rng.integers(100, 1000)),
            requested_price=price,
            market="us",
            ticker="AAPL",
        )
        report = sim.simulate_order(order, mid_price=price, avg_daily_volume=5_000_000)
        if i % 20 == 0:
            print(
                f"  Order {i:3d}: {side.value:<4s}  "
                f"{report.order.filled_quantity:>5d} @ {report.order.fill_price:>8.2f}  "
                f"commission=${report.commission:>6.2f}  "
                f"slippage={report.order.slippage_bps:>5.1f} bps  "
                f"status={report.order.status.value}"
            )

    print("\nStats:", json.dumps(sim.get_stats(), indent=2))

    # --- Quick backtest demo ---
    print("\n--- Backtest demo (500 steps) ---")
    price_series = 150.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.05, 500))
    action_series = np.random.default_rng(0).integers(0, 3, 500)

    results = simulate_backtest(
        actions=action_series,
        prices=price_series,
        market="us",
        ticker="AAPL",
        initial_capital=100_000.0,
        seed=42,
    )
    results_display = {k: v for k, v in results.items() if k != "equity_curve"}
    results_display["equity_start"] = results["equity_curve"][0]
    results_display["equity_end"] = results["equity_curve"][-1]
    print(json.dumps(results_display, indent=2))
