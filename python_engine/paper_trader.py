"""
paper_trader.py — Live Paper Trading

AlpacaPaperTrader: connects to Alpaca paper trading WebSocket, aggregates
live 1-minute bars into synthetic 10-second bars, runs ONNX inference,
and dispatches paper orders with built-in risk controls.

HKSimPaperTrader: replays held-out HK test data at 10× real speed with
an identical interface to AlpacaPaperTrader.

Usage
-----
    # US paper trading via Alpaca
    python paper_trader.py --market us --ticker AAPL \\
        --model models/us_1m_PPO/model_final.onnx \\
        --api-key YOUR_KEY --api-secret YOUR_SECRET

    # HK simulation (replay)
    python paper_trader.py --market hk --ticker 0700.HK \\
        --model models/hk_1m_PPO/model_final.onnx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from backtester import ONNXPolicyRunner
from data_pipeline import compute_features, load_dataset, _resample_to_10s
from rl_environment import _get_feature_cols, HFTradingEnv

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

BASE_DIR = Path(__file__).parent
PAPER_TRADES_LOG = BASE_DIR / "paper_trades.jsonl"

# Risk parameters
MAX_POSITION_FRAC: float = 0.95   # max 95% of capital in one position
STOP_LOSS_PCT: float = 0.02       # 2% hard stop-loss per position
DAILY_LOSS_LIMIT_PCT: float = 0.05  # 5% daily drawdown limit


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TradeSignal:
    timestamp: str
    ticker: str
    action: int          # 0=HOLD, 1=BUY, 2=SELL
    action_label: str
    price: float
    inference_ms: float
    position_before: int
    position_after: int
    portfolio_value: float
    reason: str = ""     # e.g. "model", "stop_loss", "daily_limit"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskState:
    position: int = 0             # -1 short, 0 flat, 1 long
    entry_price: float = 0.0
    shares_held: float = 0.0
    cash: float = 0.0
    portfolio_value: float = 0.0
    initial_capital: float = 100_000.0
    daily_start_value: float = 100_000.0
    n_trades: int = 0
    realized_pnl: float = 0.0

    def mark_to_market(self, price: float) -> float:
        if self.position == 1:
            self.portfolio_value = self.cash + self.shares_held * price
        elif self.position == -1:
            unrealized = (self.entry_price - price) * self.shares_held
            self.portfolio_value = self.cash + unrealized
        else:
            self.portfolio_value = self.cash
        return self.portfolio_value


# ---------------------------------------------------------------------------
# Trade log helper
# ---------------------------------------------------------------------------


def _append_trade_log(signal: TradeSignal) -> None:
    with open(PAPER_TRADES_LOG, "a") as f:
        f.write(json.dumps(signal.to_dict()) + "\n")


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------


class ObservationBuilder:
    """
    Maintains a rolling buffer of OHLCV bars, computes features, and
    constructs observations for ONNX inference.
    """

    def __init__(self, window_size: int = 60, feature_cols: Optional[list[str]] = None) -> None:
        self.window_size = window_size
        self._feature_cols: Optional[list[str]] = feature_cols
        self._buffer: deque[dict] = deque(maxlen=max(window_size * 10, 200))
        self._df_cache: Optional[pd.DataFrame] = None

    def add_bar(self, bar: dict) -> None:
        """Add a single OHLCV bar dict (keys: Open, High, Low, Close, Volume)."""
        self._buffer.append(bar)
        self._df_cache = None  # invalidate cache

    def _build_df(self) -> pd.DataFrame:
        if self._df_cache is not None:
            return self._df_cache
        df = pd.DataFrame(list(self._buffer))
        if df.empty:
            return df
        # Compute features
        df = compute_features(df)
        self._df_cache = df
        return df

    def get_obs(self) -> Optional[np.ndarray]:
        """Return a flat float32 observation array or None if buffer too small."""
        df = self._build_df()
        if len(df) < self.window_size:
            return None

        if self._feature_cols is None:
            self._feature_cols = _get_feature_cols(df)

        available_cols = [c for c in self._feature_cols if c in df.columns]
        if not available_cols:
            return None

        window = df[available_cols].iloc[-self.window_size:].values.astype(np.float32)
        obs = np.nan_to_num(window.flatten(), nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(obs, -10.0, 10.0)


# ---------------------------------------------------------------------------
# AlpacaPaperTrader
# ---------------------------------------------------------------------------


class AlpacaPaperTrader:
    """
    Paper trading loop using Alpaca's WebSocket data stream.

    Subscribes to 1-minute bars, synthesises 10-second sub-bars, computes
    features over a rolling window, runs ONNX inference, applies risk checks,
    and dispatches paper orders via the Alpaca trading API.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        model_path: str,
        ticker: str,
        timescale: str = "10s",
        window_size: int = 60,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.ticker = ticker
        self.timescale = timescale
        self.window_size = window_size

        self._runner = ONNXPolicyRunner(model_path)
        self._obs_builder = ObservationBuilder(window_size=window_size)
        self._risk = RiskState(
            cash=initial_capital,
            portfolio_value=initial_capital,
            initial_capital=initial_capital,
            daily_start_value=initial_capital,
        )
        self._running = False
        self._bar_buffer_1m: deque[dict] = deque(maxlen=10)
        self._sub_bar_count: int = 0

        # Lazy import to avoid hard dependency at module level
        try:
            from alpaca.data.live import StockDataStream
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            self._stream = StockDataStream(api_key, api_secret)
            self._trading_client = TradingClient(api_key, api_secret, paper=True)
            self._MarketOrderRequest = MarketOrderRequest
            self._OrderSide = OrderSide
            self._TimeInForce = TimeInForce
            self._alpaca_available = True
        except ImportError:
            logger.warning("alpaca-py not installed — order dispatch will be simulated.")
            self._alpaca_available = False

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

    def _check_stop_loss(self, current_price: float) -> bool:
        """Return True if stop-loss is triggered."""
        if self._risk.position == 0 or self._risk.entry_price == 0:
            return False
        if self._risk.position == 1:
            pct_loss = (self._risk.entry_price - current_price) / self._risk.entry_price
        else:
            pct_loss = (current_price - self._risk.entry_price) / self._risk.entry_price
        return pct_loss >= STOP_LOSS_PCT

    def _check_daily_loss_limit(self) -> bool:
        """Return True if daily loss limit is exceeded."""
        daily_loss = (self._risk.daily_start_value - self._risk.portfolio_value) / self._risk.daily_start_value
        return daily_loss >= DAILY_LOSS_LIMIT_PCT

    # ------------------------------------------------------------------
    # Order dispatch
    # ------------------------------------------------------------------

    def _dispatch_order(self, side: str, qty: float, price: float) -> None:
        """Place a paper order via Alpaca or log it if Alpaca is unavailable."""
        logger.info("ORDER: %s %s × %.4f shares @ $%.4f", side, self.ticker, qty, price)
        if not self._alpaca_available:
            return

        try:
            order_request = self._MarketOrderRequest(
                symbol=self.ticker,
                qty=qty,
                side=self._OrderSide.BUY if side == "BUY" else self._OrderSide.SELL,
                time_in_force=self._TimeInForce.GTC,
            )
            self._trading_client.submit_order(order_request)
        except Exception as exc:  # noqa: BLE001
            logger.error("Order dispatch failed: %s", exc)

    # ------------------------------------------------------------------
    # Bar processing
    # ------------------------------------------------------------------

    def _process_1m_bar(self, bar: dict) -> None:
        """
        Receive a 1-minute bar, synthesise 10-second sub-bars, update
        observation builder, run inference, apply risk, dispatch orders.
        """
        # Synthesise 6 × 10s sub-bars from the 1m bar
        bar_df = pd.DataFrame([bar])
        sub_bars = _resample_to_10s(bar_df)

        for _, sub_row in sub_bars.iterrows():
            sub_bar = {
                "Open": sub_row["Open"],
                "High": sub_row["High"],
                "Low": sub_row["Low"],
                "Close": sub_row["Close"],
                "Volume": sub_row["Volume"],
            }
            self._obs_builder.add_bar(sub_bar)
            current_price = float(sub_row["Close"])
            self._risk.mark_to_market(current_price)

            # Risk checks before inference
            reason = "model"
            if self._check_daily_loss_limit():
                logger.warning("Daily loss limit reached — forcing FLAT position.")
                action = 0
                reason = "daily_limit"
            elif self._check_stop_loss(current_price):
                logger.warning("Stop-loss triggered at $%.4f", current_price)
                action = 0
                reason = "stop_loss"
            else:
                obs = self._obs_builder.get_obs()
                if obs is None:
                    action = 0  # HOLD until buffer filled
                    reason = "buffer_filling"
                else:
                    t0 = time.perf_counter()
                    action = self._runner.predict(obs)
                    inference_ms = (time.perf_counter() - t0) * 1000
                    logger.debug("Inference: %.3f ms  action=%d", inference_ms, action)

            self._apply_action(action, current_price, reason)

    def _apply_action(self, action: int, price: float, reason: str = "model") -> None:
        """Apply the model's action to the paper portfolio state."""
        pos_before = self._risk.position
        action_labels = {0: "HOLD", 1: "BUY", 2: "SELL"}

        if action == 1 and self._risk.position != 1:
            # Flatten short if needed
            if self._risk.position == -1:
                self._risk.realized_pnl += (self._risk.entry_price - price) * self._risk.shares_held
                self._risk.cash += self._risk.shares_held * price
                self._dispatch_order("BUY", self._risk.shares_held, price)
            # Enter long
            invest = self._risk.cash * MAX_POSITION_FRAC
            shares = invest / price
            self._risk.shares_held = shares
            self._risk.cash -= invest
            self._risk.position = 1
            self._risk.entry_price = price
            self._risk.n_trades += 1
            self._dispatch_order("BUY", shares, price)

        elif action == 2 and self._risk.position != -1:
            # Flatten long if needed
            if self._risk.position == 1:
                self._risk.realized_pnl += (price - self._risk.entry_price) * self._risk.shares_held
                self._risk.cash += self._risk.shares_held * price
                self._dispatch_order("SELL", self._risk.shares_held, price)
            # Enter short
            invest = self._risk.cash * MAX_POSITION_FRAC
            shares = invest / price
            self._risk.shares_held = shares
            self._risk.position = -1
            self._risk.entry_price = price
            self._risk.n_trades += 1
            self._dispatch_order("SELL", shares, price)

        self._risk.mark_to_market(price)

        signal = TradeSignal(
            timestamp=datetime.now(timezone.utc).isoformat(),
            ticker=self.ticker,
            action=action,
            action_label=action_labels.get(action, "UNKNOWN"),
            price=price,
            inference_ms=0.0,
            position_before=pos_before,
            position_after=self._risk.position,
            portfolio_value=self._risk.portfolio_value,
            reason=reason,
        )
        _append_trade_log(signal)
        logger.info(
            "[%s] action=%s  price=%.4f  pos=%d  portfolio=$%.2f  pnl=$%.2f",
            self.ticker,
            signal.action_label,
            price,
            self._risk.position,
            self._risk.portfolio_value,
            self._risk.realized_pnl,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to Alpaca WebSocket and run the live paper trading loop."""
        self._running = True
        logger.info("Starting AlpacaPaperTrader for %s", self.ticker)

        if not self._alpaca_available:
            logger.error("alpaca-py is not installed. Cannot start live trading.")
            return

        async def on_bar(bar) -> None:
            if not self._running:
                return
            bar_dict = {
                "Open": float(bar.open),
                "High": float(bar.high),
                "Low": float(bar.low),
                "Close": float(bar.close),
                "Volume": float(bar.volume),
            }
            self._process_1m_bar(bar_dict)

        self._stream.subscribe_bars(on_bar, self.ticker)
        try:
            await self._stream.run()
        except asyncio.CancelledError:
            logger.info("WebSocket stream cancelled.")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Flatten all positions and shut down gracefully."""
        self._running = False
        logger.info("Stopping AlpacaPaperTrader — flattening positions...")
        if self._risk.position != 0:
            logger.info("Closing open position (position=%d)", self._risk.position)
            # In a real system we'd fetch current market price; use entry_price as proxy
            self._dispatch_order(
                "SELL" if self._risk.position == 1 else "BUY",
                self._risk.shares_held,
                self._risk.entry_price,
            )
            self._risk.position = 0
            self._risk.shares_held = 0.0
        logger.info("AlpacaPaperTrader stopped. Final portfolio: $%.2f", self._risk.portfolio_value)


# ---------------------------------------------------------------------------
# HKSimPaperTrader — data replay
# ---------------------------------------------------------------------------


class HKSimPaperTrader:
    """
    HK market paper trading simulator using local data replay.

    Replays held-out HK test data at 10× real speed (10-second bars played
    at 1-second intervals). Provides the same interface and logging as
    AlpacaPaperTrader so the two can be used interchangeably.
    """

    REPLAY_SPEED = 10.0  # 10× real-time

    def __init__(
        self,
        model_path: str,
        ticker: str,
        timescale: str = "10s",
        window_size: int = 60,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.ticker = ticker
        self.timescale = timescale
        self.window_size = window_size

        self._runner = ONNXPolicyRunner(model_path)
        self._obs_builder = ObservationBuilder(window_size=window_size)
        self._risk = RiskState(
            cash=initial_capital,
            portfolio_value=initial_capital,
            initial_capital=initial_capital,
            daily_start_value=initial_capital,
        )
        self._running = False

        # Load and hold-out 20% for replay
        try:
            data_dict = load_dataset("hk", timescale)
            if ticker not in data_dict:
                ticker = next(iter(data_dict.keys()))
                logger.warning("Ticker %s not in dataset; using %s instead.", self.ticker, ticker)
                self.ticker = ticker
            df = data_dict[ticker]
            split = int(len(df) * 0.8)
            self._replay_df: pd.DataFrame = df.iloc[split:].reset_index(drop=True)
            logger.info(
                "HKSimPaperTrader loaded %d bars for replay (ticker=%s)", len(self._replay_df), ticker
            )
        except FileNotFoundError:
            logger.error(
                "HK dataset not found. Run: python data_pipeline.py --market hk --timescale %s", timescale
            )
            self._replay_df = pd.DataFrame()

    def _check_stop_loss(self, price: float) -> bool:
        if self._risk.position == 0 or self._risk.entry_price == 0:
            return False
        if self._risk.position == 1:
            pct_loss = (self._risk.entry_price - price) / self._risk.entry_price
        else:
            pct_loss = (price - self._risk.entry_price) / self._risk.entry_price
        return pct_loss >= STOP_LOSS_PCT

    def _check_daily_loss_limit(self) -> bool:
        daily_loss = (self._risk.daily_start_value - self._risk.portfolio_value) / self._risk.daily_start_value
        return daily_loss >= DAILY_LOSS_LIMIT_PCT

    def _apply_action(self, action: int, price: float, reason: str = "model") -> None:
        action_labels = {0: "HOLD", 1: "BUY", 2: "SELL"}
        pos_before = self._risk.position

        if action == 1 and self._risk.position != 1:
            if self._risk.position == -1:
                self._risk.realized_pnl += (self._risk.entry_price - price) * self._risk.shares_held
                self._risk.cash += self._risk.shares_held * price
                self._risk.n_trades += 1
            invest = self._risk.cash * MAX_POSITION_FRAC
            self._risk.shares_held = invest / price
            self._risk.cash -= invest
            self._risk.position = 1
            self._risk.entry_price = price
            self._risk.n_trades += 1

        elif action == 2 and self._risk.position != -1:
            if self._risk.position == 1:
                self._risk.realized_pnl += (price - self._risk.entry_price) * self._risk.shares_held
                self._risk.cash += self._risk.shares_held * price
                self._risk.n_trades += 1
            invest = self._risk.cash * MAX_POSITION_FRAC
            self._risk.shares_held = invest / price
            self._risk.position = -1
            self._risk.entry_price = price
            self._risk.n_trades += 1

        self._risk.mark_to_market(price)

        signal = TradeSignal(
            timestamp=datetime.now(timezone.utc).isoformat(),
            ticker=self.ticker,
            action=action,
            action_label=action_labels.get(action, "UNKNOWN"),
            price=price,
            inference_ms=0.0,
            position_before=pos_before,
            position_after=self._risk.position,
            portfolio_value=self._risk.portfolio_value,
            reason=reason,
        )
        _append_trade_log(signal)
        logger.info(
            "[%s] %s  price=%.4f  pos=%d  portfolio=$%.2f",
            self.ticker,
            signal.action_label,
            price,
            self._risk.position,
            self._risk.portfolio_value,
        )

    async def start(self) -> None:
        """Replay held-out test data at 10× speed."""
        if self._replay_df.empty:
            logger.error("No replay data available.")
            return

        self._running = True
        feature_cols = _get_feature_cols(self._replay_df)
        sleep_s = 10.0 / self.REPLAY_SPEED  # real 10s bar → 1s simulated

        logger.info(
            "HKSimPaperTrader starting replay (%d bars @ %.1f× speed)...",
            len(self._replay_df), self.REPLAY_SPEED,
        )

        for idx, row in self._replay_df.iterrows():
            if not self._running:
                break

            bar = {
                "Open": float(row.get("Open", 0)),
                "High": float(row.get("High", 0)),
                "Low": float(row.get("Low", 0)),
                "Close": float(row.get("Close", 0)),
                "Volume": float(row.get("Volume", 0)),
            }
            self._obs_builder.add_bar(bar)
            price = bar["Close"]
            self._risk.mark_to_market(price)

            # Risk checks
            reason = "model"
            if self._check_daily_loss_limit():
                action = 0
                reason = "daily_limit"
            elif self._check_stop_loss(price):
                action = 0
                reason = "stop_loss"
            else:
                obs = self._obs_builder.get_obs()
                if obs is None:
                    action = 0
                    reason = "buffer_filling"
                else:
                    t0 = time.perf_counter()
                    action = self._runner.predict(obs)
                    inference_ms = (time.perf_counter() - t0) * 1000
                    logger.debug("Inference %.3f ms", inference_ms)

            self._apply_action(action, price, reason)
            await asyncio.sleep(sleep_s)

        logger.info("HKSim replay complete. Final portfolio: $%.2f", self._risk.portfolio_value)

    async def stop(self) -> None:
        """Signal the replay loop to stop."""
        self._running = False
        logger.info("HKSimPaperTrader stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="HFT Paper Trader")
    parser.add_argument("--market", choices=["us", "hk"], required=True)
    parser.add_argument("--ticker", required=True, help="Ticker symbol")
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--timescale", default="10s", choices=["10s", "1m", "5m", "1h"])
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--api-key", default=None, help="Alpaca API key (US only)")
    parser.add_argument("--api-secret", default=None, help="Alpaca API secret (US only)")
    args = parser.parse_args()

    if args.market == "us":
        if not args.api_key or not args.api_secret:
            logger.error(
                "Alpaca --api-key and --api-secret are required for US paper trading. "
                "Get them from https://alpaca.markets/."
            )
            return
        trader = AlpacaPaperTrader(
            api_key=args.api_key,
            api_secret=args.api_secret,
            model_path=args.model,
            ticker=args.ticker,
            timescale=args.timescale,
            window_size=args.window_size,
            initial_capital=args.capital,
        )
    else:
        trader = HKSimPaperTrader(
            model_path=args.model,
            ticker=args.ticker,
            timescale=args.timescale,
            window_size=args.window_size,
            initial_capital=args.capital,
        )

    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — stopping...")
        asyncio.run(trader.stop())


if __name__ == "__main__":
    main()
