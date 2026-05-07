"""
paper_trader_lob.py
Live paper trading loop for LOB-HFT v2.

Uses:
  - LiveLOBFeatureBuilder (live_quote_adapter.py) for real-time LOB features
  - RiskManager (risk_controls.py) for circuit-breaker protection
  - LOB PPO model (checkpoints_lob/) for action decisions
  - alpaca-py TradingClient for order submission (paper trading mode)

Trading loop:
  1. Subscribe to Alpaca NBBO quote stream for target symbols
  2. On each quote: build LOB feature obs via LiveLOBFeatureBuilder
  3. Pass obs to model.predict() -> action (0=FLAT, 1=LONG, 2=SHORT)
  4. Check RiskManager.check_and_approve()
  5. If approved and action changes position: submit market order via TradingClient
  6. Log quote-to-order latency

Startup:
  - Load PPO model + VecNormalize from checkpoints_lob/<checkpoint_dir>/
  - VecNorm: training=False, norm_reward=False
  - device="cpu" for inference

Shutdown:
  - Flatten all positions before stopping
  - Log final PnL summary
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

load_dotenv()
logger = logging.getLogger(__name__)

try:
    from live_quote_adapter import LiveLOBFeatureBuilder
    from risk_controls import RiskManager
except ImportError:
    from hft_lob.live_quote_adapter import LiveLOBFeatureBuilder
    from hft_lob.risk_controls import RiskManager

BASE_DIR = Path(__file__).parent.parent
CHECKPOINTS_LOB = BASE_DIR / "checkpoints_lob"

ACTION_MAP = {0: "FLAT", 1: "LONG", 2: "SHORT"}
POSITION_DELTA = {
    ("FLAT",  "LONG"):  +1,
    ("FLAT",  "SHORT"): -1,
    ("LONG",  "FLAT"):  -1,
    ("LONG",  "SHORT"): -2,
    ("SHORT", "FLAT"):  +1,
    ("SHORT", "LONG"):  +2,
}


class LOBPaperTrader:
    """
    Live paper trading agent for LOB-HFT v2.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        symbols: list[str],
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: str = "https://paper-api.alpaca.markets",
        qty_per_trade: int = 10,
        max_daily_loss: float = -500.0,
        max_drawdown_pct: float = -0.05,
    ):
        self.symbols = symbols
        self.qty = qty_per_trade

        # Load model
        ckpt_path = CHECKPOINTS_LOB / checkpoint_dir
        model_path = ckpt_path / "ppo_final.zip"
        vecnorm_path = ckpt_path / "vecnorm.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"LOB model not found: {model_path}")

        # Build a dummy env just to load VecNormalize
        try:
            from lob_features import FEATURE_COLS
            from lob_environment import LOBTradingEnv
            import pandas as pd
            dummy_df = pd.DataFrame(
                np.zeros((100, len(FEATURE_COLS) + 1)),
                columns=FEATURE_COLS + ["label"]
            )
            dummy_env = DummyVecEnv([lambda: LOBTradingEnv(dummy_df)])
        except ImportError:
            from hft_lob.lob_features import FEATURE_COLS
            from hft_lob.lob_environment import LOBTradingEnv
            import pandas as pd
            dummy_df = pd.DataFrame(
                np.zeros((100, len(FEATURE_COLS) + 1)),
                columns=FEATURE_COLS + ["label"]
            )
            dummy_env = DummyVecEnv([lambda: LOBTradingEnv(dummy_df)])

        if vecnorm_path.exists():
            self.env = VecNormalize.load(str(vecnorm_path), dummy_env)
            self.env.training = False
            self.env.norm_reward = False
        else:
            self.env = dummy_env
            logger.warning("vecnorm.pkl not found - running without normalisation")

        self.model = PPO.load(str(model_path), env=self.env, device="cpu")
        logger.info("LOB model loaded from %s", ckpt_path)

        # Feature builder
        self.feature_builder = LiveLOBFeatureBuilder(symbols=symbols)

        # Risk manager
        self.risk = RiskManager(
            max_daily_loss_usd=max_daily_loss,
            max_drawdown_pct=max_drawdown_pct,
        )

        # Alpaca trading client
        _api_key    = api_key    or os.environ.get("ALPACA_API_KEY", "")
        _api_secret = api_secret or os.environ.get("ALPACA_API_SECRET", "")
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            self.trading_client = TradingClient(_api_key, _api_secret, paper=True)
            self.MarketOrderRequest = MarketOrderRequest
            self.OrderSide = OrderSide
            self.TimeInForce = TimeInForce
            logger.info("Alpaca TradingClient connected (paper=True)")
        except Exception as exc:
            logger.warning("Alpaca TradingClient unavailable: %s - orders will be logged only", exc)
            self.trading_client = None

        # State
        self.positions: dict[str, str] = {s: "FLAT" for s in symbols}
        self.equity: float = 100_000.0
        self.running = False

        # Alpaca quote stream
        try:
            from alpaca.data.live import StockDataStream
            self.stream = StockDataStream(_api_key, _api_secret)
        except Exception as exc:
            logger.warning("StockDataStream unavailable: %s", exc)
            self.stream = None

    def _submit_order(self, symbol: str, side: str, qty: int) -> None:
        """Submit a market order. Logs only if TradingClient is unavailable."""
        logger.info("[order] %s %s qty=%d", side, symbol, qty)
        if self.trading_client is None:
            return
        try:
            req = self.MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=self.OrderSide.BUY if side == "BUY" else self.OrderSide.SELL,
                time_in_force=self.TimeInForce.DAY,
            )
            self.trading_client.submit_order(req)
        except Exception as exc:
            logger.error("[order] submit failed for %s: %s", symbol, exc)
            self.risk.record_error()
            return
        self.risk.record_success()

    def _on_quote(self, quote) -> None:
        """Called on every incoming NBBO quote."""
        symbol = quote.symbol
        t0 = time.perf_counter()

        # Build LOB feature observation
        obs = self.feature_builder.on_quote(symbol, quote)
        if obs is None:
            return  # Not enough history yet

        current_pos_str = self.positions.get(symbol, "FLAT")
        pos_int = {"FLAT": 0, "LONG": 1, "SHORT": -1}.get(current_pos_str, 0)
        obs_with_pos = np.append(obs, float(pos_int)).astype(np.float32)

        # Normalise using VecNorm statistics
        if isinstance(self.env, VecNormalize):
            obs_norm = self.env.normalize_obs(obs_with_pos[np.newaxis, :])[0]
        else:
            obs_norm = obs_with_pos

        # Model inference
        action_arr, _ = self.model.predict(obs_norm[np.newaxis, :], deterministic=True)
        action = int(action_arr[0])
        target_pos_str = ACTION_MAP[action]

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug("[%s] action=%s pos=%s latency=%.2fms", symbol, target_pos_str, current_pos_str, latency_ms)

        # Check risk before acting
        if not self.risk.check_and_approve(pos_int, self.equity, action):
            logger.warning("[risk] order blocked for %s (halted=%s)", symbol, self.risk.is_halted)
            return

        # Execute position change
        if target_pos_str != current_pos_str:
            delta = POSITION_DELTA.get((current_pos_str, target_pos_str), 0)
            if delta > 0:
                self._submit_order(symbol, "BUY", abs(delta) * self.qty)
            elif delta < 0:
                self._submit_order(symbol, "SELL", abs(delta) * self.qty)
            self.positions[symbol] = target_pos_str

    async def start(self) -> None:
        """Start the live paper trading loop."""
        if self.stream is None:
            raise RuntimeError("StockDataStream not available. Check alpaca-py installation.")
        self.running = True
        self.risk.reset_daily()
        logger.info("LOB paper trader starting for symbols: %s", self.symbols)

        async def _quote_handler(quote):
            try:
                self._on_quote(quote)
            except Exception as exc:
                logger.error("quote handler error: %s", exc)

        self.stream.subscribe_quotes(_quote_handler, *self.symbols)
        await self.stream.run()

    def stop(self) -> None:
        """Stop the trading loop and flatten all positions."""
        self.running = False
        logger.info("LOB paper trader stopping - flattening positions: %s", self.positions)
        for symbol, pos in self.positions.items():
            if pos == "LONG":
                self._submit_order(symbol, "SELL", self.qty)
            elif pos == "SHORT":
                self._submit_order(symbol, "BUY", self.qty)
        self.positions = {s: "FLAT" for s in self.symbols}
        if self.stream:
            asyncio.create_task(self.stream.stop_ws())
        logger.info("LOB paper trader stopped.")
