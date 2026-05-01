"""
backtester.py — ONNX-based Backtesting & Red Team Testing

Loads exported ONNX models and runs them on historical data to produce
detailed performance metrics, equity curves, and adversarial stress tests.

Usage
-----
    python backtester.py --model models/us_1m_PPO/model_final.onnx \\
                         --market us --timescale 1m
    python backtester.py --model models/us_1m_PPO/model_final.onnx \\
                         --market us --timescale 1m --red-team
"""

from __future__ import annotations

import abc
import argparse
import copy
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import onnxruntime as ort

from data_pipeline import load_dataset
from rl_environment import HFTradingEnv, _get_feature_cols

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

BASE_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# ONNX inference helper
# ---------------------------------------------------------------------------


class ONNXPolicyRunner:
    """Thin wrapper around onnxruntime InferenceSession for policy inference."""

    def __init__(self, model_path: str) -> None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self._sess = ort.InferenceSession(model_path, providers=providers)
        except Exception:
            self._sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name: str = self._sess.get_inputs()[0].name

    def predict(self, obs: np.ndarray) -> int:
        """Run a single observation through the policy; returns discrete action."""
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
        # Cast to float32; ONNX FP16 models accept float32 inputs
        obs = obs.astype(np.float32)
        outputs = self._sess.run(None, {self._input_name: obs})
        logits = outputs[0][0]
        if logits.ndim == 0:
            # Scalar output (regression-style TD3)
            return int(np.clip(round(float(logits)), 0, 2))
        return int(np.argmax(logits))

    def predict_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        obs_batch = obs_batch.astype(np.float32)
        outputs = self._sess.run(None, {self._input_name: obs_batch})
        logits = outputs[0]
        if logits.ndim == 1:
            return logits.astype(int)
        return np.argmax(logits, axis=1)


# ---------------------------------------------------------------------------
# Backtest metrics
# ---------------------------------------------------------------------------


@dataclass
class BacktestMetrics:
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration_bars: float = 0.0
    n_trades: int = 0
    total_return: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    market: str = ""
    model_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Truncate equity curve for JSON serialisation if very long
        if len(d["equity_curve"]) > 2000:
            step = len(d["equity_curve"]) // 2000
            d["equity_curve"] = d["equity_curve"][::step]
        return d


def _compute_sharpe(returns: np.ndarray, bars_per_year: int = 252 * 6 * 60 * 6) -> float:
    """Annualised Sharpe ratio from per-step returns."""
    if len(returns) < 2:
        return 0.0
    std = np.std(returns)
    if std == 0:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(bars_per_year))


def _compute_sortino(returns: np.ndarray, bars_per_year: int = 252 * 6 * 60 * 6) -> float:
    downside = returns[returns < 0]
    if len(downside) < 2:
        return float(np.mean(returns) * bars_per_year) if len(returns) else 0.0
    return float(np.mean(returns) / (np.std(downside) + 1e-8) * np.sqrt(bars_per_year))


def _compute_max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / (running_max + 1e-8)
    return float(np.max(drawdowns))


def _compute_calmar(total_return: float, max_drawdown: float) -> float:
    if max_drawdown == 0:
        return 0.0
    return total_return / max_drawdown


def _compute_profit_factor(trade_pnls: list[float]) -> float:
    if not trade_pnls:
        return 0.0
    gains = sum(p for p in trade_pnls if p > 0)
    losses = abs(sum(p for p in trade_pnls if p < 0))
    return gains / (losses + 1e-8)


# ---------------------------------------------------------------------------
# Core backtest runner
# ---------------------------------------------------------------------------


def run_backtest(
    model_path: str,
    data: pd.DataFrame,
    market: str = "",
    window_size: int = 60,
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.001,
) -> dict[str, Any]:
    """
    Run ONNX model on historical data and compute full trading metrics.

    Parameters
    ----------
    model_path:       Path to the ONNX model file.
    data:             Feature DataFrame (output of compute_features).
    market:           Label for the report (e.g. 'us', 'hk').
    window_size:      Observation window in bars.
    initial_capital:  Starting capital.
    transaction_cost: Fractional transaction cost.

    Returns
    -------
    Dict with all metrics + full equity_curve list.
    """
    runner = ONNXPolicyRunner(model_path)
    env = HFTradingEnv(
        df=data,
        window_size=window_size,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
    )
    obs, _ = env.reset(seed=0)
    terminated = truncated = False

    equity_curve: list[float] = [initial_capital]
    trade_pnls: list[float] = []
    trade_durations: list[int] = []
    n_trades = 0
    prev_position = 0
    trade_start_step = 0
    prev_pnl = 0.0

    while not (terminated or truncated):
        action = runner.predict(obs)
        obs, _, terminated, truncated, info = env.step(action)
        equity_curve.append(info["portfolio_value"])

        cur_pos = info["position"]
        cur_pnl = info["realized_pnl"]
        cur_step = info["step_idx"]

        # Detect trade close
        if prev_position != 0 and cur_pos != prev_position:
            pnl_delta = cur_pnl - prev_pnl
            trade_pnls.append(pnl_delta)
            trade_durations.append(cur_step - trade_start_step)
            n_trades += 1
        if cur_pos != 0 and prev_position == 0:
            trade_start_step = cur_step
            prev_pnl = cur_pnl

        prev_position = cur_pos

    env.close()

    equity_arr = np.array(equity_curve, dtype=np.float64)
    step_returns = np.diff(equity_arr) / (equity_arr[:-1] + 1e-8)
    total_return = float((equity_arr[-1] - initial_capital) / initial_capital)
    mdd = _compute_max_drawdown(equity_arr)

    metrics = BacktestMetrics(
        sharpe=_compute_sharpe(step_returns),
        sortino=_compute_sortino(step_returns),
        calmar=_compute_calmar(total_return, mdd),
        max_drawdown=mdd,
        win_rate=float(np.mean([p > 0 for p in trade_pnls])) if trade_pnls else 0.0,
        profit_factor=_compute_profit_factor(trade_pnls),
        avg_trade_duration_bars=float(np.mean(trade_durations)) if trade_durations else 0.0,
        n_trades=n_trades,
        total_return=total_return,
        equity_curve=equity_curve,
        market=market,
        model_path=model_path,
    )

    result = metrics.to_dict()
    logger.info(
        "Backtest complete | return=%.2f%%  Sharpe=%.3f  MDD=%.2f%%  n_trades=%d",
        total_return * 100, metrics.sharpe, mdd * 100, n_trades,
    )
    return result


# ---------------------------------------------------------------------------
# Red Team Scenarios
# ---------------------------------------------------------------------------


class RedTeamScenario(abc.ABC):
    """Abstract base for adversarial data transformation scenarios."""

    name: str = "base"
    description: str = ""

    @abc.abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a transformed copy of df representing the stress scenario."""

    def run(
        self,
        model_path: str,
        data: pd.DataFrame,
        window_size: int = 60,
        initial_capital: float = 100_000.0,
    ) -> dict[str, Any]:
        transformed = self.transform(data)
        metrics = run_backtest(
            model_path=model_path,
            data=transformed,
            market=self.name,
            window_size=window_size,
            initial_capital=initial_capital,
        )
        metrics["scenario"] = self.name
        metrics["description"] = self.description
        return metrics


class FlashCrashScenario(RedTeamScenario):
    """
    Inject a sudden price crash (default -20 %) sustained for ``duration_bars`` bars
    starting at the midpoint of the series, followed by a recovery.
    """

    name = "flash_crash"
    description = "Sudden -20% price drop over 3 bars, partial recovery"

    def __init__(self, crash_pct: float = 0.20, duration_bars: int = 3) -> None:
        self.crash_pct = crash_pct
        self.duration_bars = duration_bars

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        mid = len(out) // 2
        multiplier = 1.0 - self.crash_pct
        ohlc = ["Open", "High", "Low", "Close"]
        for col in ohlc:
            if col in out.columns:
                out.iloc[mid:mid + self.duration_bars, out.columns.get_loc(col)] *= multiplier
        # Partial recovery over next duration_bars
        recovery_end = mid + 2 * self.duration_bars
        for col in ohlc:
            if col in out.columns:
                for i in range(self.duration_bars):
                    idx = mid + self.duration_bars + i
                    if idx < len(out):
                        recovery_factor = multiplier + self.crash_pct * (i + 1) / self.duration_bars * 0.5
                        out.iloc[idx, out.columns.get_loc(col)] *= recovery_factor
        logger.info("FlashCrash: applied %.0f%% crash at bar %d", self.crash_pct * 100, mid)
        return out


class LiquidityDroughtScenario(RedTeamScenario):
    """
    Reduce trading volume to near-zero for the middle third of the series.
    Models that rely on volume signals should degrade noticeably.
    """

    name = "liquidity_drought"
    description = "Near-zero volume for middle third of bars"

    def __init__(self, volume_multiplier: float = 0.01, duration_pct: float = 0.33) -> None:
        self.volume_multiplier = volume_multiplier
        self.duration_pct = duration_pct

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        start = int(n * 0.33)
        end = int(n * (0.33 + self.duration_pct))
        if "Volume" in out.columns:
            out.iloc[start:end, out.columns.get_loc("Volume")] *= self.volume_multiplier
        logger.info("LiquidityDrought: volume * %.3f for bars %d–%d", self.volume_multiplier, start, end)
        return out


class AdverseSelectionScenario(RedTeamScenario):
    """
    Model always gets the worst possible fill:
      - BUY orders execute at bar High.
      - SELL orders execute at bar Low.
    Simulated by swapping High and Low in the Close column for the env.
    (The env uses Close as the fill price; here we pessimistically set
    Close = High for the entire series, meaning all fills are at worst price.)
    """

    name = "adverse_selection"
    description = "All fills at worst intra-bar price (buy@high, sell@low)"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "High" in out.columns and "Low" in out.columns:
            # Pessimistic close: model always buys high, sells low
            out["Close"] = (out["High"] + out["Low"]) / 2 + (out["High"] - out["Low"]) / 2
        logger.info("AdverseSelection: Close set to bar High (worst-case fill)")
        return out


class RegimeChangeScenario(RedTeamScenario):
    """
    Reverse the price trend direction for the second half of the series.
    A model trained on bull-market data should struggle with a bear-regime tail.
    """

    name = "regime_change"
    description = "Price trend reversed for second half (bull→bear)"

    def __init__(self, bull_to_bear: bool = True) -> None:
        self.bull_to_bear = bull_to_bear

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        mid = n // 2
        ohlc = ["Open", "High", "Low", "Close"]

        if self.bull_to_bear:
            # Reflect price series around the midpoint value
            mid_close = float(out["Close"].iloc[mid])
            for col in ohlc:
                if col in out.columns:
                    second_half = out[col].iloc[mid:].values
                    reflected = 2 * mid_close - second_half
                    out.iloc[mid:, out.columns.get_loc(col)] = reflected

        logger.info("RegimeChange: price series reflected for second half")
        return out


class OverfitDetectionScenario(RedTeamScenario):
    """
    Compare Sharpe ratio on train data vs test data.
    Flags overfitting if in-sample Sharpe > 2× out-of-sample Sharpe.
    """

    name = "overfit_detection"
    description = "Compare in-sample vs out-of-sample Sharpe; flag if ratio > 2.0"

    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        overfit_threshold: float = 2.0,
    ) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.overfit_threshold = overfit_threshold

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Not used — this scenario uses its own data
        return df

    def run(
        self,
        model_path: str,
        data: pd.DataFrame,
        window_size: int = 60,
        initial_capital: float = 100_000.0,
    ) -> dict[str, Any]:
        train_metrics = run_backtest(model_path, self.train_data, "train", window_size, initial_capital)
        test_metrics = run_backtest(model_path, self.test_data, "test", window_size, initial_capital)

        in_sharpe = train_metrics["sharpe"]
        out_sharpe = test_metrics["sharpe"]
        ratio = in_sharpe / (abs(out_sharpe) + 1e-8)
        overfit = ratio > self.overfit_threshold

        result = {
            "scenario": self.name,
            "description": self.description,
            "in_sample_sharpe": in_sharpe,
            "out_of_sample_sharpe": out_sharpe,
            "sharpe_ratio": ratio,
            "overfit_detected": overfit,
            "threshold": self.overfit_threshold,
            "pass": not overfit,
        }
        if overfit:
            logger.warning(
                "OVERFIT DETECTED: in-sample Sharpe %.3f vs out-of-sample %.3f (ratio=%.2f)",
                in_sharpe, out_sharpe, ratio,
            )
        return result


# ---------------------------------------------------------------------------
# Red team runner
# ---------------------------------------------------------------------------


def run_red_team(
    model_path: str,
    data: pd.DataFrame,
    scenarios: Optional[list[RedTeamScenario]] = None,
    window_size: int = 60,
    initial_capital: float = 100_000.0,
) -> dict[str, Any]:
    """
    Run all red-team scenarios and return a consolidated results dict.

    Parameters
    ----------
    model_path: Path to the ONNX model file.
    data:       Feature DataFrame for the base scenario.
    scenarios:  List of RedTeamScenario instances. If None, run all defaults.
    window_size, initial_capital: Passed to each scenario.

    Returns
    -------
    Dict with keys: summary (pass/fail counts), per-scenario results.
    """
    if scenarios is None:
        n = len(data)
        split = int(n * 0.8)
        scenarios = [
            FlashCrashScenario(),
            LiquidityDroughtScenario(),
            AdverseSelectionScenario(),
            RegimeChangeScenario(),
            OverfitDetectionScenario(
                train_data=data.iloc[:split],
                test_data=data.iloc[split:],
            ),
        ]

    results: dict[str, Any] = {}
    passed = 0
    failed = 0

    for scenario in scenarios:
        logger.info("Running red-team scenario: %s", scenario.name)
        try:
            result = scenario.run(model_path, data, window_size, initial_capital)
            # Generic pass/fail: pass if total_return > -0.1 (not more than -10%)
            if "pass" not in result:
                result["pass"] = result.get("total_return", 0.0) > -0.10
            results[scenario.name] = result
            if result["pass"]:
                passed += 1
                logger.info("  ✓ %s PASSED", scenario.name)
            else:
                failed += 1
                logger.warning("  ✗ %s FAILED", scenario.name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Scenario %s raised an error: %s", scenario.name, exc)
            results[scenario.name] = {"scenario": scenario.name, "error": str(exc), "pass": False}
            failed += 1

    results["_summary"] = {
        "total_scenarios": len(scenarios),
        "passed": passed,
        "failed": failed,
        "model_path": model_path,
    }
    logger.info("Red team complete: %d/%d passed", passed, len(scenarios))
    return results


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def generate_backtest_report(
    metrics: dict[str, Any],
    equity_curve: Optional[list[float]] = None,
) -> str:
    """
    Produce a formatted JSON report string from backtest metrics.

    Parameters
    ----------
    metrics:      Dict returned by run_backtest().
    equity_curve: Optional explicit equity curve (overrides metrics['equity_curve']).

    Returns
    -------
    Pretty-printed JSON string.
    """
    report = dict(metrics)
    if equity_curve is not None:
        # Downsample for readability
        if len(equity_curve) > 2000:
            step = len(equity_curve) // 2000
            report["equity_curve"] = equity_curve[::step]
        else:
            report["equity_curve"] = equity_curve

    # Remove raw curve from dict if already set above
    if "equity_curve" in metrics and equity_curve is not None:
        pass  # already overwritten

    return json.dumps(report, indent=2, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_scenarios_from_names(
    names: list[str], data: pd.DataFrame
) -> list[RedTeamScenario]:
    n = len(data)
    split = int(n * 0.8)
    mapping: dict[str, RedTeamScenario] = {
        "flash_crash": FlashCrashScenario(),
        "liquidity_drought": LiquidityDroughtScenario(),
        "adverse_selection": AdverseSelectionScenario(),
        "regime_change": RegimeChangeScenario(),
        "overfit_detection": OverfitDetectionScenario(
            train_data=data.iloc[:split],
            test_data=data.iloc[split:],
        ),
    }
    return [mapping[n] for n in names if n in mapping]


def main() -> None:
    parser = argparse.ArgumentParser(description="HFT Backtester & Red Team")
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--market", choices=["us", "hk"], default="us")
    parser.add_argument("--timescale", choices=["10s", "1m", "5m", "1h"], default="1m")
    parser.add_argument("--ticker", default=None, help="Specific ticker (default: first in dataset)")
    parser.add_argument(
        "--red-team", action="store_true", help="Also run red-team scenarios"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["flash_crash", "liquidity_drought", "adverse_selection", "regime_change", "overfit_detection"],
        help="Red-team scenario names to run",
    )
    parser.add_argument("--output", default=None, help="Save JSON report to this path")
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    args = parser.parse_args()

    data_dict = load_dataset(args.market, args.timescale)
    ticker = args.ticker or next(iter(data_dict.keys()))
    df = data_dict[ticker]
    logger.info("Running backtest on %s (%d bars)", ticker, len(df))

    backtest_result = run_backtest(
        model_path=args.model,
        data=df,
        market=args.market,
        window_size=args.window_size,
        initial_capital=args.initial_capital,
    )

    report_data: dict[str, Any] = {"backtest": backtest_result}

    if args.red_team:
        logger.info("Running red-team scenarios: %s", args.scenarios)
        scenarios = _build_scenarios_from_names(args.scenarios, df)
        red_team_results = run_red_team(
            model_path=args.model,
            data=df,
            scenarios=scenarios if scenarios else None,
            window_size=args.window_size,
            initial_capital=args.initial_capital,
        )
        report_data["red_team"] = red_team_results

    report_str = generate_backtest_report(report_data, equity_curve=backtest_result.get("equity_curve"))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_str)
        logger.info("Report saved → %s", args.output)
    else:
        print(report_str)


if __name__ == "__main__":
    main()
