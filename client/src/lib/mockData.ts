// Mock data generators for HFT dashboard demo
import { format, subMinutes, subDays, subHours } from "date-fns";

export type Action = "BUY" | "SELL" | "HOLD";
export type Market = "US" | "HK";

export interface CandleBar {
  time: string;
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  signal?: Action;
}

export interface PaperTrade {
  id: number;
  action: Action;
  price: number;
  quantity: number;
  pnl: number;
  cumulativePnl: number;
  time: string;
  inferenceMsec: number;
  market: Market;
  ticker: string;
}

export interface TrainingJob {
  id: string;
  market: string;
  timescale: string;
  algo: string;
  totalTimesteps: number;
  progressPct: number;
  currentReward: number;
  elapsedSeconds: number;
  status: "running" | "done" | "error";
  rewardHistory: { step: number; reward: number }[];
}

export interface ModelCheckpoint {
  id: string;
  name: string;
  market: string;
  timescale: string;
  algo: string;
  sharpe: number | null;
  createdAt: string;
}

export interface BacktestRun {
  id: number;
  market: string;
  ticker: string;
  timescale: string;
  model: string;
  startDate: string;
  endDate: string;
  sharpe: number;
  sortino: number;
  calmar: number;
  maxDrawdown: number;
  totalReturn: number;
  winRate: number;
  nTrades: number;
  equityCurve: { date: string; value: number; drawdown?: number }[];
  trades: BacktestTrade[];
  createdAt: string;
}

export interface BacktestTrade {
  id: number;
  time: string;
  action: Action;
  price: number;
  qty: number;
  pnl: number;
  cumulativePnl: number;
}

// ---- Candle generation ----

export function generateCandleBars(
  count = 120,
  basePrice = 185.5,
  volatility = 0.003
): CandleBar[] {
  const bars: CandleBar[] = [];
  let price = basePrice;
  const now = new Date();

  // Sine wave with noise
  for (let i = count; i >= 0; i--) {
    const sineWave = Math.sin(i * 0.08) * basePrice * 0.015;
    const noise = (Math.random() - 0.5) * basePrice * volatility;
    const trend = (count - i) * 0.002; // slight uptrend
    const open = price;
    const change = sineWave * 0.1 + noise + trend * 0.01;
    const close = Math.max(open + change, open * 0.97);
    const high = Math.max(open, close) + Math.random() * basePrice * 0.004;
    const low = Math.min(open, close) - Math.random() * basePrice * 0.004;
    const volume = Math.floor(50000 + Math.random() * 200000);

    const timestamp = subMinutes(now, i * 0.5);

    // Inject signals randomly
    let signal: Action | undefined;
    const r = Math.random();
    if (r < 0.06) signal = "BUY";
    else if (r < 0.12) signal = "SELL";

    bars.push({
      time: format(timestamp, "HH:mm:ss"),
      timestamp,
      open,
      high,
      low,
      close,
      volume,
      signal,
    });

    price = close;
  }

  return bars;
}

export function generateHKCandleBars(count = 120): CandleBar[] {
  return generateCandleBars(count, 382.4, 0.004);
}

// ---- Paper Trades ----

const usTickers = ["AAPL", "NVDA", "TSLA", "SPY", "QQQ"];
const hkTickers = ["0700.HK", "9988.HK", "1810.HK", "3690.HK"];

export function generatePaperTrades(count = 60, market: Market = "US"): PaperTrade[] {
  const trades: PaperTrade[] = [];
  const tickers = market === "US" ? usTickers : hkTickers;
  const basePrice = market === "US" ? 185 : 382;
  let cumPnl = 0;
  const now = new Date();

  for (let i = count; i >= 0; i--) {
    const actions: Action[] = ["BUY", "SELL", "HOLD", "HOLD", "HOLD", "BUY", "SELL"];
    const action = actions[Math.floor(Math.random() * actions.length)];
    const price = basePrice + (Math.random() - 0.5) * basePrice * 0.05;
    const quantity = Math.floor(10 + Math.random() * 90);
    const pnl = action === "HOLD" ? 0 : (Math.random() - 0.4) * price * 0.01 * quantity;
    cumPnl += pnl;

    trades.push({
      id: count - i + 1,
      action,
      price,
      quantity,
      pnl,
      cumulativePnl: cumPnl,
      time: format(subMinutes(now, i * 0.3), "HH:mm:ss"),
      inferenceMsec: 2 + Math.random() * 18,
      market,
      ticker: tickers[Math.floor(Math.random() * tickers.length)],
    });
  }

  return trades;
}

// ---- Training Jobs ----

function generateRewardHistory(steps: number, algo: string): { step: number; reward: number }[] {
  const history: { step: number; reward: number }[] = [];
  let reward = -0.5;
  const maxSteps = Math.floor(steps * 0.6);

  for (let s = 0; s <= maxSteps; s += Math.floor(maxSteps / 50)) {
    const progress = s / maxSteps;
    // Simulate learning curve
    const target = algo === "TD3" ? 1.8 : 1.5;
    reward = -0.5 + (target + 0.5) * (1 - Math.exp(-progress * 4)) + (Math.random() - 0.5) * 0.3;
    history.push({ step: s, reward: Math.round(reward * 100) / 100 });
  }
  return history;
}

export function generateTrainingJobs(): TrainingJob[] {
  return [
    {
      id: "job-001",
      market: "US",
      timescale: "1m",
      algo: "PPO",
      totalTimesteps: 2000000,
      progressPct: 67,
      currentReward: 1.23,
      elapsedSeconds: 1842,
      status: "running",
      rewardHistory: generateRewardHistory(2000000, "PPO"),
    },
    {
      id: "job-002",
      market: "HK",
      timescale: "5m",
      algo: "TD3",
      totalTimesteps: 1000000,
      progressPct: 89,
      currentReward: 1.67,
      elapsedSeconds: 3201,
      status: "running",
      rewardHistory: generateRewardHistory(1000000, "TD3"),
    },
  ];
}

export function generateModelCheckpoints(): ModelCheckpoint[] {
  const now = new Date();
  return [
    { id: "ckpt-001", name: "PPO_US_1m_v3", market: "US", timescale: "1m", algo: "PPO", sharpe: 1.82, createdAt: format(subHours(now, 2), "yyyy-MM-dd HH:mm") },
    { id: "ckpt-002", name: "TD3_HK_5m_v2", market: "HK", timescale: "5m", algo: "TD3", sharpe: 1.54, createdAt: format(subHours(now, 8), "yyyy-MM-dd HH:mm") },
    { id: "ckpt-003", name: "PPO_US_10s_v1", market: "US", timescale: "10s", algo: "PPO", sharpe: 1.31, createdAt: format(subDays(now, 1), "yyyy-MM-dd HH:mm") },
    { id: "ckpt-004", name: "TD3_US_1h_v1", market: "US", timescale: "1h", algo: "TD3", sharpe: null, createdAt: format(subDays(now, 2), "yyyy-MM-dd HH:mm") },
    { id: "ckpt-005", name: "PPO_HK_1m_v2", market: "HK", timescale: "1m", algo: "PPO", sharpe: 1.95, createdAt: format(subDays(now, 3), "yyyy-MM-dd HH:mm") },
  ];
}

// ---- Backtest ----

export function generateEquityCurve(days = 90, startValue = 100000): { date: string; value: number; drawdown?: number }[] {
  const curve: { date: string; value: number; drawdown?: number }[] = [];
  let value = startValue;
  let peak = startValue;
  const now = new Date();

  for (let i = days; i >= 0; i--) {
    const date = subDays(now, i);
    const dailyReturn = (Math.random() - 0.42) * 0.015 + 0.001;
    value = value * (1 + dailyReturn);
    if (value > peak) peak = value;
    const drawdown = (value - peak) / peak;

    curve.push({
      date: format(date, "MMM dd"),
      value: Math.round(value),
      drawdown: drawdown < -0.02 ? drawdown * 100 : undefined,
    });
  }
  return curve;
}

export function generateBacktestTrades(count = 150): BacktestTrade[] {
  const trades: BacktestTrade[] = [];
  let cumPnl = 0;
  const now = new Date();
  let price = 185;

  for (let i = 0; i < count; i++) {
    const action: Action = Math.random() > 0.5 ? "BUY" : "SELL";
    price += (Math.random() - 0.48) * 2;
    const qty = Math.floor(10 + Math.random() * 50);
    const pnl = (Math.random() - 0.4) * price * 0.008 * qty;
    cumPnl += pnl;

    trades.push({
      id: i + 1,
      time: format(subHours(now, count - i), "yyyy-MM-dd HH:mm:ss"),
      action,
      price,
      qty,
      pnl,
      cumulativePnl: cumPnl,
    });
  }
  return trades;
}

export function generateBacktestResult(
  market = "US",
  ticker = "AAPL",
  timescale = "1m",
  model = "PPO_US_1m_v3"
): BacktestRun {
  const totalReturn = 18.4 + (Math.random() - 0.3) * 10;
  const nTrades = 150 + Math.floor(Math.random() * 100);
  const now = new Date();

  return {
    id: 1,
    market,
    ticker,
    timescale,
    model,
    startDate: format(subDays(now, 90), "yyyy-MM-dd"),
    endDate: format(now, "yyyy-MM-dd"),
    sharpe: 1.82 + (Math.random() - 0.3) * 0.5,
    sortino: 2.14 + (Math.random() - 0.3) * 0.5,
    calmar: 1.43 + (Math.random() - 0.3) * 0.3,
    maxDrawdown: -(8.2 + Math.random() * 6),
    totalReturn,
    winRate: 58 + (Math.random() - 0.3) * 10,
    nTrades,
    equityCurve: generateEquityCurve(90),
    trades: generateBacktestTrades(Math.min(nTrades, 150)),
    createdAt: format(now, "yyyy-MM-dd HH:mm"),
  };
}

export function generatePastBacktestRuns(): Omit<BacktestRun, "equityCurve" | "trades">[] {
  const now = new Date();
  return [
    {
      id: 1, market: "US", ticker: "AAPL", timescale: "1m", model: "PPO_US_1m_v3",
      startDate: "2024-09-01", endDate: "2024-11-30",
      sharpe: 1.82, sortino: 2.14, calmar: 1.43, maxDrawdown: -8.2, totalReturn: 18.4, winRate: 61.2, nTrades: 234,
      createdAt: format(subHours(now, 1), "yyyy-MM-dd HH:mm"),
    },
    {
      id: 2, market: "HK", ticker: "0700.HK", timescale: "5m", model: "PPO_HK_1m_v2",
      startDate: "2024-08-01", endDate: "2024-10-31",
      sharpe: 1.95, sortino: 2.31, calmar: 1.67, maxDrawdown: -6.8, totalReturn: 22.1, winRate: 64.5, nTrades: 187,
      createdAt: format(subDays(now, 1), "yyyy-MM-dd HH:mm"),
    },
    {
      id: 3, market: "US", ticker: "NVDA", timescale: "10s", model: "PPO_US_10s_v1",
      startDate: "2024-07-01", endDate: "2024-09-30",
      sharpe: 1.31, sortino: 1.58, calmar: 0.98, maxDrawdown: -12.4, totalReturn: 11.7, winRate: 55.1, nTrades: 892,
      createdAt: format(subDays(now, 3), "yyyy-MM-dd HH:mm"),
    },
  ];
}

// ---- Red Team ----

export interface RedTeamScenario {
  id: string;
  icon: string;
  name: string;
  description: string;
}

export interface RedTeamScenarioResult {
  scenarioId: string;
  passed: boolean;
  metric: string;
  detail: string;
}

export const RED_TEAM_SCENARIOS: RedTeamScenario[] = [
  { id: "flash_crash", icon: "⚡", name: "Flash Crash", description: "Injects sudden -20% price drop over 3 bars" },
  { id: "liquidity_drought", icon: "💧", name: "Liquidity Drought", description: "Near-zero volume for extended period" },
  { id: "adverse_selection", icon: "🎯", name: "Adverse Selection", description: "Model always gets worst fill price" },
  { id: "regime_change", icon: "📉", name: "Regime Change", description: "Sudden bull-to-bear market reversal" },
  { id: "overfitting", icon: "🔬", name: "Overfitting Detection", description: "Compares in-sample vs out-of-sample Sharpe" },
];

export function generateRedTeamResults(): RedTeamScenarioResult[] {
  return [
    {
      scenarioId: "flash_crash",
      passed: true,
      metric: "Return during crash: -3.8%",
      detail: "Model successfully reduced position size to 12% before crash. Max drawdown was contained to -3.8% vs benchmark -18.2%. Stop-loss triggered at -2.1% within 1 bar.",
    },
    {
      scenarioId: "liquidity_drought",
      passed: false,
      metric: "Slippage: 4.2x normal",
      detail: "Model attempted 23 trades during drought period. Average slippage was 4.2x normal levels. Position was not reduced. Consider adding volume filter to signal generation.",
    },
    {
      scenarioId: "adverse_selection",
      passed: true,
      metric: "Sharpe degradation: 1.3x",
      detail: "Under worst-fill conditions, Sharpe ratio dropped from 1.82 to 1.41 (1.3x degradation). Win rate decreased from 61% to 54%. Strategy remains profitable but with reduced edge.",
    },
    {
      scenarioId: "regime_change",
      passed: false,
      metric: "Return: -14.2% in 48h",
      detail: "Model failed to adapt to regime change within 48 hours. No regime detection mechanism present. Continued making long-biased trades during bearish reversal. Recommend adding regime classifier.",
    },
    {
      scenarioId: "overfitting",
      passed: true,
      metric: "IS/OOS Sharpe: 1.82/1.71",
      detail: "In-sample Sharpe: 1.82, Out-of-sample Sharpe: 1.71 (6.0% degradation). Degradation is within acceptable range (<15%). Model generalizes well to unseen data.",
    },
  ];
}
