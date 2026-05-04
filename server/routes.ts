import type { Express, Request, Response } from 'express';
import type { ParamsDictionary } from 'express-serve-static-core';
import type { ParsedQs } from 'qs';
import type { Server } from 'node:http';
import { WebSocketServer, WebSocket } from 'ws';
import { randomUUID } from 'node:crypto';
import { storage } from './storage';
import type {
  InsertTradingSettings,
  InsertTrainingJob,
  InsertBacktestResult,
  InsertRedTeamResult,
  InsertPaperTrade,
} from '@shared/schema';

// ─── Python FastAPI base URL ──────────────────────────────────────────────────
const PYTHON_API = process.env.PYTHON_API_URL || 'http://localhost:8001';

// ─── Helper: call Python API with mock fallback ───────────────────────────────
async function callPython<T>(
  path: string,
  options: RequestInit = {},
): Promise<{ data: T | null; ok: boolean; mock: boolean }> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    const res = await fetch(`${PYTHON_API}${path}`, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!res.ok) throw new Error(`Python API ${res.status}`);
    const data = (await res.json()) as T;
    return { data, ok: true, mock: false };
  } catch {
    return { data: null, ok: false, mock: true };
  }
}

// ─── Mock data generators ─────────────────────────────────────────────────────

function mockJob(overrides: Partial<InsertTrainingJob> = {}): Record<string, unknown> {
  const jobId = randomUUID();
  return {
    jobId,
    market: 'us',
    timescale: '1m',
    algo: 'PPO',
    totalTimesteps: 1_000_000,
    status: 'running',
    progressPct: Math.random() * 60,
    currentReward: +(Math.random() * 2 - 0.5).toFixed(4),
    modelPath: null,
    errorMsg: null,
    startedAt: new Date().toISOString(),
    completedAt: null,
    createdAt: new Date().toISOString(),
    ...overrides,
  };
}

function mockEquityCurve(n = 120): number[] {
  const curve: number[] = [100_000];
  for (let i = 1; i < n; i++) {
    const prev = curve[i - 1];
    curve.push(+(prev * (1 + (Math.random() - 0.46) * 0.005)).toFixed(2));
  }
  return curve;
}

function mockBacktestResult(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    market: 'us',
    timescale: '1m',
    ticker: 'AAPL',
    modelPath: 'models/ppo_us_1m_demo.zip',
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    sharpeRatio: +(1.2 + Math.random() * 0.8).toFixed(3),
    sortinoRatio: +(1.8 + Math.random() * 0.8).toFixed(3),
    calmarRatio: +(0.9 + Math.random() * 0.4).toFixed(3),
    maxDrawdownPct: +(0.05 + Math.random() * 0.1).toFixed(4),
    totalReturnPct: +(0.12 + Math.random() * 0.2).toFixed(4),
    winRate: +(0.52 + Math.random() * 0.1).toFixed(4),
    nTrades: Math.floor(200 + Math.random() * 500),
    avgPnlPerTrade: +(12 + Math.random() * 40).toFixed(2),
    profitFactor: +(1.3 + Math.random() * 0.5).toFixed(3),
    equityCurve: JSON.stringify(mockEquityCurve()),
    createdAt: new Date().toISOString(),
    ...overrides,
  };
}

const SCENARIO_METRICS: Record<string, (passed: boolean) => string> = {
  flash_crash: (p) => p ? 'Return during crash: -3.8%' : 'Return during crash: -16.2%',
  liquidity_drought: (p) => p ? 'Slippage: 1.8x normal' : 'Slippage: 4.9x normal',
  adverse_selection: (p) => p ? 'Sharpe degradation: 1.2x' : 'Sharpe degradation: 2.4x',
  regime_change: (p) => p ? 'Return in shift: -2.1%' : 'Return in shift: -14.7%',
  overfitting: (p) => p ? 'IS/OOS Sharpe: 1.82/1.71' : 'IS/OOS Sharpe: 2.14/0.81',
};

const SCENARIO_DETAILS: Record<string, (passed: boolean) => string> = {
  flash_crash: (p) => p ? 'Model reduced exposure before crash; stop-loss triggered within 1 bar.' : 'Model continued buying during crash; no defensive behavior observed.',
  liquidity_drought: (p) => p ? 'Model correctly reduced trade frequency in low-volume periods.' : 'Model attempted 23 trades during drought; slippage 4.9x normal levels.',
  adverse_selection: (p) => p ? 'Strategy profitable even with worst fills; robust edge.' : 'Worst-fill conditions eliminated model edge; Sharpe fell below 1.0.',
  regime_change: (p) => p ? 'Model adapted within 15 bars; loss well-contained.' : 'No regime detection; continued long-biased trades in bearish environment.',
  overfitting: (p) => p ? 'In/out-of-sample Sharpe within 6%; model generalizes well.' : 'Out-of-sample Sharpe 62% lower than in-sample; severe overfitting detected.',
};

function mockRedTeamResult(
  scenario: string,
  backtestId: number | null = null,
): Record<string, unknown> {
  const passed = Math.random() > 0.35;
  const metricFn = SCENARIO_METRICS[scenario] ?? ((p: boolean) => p ? 'Test passed' : 'Test failed');
  const detailFn = SCENARIO_DETAILS[scenario] ?? (() => '');
  return {
    backtestId,
    scenario,
    scenarioId: scenario,
    passed,
    metric: metricFn(passed),
    detail: detailFn(passed),
    metrics: JSON.stringify({
      maxLoss: +(-0.03 - Math.random() * 0.05).toFixed(4),
      volatility: +(0.015 + Math.random() * 0.02).toFixed(4),
      recoveryBars: Math.floor(5 + Math.random() * 30),
    }),
    createdAt: new Date().toISOString(),
  };
}

function mockPriceTick(
  ticker: string,
  basePrice: number,
  t: number,
): Record<string, unknown> {
  const amplitude = basePrice * 0.003;
  const price = +(basePrice + amplitude * Math.sin(t / 10) + (Math.random() - 0.5) * amplitude).toFixed(2);
  return {
    ticker,
    open: price,
    high: +(price * (1 + Math.random() * 0.001)).toFixed(2),
    low: +(price * (1 - Math.random() * 0.001)).toFixed(2),
    close: +(price + (Math.random() - 0.5) * amplitude * 0.3).toFixed(2),
    volume: Math.floor(1000 + Math.random() * 9000),
    timestamp: new Date().toISOString(),
  };
}

// ─── WebSocket setup ──────────────────────────────────────────────────────────

const BASE_PRICES: Record<string, number> = {
  AAPL: 182.5,
  TSLA: 245.3,
  NVDA: 875.2,
  SPY: 510.4,
  QQQ: 435.6,
  '0700.HK': 312.4,
  '9988.HK': 75.8,
};

// Track all connected WS clients with their subscribed tickers
interface WsClient {
  ws: WebSocket;
  ticker: string;
  market: string;
  tick: number;
}
const wsClients = new Set<WsClient>();

function broadcast(data: unknown): void {
  const msg = JSON.stringify(data);
  for (const client of Array.from(wsClients)) {
    if (client.ws.readyState === WebSocket.OPEN) {
      client.ws.send(msg);
    }
  }
}

// Called from paper trade creation to broadcast new trades
export function broadcastTrade(trade: unknown): void {
  const msg = JSON.stringify({ type: 'trade', data: trade });
  for (const client of Array.from(wsClients)) {
    if (client.ws.readyState === WebSocket.OPEN) {
      client.ws.send(msg);
    }
  }
}

// ─── Route registration ───────────────────────────────────────────────────────

export async function registerRoutes(httpServer: Server, app: Express): Promise<Server> {
  // CORS headers
  app.use((_req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PATCH,DELETE,OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    next();
  });

  app.options('/{*splat}', (_req, res) => res.sendStatus(204));

  // ── WebSocket server ────────────────────────────────────────────────────────
  const wss = new WebSocketServer({ server: httpServer, path: '/ws' });

  wss.on('connection', (ws) => {
    const client: WsClient = { ws, ticker: 'AAPL', market: 'us', tick: 0 };
    wsClients.add(client);

    ws.on('message', (raw) => {
      try {
        const msg = JSON.parse(raw.toString()) as { type?: string; ticker?: string; market?: string };
        if (msg.type === 'subscribe' && msg.ticker) {
          client.ticker = msg.ticker;
          client.market = msg.market ?? 'us';
        }
      } catch {
        // ignore malformed messages
      }
    });

    ws.on('close', () => wsClients.delete(client));
    ws.on('error', () => wsClients.delete(client));
  });

  // Price tick interval — send to each client based on their subscribed ticker
  setInterval(() => {
    for (const client of Array.from(wsClients)) {
      if (client.ws.readyState !== WebSocket.OPEN) continue;
      const base = BASE_PRICES[client.ticker] ?? 100;
      client.tick++;
      const tick = mockPriceTick(client.ticker, base, client.tick);
      // slowly drift the base price
      BASE_PRICES[client.ticker] = (BASE_PRICES[client.ticker] ?? 100) * (1 + (Math.random() - 0.499) * 0.0002);
      client.ws.send(JSON.stringify({ type: 'tick', data: tick }));
    }
  }, 500);

  // ── Settings routes ─────────────────────────────────────────────────────────

  app.get('/api/settings', async (_req: Request, res: Response) => {
    try {
      const settings = await storage.getAllSettings();
      res.json(settings);
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  app.post('/api/settings', async (req: Request, res: Response) => {
    try {
      const data = req.body as InsertTradingSettings;
      const created = await storage.createSettings(data);
      res.status(201).json(created);
    } catch (err) {
      res.status(400).json({ error: String(err) });
    }
  });

  app.patch('/api/settings/:id', async (req: Request<ParamsDictionary, unknown, unknown, ParsedQs>, res: Response) => {
    try {
      const id = parseInt(String(req.params.id), 10);
      const updated = await storage.updateSettings(id, req.body as Partial<InsertTradingSettings>);
      if (!updated) return res.status(404).json({ error: 'Not found' });
      res.json(updated);
    } catch (err) {
      res.status(400).json({ error: String(err) });
    }
  });

  app.delete('/api/settings/:id', async (req: Request<ParamsDictionary, unknown, unknown, ParsedQs>, res: Response) => {
    try {
      const id = parseInt(String(req.params.id), 10);
      await storage.deleteSettings(id);
      res.sendStatus(204);
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  // ── Training routes ─────────────────────────────────────────────────────────

  app.get('/api/training/jobs', async (_req: Request, res: Response) => {
    try {
      const jobs = await storage.getAllJobs();
      res.json(jobs);
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  app.post('/api/training/start', async (req: Request, res: Response) => {
    try {
      const { market = 'us', timescale = '1m', algo = 'PPO', timesteps = 1_000_000 } =
        req.body as { market?: string; timescale?: string; algo?: string; timesteps?: number };

      const jobId = randomUUID();
      const now = new Date().toISOString();

      const { data, mock } = await callPython<{ job_id: string }>('/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ market, timescale, algo, total_timesteps: timesteps }),
      });

      const resolvedJobId = (!mock && data?.job_id) ? data.job_id : jobId;

      const jobData: InsertTrainingJob = {
        jobId: resolvedJobId,
        market,
        timescale,
        algo,
        totalTimesteps: timesteps,
        status: mock ? 'running' : 'pending',
        progressPct: mock ? +(Math.random() * 10).toFixed(1) : 0,
        currentReward: null,
        modelPath: null,
        errorMsg: null,
        startedAt: now,
        completedAt: null,
        createdAt: now,
      };

      const job = await storage.createJob(jobData);
      res.status(201).json({ ...job, _mock: mock });
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  app.get('/api/training/jobs/:jobId/status', async (req: Request<ParamsDictionary, unknown, unknown, ParsedQs>, res: Response) => {
    try {
      const jobId = String(req.params.jobId);
      const existing = await storage.getJobByJobId(jobId);
      if (!existing) return res.status(404).json({ error: 'Job not found' });

      const { data, mock } = await callPython<{
        status: string;
        progress_pct: number;
        current_reward?: number;
        model_path?: string;
        error_msg?: string;
        completed_at?: string;
      }>(`/train/${jobId}/status`);

      if (!mock && data) {
        const updated = await storage.updateJob(jobId, {
          status: data.status,
          progressPct: data.progress_pct,
          currentReward: data.current_reward ?? null,
          modelPath: data.model_path ?? null,
          errorMsg: data.error_msg ?? null,
          completedAt: data.completed_at ?? null,
        });
        return res.json(updated);
      }

      // Mock: simulate progress
      const newPct = Math.min(100, existing.progressPct + +(Math.random() * 3).toFixed(1));
      const isDone = newPct >= 100;
      const updated = await storage.updateJob(jobId, {
        progressPct: newPct,
        status: isDone ? 'done' : 'running',
        currentReward: +(Math.random() * 2 - 0.3).toFixed(4),
        completedAt: isDone ? new Date().toISOString() : null,
        modelPath: isDone ? `models/ppo_${existing.market}_${existing.timescale}_${jobId.slice(0, 8)}.zip` : null,
      });
      res.json({ ...updated, _mock: true });
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  app.delete('/api/training/jobs/:jobId', async (req: Request<ParamsDictionary, unknown, unknown, ParsedQs>, res: Response) => {
    try {
      const jobId = String(req.params.jobId);
      await storage.deleteJob(jobId);
      res.sendStatus(204);
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  // ── Backtest routes ─────────────────────────────────────────────────────────

  app.get('/api/backtest/results', async (_req: Request, res: Response) => {
    try {
      const results = await storage.getAllBacktestResults();
      res.json(results);
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  app.post('/api/backtest/run', async (req: Request, res: Response) => {
    try {
      const { market = 'us', timescale = '1m', ticker = 'AAPL', modelPath = '', startDate, endDate } =
        req.body as {
          market?: string;
          timescale?: string;
          ticker?: string;
          modelPath?: string;
          startDate?: string;
          endDate?: string;
        };

      const { data, mock } = await callPython<Record<string, unknown>>('/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ market, timescale, ticker, model_path: modelPath, start_date: startDate, end_date: endDate }),
      });

      const resultData: InsertBacktestResult = mock
        ? (mockBacktestResult({ market, timescale, ticker, modelPath, startDate: startDate ?? '2023-01-01', endDate: endDate ?? '2023-12-31' }) as InsertBacktestResult)
        : {
            market,
            timescale,
            ticker,
            modelPath,
            startDate: startDate ?? '',
            endDate: endDate ?? '',
            sharpeRatio: (data?.sharpe_ratio as number) ?? null,
            sortinoRatio: (data?.sortino_ratio as number) ?? null,
            calmarRatio: (data?.calmar_ratio as number) ?? null,
            maxDrawdownPct: (data?.max_drawdown_pct as number) ?? null,
            totalReturnPct: (data?.total_return_pct as number) ?? null,
            winRate: (data?.win_rate as number) ?? null,
            nTrades: (data?.n_trades as number) ?? null,
            avgPnlPerTrade: (data?.avg_pnl_per_trade as number) ?? null,
            profitFactor: (data?.profit_factor as number) ?? null,
            equityCurve: data?.equity_curve ? JSON.stringify(data.equity_curve) : null,
            createdAt: new Date().toISOString(),
          };

      const result = await storage.createBacktestResult(resultData);
      res.status(201).json({ ...result, _mock: mock });
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  app.get('/api/backtest/results/:id', async (req: Request<ParamsDictionary, unknown, unknown, ParsedQs>, res: Response) => {
    try {
      const id = parseInt(String(req.params.id), 10);
      const result = await storage.getBacktestResultById(id);
      if (!result) return res.status(404).json({ error: 'Not found' });
      res.json(result);
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  app.delete('/api/backtest/results/:id', async (req: Request<ParamsDictionary, unknown, unknown, ParsedQs>, res: Response) => {
    try {
      const id = parseInt(String(req.params.id), 10);
      await storage.deleteBacktestResult(id);
      res.sendStatus(204);
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  // ── Red Team routes ─────────────────────────────────────────────────────────

  app.post('/api/redteam/run', async (req: Request, res: Response) => {
    try {
      const { backtestId, market = 'us', timescale = '1m', modelPath = '', scenarios = [] } =
        req.body as {
          backtestId?: number;
          market?: string;
          timescale?: string;
          modelPath?: string;
          scenarios?: string[];
        };

      const { data, mock } = await callPython<{ results: Array<{ scenario: string; passed: boolean; metrics: unknown }> }>('/redteam', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backtest_id: backtestId, market, timescale, model_path: modelPath, scenarios }),
      });

      const now = new Date().toISOString();
      const scenarioList = scenarios.length > 0 ? scenarios : ['flash_crash', 'liquidity_drought', 'adverse_selection', 'regime_change', 'overfitting'];

      // Build enriched results (includes scenarioId, metric, detail for frontend)
      const enriched: Record<string, unknown>[] = [];

      if (!mock && data?.results) {
        for (const r of data.results) {
          const item: InsertRedTeamResult = {
            backtestId: backtestId ?? null,
            scenario: r.scenario,
            passed: r.passed,
            metrics: JSON.stringify(r.metrics),
            createdAt: now,
          };
          const saved = await storage.createRedTeamResult(item);
          enriched.push({ ...saved, scenarioId: r.scenario, metric: '', detail: '' });
        }
      } else {
        for (const scenario of scenarioList) {
          const mockData = mockRedTeamResult(scenario, backtestId ?? null);
          const item: InsertRedTeamResult = {
            backtestId: (mockData.backtestId as number | null),
            scenario: mockData.scenario as string,
            passed: mockData.passed as boolean,
            metrics: mockData.metrics as string,
            createdAt: mockData.createdAt as string,
          };
          await storage.createRedTeamResult(item);
          // Return enriched version with all frontend fields
          enriched.push(mockData);
        }
      }

      res.status(201).json({ results: enriched, _mock: mock });
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  app.get('/api/redteam/results/:backtestId', async (req: Request<ParamsDictionary, unknown, unknown, ParsedQs>, res: Response) => {
    try {
      const backtestId = parseInt(String(req.params.backtestId), 10);
      const results = await storage.getRedTeamResultsByBacktestId(backtestId);
      res.json(results);
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  // ── Paper Trading routes ────────────────────────────────────────────────────

  app.get('/api/paper/trades', async (req: Request, res: Response) => {
    try {
      const limit = Math.min(parseInt((req.query.limit as string) || '500', 10), 1000);
      const trades = await storage.getRecentPaperTrades(limit);
      res.json(trades);
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  app.get('/api/paper/trades/recent', async (req: Request, res: Response) => {
    try {
      const market = req.query.market as string | undefined;
      const limit = Math.min(parseInt((req.query.limit as string) || '50', 10), 500);
      const trades = market
        ? await storage.getPaperTradesByMarket(market, limit)
        : await storage.getRecentPaperTrades(limit);
      res.json(trades);
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  app.post('/api/paper/start', async (req: Request, res: Response) => {
    try {
      const { market = 'us', ticker = 'AAPL', modelPath = '', alpacaApiKey, alpacaApiSecret } =
        req.body as {
          market?: string;
          ticker?: string;
          modelPath?: string;
          alpacaApiKey?: string;
          alpacaApiSecret?: string;
        };

      const { data, mock } = await callPython<{ status: string; session_id: string }>('/paper/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ market, ticker, model_path: modelPath, alpaca_api_key: alpacaApiKey, alpaca_api_secret: alpacaApiSecret }),
      });

      if (mock) {
        return res.json({ status: 'started', session_id: randomUUID(), _mock: true });
      }
      res.json(data);
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  app.post('/api/paper/stop', async (_req: Request, res: Response) => {
    try {
      const { data, mock } = await callPython<{ status: string }>('/paper/stop', { method: 'POST' });
      if (mock) return res.json({ status: 'stopped', _mock: true });
      res.json(data);
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  app.post('/api/paper/trade', async (req: Request, res: Response) => {
    try {
      const tradeData = req.body as InsertPaperTrade;
      if (!tradeData.timestamp) {
        tradeData.timestamp = new Date().toISOString();
      }
      const trade = await storage.createPaperTrade(tradeData);
      // Broadcast to all WebSocket clients
      broadcastTrade(trade);
      res.status(201).json(trade);
    } catch (err) {
      res.status(400).json({ error: String(err) });
    }
  });

  // ── Models route ────────────────────────────────────────────────────────────

  app.get('/api/models', async (_req: Request, res: Response) => {
    try {
      const { data, mock } = await callPython<{ models: string[] }>('/models');
      if (!mock && data) return res.json(data);

      // Mock model list
      res.json({
        models: [
          'models/ppo_us_1m_demo.zip',
          'models/ppo_us_5m_v2.zip',
          'models/ppo_hk_1m_v1.zip',
          'models/a2c_us_10s_v1.zip',
        ],
        _mock: true,
      });
    } catch (err) {
      res.status(500).json({ error: String(err) });
    }
  });

  // ── Health check ────────────────────────────────────────────────────────────

  app.get('/api/health', async (_req: Request, res: Response) => {
    const { ok } = await callPython('/health');
    res.json({
      status: 'ok',
      pythonApi: ok ? 'connected' : 'unavailable (mock mode)',
      timestamp: new Date().toISOString(),
    });
  });


  // ─── Data Manager ────────────────────────────────────────────────────────────
  
  app.post('/api/data/download', async (req, res) => {
    const { market = 'us', tickers = ['AAPL'], timescale = '1m', start_date, end_date, train_ratio = 0.6, val_ratio = 0.3 } = req.body;
    const { data, mock } = await callPython('/data/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ market, tickers, timescale, start_date, end_date, train_ratio, val_ratio }),
    });
    if (mock || !data) {
      return res.json({ job_id: randomUUID(), status: 'started', mock: true });
    }
    res.json(data);
  });

  app.get('/api/data/available', async (_req, res) => {
    const { data, mock } = await callPython<{ datasets: unknown[] }>('/data/available');
    if (mock || !data) {
      return res.json({ datasets: [
        { ticker: 'AAPL', market: 'us', timescale: '1m', n_bars: 48200, n_train: 28920, n_val: 14460, n_test: 4820, size_mb: 12.4, created_at: new Date().toISOString(), file_path: 'data/aapl_us_1m.parquet' },
        { ticker: 'NVDA', market: 'us', timescale: '1m', n_bars: 51000, n_train: 30600, n_val: 15300, n_test: 5100, size_mb: 13.1, created_at: new Date().toISOString(), file_path: 'data/nvda_us_1m.parquet' },
        { ticker: '0700.HK', market: 'hk', timescale: '1m', n_bars: 39600, n_train: 23760, n_val: 11880, n_test: 3960, size_mb: 9.8, created_at: new Date().toISOString(), file_path: 'data/0700.HK_hk_1m.parquet' },
      ], mock: true });
    }
    res.json(data);
  });

  app.delete('/api/data/dataset', async (req, res) => {
    const { data, mock } = await callPython('/data/dataset', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    if (mock || !data) return res.json({ ok: true, mock: true });
    res.json(data);
  });

  app.get('/api/data/preview', async (req, res) => {
    const { ticker = 'AAPL', market = 'us', timescale = '1m' } = req.query as Record<string, string>;
    const { data, mock } = await callPython<unknown>(`/data/preview?ticker=${ticker}&market=${market}&timescale=${timescale}`);
    if (mock || !data) {
      const bars = Array.from({ length: 200 }, (_, i) => ({
        t: new Date(Date.now() - (200 - i) * 60000).toISOString(),
        c: 150 + Math.sin(i / 20) * 10 + Math.random() * 3,
        v: 100000 + Math.random() * 50000,
      }));
      return res.json({ bars, stats: { total_bars: 48200, date_from: '2023-01-01', date_to: '2024-01-01', missing_pct: 0.3, avg_volume: 125000, ann_volatility: 0.284 }, mock: true });
    }
    res.json(data);
  });

  app.get('/api/data/jobs', async (_req, res) => {
    const { data, mock } = await callPython<unknown>('/data/jobs');
    if (mock || !data) return res.json({ jobs: [], mock: true });
    res.json(data);
  });

  // ─── HPO routes ───────────────────────────────────────────────────────────────

  app.post('/api/hpo/run', async (req, res) => {
    const { data, mock } = await callPython('/hpo/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    if (mock || !data) return res.json({ study_id: randomUUID(), status: 'started', mock: true });
    res.json(data);
  });

  app.get('/api/hpo/status/:study_id', async (req, res) => {
    const { data, mock } = await callPython<unknown>(`/hpo/status/${req.params.study_id}`);
    if (mock || !data) {
      return res.json({
        study_id: req.params.study_id,
        study_name: `hft_us_1m_PPO_demo`,
        n_trials_total: 50, n_completed: Math.floor(Math.random() * 30) + 5,
        n_pruned: 3, n_failed: 1,
        best_value: 1.24,
        best_params: { learning_rate: 3e-4, n_steps: 2048, batch_size: 256, gamma: 0.99, net_arch: 'medium' },
        status: 'running', eta_secs: 3600, mock: true,
      });
    }
    res.json(data);
  });

  app.get('/api/hpo/trials/:study_id', async (req, res) => {
    const { data, mock } = await callPython<unknown>(`/hpo/trials/${req.params.study_id}`);
    if (mock || !data) {
      const trials = Array.from({ length: 20 }, (_, i) => ({
        number: i, status: i < 17 ? 'complete' : ['pruned', 'complete', 'failed'][i % 3],
        value: 0.3 + Math.random() * 1.2, duration_secs: 30 + Math.random() * 120,
        params: { learning_rate: 1e-4 * (1 + Math.random()), n_steps: [512,1024,2048][i%3], batch_size: [64,128,256][i%3], gamma: 0.97 + Math.random()*0.02, net_arch: ['small','medium','large'][i%3] },
      }));
      return res.json({ trials, mock: true });
    }
    res.json(data);
  });

  app.post('/api/hpo/stop/:study_id', async (req, res) => {
    const { data, mock } = await callPython(`/hpo/stop/${req.params.study_id}`, { method: 'POST' });
    if (mock || !data) return res.json({ ok: true, mock: true });
    res.json(data);
  });

  app.get('/api/hpo/studies', async (_req, res) => {
    const { data, mock } = await callPython<unknown>('/hpo/studies');
    if (mock || !data) return res.json({ studies: [], mock: true });
    res.json(data);
  });

  // ─── Meta Controller routes ───────────────────────────────────────────────────
  
  app.post('/api/meta/register', async (req, res) => {
    const { data, mock } = await callPython('/meta/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    if (mock || !data) return res.json({ ok: true, mock: true });
    res.json(data);
  });

  app.post('/api/meta/unregister', async (req, res) => {
    const { data, mock } = await callPython('/meta/unregister', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    if (mock || !data) return res.json({ ok: true, mock: true });
    res.json(data);
  });

  app.get('/api/meta/signals', async (req, res) => {
    const limit = req.query.limit || 20;
    const { data, mock } = await callPython<unknown>(`/meta/signals?limit=${limit}`);
    if (mock || !data) {
      const signals = Array.from({ length: 20 }, (_, i) => ({
        timestamp: new Date(Date.now() - i * 60000).toISOString(),
        action: ['BUY','HOLD','SELL','HOLD','BUY'][i % 5],
        confidence: 0.45 + Math.random() * 0.45,
        regime: 'bull', kelly_fraction: 0.15 + Math.random() * 0.08,
        price: 191 + Math.random() * 4,
      }));
      return res.json({ signals, mock: true });
    }
    res.json(data);
  });

  // ─── Continuous Learning configure ───────────────────────────────────────────

  app.post('/api/continuous/configure', async (req, res) => {
    const { data, mock } = await callPython('/continuous/configure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    if (mock || !data) return res.json({ ok: true, mock: true });
    res.json(data);
  });

  return httpServer;
}
