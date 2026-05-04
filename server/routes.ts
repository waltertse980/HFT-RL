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

// ─── Helper: call Python API — throws typed error when offline ────────────────
async function callPython<T>(
  path: string,
  options: RequestInit = {},
): Promise<{ data: T; ok: true }> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000); // 10s timeout for long ops
  try {
    const res = await fetch(`${PYTHON_API}${path}`, { ...options, signal: controller.signal });
    clearTimeout(timeout);
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`Python API error ${res.status}: ${text}`);
    }
    const data = (await res.json()) as T;
    return { data, ok: true };
  } catch (err: unknown) {
    clearTimeout(timeout);
    const msg = err instanceof Error ? err.message : String(err);
    if (
      msg.includes('fetch failed') ||
      msg.includes('ECONNREFUSED') ||
      msg.includes('abort') ||
      msg.includes('AbortError')
    ) {
      throw {
        engineOffline: true,
        message: 'Python engine is offline. Start it with: uvicorn api_server:app --port 8001',
      };
    }
    throw { engineOffline: false, message: msg };
  }
}

// Helper to handle engine-offline errors uniformly
function handleEngineError(res: Response, err: unknown): void {
  if (err && typeof err === 'object' && 'engineOffline' in err) {
    res.status(503).json({
      error: 'ENGINE_OFFLINE',
      message: (err as unknown as { message: string }).message,
      hint: 'Start the Python engine: cd python_engine && uvicorn api_server:app --port 8001 --reload',
    });
  } else {
    res.status(500).json({
      error: 'PYTHON_ERROR',
      message: String((err as { message?: string })?.message ?? err),
    });
  }
}

// ─── Mock price tick (WebSocket only — clearly labelled) ──────────────────────

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

  // ── WebSocket server ──────────────────────────────────────────────────────────
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

  // Price tick interval — mock data clearly labelled; real data would come from Python WS when connected
  setInterval(() => {
    for (const client of Array.from(wsClients)) {
      if (client.ws.readyState !== WebSocket.OPEN) continue;
      const base = BASE_PRICES[client.ticker] ?? 100;
      client.tick++;
      const tick = mockPriceTick(client.ticker, base, client.tick);
      // slowly drift the base price
      BASE_PRICES[client.ticker] = (BASE_PRICES[client.ticker] ?? 100) * (1 + (Math.random() - 0.499) * 0.0002);
      // mock: true so the frontend can display a "MOCK DATA" label on the chart
      client.ws.send(JSON.stringify({ type: 'tick', mock: true, data: tick }));
    }
  }, 1000); // 1s for mock (real data comes from Python WS when connected)

  // ── Health check ──────────────────────────────────────────────────────────────

  app.get('/api/health', async (_req: Request, res: Response) => {
    try {
      const result = await callPython<{ status: string }>('/health');
      res.json({
        status: 'ok',
        pythonApi: 'connected',
        pythonStatus: result.data,
        timestamp: new Date().toISOString(),
      });
    } catch {
      res.json({
        status: 'ok',
        pythonApi: 'offline',
        message: 'Python engine not reachable on port 8001',
        timestamp: new Date().toISOString(),
      });
    }
  });

  // ── Settings routes ───────────────────────────────────────────────────────────

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

  app.patch(
    '/api/settings/:id',
    async (req: Request<ParamsDictionary, unknown, unknown, ParsedQs>, res: Response) => {
      try {
        const id = parseInt(String(req.params.id), 10);
        const updated = await storage.updateSettings(id, req.body as Partial<InsertTradingSettings>);
        if (!updated) return res.status(404).json({ error: 'Not found' });
        res.json(updated);
      } catch (err) {
        res.status(400).json({ error: String(err) });
      }
    },
  );

  app.delete(
    '/api/settings/:id',
    async (req: Request<ParamsDictionary, unknown, unknown, ParsedQs>, res: Response) => {
      try {
        const id = parseInt(String(req.params.id), 10);
        await storage.deleteSettings(id);
        res.sendStatus(204);
      } catch (err) {
        res.status(500).json({ error: String(err) });
      }
    },
  );

  // Test connection — proxies credentials to Python for validation
  app.post('/api/settings/test-connection', async (req: Request, res: Response) => {
    try {
      const { api_key, api_secret } = req.body as { api_key: string; api_secret: string };
      if (!api_key || !api_secret) {
        return res
          .status(400)
          .json({ error: 'MISSING_CREDENTIALS', message: 'API key and secret are required' });
      }
      const result = await callPython<{
        connected: boolean;
        account_status: string;
        buying_power: number;
        portfolio_value: number;
      }>('/settings/test-connection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key, api_secret }),
      });
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  // Global settings (initial_capital, etc.) — stored in python_engine/global_config.json via Python
  app.get('/api/settings/global', async (_req: Request, res: Response) => {
    try {
      const result = await callPython<Record<string, unknown>>('/settings/global');
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  app.post('/api/settings/global', async (req: Request, res: Response) => {
    try {
      const result = await callPython<Record<string, unknown>>('/settings/global', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body),
      });
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  // ── Training routes ───────────────────────────────────────────────────────────

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
      const {
        market = 'us',
        timescale = '1m',
        algo = 'PPO',
        timesteps = 1_000_000,
      } = req.body as { market?: string; timescale?: string; algo?: string; timesteps?: number };

      const now = new Date().toISOString();

      const result = await callPython<{ job_id: string }>('/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ market, timescale, algo, total_timesteps: timesteps }),
      });

      const jobData: InsertTrainingJob = {
        jobId: result.data.job_id,
        market,
        timescale,
        algo,
        totalTimesteps: timesteps,
        status: 'pending',
        progressPct: 0,
        currentReward: null,
        modelPath: null,
        errorMsg: null,
        startedAt: now,
        completedAt: null,
        createdAt: now,
      };

      const job = await storage.createJob(jobData);
      res.status(201).json(job);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  app.get(
    '/api/training/jobs/:jobId/status',
    async (req: Request<ParamsDictionary, unknown, unknown, ParsedQs>, res: Response) => {
      try {
        const jobId = String(req.params.jobId);
        const existing = await storage.getJobByJobId(jobId);
        if (!existing) return res.status(404).json({ error: 'Job not found' });

        const result = await callPython<{
          status: string;
          progress_pct: number;
          current_reward?: number;
          model_path?: string;
          error_msg?: string;
          completed_at?: string;
          reward_history?: { step: number; reward: number }[];
        }>(`/train/${jobId}/status`);

        const data = result.data;
        const updated = await storage.updateJob(jobId, {
          status: data.status,
          progressPct: data.progress_pct,
          currentReward: data.current_reward ?? null,
          modelPath: data.model_path ?? null,
          errorMsg: data.error_msg ?? null,
          completedAt: data.completed_at ?? null,
        });
        return res.json({ ...updated, rewardHistory: data.reward_history ?? [] });
      } catch (err) {
        handleEngineError(res as unknown as Response, err);
      }
    },
  );

  app.delete(
    '/api/training/jobs/:jobId',
    async (req: Request<ParamsDictionary, unknown, unknown, ParsedQs>, res: Response) => {
      try {
        const jobId = String(req.params.jobId);
        await storage.deleteJob(jobId);
        res.sendStatus(204);
      } catch (err) {
        res.status(500).json({ error: String(err) });
      }
    },
  );

  app.post('/api/training/stop/:jobId', async (req: Request<ParamsDictionary>, res: Response) => {
    try {
      const result = await callPython<{ ok: boolean }>(`/train/${req.params.jobId}/stop`, {
        method: 'POST',
      });
      await storage.updateJob(String(req.params.jobId), { status: 'stopped' });
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  app.post(
    '/api/training/evaluate/:modelName',
    async (req: Request<ParamsDictionary>, res: Response) => {
      try {
        const result = await callPython<Record<string, unknown>>(
          `/train/evaluate/${req.params.modelName}`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req.body),
          },
        );
        res.json(result.data);
      } catch (err) {
        handleEngineError(res as unknown as Response, err);
      }
    },
  );

  app.post('/api/training/export-onnx', async (req: Request, res: Response) => {
    try {
      const result = await callPython<{ onnx_path: string }>('/train/export-onnx', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body),
      });
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  // ── Models route ──────────────────────────────────────────────────────────────

  app.get('/api/models', async (_req: Request, res: Response) => {
    try {
      const result = await callPython<{ models: string[] }>('/models');
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  // ── Backtest routes ───────────────────────────────────────────────────────────

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
      const result = await callPython<Record<string, unknown>>('/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body),
      });
      const data = result.data;
      const {
        market = 'us',
        timescale = '1m',
        ticker = 'AAPL',
        modelPath = '',
        startDate = '',
        endDate = '',
      } = req.body as Record<string, string>;

      const saved = await storage.createBacktestResult({
        market,
        timescale,
        ticker,
        modelPath,
        startDate: startDate || (data.start_date as string) || '',
        endDate: endDate || (data.end_date as string) || '',
        sharpeRatio: (data.sharpe_ratio as number) ?? null,
        sortinoRatio: (data.sortino_ratio as number) ?? null,
        calmarRatio: (data.calmar_ratio as number) ?? null,
        maxDrawdownPct: (data.max_drawdown_pct as number) ?? null,
        totalReturnPct: (data.total_return_pct as number) ?? null,
        winRate: (data.win_rate as number) ?? null,
        nTrades: (data.n_trades as number) ?? null,
        avgPnlPerTrade: (data.avg_pnl_per_trade as number) ?? null,
        profitFactor: (data.profit_factor as number) ?? null,
        equityCurve: data.equity_curve ? JSON.stringify(data.equity_curve) : null,
        createdAt: new Date().toISOString(),
      } as InsertBacktestResult);
      res.status(201).json(saved);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  app.get(
    '/api/backtest/results/:id',
    async (req: Request<ParamsDictionary, unknown, unknown, ParsedQs>, res: Response) => {
      try {
        const id = parseInt(String(req.params.id), 10);
        const result = await storage.getBacktestResultById(id);
        if (!result) return res.status(404).json({ error: 'Not found' });
        res.json(result);
      } catch (err) {
        res.status(500).json({ error: String(err) });
      }
    },
  );

  app.delete(
    '/api/backtest/results/:id',
    async (req: Request<ParamsDictionary, unknown, unknown, ParsedQs>, res: Response) => {
      try {
        const id = parseInt(String(req.params.id), 10);
        await storage.deleteBacktestResult(id);
        res.sendStatus(204);
      } catch (err) {
        res.status(500).json({ error: String(err) });
      }
    },
  );

  // ── Red Team routes ───────────────────────────────────────────────────────────

  app.post('/api/redteam/run', async (req: Request, res: Response) => {
    try {
      const result = await callPython<{
        results: Array<{
          scenario: string;
          passed: boolean;
          metric: string;
          detail: string;
          metrics: Record<string, unknown>;
        }>;
      }>('/redteam', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body),
      });
      const now = new Date().toISOString();
      const enriched = [];
      for (const r of result.data.results) {
        const saved = await storage.createRedTeamResult({
          backtestId: (req.body as { backtestId?: number }).backtestId ?? null,
          scenario: r.scenario,
          passed: r.passed,
          metrics: JSON.stringify(r.metrics),
          createdAt: now,
        } as InsertRedTeamResult);
        enriched.push({ ...saved, metric: r.metric, detail: r.detail });
      }
      res.status(201).json({ results: enriched });
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  app.get(
    '/api/redteam/results/:backtestId',
    async (req: Request<ParamsDictionary, unknown, unknown, ParsedQs>, res: Response) => {
      try {
        const backtestId = parseInt(String(req.params.backtestId), 10);
        const results = await storage.getRedTeamResultsByBacktestId(backtestId);
        res.json(results);
      } catch (err) {
        res.status(500).json({ error: String(err) });
      }
    },
  );

  // ── Paper Trading routes ──────────────────────────────────────────────────────

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
      const {
        market = 'us',
        ticker = 'AAPL',
        modelPath = '',
        alpacaApiKey,
        alpacaApiSecret,
      } = req.body as {
        market?: string;
        ticker?: string;
        modelPath?: string;
        alpacaApiKey?: string;
        alpacaApiSecret?: string;
      };

      const result = await callPython<{ status: string; session_id: string }>('/paper/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          market,
          ticker,
          model_path: modelPath,
          alpaca_api_key: alpacaApiKey,
          alpaca_api_secret: alpacaApiSecret,
        }),
      });
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  app.post('/api/paper/stop', async (_req: Request, res: Response) => {
    try {
      const result = await callPython<{ status: string }>('/paper/stop', { method: 'POST' });
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  app.post('/api/paper/trade', async (req: Request, res: Response) => {
    try {
      const tradeData = req.body as InsertPaperTrade;
      if (!tradeData.timestamp) {
        tradeData.timestamp = new Date().toISOString();
      }
      const trade = await storage.createPaperTrade(tradeData);
      broadcastTrade(trade);
      res.status(201).json(trade);
    } catch (err) {
      res.status(400).json({ error: String(err) });
    }
  });

  // ── Data Manager routes ───────────────────────────────────────────────────────

  app.post('/api/data/download', async (req: Request, res: Response) => {
    try {
      const result = await callPython<{ job_id: string; status: string }>('/data/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body),
      });
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  app.get('/api/data/available', async (_req: Request, res: Response) => {
    try {
      const result = await callPython<{ datasets: unknown[] }>('/data/available');
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  app.delete('/api/data/dataset', async (req: Request, res: Response) => {
    try {
      const result = await callPython<{ ok: boolean }>('/data/dataset', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body),
      });
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  app.get('/api/data/preview', async (req: Request, res: Response) => {
    try {
      const {
        ticker = 'AAPL',
        market = 'us',
        timescale = '1m',
      } = req.query as Record<string, string>;
      const result = await callPython<unknown>(
        `/data/preview?ticker=${ticker}&market=${market}&timescale=${timescale}`,
      );
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  app.get('/api/data/jobs', async (_req: Request, res: Response) => {
    try {
      const result = await callPython<{ jobs: unknown[] }>('/data/jobs');
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  // ── HPO routes ────────────────────────────────────────────────────────────────

  app.post('/api/hpo/run', async (req: Request, res: Response) => {
    try {
      const result = await callPython<{ study_id: string; status: string }>('/hpo/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body),
      });
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  app.get('/api/hpo/status/:study_id', async (req: Request<ParamsDictionary>, res: Response) => {
    try {
      const result = await callPython<unknown>(`/hpo/status/${req.params.study_id}`);
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  app.get('/api/hpo/trials/:study_id', async (req: Request<ParamsDictionary>, res: Response) => {
    try {
      const result = await callPython<unknown>(`/hpo/trials/${req.params.study_id}`);
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  app.post('/api/hpo/stop/:study_id', async (req: Request<ParamsDictionary>, res: Response) => {
    try {
      const result = await callPython<{ ok: boolean }>(
        `/hpo/stop/${req.params.study_id}`,
        { method: 'POST' },
      );
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  app.get('/api/hpo/studies', async (_req: Request, res: Response) => {
    try {
      const result = await callPython<unknown>('/hpo/studies');
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  // ── Meta Controller routes ────────────────────────────────────────────────────

  app.post('/api/meta/register', async (req: Request, res: Response) => {
    try {
      const result = await callPython<Record<string, unknown>>('/meta/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body),
      });
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  app.post('/api/meta/unregister', async (req: Request, res: Response) => {
    try {
      const result = await callPython<Record<string, unknown>>('/meta/unregister', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body),
      });
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  app.get('/api/meta/signals', async (req: Request, res: Response) => {
    try {
      const limit = String(req.query.limit || 20);
      const result = await callPython<unknown>(`/meta/signals?limit=${limit}`);
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  // ── Continuous Learning routes ────────────────────────────────────────────────

  app.post('/api/continuous/configure', async (req: Request, res: Response) => {
    try {
      const result = await callPython<Record<string, unknown>>('/continuous/configure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body),
      });
      res.json(result.data);
    } catch (err) {
      handleEngineError(res as unknown as Response, err);
    }
  });

  return httpServer;
}
