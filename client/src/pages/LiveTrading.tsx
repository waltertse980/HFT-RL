import { useState, useEffect, useRef, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  Play,
  Square,
  RefreshCw,
  Wifi,
  WifiOff,
  AlertTriangle,
  Loader2,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

// ---- Types ----
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

// ---- Constants ----
const US_TICKERS = ["AAPL", "NVDA", "TSLA", "SPY", "QQQ"];
const HK_TICKERS = ["0700.HK", "9988.HK", "1810.HK", "3690.HK", "9999.HK"];
const TIMESCALES = ["10s", "1m", "5m", "1h"];

// ---- ENGINE_OFFLINE banner ----
function EngineOfflineBanner({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-3 rounded-md border border-destructive/40 bg-destructive/10 px-4 py-3 text-xs text-destructive font-mono shrink-0 mx-4 mt-2">
      <AlertTriangle size={14} className="shrink-0" />
      <span className="font-sans">{message}</span>
      <code className="ml-auto bg-destructive/10 px-2 py-0.5 rounded text-[10px]">
        uvicorn api_server:app --port 8001 --reload
      </code>
    </div>
  );
}

// ---- KPI Card ----
function KpiCard({
  title,
  value,
  delta,
  deltaLabel,
  sub,
  testId,
}: {
  title: string;
  value: React.ReactNode;
  delta?: number;
  deltaLabel?: string;
  sub?: string;
  testId?: string;
}) {
  const isPositive = delta !== undefined && delta >= 0;
  return (
    <Card data-testid={testId} className="bg-card border-border p-0">
      <CardContent className="p-4">
        <p className="text-xs text-muted-foreground font-sans uppercase tracking-wide mb-1">{title}</p>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-mono font-semibold text-foreground">{value}</span>
          {delta !== undefined && (
            <span
              className={cn(
                "flex items-center gap-0.5 text-sm font-mono",
                isPositive ? "text-success" : "text-destructive"
              )}
            >
              {isPositive ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
              {Math.abs(delta).toFixed(2)}%
            </span>
          )}
        </div>
        {deltaLabel && (
          <p className="text-xs text-muted-foreground mt-1 font-mono">{deltaLabel}</p>
        )}
        {sub && <p className="text-xs text-muted-foreground mt-1 font-sans">{sub}</p>}
      </CardContent>
    </Card>
  );
}

// ---- Position Badge ----
function PositionBadge({ pos }: { pos: "LONG" | "SHORT" | "FLAT" }) {
  const colors = {
    LONG: "bg-success/10 text-success border-success/30",
    SHORT: "bg-destructive/10 text-destructive border-destructive/30",
    FLAT: "bg-muted text-muted-foreground border-border",
  };
  return (
    <span className={cn("px-2 py-0.5 rounded text-xs font-mono font-semibold border", colors[pos])}>
      {pos}
    </span>
  );
}

// ---- Candlestick Tooltip ----
function CandleTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: any[];
}) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0]?.payload as CandleBar;
  if (!d) return null;
  return (
    <div className="bg-card border border-border rounded-md p-3 text-xs font-mono shadow-lg min-w-36">
      <p className="text-muted-foreground mb-1">{d.time}</p>
      <div className="space-y-0.5">
        <p>
          O: <span className="text-foreground">${d.open.toFixed(2)}</span>
        </p>
        <p>
          H: <span className="text-success">${d.high.toFixed(2)}</span>
        </p>
        <p>
          L: <span className="text-destructive">${d.low.toFixed(2)}</span>
        </p>
        <p>
          C: <span className="text-foreground">${d.close.toFixed(2)}</span>
        </p>
        <p>
          Vol: <span className="text-muted-foreground">{(d.volume / 1000).toFixed(0)}K</span>
        </p>
        {d.signal && (
          <p
            className={cn(
              "font-semibold mt-1",
              d.signal === "BUY"
                ? "text-success"
                : d.signal === "SELL"
                ? "text-destructive"
                : "text-warning"
            )}
          >
            ▶ {d.signal}
          </p>
        )}
      </div>
    </div>
  );
}

// ---- Signal Dot ----
function SignalDot(props: any) {
  const { cx, cy, payload } = props;
  if (!payload?.signal || payload.signal === "HOLD") return null;
  const isBuy = payload.signal === "BUY";
  return (
    <g>
      <polygon
        points={
          isBuy
            ? `${cx},${cy - 14} ${cx - 6},${cy - 6} ${cx + 6},${cy - 6}`
            : `${cx},${cy + 14} ${cx - 6},${cy + 6} ${cx + 6},${cy + 6}`
        }
        fill={isBuy ? "#22c55e" : "#ef4444"}
        opacity={0.9}
      />
    </g>
  );
}

// ---- Inference Meter ----
function InferenceMeter({ ms }: { ms: number }) {
  const max = 30;
  const pct = Math.min((ms / max) * 100, 100);
  const color =
    ms < 8 ? "bg-success" : ms < 15 ? "bg-warning" : "bg-destructive";
  return (
    <div data-testid="inference-meter" className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground font-sans">Inference Latency</span>
        <span
          className={cn(
            "font-mono font-semibold",
            ms < 8 ? "text-success" : ms < 15 ? "text-warning" : "text-destructive"
          )}
        >
          {ms.toFixed(1)} ms
        </span>
      </div>
      <div className="h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all duration-300", color)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function actionClass(action: Action) {
  if (action === "BUY") return "text-success";
  if (action === "SELL") return "text-destructive";
  return "text-warning";
}
function actionRowBg(action: Action) {
  if (action === "BUY") return "bg-success/5";
  if (action === "SELL") return "bg-destructive/5";
  return "";
}

// ---- Main Component ----
export default function LiveTrading() {
  const qc = useQueryClient();

  const [market, setMarket] = useState<Market>("US");
  const [ticker, setTicker] = useState("AAPL");
  const [timescale, setTimescale] = useState("1m");
  const [isRunning, setIsRunning] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [wsMock, setWsMock] = useState(false);
  const [candles, setCandles] = useState<CandleBar[]>([]);
  const [engineError, setEngineError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelay = useRef(1000);

  // ---- Fetch model list from engine ----
  const { data: modelsData } = useQuery<{ models: string[] }>({
    queryKey: ["/api/models"],
    retry: false,
  });
  const modelList = modelsData?.models ?? [];
  const [model, setModel] = useState("");
  const resolvedModel = model || modelList[0] || "__none__";

  // ---- Poll real paper trades while running ----
  const { data: tradesData, refetch: refetchTrades } = useQuery<{
    trades: PaperTrade[];
  }>({
    queryKey: ["/api/paper/trades"],
    enabled: isRunning,
    refetchInterval: isRunning ? 2000 : false,
    retry: false,
  });
  const trades: PaperTrade[] = tradesData?.trades ?? [];

  // ---- Poll paper trading status ----
  const { data: paperStatus } = useQuery<{
    running: boolean;
    session_id: string | null;
    portfolio_value: number;
    daily_pnl: number;
    position: "LONG" | "SHORT" | "FLAT";
    entry_price: number | null;
    trade_count: number;
  }>({
    queryKey: ["/api/paper/status"],
    enabled: isRunning,
    refetchInterval: isRunning ? 2000 : false,
    retry: false,
  });

  // ---- Start paper trading ----
  const startMutation = useMutation({
    mutationFn: async () => {
      // Use fetch directly so we can inspect the response body for ENGINE_OFFLINE
      const res = await fetch("/api/paper/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          market: market.toLowerCase(),
          ticker,
          timescale,
          modelPath: resolvedModel === "__none__" ? "" : resolvedModel,
        }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        if (body?.error === "ENGINE_OFFLINE" || res.status === 503) throw new Error("ENGINE_OFFLINE");
        throw new Error(body?.message ?? `HTTP ${res.status}`);
      }
      return body as { status: string; session_id: string };
    },
    onSuccess: (data) => {
      setIsRunning(true);
      setSessionId(data.session_id);
      setEngineError(null);
      qc.invalidateQueries({ queryKey: ["/api/paper/trades"] });
    },
    onError: (err: Error) => {
      if (err.message === "ENGINE_OFFLINE") {
        setEngineError(
          "Python engine is offline. Start it with the command on the right, then retry."
        );
      } else {
        setEngineError(`Failed to start paper trading: ${err.message}`);
      }
    },
  });

  // ---- Stop paper trading ----
  const stopMutation = useMutation({
    mutationFn: async () => {
      const res = await fetch("/api/paper/stop", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body?.message ?? `HTTP ${res.status}`);
      return body;
    },
    onSuccess: () => {
      setIsRunning(false);
      setSessionId(null);
      setEngineError(null);
    },
    onError: (err: Error) => {
      // Force-stop on client side even if engine returns error
      setIsRunning(false);
      setSessionId(null);
      setEngineError(`Stop request failed: ${err.message}`);
    },
  });

  // ---- Reset candles + trades when market changes ----
  useEffect(() => {
    setCandles([]);
    setTicker(market === "US" ? "AAPL" : "0700.HK");
  }, [market]);

  // ---- WebSocket — connects to dashboard Express WS ----
  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    try {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${protocol}//${window.location.host}`);
      wsRef.current = ws;
      ws.onopen = () => {
        setWsConnected(true);
        reconnectDelay.current = 1000;
      };
      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === "tick") {
            setWsMock(!!msg.mock);
            const d = msg.data;
            if (!d) return;
            const bar: CandleBar = {
              time: d.time ?? new Date().toLocaleTimeString("en-US", { hour12: false }),
              timestamp: new Date(),
              open: d.open ?? d.close,
              high: d.high ?? d.close,
              low: d.low ?? d.close,
              close: d.close,
              volume: d.volume ?? 0,
              signal: d.signal,
            };
            setCandles((prev) => [...prev.slice(-119), bar]);
          }
        } catch {}
      };
      ws.onclose = () => {
        setWsConnected(false);
        reconnectTimer.current = setTimeout(() => {
          reconnectDelay.current = Math.min(reconnectDelay.current * 2, 30000);
          connectWs();
        }, reconnectDelay.current);
      };
      ws.onerror = () => ws.close();
    } catch {}
  }, []);

  useEffect(() => {
    connectWs();
    return () => {
      wsRef.current?.close();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };
  }, [connectWs]);

  // ---- Derived KPIs from real data ----
  const lastCandle = candles[candles.length - 1];
  const firstCandle = candles[0];
  const todayPct =
    lastCandle && firstCandle && firstCandle.close > 0
      ? ((lastCandle.close - firstCandle.close) / firstCandle.close) * 100
      : 0;

  const portfolioValue = paperStatus?.portfolio_value ?? 0;
  const dailyPnl = paperStatus?.daily_pnl ?? 0;
  const position = paperStatus?.position ?? "FLAT";
  const entryPrice = paperStatus?.entry_price ?? null;
  const tradeCount = paperStatus?.trade_count ?? trades.filter((t) => t.action !== "HOLD").length;

  const avgInference =
    trades.length > 0
      ? trades.slice(-10).reduce((s, t) => s + (t.inferenceMsec ?? 0), 0) /
        Math.min(trades.length, 10)
      : 0;

  const currency = market === "HK" ? "HK$" : "$";
  const tickers = market === "US" ? US_TICKERS : HK_TICKERS;

  // ---- Chart data ----
  const chartData = candles.map((c) => ({
    ...c,
    bullBody: c.close >= c.open ? c.close - c.open : 0,
    bearBody: c.open > c.close ? c.open - c.close : 0,
    baseValue: Math.min(c.open, c.close),
  }));

  const isPending = startMutation.isPending || stopMutation.isPending;

  return (
    <div className="flex flex-col h-full overflow-hidden" data-testid="page-live-trading">
      {/* Top Header */}
      <div className="border-b border-border px-6 py-3 flex items-center justify-between shrink-0">
        <h1 className="text-sm font-sans font-semibold text-foreground flex items-center gap-2">
          Live Trading
          {wsConnected ? (
            <span className="flex items-center gap-1 text-xs font-mono">
              <Wifi size={12} className="text-success" />
              <span className={wsMock ? "text-warning" : "text-success"}>
                {wsMock ? "WS (Mock Feed)" : "WS Live"}
              </span>
            </span>
          ) : (
            <span className="flex items-center gap-1 text-muted-foreground text-xs font-mono">
              <WifiOff size={12} /> Connecting...
            </span>
          )}
          {isRunning && (
            <Badge className="bg-success/10 text-success border-success/30 text-xs font-mono ml-1">
              LIVE
            </Badge>
          )}
        </h1>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground font-mono">
            {new Date().toLocaleTimeString("en-US", { hour12: false })}
          </span>
        </div>
      </div>

      {/* Engine error banner */}
      {engineError && <EngineOfflineBanner message={engineError} />}

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* KPI Cards */}
        <div className="grid grid-cols-3 gap-3" data-testid="kpi-row">
          <KpiCard
            title="Portfolio Value"
            value={
              portfolioValue > 0
                ? `${currency}${portfolioValue.toLocaleString("en-US", {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}`
                : isRunning
                ? "Loading…"
                : "—"
            }
            delta={isRunning ? todayPct : undefined}
            deltaLabel={
              isRunning
                ? `${todayPct >= 0 ? "+" : ""}${todayPct.toFixed(2)}% today`
                : undefined
            }
            testId="kpi-portfolio-value"
          />
          <Card data-testid="kpi-position" className="bg-card border-border">
            <CardContent className="p-4">
              <p className="text-xs text-muted-foreground font-sans uppercase tracking-wide mb-1">
                Active Position
              </p>
              <div className="flex items-center gap-2 mb-1">
                <PositionBadge pos={position} />
                <span className="font-mono text-sm text-foreground">{isRunning ? ticker : "—"}</span>
              </div>
              {isRunning ? (
                <p className="text-xs font-mono text-muted-foreground">
                  {entryPrice ? (
                    <>
                      Entry:{" "}
                      <span className="text-foreground">
                        {currency}
                        {entryPrice.toFixed(2)}
                      </span>
                      <span
                        className={cn(
                          "ml-3",
                          dailyPnl >= 0 ? "text-success" : "text-destructive"
                        )}
                      >
                        uPnL: {dailyPnl >= 0 ? "+" : ""}
                        {currency}
                        {Math.abs(dailyPnl).toFixed(2)}
                      </span>
                    </>
                  ) : (
                    "Waiting for first signal…"
                  )}
                </p>
              ) : (
                <p className="text-xs text-muted-foreground font-sans">
                  No open position — press Start
                </p>
              )}
            </CardContent>
          </Card>
          <KpiCard
            title="Today's P&L"
            value={
              isRunning ? (
                <span className={dailyPnl >= 0 ? "text-success" : "text-destructive"}>
                  {dailyPnl >= 0 ? "+" : ""}
                  {currency}
                  {Math.abs(dailyPnl).toFixed(2)}
                </span>
              ) : (
                "—"
              )
            }
            sub={isRunning ? `${tradeCount} trades today` : "Start paper trading to see P&L"}
            testId="kpi-daily-pnl"
          />
        </div>

        {/* Main area: chart + right panel */}
        <div
          className="flex gap-4 min-h-0"
          style={{ height: "calc(100vh - 280px)" }}
        >
          {/* Chart */}
          <div className="flex-1 min-w-0" data-testid="main-chart">
            <Card className="bg-card border-border h-full flex flex-col">
              <CardHeader className="py-3 px-4 border-b border-border shrink-0">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-sans font-medium text-foreground">
                    {ticker} · {timescale}
                    {wsMock && (
                      <span className="ml-2 text-xs font-mono text-warning">[MOCK FEED — engine offline]</span>
                    )}
                  </CardTitle>
                  <div className="flex items-center gap-2">
                    {lastCandle && (
                      <>
                        <span className="text-xs font-mono text-muted-foreground">
                          Last:{" "}
                          <span className="text-foreground">
                            {currency}
                            {lastCandle.close.toFixed(2)}
                          </span>
                        </span>
                        <span
                          className={cn(
                            "text-xs font-mono",
                            todayPct >= 0 ? "text-success" : "text-destructive"
                          )}
                        >
                          {todayPct >= 0 ? "+" : ""}
                          {todayPct.toFixed(2)}%
                        </span>
                      </>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="flex-1 p-2 min-h-0">
                {chartData.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-center">
                    <WifiOff size={32} className="text-muted-foreground opacity-30 mb-3" />
                    <p className="text-sm text-muted-foreground font-sans">
                      {wsConnected
                        ? "Waiting for price data…"
                        : "Connecting to price feed…"}
                    </p>
                    <p className="text-xs text-muted-foreground/60 font-sans mt-1">
                      Start the Python engine and click Start to begin
                    </p>
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart
                      data={chartData}
                      margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="hsl(222 35% 18%)"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="time"
                        tick={{ fill: "#64748b", fontSize: 10, fontFamily: "'JetBrains Mono'" }}
                        tickLine={false}
                        axisLine={false}
                        interval={Math.floor(chartData.length / 8)}
                      />
                      <YAxis
                        tick={{ fill: "#64748b", fontSize: 10, fontFamily: "'JetBrains Mono'" }}
                        tickLine={false}
                        axisLine={false}
                        tickFormatter={(v) => `${currency}${v.toFixed(0)}`}
                        domain={([dataMin, dataMax]: [number, number]) => {
                          const padding =
                            (dataMax - dataMin) * 0.15 || dataMin * 0.01;
                          return [
                            Math.floor(dataMin - padding),
                            Math.ceil(dataMax + padding),
                          ];
                        }}
                        width={60}
                      />
                      <Tooltip content={<CandleTooltip />} />
                      <Bar dataKey="bullBody" stackId="candle" fill="transparent" barSize={6}>
                        {chartData.map((entry, idx) => (
                          <Cell
                            key={idx}
                            fill={
                              entry.close >= entry.open ? "#22c55e" : "transparent"
                            }
                            fillOpacity={entry.close >= entry.open ? 0.85 : 0}
                          />
                        ))}
                      </Bar>
                      <Bar dataKey="bearBody" stackId="candle2" fill="transparent" barSize={6}>
                        {chartData.map((entry, idx) => (
                          <Cell
                            key={idx}
                            fill={
                              entry.open > entry.close ? "#ef4444" : "transparent"
                            }
                            fillOpacity={entry.open > entry.close ? 0.85 : 0}
                          />
                        ))}
                      </Bar>
                      <Line
                        type="monotone"
                        dataKey="close"
                        stroke="#3b82f6"
                        strokeWidth={1.5}
                        dot={<SignalDot />}
                        activeDot={{ r: 3, fill: "#3b82f6" }}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Right panel */}
          <div className="w-72 xl:w-80 shrink-0 flex flex-col gap-3 overflow-y-auto">
            {/* Controls */}
            <Card className="bg-card border-border">
              <CardContent className="p-3 space-y-3">
                {/* Market */}
                <div>
                  <p className="text-xs text-muted-foreground mb-1 font-sans">Market</p>
                  <Tabs
                    value={market}
                    onValueChange={(v) => setMarket(v as Market)}
                    data-testid="market-tabs"
                  >
                    <TabsList className="w-full bg-muted h-8">
                      <TabsTrigger value="US" className="flex-1 text-xs" disabled={isRunning}>
                        US
                      </TabsTrigger>
                      <TabsTrigger value="HK" className="flex-1 text-xs" disabled={isRunning}>
                        HK
                      </TabsTrigger>
                    </TabsList>
                  </Tabs>
                </div>

                {/* Ticker */}
                <div>
                  <p className="text-xs text-muted-foreground mb-1 font-sans">Ticker</p>
                  <Select
                    value={ticker}
                    onValueChange={setTicker}
                    disabled={isRunning}
                    data-testid="ticker-select"
                  >
                    <SelectTrigger className="h-8 text-xs font-mono bg-muted border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-popover border-border">
                      {tickers.map((t) => (
                        <SelectItem key={t} value={t} className="text-xs font-mono">
                          {t}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Model */}
                <div>
                  <p className="text-xs text-muted-foreground mb-1 font-sans">Model</p>
                  <Select
                    value={resolvedModel}
                    onValueChange={setModel}
                    disabled={isRunning}
                    data-testid="model-select"
                  >
                    <SelectTrigger className="h-8 text-xs font-mono bg-muted border-border">
                      <SelectValue placeholder={modelList.length === 0 ? "— no models —" : undefined} />
                    </SelectTrigger>
                    <SelectContent className="bg-popover border-border">
                      {modelList.length === 0 ? (
                        <SelectItem value="__no_models__" className="text-xs font-mono text-muted-foreground" disabled>
                          — no models — train first
                        </SelectItem>
                      ) : (
                        modelList.map((m) => (
                          <SelectItem key={m} value={m} className="text-xs font-mono">
                            {m}
                          </SelectItem>
                        ))
                      )}
                    </SelectContent>
                  </Select>
                </div>

                {/* Timescale */}
                <div>
                  <p className="text-xs text-muted-foreground mb-1 font-sans">Timescale</p>
                  <Tabs
                    value={timescale}
                    onValueChange={setTimescale}
                    data-testid="timescale-tabs"
                  >
                    <TabsList className="w-full bg-muted h-8">
                      {TIMESCALES.map((ts) => (
                        <TabsTrigger
                          key={ts}
                          value={ts}
                          className="flex-1 text-xs"
                          disabled={isRunning}
                        >
                          {ts}
                        </TabsTrigger>
                      ))}
                    </TabsList>
                  </Tabs>
                </div>

                {/* Inference Meter */}
                <InferenceMeter ms={avgInference} />

                {/* Start/Stop */}
                <div className="flex gap-2">
                  <Button
                    data-testid="btn-start-trading"
                    size="sm"
                    className="flex-1 bg-success hover:bg-success/90 text-white text-xs"
                    onClick={() => {
                      setEngineError(null);
                      startMutation.mutate();
                    }}
                    disabled={isRunning || isPending || !resolvedModel}
                  >
                    {startMutation.isPending ? (
                      <Loader2 size={12} className="mr-1 animate-spin" />
                    ) : (
                      <Play size={12} className="mr-1" />
                    )}
                    Start
                  </Button>
                  <Button
                    data-testid="btn-stop-trading"
                    size="sm"
                    variant="destructive"
                    className="flex-1 text-xs"
                    onClick={() => stopMutation.mutate()}
                    disabled={!isRunning || isPending}
                  >
                    {stopMutation.isPending ? (
                      <Loader2 size={12} className="mr-1 animate-spin" />
                    ) : (
                      <Square size={12} className="mr-1" />
                    )}
                    Stop
                  </Button>
                </div>

                {/* Session ID */}
                {sessionId && (
                  <p className="text-xs text-muted-foreground font-mono truncate">
                    Session: <span className="text-foreground">{sessionId.slice(0, 12)}…</span>
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Recent Signals — from real paper trades */}
            <Card className="bg-card border-border flex-1 min-h-0 flex flex-col">
              <CardHeader className="py-2 px-3 border-b border-border shrink-0">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-xs font-sans font-medium text-foreground">
                    Recent Signals
                  </CardTitle>
                  <button
                    onClick={() => refetchTrades()}
                    className="text-muted-foreground hover:text-foreground transition-colors"
                  >
                    <RefreshCw size={12} />
                  </button>
                </div>
              </CardHeader>
              <div className="flex-1 overflow-y-auto">
                {trades.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-center py-8">
                    <p className="text-xs text-muted-foreground font-sans">
                      {isRunning
                        ? "Waiting for signals…"
                        : "Start paper trading to see signals"}
                    </p>
                  </div>
                ) : (
                  <table className="w-full text-xs" data-testid="signals-table">
                    <thead className="sticky top-0 bg-card border-b border-border">
                      <tr>
                        <th className="text-left px-2 py-1.5 text-muted-foreground font-sans font-medium">
                          Action
                        </th>
                        <th className="text-right px-2 py-1.5 text-muted-foreground font-sans font-medium">
                          Price
                        </th>
                        <th className="text-right px-2 py-1.5 text-muted-foreground font-sans font-medium">
                          P&amp;L
                        </th>
                        <th className="text-right px-2 py-1.5 text-muted-foreground font-sans font-medium">
                          ms
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...trades].reverse().slice(0, 20).map((trade, idx) => (
                        <tr
                          key={trade.id}
                          className={cn(
                            "border-b border-border/50 transition-colors hover:bg-accent/30",
                            actionRowBg(trade.action)
                          )}
                          data-testid={`signal-row-${idx}`}
                        >
                          <td className="px-2 py-1.5">
                            <span
                              className={cn("font-mono font-semibold", actionClass(trade.action))}
                            >
                              {trade.action}
                            </span>
                          </td>
                          <td className="text-right px-2 py-1.5 font-mono text-foreground">
                            {currency}
                            {trade.price.toFixed(2)}
                          </td>
                          <td
                            className={cn(
                              "text-right px-2 py-1.5 font-mono",
                              trade.pnl >= 0 ? "text-success" : "text-destructive"
                            )}
                          >
                            {trade.pnl >= 0 ? "+" : ""}
                            {trade.pnl.toFixed(2)}
                          </td>
                          <td className="text-right px-2 py-1.5 font-mono text-muted-foreground">
                            {(trade.inferenceMsec ?? 0).toFixed(1)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
