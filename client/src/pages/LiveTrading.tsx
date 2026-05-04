import { useState, useEffect, useRef, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
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
  ReferenceLine,
} from "recharts";
import { TrendingUp, TrendingDown, Minus, Play, Square, RefreshCw, Wifi, WifiOff } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import {
  generateCandleBars,
  generateHKCandleBars,
  generatePaperTrades,
  type CandleBar,
  type PaperTrade,
  type Market,
  type Action,
} from "@/lib/mockData";

// ---- Tickers ----
const US_TICKERS = ["AAPL", "NVDA", "TSLA", "SPY", "QQQ"];
const HK_TICKERS = ["0700.HK", "9988.HK", "1810.HK", "3690.HK", "9999.HK"];
const TIMESCALES = ["10s", "1m", "5m", "1h"];

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
    <Card
      data-testid={testId}
      className="bg-card border-border p-0"
    >
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

// ---- Candlestick Custom Tooltip ----
function CandleTooltip({ active, payload }: { active?: boolean; payload?: any[] }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0]?.payload as CandleBar;
  if (!d) return null;

  return (
    <div className="bg-card border border-border rounded-md p-3 text-xs font-mono shadow-lg min-w-36">
      <p className="text-muted-foreground mb-1">{d.time}</p>
      <div className="space-y-0.5">
        <p>O: <span className="text-foreground">${d.open.toFixed(2)}</span></p>
        <p>H: <span className="text-success">${d.high.toFixed(2)}</span></p>
        <p>L: <span className="text-destructive">${d.low.toFixed(2)}</span></p>
        <p>C: <span className="text-foreground">${d.close.toFixed(2)}</span></p>
        <p>Vol: <span className="text-muted-foreground">{(d.volume / 1000).toFixed(0)}K</span></p>
        {d.signal && (
          <p className={cn("font-semibold mt-1", d.signal === "BUY" ? "text-success" : d.signal === "SELL" ? "text-destructive" : "text-warning")}>
            ▶ {d.signal}
          </p>
        )}
      </div>
    </div>
  );
}

// ---- Signal Dot on chart ----
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

// ---- Action color ----
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

// ---- Inference Meter ----
function InferenceMeter({ ms }: { ms: number }) {
  // Target <20ms, good <10ms
  const max = 30;
  const pct = Math.min((ms / max) * 100, 100);
  const color = ms < 8 ? "bg-success" : ms < 15 ? "bg-warning" : "bg-destructive";

  return (
    <div data-testid="inference-meter" className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground font-sans">Inference Latency</span>
        <span className={cn("font-mono font-semibold", ms < 8 ? "text-success" : ms < 15 ? "text-warning" : "text-destructive")}>
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

export default function LiveTrading() {
  const [market, setMarket] = useState<Market>("US");
  const [ticker, setTicker] = useState("AAPL");
  const [model, setModel] = useState("PPO_US_1m_v3");
  const [timescale, setTimescale] = useState("1m");
  const [isRunning, setIsRunning] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [candles, setCandles] = useState<CandleBar[]>(() =>
    generateCandleBars(120, 185.5, 0.003)
  );
  const [trades, setTrades] = useState<PaperTrade[]>(() => generatePaperTrades(60, "US"));
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelay = useRef(1000);

  // Fetch models
  const { data: modelsData } = useQuery<{ models: string[] }>({
    queryKey: ["/api/models"],
    retry: false,
  });
  const STATIC_MODELS = ["PPO_US_1m_v3", "TD3_HK_5m_v2", "PPO_US_10s_v1", "PPO_HK_1m_v2", "TD3_US_1h_v1"];
  const models = (modelsData?.models && modelsData.models.length > 0) ? modelsData.models : STATIC_MODELS;
  // Ensure model has a valid default when STATIC_MODELS is used
  const resolvedModel = models.includes(model) ? model : models[0] ?? "PPO_US_1m_v3";

  // Regenerate candles on market change
  useEffect(() => {
    const bars = market === "HK" ? generateHKCandleBars(120) : generateCandleBars(120, 185.5, 0.003);
    setCandles(bars);
    setTrades(generatePaperTrades(60, market));
    setTicker(market === "US" ? "AAPL" : "0700.HK");
  }, [market]);

  // WebSocket connection
  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    try {
      const ws = new WebSocket("ws://localhost:5000");
      wsRef.current = ws;
      ws.onopen = () => {
        setWsConnected(true);
        reconnectDelay.current = 1000;
      };
      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          if (data.type === "tick") {
            setCandles((prev) => {
              const next = [...prev.slice(-119), data.bar as CandleBar];
              return next;
            });
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

  // Simulate live ticks when running and WS not connected
  useEffect(() => {
    if (!isRunning || wsConnected) return;
    const interval = setInterval(() => {
      setCandles((prev) => {
        const last = prev[prev.length - 1];
        const change = (Math.random() - 0.48) * last.close * 0.003;
        const close = last.close + change;
        const high = Math.max(close, last.close) + Math.random() * last.close * 0.002;
        const low = Math.min(close, last.close) - Math.random() * last.close * 0.002;
        const now = new Date();
        const bar: CandleBar = {
          time: now.toLocaleTimeString("en-US", { hour12: false }),
          timestamp: now,
          open: last.close,
          high,
          low,
          close,
          volume: Math.floor(50000 + Math.random() * 100000),
          signal: Math.random() < 0.07 ? "BUY" : Math.random() < 0.07 ? "SELL" : undefined,
        };
        return [...prev.slice(-119), bar];
      });
    }, timescale === "10s" ? 500 : 1000);
    return () => clearInterval(interval);
  }, [isRunning, wsConnected, timescale]);

  // Poll trades
  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(() => {
      setTrades((prev) => {
        const last = prev[prev.length - 1];
        const actions: Action[] = ["BUY", "SELL", "HOLD", "HOLD"];
        const action = actions[Math.floor(Math.random() * actions.length)];
        const price = candles[candles.length - 1]?.close ?? 185;
        const pnl = action === "HOLD" ? 0 : (Math.random() - 0.4) * price * 0.005 * 10;
        const newTrade: PaperTrade = {
          id: (last?.id ?? 0) + 1,
          action,
          price,
          quantity: Math.floor(10 + Math.random() * 40),
          pnl,
          cumulativePnl: (last?.cumulativePnl ?? 0) + pnl,
          time: new Date().toLocaleTimeString("en-US", { hour12: false }),
          inferenceMsec: 2 + Math.random() * 15,
          market,
          ticker,
        };
        return [...prev.slice(-59), newTrade];
      });
    }, 5000);
    return () => clearInterval(interval);
  }, [isRunning, candles, market, ticker]);

  // Derived KPIs
  const lastCandle = candles[candles.length - 1];
  const firstCandle = candles[0];
  const portfolioValue = 102483.5 + (lastCandle?.close ?? 185) * 10 - 185 * 10;
  const todayPct = ((lastCandle?.close - firstCandle?.close) / firstCandle?.close) * 100 || 0;
  const todayPnl = trades.reduce((s, t) => s + t.pnl, 0);
  const tradeCount = trades.filter((t) => t.action !== "HOLD").length;
  const lastTrade = trades[trades.length - 1];
  const avgInference = trades.slice(-10).reduce((s, t) => s + t.inferenceMsec, 0) / 10;

  const tickers = market === "US" ? US_TICKERS : HK_TICKERS;
  const currency = market === "HK" ? "HK$" : "$";

  // Chart data: transform candles for Recharts
  // We pass a special "body" value array = [open, close] for the stacked bar trick
  const chartData = candles.map((c) => ({
    ...c,
    bullBody: c.close >= c.open ? c.close - c.open : 0,
    bearBody: c.open > c.close ? c.open - c.close : 0,
    baseValue: Math.min(c.open, c.close),
    wickHigh: c.high,
    wickLow: c.low,
  }));

  return (
    <div className="flex flex-col h-full overflow-hidden" data-testid="page-live-trading">
      {/* Top Header */}
      <div className="border-b border-border px-6 py-3 flex items-center justify-between shrink-0">
        <h1 className="text-sm font-sans font-semibold text-foreground flex items-center gap-2">
          Live Trading
          {wsConnected ? (
            <span className="flex items-center gap-1 text-success text-xs font-mono">
              <Wifi size={12} /> WS Live
            </span>
          ) : (
            <span className="flex items-center gap-1 text-muted-foreground text-xs font-mono">
              <WifiOff size={12} /> Mock Data
            </span>
          )}
        </h1>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground font-mono">
            {new Date().toLocaleTimeString("en-US", { hour12: false })}
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* KPI Cards */}
        <div className="grid grid-cols-3 gap-3" data-testid="kpi-row">
          <KpiCard
            title="Portfolio Value"
            value={`$${portfolioValue.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
            delta={todayPct}
            deltaLabel={`${todayPct >= 0 ? "+" : ""}${todayPct.toFixed(2)}% today`}
            testId="kpi-portfolio-value"
          />
          <Card data-testid="kpi-position" className="bg-card border-border">
            <CardContent className="p-4">
              <p className="text-xs text-muted-foreground font-sans uppercase tracking-wide mb-1">Active Position</p>
              <div className="flex items-center gap-2 mb-1">
                <PositionBadge pos={isRunning ? "LONG" : "FLAT"} />
                <span className="font-mono text-sm text-foreground">{ticker}</span>
              </div>
              {isRunning ? (
                <p className="text-xs font-mono text-muted-foreground">
                  Entry: <span className="text-foreground">{currency}{(lastCandle?.open ?? 185).toFixed(2)}</span>
                  <span className={cn("ml-3", todayPnl >= 0 ? "text-success" : "text-destructive")}>
                    uPnL: {todayPnl >= 0 ? "+" : ""}{currency}{Math.abs(todayPnl * 0.3).toFixed(2)}
                  </span>
                </p>
              ) : (
                <p className="text-xs text-muted-foreground font-sans">No open position — press Start</p>
              )}
            </CardContent>
          </Card>
          <KpiCard
            title="Today's P&L"
            value={
              <span className={todayPnl >= 0 ? "text-success" : "text-destructive"}>
                {todayPnl >= 0 ? "+" : ""}{currency}{Math.abs(todayPnl).toFixed(2)}
              </span>
            }
            sub={`${tradeCount} trades today`}
            testId="kpi-daily-pnl"
          />
        </div>

        {/* Main area: chart + right panel */}
        <div className="flex gap-4 min-h-0" style={{ height: "calc(100vh - 280px)" }}>
          {/* Chart */}
          <div className="flex-1 min-w-0" data-testid="main-chart">
            <Card className="bg-card border-border h-full flex flex-col">
              <CardHeader className="py-3 px-4 border-b border-border shrink-0">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-sans font-medium text-foreground">
                    {ticker} · {timescale}
                  </CardTitle>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-muted-foreground">
                      Last: <span className="text-foreground">{currency}{(lastCandle?.close ?? 185).toFixed(2)}</span>
                    </span>
                    <span className={cn("text-xs font-mono", todayPct >= 0 ? "text-success" : "text-destructive")}>
                      {todayPct >= 0 ? "+" : ""}{todayPct.toFixed(2)}%
                    </span>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="flex-1 p-2 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(222 35% 18%)" vertical={false} />
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
                          const padding = (dataMax - dataMin) * 0.15 || dataMin * 0.01;
                          return [Math.floor(dataMin - padding), Math.ceil(dataMax + padding)];
                        }}
                      width={60}
                    />
                    <Tooltip content={<CandleTooltip />} />
                    {/* Wick high-low */}
                    <Bar dataKey="wickHigh" fill="transparent" barSize={1} />
                    {/* Candle bodies: bullish */}
                    <Bar dataKey="bullBody" stackId="candle" fill="transparent" barSize={6}>
                      {chartData.map((entry, idx) => (
                        <Cell
                          key={idx}
                          fill={entry.close >= entry.open ? "#22c55e" : "transparent"}
                          fillOpacity={entry.close >= entry.open ? 0.85 : 0}
                        />
                      ))}
                    </Bar>
                    {/* Candle bodies: bearish */}
                    <Bar dataKey="bearBody" stackId="candle2" fill="transparent" barSize={6}>
                      {chartData.map((entry, idx) => (
                        <Cell
                          key={idx}
                          fill={entry.open > entry.close ? "#ef4444" : "transparent"}
                          fillOpacity={entry.open > entry.close ? 0.85 : 0}
                        />
                      ))}
                    </Bar>
                    {/* Close line */}
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
                  <Tabs value={market} onValueChange={(v) => setMarket(v as Market)} data-testid="market-tabs">
                    <TabsList className="w-full bg-muted h-8">
                      <TabsTrigger value="US" className="flex-1 text-xs">US</TabsTrigger>
                      <TabsTrigger value="HK" className="flex-1 text-xs">HK</TabsTrigger>
                    </TabsList>
                  </Tabs>
                </div>

                {/* Ticker */}
                <div>
                  <p className="text-xs text-muted-foreground mb-1 font-sans">Ticker</p>
                  <Select value={ticker} onValueChange={setTicker} data-testid="ticker-select">
                    <SelectTrigger className="h-8 text-xs font-mono bg-muted border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-popover border-border">
                      {tickers.map((t) => (
                        <SelectItem key={t} value={t} className="text-xs font-mono">{t}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Model */}
                <div>
                  <p className="text-xs text-muted-foreground mb-1 font-sans">Model</p>
                  <Select value={resolvedModel} onValueChange={setModel} data-testid="model-select">
                    <SelectTrigger className="h-8 text-xs font-mono bg-muted border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-popover border-border">
                      {models.map((m) => (
                        <SelectItem key={m} value={m} className="text-xs font-mono">{m}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Timescale */}
                <div>
                  <p className="text-xs text-muted-foreground mb-1 font-sans">Timescale</p>
                  <Tabs value={timescale} onValueChange={setTimescale} data-testid="timescale-tabs">
                    <TabsList className="w-full bg-muted h-8">
                      {TIMESCALES.map((ts) => (
                        <TabsTrigger key={ts} value={ts} className="flex-1 text-xs">{ts}</TabsTrigger>
                      ))}
                    </TabsList>
                  </Tabs>
                </div>

                {/* Inference Meter */}
                <InferenceMeter ms={avgInference || 8.4} />

                {/* Start/Stop */}
                <div className="flex gap-2">
                  <Button
                    data-testid="btn-start-trading"
                    size="sm"
                    className="flex-1 bg-success hover:bg-success/90 text-white text-xs"
                    onClick={() => setIsRunning(true)}
                    disabled={isRunning}
                  >
                    <Play size={12} className="mr-1" />
                    Start
                  </Button>
                  <Button
                    data-testid="btn-stop-trading"
                    size="sm"
                    variant="destructive"
                    className="flex-1 text-xs"
                    onClick={() => setIsRunning(false)}
                    disabled={!isRunning}
                  >
                    <Square size={12} className="mr-1" />
                    Stop
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Recent Signals */}
            <Card className="bg-card border-border flex-1 min-h-0 flex flex-col">
              <CardHeader className="py-2 px-3 border-b border-border shrink-0">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-xs font-sans font-medium text-foreground">Recent Signals</CardTitle>
                  <RefreshCw size={12} className="text-muted-foreground" />
                </div>
              </CardHeader>
              <div className="flex-1 overflow-y-auto">
                <table className="w-full text-xs" data-testid="signals-table">
                  <thead className="sticky top-0 bg-card border-b border-border">
                    <tr>
                      <th className="text-left px-2 py-1.5 text-muted-foreground font-sans font-medium">Action</th>
                      <th className="text-right px-2 py-1.5 text-muted-foreground font-sans font-medium">Price</th>
                      <th className="text-right px-2 py-1.5 text-muted-foreground font-sans font-medium">P&L</th>
                      <th className="text-right px-2 py-1.5 text-muted-foreground font-sans font-medium">ms</th>
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
                          <span className={cn("font-mono font-semibold", actionClass(trade.action))}>
                            {trade.action}
                          </span>
                        </td>
                        <td className="text-right px-2 py-1.5 font-mono text-foreground">
                          {currency}{trade.price.toFixed(2)}
                        </td>
                        <td className={cn("text-right px-2 py-1.5 font-mono", trade.pnl >= 0 ? "text-success" : "text-destructive")}>
                          {trade.pnl >= 0 ? "+" : ""}{trade.pnl.toFixed(2)}
                        </td>
                        <td className="text-right px-2 py-1.5 font-mono text-muted-foreground">
                          {trade.inferenceMsec.toFixed(1)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
