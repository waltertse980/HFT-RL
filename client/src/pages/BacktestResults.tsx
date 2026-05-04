import { useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Play, Download, ChevronLeft, ChevronRight, TrendingUp, TrendingDown, AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import {
  generateBacktestResult,
  generatePastBacktestRuns,
  type BacktestRun,
  type Action,
} from "@/lib/mockData";

const US_TICKERS = ["AAPL", "NVDA", "TSLA", "SPY", "QQQ"];
const HK_TICKERS = ["0700.HK", "9988.HK", "1810.HK", "3690.HK"];
const TIMESCALES = ["10s", "1m", "5m", "1h"];
const MODELS = ["PPO_US_1m_v3", "TD3_HK_5m_v2", "PPO_US_10s_v1", "PPO_HK_1m_v2"];
const MARKETS = ["US", "HK"];

function actionRowBg(action: Action) {
  if (action === "BUY") return "bg-success/5";
  if (action === "SELL") return "bg-destructive/5";
  return "";
}
function actionClass(action: Action) {
  if (action === "BUY") return "text-success";
  if (action === "SELL") return "text-destructive";
  return "text-warning";
}

// ---- Metric Card ----
function MetricCard({
  label,
  value,
  target,
  good,
  bad,
  suffix = "",
  testId,
}: {
  label: string;
  value: number;
  target?: string;
  good?: boolean;
  bad?: boolean;
  suffix?: string;
  testId?: string;
}) {
  return (
    <Card className="bg-card border-border" data-testid={testId}>
      <CardContent className="p-4">
        <p className="text-xs text-muted-foreground font-sans uppercase tracking-wide mb-1">{label}</p>
        <p
          className={cn(
            "text-2xl font-mono font-semibold",
            good === true ? "text-success" : bad === true ? "text-destructive" : "text-foreground"
          )}
        >
          {value.toFixed(2)}{suffix}
        </p>
        {target && <p className="text-xs text-muted-foreground font-sans mt-1">{target}</p>}
      </CardContent>
    </Card>
  );
}

// ---- Equity Curve Tooltip ----
function EquityTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-card border border-border rounded-md p-3 text-xs font-mono shadow-lg">
      <p className="text-muted-foreground mb-1">{label}</p>
      <p className="text-primary">${payload[0]?.value?.toLocaleString()}</p>
    </div>
  );
}

// ---- Past Runs Sidebar ----
function PastRunsSidebar({
  runs,
  selectedId,
  onSelect,
}: {
  runs: Omit<BacktestRun, "equityCurve" | "trades">[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}) {
  return (
    <div className="w-52 shrink-0 border-r border-border overflow-y-auto" data-testid="past-runs-sidebar">
      <div className="p-3 border-b border-border">
        <p className="text-xs font-sans font-semibold text-muted-foreground uppercase tracking-wide">Past Results</p>
      </div>
      <div className="p-2 space-y-1">
        {runs.map((run) => (
          <button
            key={run.id}
            data-testid={`past-run-${run.id}`}
            onClick={() => onSelect(run.id)}
            className={cn(
              "w-full text-left px-3 py-2 rounded-md text-xs transition-colors",
              selectedId === run.id
                ? "bg-primary/10 border border-primary/30"
                : "hover:bg-accent/50 border border-transparent"
            )}
          >
            <p className="font-mono text-foreground font-medium">{run.ticker}</p>
            <p className="text-muted-foreground">{run.market} · {run.timescale}</p>
            <p className={cn("font-mono", run.totalReturn > 0 ? "text-success" : "text-destructive")}>
              {run.totalReturn > 0 ? "+" : ""}{run.totalReturn.toFixed(1)}%
            </p>
            <p className="text-muted-foreground text-xs mt-0.5">{run.createdAt}</p>
          </button>
        ))}
      </div>
    </div>
  );
}

const ROWS_PER_PAGE = 50;

export default function BacktestResults() {
  const [market, setMarket] = useState("US");
  const [ticker, setTicker] = useState("AAPL");
  const [timescale, setTimescale] = useState("1m");
  const [model, setModel] = useState("PPO_US_1m_v3");
  const [startDate, setStartDate] = useState("2024-09-01");
  const [endDate, setEndDate] = useState("2024-11-30");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<BacktestRun | null>(null);
  const [pastRuns] = useState(generatePastBacktestRuns());
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [page, setPage] = useState(0);

  const tickers = market === "US" ? US_TICKERS : HK_TICKERS;

  const handleRunBacktest = async () => {
    setIsLoading(true);
    setResult(null);
    setPage(0);
    await new Promise((r) => setTimeout(r, 1800));
    const res = generateBacktestResult(market, ticker, timescale, model);
    setResult(res);
    setSelectedRunId(res.id);
    setIsLoading(false);
  };

  const handleSelectPast = (id: number) => {
    const run = pastRuns.find((r) => r.id === id);
    if (!run) return;
    setSelectedRunId(id);
    setIsLoading(true);
    setResult(null);
    setTimeout(() => {
      const res = generateBacktestResult(run.market, run.ticker, run.timescale, run.model);
      setResult({ ...res, ...run, id, equityCurve: res.equityCurve, trades: res.trades });
      setIsLoading(false);
    }, 600);
  };

  const handleExportCSV = () => {
    if (!result) return;
    const header = "id,time,action,price,qty,pnl,cumulative_pnl";
    const rows = result.trades.map(
      (t) => `${t.id},${t.time},${t.action},${t.price.toFixed(2)},${t.qty},${t.pnl.toFixed(2)},${t.cumulativePnl.toFixed(2)}`
    );
    const csv = [header, ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `backtest_${result.ticker}_${result.startDate}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const trades = result?.trades ?? [];
  const totalPages = Math.ceil(trades.length / ROWS_PER_PAGE);
  const pageTrades = trades.slice(page * ROWS_PER_PAGE, (page + 1) * ROWS_PER_PAGE);

  return (
    <div className="flex flex-col h-full overflow-hidden" data-testid="page-backtest-results">
      {/* Header */}
      <div className="border-b border-border px-6 py-3 shrink-0">
        <h1 className="text-sm font-sans font-semibold text-foreground">Backtest Results</h1>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Past Runs */}
        <PastRunsSidebar runs={pastRuns} selectedId={selectedRunId} onSelect={handleSelectPast} />

        {/* Main content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 min-w-0">
          {/* Run Config */}
          <Card className="bg-card border-border">
            <CardContent className="p-4">
              <div className="flex flex-wrap items-end gap-3">
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground font-sans">Market</label>
                  <Select value={market} onValueChange={(v) => { setMarket(v); setTicker(v === "US" ? "AAPL" : "0700.HK"); }} data-testid="bt-market-select">
                    <SelectTrigger className="h-8 w-24 text-xs font-mono bg-muted border-border"><SelectValue /></SelectTrigger>
                    <SelectContent className="bg-popover border-border">
                      {MARKETS.map((m) => <SelectItem key={m} value={m} className="text-xs font-mono">{m}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground font-sans">Ticker</label>
                  <Select value={ticker} onValueChange={setTicker} data-testid="bt-ticker-select">
                    <SelectTrigger className="h-8 w-28 text-xs font-mono bg-muted border-border"><SelectValue /></SelectTrigger>
                    <SelectContent className="bg-popover border-border">
                      {tickers.map((t) => <SelectItem key={t} value={t} className="text-xs font-mono">{t}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground font-sans">Timescale</label>
                  <Select value={timescale} onValueChange={setTimescale} data-testid="bt-timescale-select">
                    <SelectTrigger className="h-8 w-24 text-xs font-mono bg-muted border-border"><SelectValue /></SelectTrigger>
                    <SelectContent className="bg-popover border-border">
                      {TIMESCALES.map((t) => <SelectItem key={t} value={t} className="text-xs font-mono">{t}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground font-sans">Model</label>
                  <Select value={model} onValueChange={setModel} data-testid="bt-model-select">
                    <SelectTrigger className="h-8 w-36 text-xs font-mono bg-muted border-border"><SelectValue /></SelectTrigger>
                    <SelectContent className="bg-popover border-border">
                      {MODELS.map((m) => <SelectItem key={m} value={m} className="text-xs font-mono">{m}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground font-sans">Start Date</label>
                  <input
                    data-testid="bt-start-date"
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="h-8 w-36 text-xs font-mono bg-muted border border-border rounded-md px-2 text-foreground"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground font-sans">End Date</label>
                  <input
                    data-testid="bt-end-date"
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="h-8 w-36 text-xs font-mono bg-muted border border-border rounded-md px-2 text-foreground"
                  />
                </div>
                <Button
                  data-testid="btn-run-backtest"
                  className="bg-primary hover:bg-primary/90 text-primary-foreground text-xs h-8"
                  onClick={handleRunBacktest}
                  disabled={isLoading}
                >
                  <Play size={12} className="mr-1" />
                  {isLoading ? "Running..." : "Run Backtest"}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Loading */}
          {isLoading && (
            <div className="space-y-4" data-testid="backtest-loading-skeleton">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {Array.from({ length: 6 }).map((_, i) => (
                  <Card key={i} className="bg-card border-border">
                    <CardContent className="p-4 space-y-2">
                      <Skeleton className="h-3 w-20 bg-muted" />
                      <Skeleton className="h-8 w-32 bg-muted" />
                    </CardContent>
                  </Card>
                ))}
              </div>
              <Card className="bg-card border-border">
                <CardContent className="p-4">
                  <Skeleton className="h-48 w-full bg-muted" />
                </CardContent>
              </Card>
            </div>
          )}

          {/* Results */}
          {result && !isLoading && (
            <>
              {/* Metrics Grid */}
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3" data-testid="metrics-grid">
                <MetricCard label="Sharpe Ratio" value={result.sharpe} target="Target: >1.5" good={result.sharpe > 1.5} bad={result.sharpe <= 1.0} testId="metric-sharpe" />
                <MetricCard label="Sortino Ratio" value={result.sortino} good={result.sortino > 1.5} testId="metric-sortino" />
                <MetricCard label="Calmar Ratio" value={result.calmar} good={result.calmar > 1.0} testId="metric-calmar" />
                <MetricCard label="Max Drawdown" value={result.maxDrawdown} suffix="%" bad={Math.abs(result.maxDrawdown) > 15} good={Math.abs(result.maxDrawdown) <= 10} testId="metric-drawdown" />
                <MetricCard label="Total Return" value={result.totalReturn} suffix="%" good={result.totalReturn > 0} bad={result.totalReturn < 0} testId="metric-return" />
                <Card className="bg-card border-border" data-testid="metric-win-rate">
                  <CardContent className="p-4">
                    <p className="text-xs text-muted-foreground font-sans uppercase tracking-wide mb-1">Win Rate</p>
                    <p className={cn("text-2xl font-mono font-semibold", result.winRate > 55 ? "text-success" : "text-warning")}>
                      {result.winRate.toFixed(1)}%
                    </p>
                    <p className="text-xs text-muted-foreground font-sans mt-1">{result.nTrades} trades</p>
                  </CardContent>
                </Card>
              </div>

              {/* Equity Curve */}
              <Card className="bg-card border-border" data-testid="equity-curve-chart">
                <CardHeader className="py-3 px-4 border-b border-border">
                  <CardTitle className="text-sm font-sans font-medium flex items-center justify-between">
                    Equity Curve
                    <span className="text-xs font-mono text-muted-foreground">{result.startDate} → {result.endDate}</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-3">
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={result.equityCurve} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                        <defs>
                          <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(222 35% 18%)" vertical={false} />
                        <XAxis
                          dataKey="date"
                          tick={{ fill: "#64748b", fontSize: 10, fontFamily: "'JetBrains Mono'" }}
                          tickLine={false}
                          axisLine={false}
                          interval={Math.floor(result.equityCurve.length / 6)}
                        />
                        <YAxis
                          tick={{ fill: "#64748b", fontSize: 10, fontFamily: "'JetBrains Mono'" }}
                          tickLine={false}
                          axisLine={false}
                          tickFormatter={(v) => `$${(v / 1000).toFixed(0)}K`}
                          width={55}
                        />
                        <Tooltip content={<EquityTooltip />} />
                        <Area
                          type="monotone"
                          dataKey="value"
                          stroke="#3b82f6"
                          strokeWidth={2}
                          fill="url(#equityGradient)"
                        />
                        {/* Drawdown reference lines */}
                        {result.equityCurve
                          .filter((d) => d.drawdown !== undefined && d.drawdown < -5)
                          .slice(0, 5)
                          .map((d, i) => (
                            <ReferenceLine key={i} x={d.date} stroke="#ef4444" strokeOpacity={0.3} strokeDasharray="2 2" />
                          ))}
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* Trade Log */}
              <Card className="bg-card border-border" data-testid="trade-log-table">
                <CardHeader className="py-3 px-4 border-b border-border">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-sans font-medium">
                      Trade Log <span className="text-muted-foreground font-normal text-xs ml-1">({trades.length} trades)</span>
                    </CardTitle>
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-xs h-7 border-border"
                      onClick={handleExportCSV}
                      data-testid="btn-export-csv"
                    >
                      <Download size={11} className="mr-1" />
                      Export CSV
                    </Button>
                  </div>
                </CardHeader>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead className="border-b border-border">
                      <tr>
                        {["#", "Time", "Action", "Price", "Qty", "P&L", "Cumulative P&L"].map((h) => (
                          <th key={h} className="px-3 py-2 text-left text-muted-foreground font-sans font-medium">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {pageTrades.map((trade) => (
                        <tr
                          key={trade.id}
                          className={cn("border-b border-border/50 hover:bg-accent/20 transition-colors", actionRowBg(trade.action))}
                          data-testid={`trade-row-${trade.id}`}
                        >
                          <td className="px-3 py-1.5 font-mono text-muted-foreground">{trade.id}</td>
                          <td className="px-3 py-1.5 font-mono text-muted-foreground">{trade.time}</td>
                          <td className={cn("px-3 py-1.5 font-mono font-semibold", actionClass(trade.action))}>{trade.action}</td>
                          <td className="px-3 py-1.5 font-mono text-foreground">${trade.price.toFixed(2)}</td>
                          <td className="px-3 py-1.5 font-mono text-foreground">{trade.qty}</td>
                          <td className={cn("px-3 py-1.5 font-mono", trade.pnl >= 0 ? "text-success" : "text-destructive")}>
                            {trade.pnl >= 0 ? "+" : ""}{trade.pnl.toFixed(2)}
                          </td>
                          <td className={cn("px-3 py-1.5 font-mono", trade.cumulativePnl >= 0 ? "text-success" : "text-destructive")}>
                            {trade.cumulativePnl >= 0 ? "+" : ""}{trade.cumulativePnl.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="flex items-center justify-between px-4 py-3 border-t border-border" data-testid="trade-log-pagination">
                    <p className="text-xs text-muted-foreground font-sans">
                      Page {page + 1} of {totalPages} · {trades.length} trades
                    </p>
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        className="h-7 text-xs border-border"
                        onClick={() => setPage((p) => Math.max(0, p - 1))}
                        disabled={page === 0}
                        data-testid="btn-prev-page"
                      >
                        <ChevronLeft size={12} />
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        className="h-7 text-xs border-border"
                        onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                        disabled={page === totalPages - 1}
                        data-testid="btn-next-page"
                      >
                        <ChevronRight size={12} />
                      </Button>
                    </div>
                  </div>
                )}
              </Card>
            </>
          )}

          {/* Empty state */}
          {!result && !isLoading && (
            <div className="flex flex-col items-center justify-center py-20 text-center" data-testid="backtest-empty-state">
              <TrendingUp size={40} className="text-muted-foreground mb-3 opacity-30" />
              <p className="text-sm text-muted-foreground font-sans">Configure parameters above and run a backtest</p>
              <p className="text-xs text-muted-foreground/60 font-sans mt-1">Results will appear here</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
