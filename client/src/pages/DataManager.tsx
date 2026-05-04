import { useState, useEffect, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Database,
  Download,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Trash2,
  Play,
  TrendingUp,
  Clock,
  BarChart2,
  FileText,
} from "lucide-react";

// ─── Types ───────────────────────────────────────────────────────────────────

interface DatasetInfo {
  ticker: string;
  market: string;
  timescale: string;
  n_bars: number;
  n_train: number;
  n_val: number;
  n_test: number;
  size_mb: number;
  created_at: string;
  file_path: string;
}

interface DataJob {
  job_id: string;
  ticker: string;
  timescale: string;
  status: "downloading" | "processing" | "splitting" | "done" | "error";
  progress_pct: number;
  elapsed_secs: number;
  error_msg?: string;
}

interface PreviewBar {
  t: string;
  c: number;
  v: number;
}

interface PreviewStats {
  total_bars: number;
  date_from: string;
  date_to: string;
  missing_pct: number;
  avg_volume: number;
  ann_volatility: number;
}

interface PreviewResponse {
  bars: PreviewBar[];
  stats: PreviewStats;
}

interface AvailableResponse {
  datasets: DatasetInfo[];
}

interface JobsResponse {
  jobs: DataJob[];
}

interface DownloadResponse {
  job_id: string;
}

// ─── Mock data ───────────────────────────────────────────────────────────────

const MOCK_DATASETS: DatasetInfo[] = [
  {
    ticker: "AAPL",
    market: "us",
    timescale: "1m",
    n_bars: 48200,
    n_train: 28920,
    n_val: 14460,
    n_test: 4820,
    size_mb: 12.4,
    created_at: "2024-01-15",
    file_path: "data/aapl_us_1m.parquet",
  },
  {
    ticker: "NVDA",
    market: "us",
    timescale: "1m",
    n_bars: 51000,
    n_train: 30600,
    n_val: 15300,
    n_test: 5100,
    size_mb: 13.1,
    created_at: "2024-01-15",
    file_path: "data/nvda_us_1m.parquet",
  },
  {
    ticker: "0700.HK",
    market: "hk",
    timescale: "1m",
    n_bars: 39600,
    n_train: 23760,
    n_val: 11880,
    n_test: 3960,
    size_mb: 9.8,
    created_at: "2024-01-14",
    file_path: "data/0700.HK_hk_1m.parquet",
  },
];

// Generate mock preview bars for a given ticker
function generateMockBars(ticker: string): PreviewBar[] {
  const bars: PreviewBar[] = [];
  const seed = ticker.charCodeAt(0);
  let price = 150 + seed;
  const now = Date.now();
  for (let i = 199; i >= 0; i--) {
    price = price + (Math.random() - 0.49) * 2;
    bars.push({
      t: new Date(now - i * 60_000).toISOString(),
      c: Math.max(1, parseFloat(price.toFixed(2))),
      v: Math.floor(50000 + Math.random() * 200000),
    });
  }
  return bars;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function today(): string {
  return new Date().toISOString().split("T")[0];
}

function oneYearAgo(): string {
  const d = new Date();
  d.setFullYear(d.getFullYear() - 1);
  return d.toISOString().split("T")[0];
}

function fmtNum(n: number): string {
  return n.toLocaleString();
}

function fmtElapsed(secs: number): string {
  if (secs < 60) return `${secs}s`;
  return `${Math.floor(secs / 60)}m ${secs % 60}s`;
}

function statusColor(
  status: DataJob["status"]
): "default" | "secondary" | "destructive" | "outline" {
  switch (status) {
    case "downloading":
      return "outline";
    case "processing":
      return "secondary";
    case "splitting":
      return "secondary";
    case "done":
      return "default";
    case "error":
      return "destructive";
    default:
      return "default";
  }
}

function statusLabel(status: DataJob["status"]): string {
  return status.charAt(0).toUpperCase() + status.slice(1);
}

// ─── Component ───────────────────────────────────────────────────────────────

export default function DataManager() {
  const qc = useQueryClient();

  // ── Config state
  const [market, setMarket] = useState<"us" | "hk">("us");
  const [tickerInput, setTickerInput] = useState("AAPL, NVDA, META");
  const [timescale, setTimescale] = useState("1m");
  const [startDate, setStartDate] = useState(oneYearAgo());
  const [endDate, setEndDate] = useState(today());
  const [trainPct, setTrainPct] = useState(60);
  const [valPct, setValPct] = useState(30);
  const testPct = Math.max(0, 100 - trainPct - valPct);

  // ── Selected dataset for preview
  const [selectedDataset, setSelectedDataset] = useState<DatasetInfo | null>(null);

  // ── Jobs that should be hidden (completed > 5s ago)
  const [hiddenJobs, setHiddenJobs] = useState<Set<string>>(new Set());

  // ─── Queries ─────────────────────────────────────────────────────────────

  const availableQuery = useQuery<AvailableResponse>({
    queryKey: ["/api/data/available"],
    refetchInterval: 10_000,
    retry: false,
  });

  const datasets: DatasetInfo[] =
    availableQuery.data?.datasets ?? MOCK_DATASETS;

  const jobsQuery = useQuery<JobsResponse>({
    queryKey: ["/api/data/jobs"],
    refetchInterval: 2_000,
    retry: false,
  });

  const previewQuery = useQuery<PreviewResponse>({
    queryKey: selectedDataset
      ? [
          `/api/data/preview?ticker=${selectedDataset.ticker}&market=${selectedDataset.market}&timescale=${selectedDataset.timescale}`,
        ]
      : ["__disabled__"],
    enabled: !!selectedDataset,
    retry: false,
  });

  // ─── Mutations ───────────────────────────────────────────────────────────

  const downloadMutation = useMutation<DownloadResponse, Error, Record<string, unknown>>({
    mutationFn: async (body) => {
      const res = await apiRequest("POST", "/api/data/download", body);
      return res.json();
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["/api/data/jobs"] });
    },
  });

  const deleteMutation = useMutation<{ ok: boolean }, Error, { ticker: string; market: string; timescale: string }>({
    mutationFn: async (body) => {
      const res = await apiRequest("DELETE", "/api/data/dataset", body);
      return res.json();
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["/api/data/available"] });
      if (selectedDataset) setSelectedDataset(null);
    },
  });

  // ─── Auto-hide completed jobs ─────────────────────────────────────────────

  const jobs: DataJob[] = jobsQuery.data?.jobs ?? [];

  useEffect(() => {
    jobs.forEach((job) => {
      if ((job.status === "done" || job.status === "error") && !hiddenJobs.has(job.job_id)) {
        const timer = setTimeout(() => {
          setHiddenJobs((prev) => new Set([...prev, job.job_id]));
        }, 5_000);
        return () => clearTimeout(timer);
      }
    });
  }, [jobs, hiddenJobs]);

  const visibleJobs = jobs.filter((j) => !hiddenJobs.has(j.job_id));

  // ─── Handlers ─────────────────────────────────────────────────────────────

  const handleDownload = useCallback(() => {
    const tickers = tickerInput
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    downloadMutation.mutate({
      market,
      tickers,
      timescale,
      start_date: startDate,
      end_date: endDate,
      train_ratio: trainPct / 100,
      val_ratio: valPct / 100,
    });
  }, [tickerInput, market, timescale, startDate, endDate, trainPct, valPct, downloadMutation]);

  const handleRefresh = useCallback(() => {
    qc.invalidateQueries({ queryKey: ["/api/data/available"] });
  }, [qc]);

  const handleUseForTraining = useCallback(
    (ds: DatasetInfo) => {
      const params = new URLSearchParams({
        ticker: ds.ticker,
        market: ds.market,
        timescale: ds.timescale,
      });
      window.location.href = `/training?${params.toString()}`;
    },
    []
  );

  // ─── Preview data ─────────────────────────────────────────────────────────

  const previewBars: PreviewBar[] =
    previewQuery.data?.bars?.slice(-200) ??
    (selectedDataset ? generateMockBars(selectedDataset.ticker) : []);

  const previewStats: PreviewStats | null =
    previewQuery.data?.stats ??
    (selectedDataset
      ? {
          total_bars: selectedDataset.n_bars,
          date_from: selectedDataset.created_at,
          date_to: today(),
          missing_pct: 0.4,
          avg_volume: 1_250_000,
          ann_volatility: 0.28,
        }
      : null);

  const chartData = previewBars.map((b) => ({
    time: new Date(b.t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    price: b.c,
  }));

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-background text-foreground p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3 mb-2">
        <Database className="w-6 h-6 text-primary" />
        <h1 className="text-xl font-semibold tracking-tight">Data Manager</h1>
        <Badge variant="outline" className="text-xs font-mono text-muted-foreground">
          {datasets.length} dataset{datasets.length !== 1 ? "s" : ""}
        </Badge>
      </div>

      {/* 3-column grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* ── Left: Config ──────────────────────────────────────────────── */}
        <Card className="bg-card border border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <FileText className="w-4 h-4 text-primary" />
              Data Source Config
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Market toggle */}
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Market</Label>
              <div className="flex gap-2">
                {(["us", "hk"] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setMarket(m)}
                    className={`flex-1 py-1.5 rounded-md border text-sm font-semibold uppercase tracking-wider transition-colors ${
                      market === m
                        ? "bg-primary/10 text-primary border-primary/30"
                        : "bg-card text-muted-foreground border-border hover:border-primary/20"
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>

            {/* Ticker input */}
            <div className="space-y-1.5">
              <Label htmlFor="ticker-input" className="text-xs text-muted-foreground">
                Tickers (comma-separated)
              </Label>
              <Input
                id="ticker-input"
                value={tickerInput}
                onChange={(e) => setTickerInput(e.target.value)}
                placeholder="AAPL, NVDA, META"
                className="font-mono text-sm bg-background border-border"
              />
            </div>

            {/* Timescale */}
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Timescale</Label>
              <Select value={timescale} onValueChange={setTimescale}>
                <SelectTrigger className="bg-background border-border text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {["10s", "1m", "5m", "1h"].map((ts) => (
                    <SelectItem key={ts} value={ts} className="font-mono">
                      {ts}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Date range */}
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Date Range</Label>
              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label className="text-xs text-muted-foreground/70">Start</Label>
                  <Input
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="text-xs font-mono bg-background border-border"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-xs text-muted-foreground/70">End</Label>
                  <Input
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="text-xs font-mono bg-background border-border"
                  />
                </div>
              </div>
            </div>

            <Separator className="bg-border" />

            {/* Split ratios */}
            <div className="space-y-3">
              <Label className="text-xs text-muted-foreground">Split Ratios</Label>

              {/* Train */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Train</span>
                  <span className="font-mono text-primary">{trainPct}%</span>
                </div>
                <input
                  type="range"
                  min={10}
                  max={90}
                  value={trainPct}
                  onChange={(e) => {
                    const v = parseInt(e.target.value);
                    setTrainPct(v);
                    if (v + valPct > 95) setValPct(Math.max(5, 95 - v));
                  }}
                  className="w-full accent-primary"
                />
              </div>

              {/* Val */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Validation</span>
                  <span className="font-mono text-primary">{valPct}%</span>
                </div>
                <input
                  type="range"
                  min={5}
                  max={50}
                  value={valPct}
                  onChange={(e) => {
                    const v = parseInt(e.target.value);
                    setValPct(v);
                    if (trainPct + v > 95) setTrainPct(Math.max(10, 95 - v));
                  }}
                  className="w-full accent-primary"
                />
              </div>

              {/* Test (read-only) */}
              <div className="flex justify-between text-xs items-center">
                <span className="text-muted-foreground">Test (auto)</span>
                <span className={`font-mono font-semibold ${testPct < 0 ? "text-destructive" : "text-muted-foreground"}`}>
                  {testPct}%
                </span>
              </div>

              {/* Visual split bar */}
              <div className="flex h-2 rounded-full overflow-hidden gap-px">
                <div className="bg-primary transition-all" style={{ width: `${trainPct}%` }} />
                <div className="bg-blue-400 transition-all" style={{ width: `${valPct}%` }} />
                <div className="bg-muted-foreground/40 transition-all" style={{ width: `${testPct}%` }} />
              </div>
              <div className="flex justify-between text-[10px] text-muted-foreground/60">
                <span>Train</span>
                <span>Val</span>
                <span>Test</span>
              </div>
            </div>

            <Separator className="bg-border" />

            {/* Action buttons */}
            <div className="flex flex-col gap-2">
              <Button
                onClick={handleDownload}
                disabled={downloadMutation.isPending}
                className="w-full gap-2 text-sm"
              >
                {downloadMutation.isPending ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Download className="w-4 h-4" />
                )}
                {downloadMutation.isPending ? "Queuing…" : "Download & Process"}
              </Button>
              <Button
                variant="outline"
                onClick={handleRefresh}
                disabled={availableQuery.isFetching}
                className="w-full gap-2 text-sm border-border"
              >
                <RefreshCw className={`w-4 h-4 ${availableQuery.isFetching ? "animate-spin" : ""}`} />
                Refresh Available
              </Button>

              {/* Download error */}
              {downloadMutation.isError && (
                <div className="flex items-center gap-2 text-xs text-destructive bg-destructive/10 border border-destructive/20 rounded-md px-3 py-2">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                  {downloadMutation.error.message}
                </div>
              )}
              {downloadMutation.isSuccess && (
                <div className="flex items-center gap-2 text-xs text-green-400 bg-green-400/10 border border-green-400/20 rounded-md px-3 py-2">
                  <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                  Job queued: {downloadMutation.data?.job_id}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* ── Middle: Available Datasets ────────────────────────────────── */}
        <Card className="bg-card border border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <Database className="w-4 h-4 text-primary" />
              Available Datasets
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {availableQuery.isLoading ? (
              <div className="px-4 pb-4 space-y-2">
                {[0, 1, 2].map((i) => (
                  <div key={i} className="space-y-1.5">
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-3 w-3/4" />
                  </div>
                ))}
              </div>
            ) : datasets.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 px-6 text-center text-muted-foreground">
                <Database className="w-10 h-10 mb-3 opacity-30" />
                <p className="text-sm">No datasets downloaded yet.</p>
                <p className="text-xs mt-1 opacity-70">Configure and download above.</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border text-muted-foreground">
                      <th className="text-left px-4 py-2 font-medium">Ticker</th>
                      <th className="text-left px-4 py-2 font-medium">Scale</th>
                      <th className="text-right px-4 py-2 font-medium">Bars</th>
                      <th className="text-right px-4 py-2 font-medium">T/V/T</th>
                      <th className="text-right px-4 py-2 font-medium">MB</th>
                      <th className="text-right px-4 py-2 font-medium">Date</th>
                      <th className="px-4 py-2" />
                    </tr>
                  </thead>
                  <tbody>
                    {datasets.map((ds) => {
                      const isSelected =
                        selectedDataset?.ticker === ds.ticker &&
                        selectedDataset?.market === ds.market &&
                        selectedDataset?.timescale === ds.timescale;
                      return (
                        <tr
                          key={`${ds.ticker}-${ds.market}-${ds.timescale}`}
                          onClick={() => setSelectedDataset(isSelected ? null : ds)}
                          className={`border-b border-border/50 cursor-pointer transition-colors hover:bg-primary/5 ${
                            isSelected ? "bg-primary/5 border-l-2 border-primary" : ""
                          }`}
                        >
                          <td className="px-4 py-2.5">
                            <div className="font-mono font-semibold text-foreground">
                              {ds.ticker}
                            </div>
                            <div className="text-muted-foreground/70 uppercase text-[10px]">
                              {ds.market}
                            </div>
                          </td>
                          <td className="px-4 py-2.5">
                            <Badge variant="outline" className="font-mono text-[10px] px-1.5">
                              {ds.timescale}
                            </Badge>
                          </td>
                          <td className="px-4 py-2.5 text-right font-mono text-foreground">
                            {fmtNum(ds.n_bars)}
                          </td>
                          <td className="px-4 py-2.5 text-right font-mono text-muted-foreground">
                            <span className="text-primary">{(ds.n_train / 1000).toFixed(0)}k</span>
                            /
                            <span className="text-blue-400">{(ds.n_val / 1000).toFixed(0)}k</span>
                            /
                            <span>{(ds.n_test / 1000).toFixed(0)}k</span>
                          </td>
                          <td className="px-4 py-2.5 text-right font-mono text-muted-foreground">
                            {ds.size_mb.toFixed(1)}
                          </td>
                          <td className="px-4 py-2.5 text-right font-mono text-muted-foreground/70 text-[10px]">
                            {ds.created_at}
                          </td>
                          <td className="px-3 py-2.5">
                            <div className="flex items-center gap-1.5 justify-end">
                              <button
                                title="Use for Training"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleUseForTraining(ds);
                                }}
                                className="p-1 rounded text-primary hover:bg-primary/10 transition-colors"
                              >
                                <Play className="w-3.5 h-3.5" />
                              </button>
                              <button
                                title="Delete Dataset"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  deleteMutation.mutate({
                                    ticker: ds.ticker,
                                    market: ds.market,
                                    timescale: ds.timescale,
                                  });
                                }}
                                className="p-1 rounded text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>

        {/* ── Right: Dataset Preview ─────────────────────────────────────── */}
        <Card className="bg-card border border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-primary" />
              Dataset Preview
              {selectedDataset && (
                <span className="font-mono text-muted-foreground font-normal">
                  — {selectedDataset.ticker} ({selectedDataset.timescale})
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {!selectedDataset ? (
              <div className="flex flex-col items-center justify-center h-[220px] text-muted-foreground">
                <BarChart2 className="w-10 h-10 mb-3 opacity-30" />
                <p className="text-sm">Click a dataset to preview</p>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Chart */}
                <div className="h-[200px]">
                  {previewQuery.isLoading ? (
                    <Skeleton className="w-full h-full" />
                  ) : (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis
                          dataKey="time"
                          tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                          interval="preserveStartEnd"
                          tickLine={false}
                          axisLine={false}
                        />
                        <YAxis
                          domain={["auto", "auto"]}
                          tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                          tickLine={false}
                          axisLine={false}
                          width={50}
                          tickFormatter={(v) => `$${v.toFixed(0)}`}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "hsl(var(--card))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: "6px",
                            fontSize: "11px",
                          }}
                          formatter={(value: number) => [`$${value.toFixed(2)}`, "Close"]}
                          labelStyle={{ color: "hsl(var(--muted-foreground))" }}
                        />
                        <Line
                          type="monotone"
                          dataKey="price"
                          stroke="hsl(var(--primary))"
                          strokeWidth={1.5}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </div>

                <Separator className="bg-border" />

                {/* Stats grid */}
                {previewStats && (
                  <div className="grid grid-cols-2 gap-x-4 gap-y-2.5">
                    {[
                      {
                        label: "Total Bars",
                        value: fmtNum(previewStats.total_bars),
                        icon: <BarChart2 className="w-3 h-3" />,
                      },
                      {
                        label: "Date Range",
                        value: `${previewStats.date_from} → ${previewStats.date_to}`,
                        icon: <Clock className="w-3 h-3" />,
                      },
                      {
                        label: "Missing %",
                        value: `${previewStats.missing_pct.toFixed(2)}%`,
                        icon: <AlertCircle className="w-3 h-3" />,
                        warn: previewStats.missing_pct > 2,
                      },
                      {
                        label: "Avg Volume",
                        value:
                          previewStats.avg_volume > 1_000_000
                            ? `${(previewStats.avg_volume / 1_000_000).toFixed(2)}M`
                            : `${(previewStats.avg_volume / 1_000).toFixed(0)}K`,
                        icon: <TrendingUp className="w-3 h-3" />,
                      },
                      {
                        label: "Ann. Volatility",
                        value: `${(previewStats.ann_volatility * 100).toFixed(1)}%`,
                        icon: <TrendingUp className="w-3 h-3" />,
                        warn: previewStats.ann_volatility > 0.5,
                      },
                    ].map((stat) => (
                      <div key={stat.label} className="space-y-0.5">
                        <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
                          {stat.icon}
                          {stat.label}
                        </div>
                        <div
                          className={`text-xs font-mono font-medium ${
                            stat.warn ? "text-yellow-400" : "text-foreground"
                          }`}
                        >
                          {stat.value}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* ── Bottom: Download Progress panel ─────────────────────────────── */}
      {visibleJobs.length > 0 && (
        <Card className="bg-card border border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <Download className="w-4 h-4 text-primary" />
              Active Jobs
              <Badge variant="secondary" className="text-xs font-mono">
                {visibleJobs.filter((j) => j.status !== "done" && j.status !== "error").length} running
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {visibleJobs.map((job) => (
                <div key={job.job_id} className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm font-semibold text-foreground">
                        {job.ticker}
                      </span>
                      <Badge variant="outline" className="text-[10px] font-mono px-1.5">
                        {job.timescale}
                      </Badge>
                      <Badge
                        variant={statusColor(job.status)}
                        className={`text-[10px] font-mono px-1.5 ${
                          job.status === "downloading"
                            ? "border-yellow-500/30 text-yellow-400 bg-yellow-400/10"
                            : job.status === "processing"
                            ? "border-primary/30 text-primary bg-primary/10"
                            : job.status === "splitting"
                            ? "border-purple-500/30 text-purple-400 bg-purple-400/10"
                            : job.status === "done"
                            ? "border-green-500/30 text-green-400 bg-green-400/10"
                            : "border-destructive/30 text-destructive bg-destructive/10"
                        }`}
                      >
                        {job.status === "done" && <CheckCircle2 className="w-2.5 h-2.5 mr-1" />}
                        {job.status === "error" && <AlertCircle className="w-2.5 h-2.5 mr-1" />}
                        {statusLabel(job.status)}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-muted-foreground font-mono">
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {fmtElapsed(job.elapsed_secs)}
                      </span>
                      <span>{job.progress_pct.toFixed(0)}%</span>
                    </div>
                  </div>
                  <Progress value={job.progress_pct} className="h-1.5" />
                  {job.error_msg && (
                    <p className="text-xs text-destructive font-mono">{job.error_msg}</p>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
