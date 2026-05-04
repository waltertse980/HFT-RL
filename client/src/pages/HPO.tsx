import { useState, useCallback, useMemo } from "react";
import { useLocation } from "wouter";
import {
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
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
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import {
  Zap,
  Play,
  Square,
  RefreshCw,
  Trophy,
  TrendingUp,
  Clock,
  ChevronDown,
  ChevronUp,
  Settings2,
  Download,
  AlertCircle,
} from "lucide-react";
import { useEffect, useRef } from "react";

// ─── API fetch helper ─────────────────────────────────────────────────────────

async function apiFetch<T>(url: string, options?: RequestInit): Promise<{ data: T | null; engineOffline: boolean; error: string | null }> {
  try {
    const res = await fetch(url, options);
    if (res.status === 503) return { data: null, engineOffline: true, error: 'ENGINE_OFFLINE' };
    if (!res.ok) {
      const body = await res.json().catch(() => ({})) as { message?: string };
      return { data: null, engineOffline: false, error: body.message ?? `HTTP ${res.status}` };
    }
    return { data: await res.json() as T, engineOffline: false, error: null };
  } catch (err) {
    return { data: null, engineOffline: false, error: String(err) };
  }
}

// ─── EngineOfflineBanner ──────────────────────────────────────────────────────

function EngineOfflineBanner() {
  return (
    <div className="mx-4 mt-4 p-3 rounded-lg border border-destructive/40 bg-destructive/10 flex items-start gap-3">
      <AlertCircle size={16} className="text-destructive mt-0.5 shrink-0" />
      <div>
        <p className="text-sm font-semibold text-destructive">Python engine is offline</p>
        <p className="text-xs text-muted-foreground mt-0.5 font-mono">
          cd python_engine && uvicorn api_server:app --port 8001 --reload
        </p>
      </div>
    </div>
  );
}

// ─── Types ────────────────────────────────────────────────────────────────────

interface HpoTrial {
  number: number;
  status: "complete" | "pruned" | "failed";
  value: number;
  params: {
    learning_rate: number;
    n_steps: number;
    batch_size: number;
    gamma: number;
    net_arch: string;
  };
  duration_secs: number;
}

interface HpoStatus {
  study_id: string;
  study_name: string;
  n_trials_total: number;
  n_completed: number;
  n_pruned: number;
  n_failed: number;
  best_value: number | null;
  best_params: HpoTrial["params"] | null;
  status: "running" | "done" | "error";
  eta_secs: number | null;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function statusColor(status: HpoTrial["status"]) {
  if (status === "complete") return "text-green-400";
  if (status === "pruned") return "text-amber-400/70";
  return "text-destructive/70";
}

function statusDot(status: HpoTrial["status"]) {
  if (status === "complete") return "#22c55e";
  if (status === "pruned") return "#f59e0b";
  return "#ef4444";
}

function statusBadgeVariant(status: HpoTrial["status"]) {
  if (status === "complete") return "bg-green-500/10 text-green-400 border-green-500/30";
  if (status === "pruned") return "bg-amber-500/10 text-amber-400 border-amber-500/30";
  return "bg-destructive/10 text-destructive border-destructive/30";
}

function formatEta(secs: number): string {
  if (secs <= 0) return "—";
  if (secs < 60) return `${Math.round(secs)}s`;
  if (secs < 3600) return `${Math.round(secs / 60)}m`;
  return `${(secs / 3600).toFixed(1)}h`;
}

function formatDuration(secs: number): string {
  if (secs < 60) return `${Math.round(secs)}s`;
  return `${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s`;
}

type SortKey = "number" | "status" | "value" | "duration_secs";
type SortDir = "asc" | "desc";

// ─── Sub-components ────────────────────────────────────────────────────────────

function TrialTooltip({ active, payload }: { active?: boolean; payload?: any[] }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload as HpoTrial;
  if (!d) return null;
  return (
    <div className="bg-card border border-border rounded-md p-3 text-xs font-mono shadow-lg min-w-40">
      <p className="text-muted-foreground mb-1">Trial #{d.number}</p>
      <p className={cn("font-semibold mb-1", statusColor(d.status))}>{d.status.toUpperCase()}</p>
      <p>Value: <span className="text-foreground">{d.value.toFixed(4)}</span></p>
      <p>LR: <span className="text-primary">{d.params.learning_rate.toExponential(2)}</span></p>
      <p>N Steps: <span className="text-foreground">{d.params.n_steps}</span></p>
      <p>Batch: <span className="text-foreground">{d.params.batch_size}</span></p>
      <p>γ: <span className="text-foreground">{d.params.gamma.toFixed(4)}</span></p>
      <p>Arch: <span className="text-foreground">{d.params.net_arch}</span></p>
      <p className="mt-1 text-muted-foreground">{formatDuration(d.duration_secs)}</p>
    </div>
  );
}

function ParamRow({ name, value }: { name: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-border/40 last:border-0">
      <span className="text-xs text-muted-foreground font-sans">{name}</span>
      <span className="text-xs text-primary font-mono">{String(value)}</span>
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────

export default function HPO() {
  const [, setLocation] = useLocation();

  // ── Config state ──────────────────────────────────────────────────────────
  const [market, setMarket] = useState<"US" | "HK">("US");
  const [timescale, setTimescale] = useState("1m");
  const [algo, setAlgo] = useState("PPO");
  const [nTrials, setNTrials] = useState(50);
  const [timestepsPerTrial, setTimestepsPerTrial] = useState("200K");
  const [nJobs, setNJobs] = useState("1");
  const [storage, setStorage] = useState("");

  // ── Study state ───────────────────────────────────────────────────────────
  const [activeStudyId, setActiveStudyId] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [runPending, setRunPending] = useState(false);
  const [stopPending, setStopPending] = useState(false);

  // ── Real data from API ────────────────────────────────────────────────────
  const [statusData, setStatusData] = useState<HpoStatus | null>(null);
  const [trials, setTrials] = useState<HpoTrial[]>([]);
  const [statusEngineOffline, setStatusEngineOffline] = useState(false);
  const [trialsEngineOffline, setTrialsEngineOffline] = useState(false);

  // ── Sort / pagination state ───────────────────────────────────────────────
  const [sortKey, setSortKey] = useState<SortKey>("number");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 10;

  // ── Toast ─────────────────────────────────────────────────────────────────
  const [toast, setToast] = useState<{ message: string } | null>(null);
  const showToast = useCallback((message: string) => {
    setToast({ message });
    setTimeout(() => setToast(null), 5000);
  }, []);

  // ── Polling refs ──────────────────────────────────────────────────────────
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  // ── Fetch status ──────────────────────────────────────────────────────────
  const fetchStatus = useCallback(async (studyId: string) => {
    const { data, engineOffline } = await apiFetch<HpoStatus>(`/api/hpo/status/${studyId}`);
    if (engineOffline) {
      setStatusEngineOffline(true);
      setIsRunning(false);
      stopPolling();
      return;
    }
    setStatusEngineOffline(false);
    if (data) {
      setStatusData(data);
      if (data.status !== "running") {
        setIsRunning(false);
        stopPolling();
      }
    }
  }, [stopPolling]);

  // ── Fetch trials ──────────────────────────────────────────────────────────
  const fetchTrials = useCallback(async (studyId: string) => {
    const { data, engineOffline } = await apiFetch<{ trials: HpoTrial[] }>(`/api/hpo/trials/${studyId}`);
    if (engineOffline) {
      setTrialsEngineOffline(true);
      setTrials([]);
      return;
    }
    setTrialsEngineOffline(false);
    setTrials(data?.trials ?? []);
  }, []);

  // ── Start polling when we have an active study ────────────────────────────
  useEffect(() => {
    if (!activeStudyId || !isRunning) return;
    // Fetch immediately
    fetchStatus(activeStudyId);
    fetchTrials(activeStudyId);
    // Then poll
    pollRef.current = setInterval(() => {
      fetchStatus(activeStudyId);
      fetchTrials(activeStudyId);
    }, 2000);
    return () => stopPolling();
  }, [activeStudyId, isRunning, fetchStatus, fetchTrials, stopPolling]);

  // ── Fetch once when study completes (not running) ─────────────────────────
  useEffect(() => {
    if (!activeStudyId || isRunning) return;
    fetchStatus(activeStudyId);
    fetchTrials(activeStudyId);
  }, [activeStudyId, isRunning, fetchStatus, fetchTrials]);

  // ── API: run HPO ──────────────────────────────────────────────────────────
  const handleRun = useCallback(async () => {
    setRunPending(true);
    const { data, engineOffline, error } = await apiFetch<{ study_id: string; status: string }>(
      "/api/hpo/run",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          market,
          timescale,
          algo,
          n_trials: nTrials,
          timesteps_per_trial: timestepsPerTrial,
          n_jobs: parseInt(nJobs),
          storage: storage || undefined,
        }),
      }
    );
    setRunPending(false);

    if (engineOffline) {
      showToast("Cannot run HPO: Python engine is offline. Start it first.");
      setIsRunning(false);
      return;
    }

    if (error || !data?.study_id) {
      showToast(error ?? "Failed to start HPO study");
      setIsRunning(false);
      return;
    }

    setActiveStudyId(data.study_id);
    setStatusData(null);
    setTrials([]);
    setStatusEngineOffline(false);
    setTrialsEngineOffline(false);
    setPage(0);
    setIsRunning(true);
  }, [market, timescale, algo, nTrials, timestepsPerTrial, nJobs, storage, showToast]);

  // ── API: stop HPO ─────────────────────────────────────────────────────────
  const handleStop = useCallback(async () => {
    if (!activeStudyId) return;
    setStopPending(true);
    await apiFetch(`/api/hpo/stop/${activeStudyId}`, { method: "POST" });
    setStopPending(false);
    setIsRunning(false);
    stopPolling();
    // Fetch final state
    fetchStatus(activeStudyId);
    fetchTrials(activeStudyId);
  }, [activeStudyId, stopPolling, fetchStatus, fetchTrials]);

  // ── Derived best trial ────────────────────────────────────────────────────
  const bestTrial = useMemo(() => {
    const completed = trials.filter((t) => t.status === "complete");
    return completed.reduce<HpoTrial | null>(
      (best, t) => (best === null || t.value > best.value ? t : best),
      null
    );
  }, [trials]);

  // ── Chart data ────────────────────────────────────────────────────────────
  const chartData = useMemo(() => {
    return trials.map((t) => ({
      ...t,
      isBest: bestTrial?.number === t.number,
    }));
  }, [trials, bestTrial]);

  // Running best line
  const runningBestData = useMemo(() => {
    let best: number | null = null;
    return trials.map((t) => {
      if (t.status === "complete" && (best === null || t.value > best)) {
        best = t.value;
      }
      return { number: t.number, best };
    });
  }, [trials]);

  // ── Sort/filter trials ────────────────────────────────────────────────────
  const sortedTrials = useMemo(() => {
    const arr = [...trials];
    arr.sort((a, b) => {
      let av: string | number = a[sortKey] as string | number;
      let bv: string | number = b[sortKey] as string | number;
      if (typeof av === "string") av = av.toLowerCase();
      if (typeof bv === "string") bv = bv.toLowerCase();
      if (av < bv) return sortDir === "asc" ? -1 : 1;
      if (av > bv) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
    return arr;
  }, [trials, sortKey, sortDir]);

  const pagedTrials = sortedTrials.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = Math.ceil(sortedTrials.length / PAGE_SIZE);

  const handleSort = useCallback(
    (key: SortKey) => {
      if (sortKey === key) {
        setSortDir((d) => (d === "asc" ? "desc" : "asc"));
      } else {
        setSortKey(key);
        setSortDir("asc");
      }
    },
    [sortKey]
  );

  // ── Apply / Export ────────────────────────────────────────────────────────
  const handleApplyToTraining = () => {
    if (!bestTrial) return;
    localStorage.setItem("hpo_best_params", JSON.stringify(bestTrial.params));
    setLocation("/training");
  };

  const handleExportJson = () => {
    if (!bestTrial) return;
    const blob = new Blob([JSON.stringify(bestTrial.params, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `hpo_best_params_${activeStudyId ?? "export"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── Progress ──────────────────────────────────────────────────────────────
  const progressPct = statusData
    ? Math.round((statusData.n_completed / Math.max(statusData.n_trials_total, 1)) * 100)
    : 0;

  // ── Engine offline: any endpoint offline means we show the banner ─────────
  const anyEngineOffline = statusEngineOffline || trialsEngineOffline;

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full overflow-hidden" data-testid="page-hpo">
      {/* Toast */}
      {toast && (
        <div className="fixed top-4 right-4 z-50 px-4 py-3 rounded-lg border bg-destructive/10 border-destructive/40 text-destructive text-sm font-medium shadow-lg flex items-center gap-2">
          <AlertCircle size={14} className="shrink-0" />
          {toast.message}
        </div>
      )}

      {/* Engine offline banner */}
      {anyEngineOffline && <EngineOfflineBanner />}

      {/* Page header */}
      <div className="border-b border-border px-6 py-3 flex items-center justify-between shrink-0">
        <h1 className="text-sm font-sans font-semibold text-foreground flex items-center gap-2">
          <Zap size={14} className="text-primary" />
          Hyperparameter Optimisation
          {isRunning && (
            <span className="flex items-center gap-1 text-amber-400 text-xs font-mono animate-pulse">
              <RefreshCw size={10} className="animate-spin" /> Running
            </span>
          )}
          {statusData?.status === "done" && (
            <span className="flex items-center gap-1 text-green-400 text-xs font-mono">
              <Trophy size={10} /> Done
            </span>
          )}
        </h1>
        <span className="text-xs text-muted-foreground font-mono">
          {activeStudyId
            ? `study: ${statusData?.study_name ?? activeStudyId}`
            : "no active study"}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* ── Run Configuration ──────────────────────────────────────────── */}
        <Card className="bg-card border-border">
          <CardHeader className="py-3 px-4 border-b border-border">
            <CardTitle className="text-sm font-sans font-medium text-foreground flex items-center gap-2">
              <Settings2 size={13} /> Run Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4">
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-x-4 gap-y-3">
              {/* Market */}
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">Market</Label>
                <Tabs value={market} onValueChange={(v) => setMarket(v as "US" | "HK")}>
                  <TabsList className="w-full bg-muted h-8">
                    <TabsTrigger value="US" className="flex-1 text-xs">US</TabsTrigger>
                    <TabsTrigger value="HK" className="flex-1 text-xs">HK</TabsTrigger>
                  </TabsList>
                </Tabs>
              </div>

              {/* Timescale */}
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">Timescale</Label>
                <Select value={timescale} onValueChange={setTimescale}>
                  <SelectTrigger className="h-8 text-xs font-mono bg-muted border-border">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border">
                    {["10s", "1m", "5m", "1h"].map((ts) => (
                      <SelectItem key={ts} value={ts} className="text-xs font-mono">{ts}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Algorithm */}
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">Algorithm</Label>
                <Select value={algo} onValueChange={setAlgo}>
                  <SelectTrigger className="h-8 text-xs font-mono bg-muted border-border">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border">
                    <SelectItem value="PPO" className="text-xs font-mono">PPO</SelectItem>
                    <SelectItem value="TD3" className="text-xs font-mono">TD3</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* N Trials */}
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">N Trials</Label>
                <Input
                  type="number"
                  min={5}
                  max={200}
                  value={nTrials}
                  onChange={(e) => setNTrials(Math.min(200, Math.max(5, parseInt(e.target.value) || 50)))}
                  className="h-8 text-xs font-mono bg-muted border-border"
                />
              </div>

              {/* Timesteps per Trial */}
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">Timesteps / Trial</Label>
                <Select value={timestepsPerTrial} onValueChange={setTimestepsPerTrial}>
                  <SelectTrigger className="h-8 text-xs font-mono bg-muted border-border">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border">
                    {["50K", "100K", "200K", "500K"].map((v) => (
                      <SelectItem key={v} value={v} className="text-xs font-mono">{v}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Parallel Jobs */}
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">
                  Parallel Jobs
                  <span className="ml-1 text-muted-foreground/60">(SQLite)</span>
                </Label>
                <Select value={nJobs} onValueChange={setNJobs}>
                  <SelectTrigger className="h-8 text-xs font-mono bg-muted border-border">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border">
                    {["1", "2", "4"].map((v) => (
                      <SelectItem key={v} value={v} className="text-xs font-mono">{v}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Storage */}
              <div className="space-y-1 lg:col-span-1">
                <Label className="text-xs text-muted-foreground">Storage</Label>
                <Input
                  type="text"
                  placeholder="sqlite:///hpo.db (optional — for resume)"
                  value={storage}
                  onChange={(e) => setStorage(e.target.value)}
                  className="h-8 text-xs font-mono bg-muted border-border placeholder:text-muted-foreground/40"
                />
              </div>

              {/* Actions */}
              <div className="space-y-1 flex flex-col justify-end gap-2">
                <div className="flex gap-2 mt-auto">
                  <Button
                    className="flex-1 h-8 text-xs bg-primary hover:bg-primary/90 text-primary-foreground"
                    onClick={handleRun}
                    disabled={isRunning || runPending}
                    data-testid="btn-run-hpo"
                  >
                    <Play size={11} className="mr-1" />
                    Run HPO
                  </Button>
                  {isRunning && (
                    <Button
                      variant="destructive"
                      size="sm"
                      className="h-8 text-xs px-3"
                      onClick={handleStop}
                      disabled={stopPending}
                      data-testid="btn-stop-hpo"
                    >
                      <Square size={11} className="mr-1" />
                      Stop
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* ── Main body ──────────────────────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
          {/* Left 60% — Optimisation Progress */}
          <div className="lg:col-span-3 space-y-4">
            {/* Study overview card */}
            <Card className="bg-card border-border">
              <CardHeader className="py-3 px-4 border-b border-border">
                <CardTitle className="text-sm font-sans font-medium text-foreground flex items-center gap-2">
                  <TrendingUp size={13} /> Optimisation Progress
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4 space-y-4">
                {statusEngineOffline ? (
                  <p className="text-xs text-destructive font-sans text-center py-6">
                    Engine offline — status unavailable.
                  </p>
                ) : statusData ? (
                  <>
                    {/* Study meta */}
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                      <div>
                        <p className="text-xs text-muted-foreground mb-0.5">Study</p>
                        <p className="text-xs font-mono text-foreground truncate">{statusData.study_name}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground mb-0.5">Algorithm</p>
                        <p className="text-xs font-mono text-foreground">{algo}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground mb-0.5">Best Value</p>
                        <p className="text-xs font-mono text-primary font-semibold">
                          {statusData.best_value !== null ? statusData.best_value.toFixed(4) : "—"}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground mb-0.5">Completed</p>
                        <p className="text-xs font-mono text-green-400">{statusData.n_completed}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground mb-0.5">Pruned</p>
                        <p className="text-xs font-mono text-amber-400">{statusData.n_pruned}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground mb-0.5">Failed</p>
                        <p className="text-xs font-mono text-destructive">{statusData.n_failed}</p>
                      </div>
                    </div>

                    {/* Progress bar */}
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground font-sans">
                          {statusData.n_completed} / {statusData.n_trials_total} trials
                        </span>
                        <span className="font-mono text-foreground">{progressPct}%</span>
                      </div>
                      <Progress value={progressPct} className="h-2" />
                    </div>

                    {/* ETA */}
                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground font-mono">
                      <Clock size={11} />
                      ETA:{" "}
                      <span className="text-foreground">
                        {statusData.eta_secs !== null ? formatEta(statusData.eta_secs) : "—"}
                      </span>
                      <span className="ml-2">
                        Status:{" "}
                        <Badge
                          className={cn(
                            "text-xs px-1.5 py-0 font-mono border",
                            statusData.status === "running"
                              ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                              : statusData.status === "done"
                              ? "bg-green-500/10 text-green-400 border-green-500/30"
                              : "bg-destructive/10 text-destructive border-destructive/30"
                          )}
                        >
                          {statusData.status}
                        </Badge>
                      </span>
                    </div>
                  </>
                ) : (
                  <p className="text-xs text-muted-foreground font-sans text-center py-6">
                    No active study. Configure and click <span className="text-primary">Run HPO</span>.
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Trial history chart */}
            <Card className="bg-card border-border">
              <CardHeader className="py-3 px-4 border-b border-border">
                <CardTitle className="text-sm font-sans font-medium text-foreground">
                  Trial Objective Values
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4">
                {trials.length > 0 ? (
                  <ResponsiveContainer width="100%" height={280}>
                    <ScatterChart margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(222 35% 18%)" vertical={false} />
                      <XAxis
                        dataKey="number"
                        name="Trial"
                        type="number"
                        tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }}
                        tickLine={false}
                        axisLine={false}
                        label={{ value: "Trial #", position: "insideBottom", offset: -4, fill: "#64748b", fontSize: 10 }}
                      />
                      <YAxis
                        dataKey="value"
                        name="Value"
                        tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }}
                        tickLine={false}
                        axisLine={false}
                        width={50}
                        tickFormatter={(v) => v.toFixed(2)}
                      />
                      <Tooltip content={<TrialTooltip />} />
                      <Scatter data={chartData} isAnimationActive={false}>
                        {chartData.map((entry, idx) => (
                          <Cell
                            key={idx}
                            fill={entry.isBest ? "#eab308" : statusDot(entry.status)}
                            opacity={entry.isBest ? 1 : 0.75}
                            r={entry.isBest ? 8 : 5}
                          />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-[280px] flex items-center justify-center text-xs text-muted-foreground font-sans">
                    No trials yet
                  </div>
                )}

                {/* Running best overlay */}
                {runningBestData.filter((d) => d.best !== null).length > 1 && (
                  <div className="mt-1">
                    <p className="text-xs text-muted-foreground mb-1">Running Best</p>
                    <ResponsiveContainer width="100%" height={80}>
                      <LineChart
                        data={runningBestData.filter((d) => d.best !== null)}
                        margin={{ top: 4, right: 8, left: 0, bottom: 0 }}
                      >
                        <XAxis dataKey="number" hide />
                        <YAxis
                          tick={{ fill: "#64748b", fontSize: 9, fontFamily: "JetBrains Mono" }}
                          tickLine={false}
                          axisLine={false}
                          width={40}
                          tickFormatter={(v) => v.toFixed(2)}
                        />
                        <Line
                          type="stepAfter"
                          dataKey="best"
                          stroke="#eab308"
                          strokeWidth={2}
                          dot={false}
                          isAnimationActive={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Legend */}
                <div className="flex items-center gap-4 mt-2 text-xs font-mono">
                  {[
                    { label: "Completed", color: "#22c55e" },
                    { label: "Pruned", color: "#f59e0b" },
                    { label: "Failed", color: "#ef4444" },
                    { label: "Best", color: "#eab308" },
                  ].map((l) => (
                    <span key={l.label} className="flex items-center gap-1 text-muted-foreground">
                      <span
                        className="inline-block w-2.5 h-2.5 rounded-full"
                        style={{ backgroundColor: l.color }}
                      />
                      {l.label}
                    </span>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right 40% — Best Parameters */}
          <div className="lg:col-span-2">
            <Card className="bg-card border-border h-full">
              <CardHeader className="py-3 px-4 border-b border-border">
                <CardTitle className="text-sm font-sans font-medium text-foreground flex items-center gap-2">
                  <Trophy size={13} className="text-yellow-500" /> Best Parameters
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4 space-y-4">
                {anyEngineOffline ? (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <AlertCircle size={28} className="text-destructive/40 mb-3" />
                    <p className="text-xs text-destructive font-sans">
                      Best parameters not available — engine offline
                    </p>
                  </div>
                ) : bestTrial ? (
                  <>
                    <div className="rounded-md border border-yellow-500/30 bg-yellow-500/5 p-3 space-y-0.5">
                      <ParamRow name="learning_rate" value={bestTrial.params.learning_rate.toExponential(2)} />
                      <ParamRow name="n_steps" value={bestTrial.params.n_steps} />
                      <ParamRow name="batch_size" value={bestTrial.params.batch_size} />
                      <ParamRow name="gamma" value={bestTrial.params.gamma.toFixed(4)} />
                      <ParamRow name="net_arch" value={bestTrial.params.net_arch} />
                    </div>

                    <div className="rounded-md border border-border bg-muted/30 p-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-muted-foreground">Best Trial #</span>
                        <span className="text-xs font-mono text-foreground">{bestTrial.number}</span>
                      </div>
                      <div className="flex items-center justify-between mt-1">
                        <span className="text-xs text-muted-foreground">Objective Value</span>
                        <span className="text-xs font-mono text-primary font-semibold">
                          {bestTrial.value.toFixed(4)}
                        </span>
                      </div>
                      <div className="flex items-center justify-between mt-1">
                        <span className="text-xs text-muted-foreground">Duration</span>
                        <span className="text-xs font-mono text-foreground">
                          {formatDuration(bestTrial.duration_secs)}
                        </span>
                      </div>
                    </div>

                    <Separator className="bg-border" />

                    <div className="space-y-2">
                      <Button
                        className="w-full h-8 text-xs bg-primary hover:bg-primary/90 text-primary-foreground"
                        onClick={handleApplyToTraining}
                        data-testid="btn-apply-params"
                      >
                        <Zap size={11} className="mr-1.5" />
                        Apply to Training
                      </Button>
                      <Button
                        variant="outline"
                        className="w-full h-8 text-xs border-border text-foreground hover:bg-accent"
                        onClick={handleExportJson}
                        data-testid="btn-export-json"
                      >
                        <Download size={11} className="mr-1.5" />
                        Export JSON
                      </Button>
                    </div>
                  </>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <Trophy size={28} className="text-muted-foreground/30 mb-3" />
                    <p className="text-xs text-muted-foreground font-sans">
                      Best parameters will appear here once trials complete.
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* ── Trial History Table ─────────────────────────────────────────── */}
        <Card className="bg-card border-border">
          <CardHeader className="py-3 px-4 border-b border-border">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-sans font-medium text-foreground">
                Trial History
                <span className="ml-2 text-xs font-mono text-muted-foreground">
                  ({trials.length} trials)
                </span>
              </CardTitle>
              {trials.length > 0 && (
                <span className="text-xs text-muted-foreground font-mono">
                  Page {page + 1} / {Math.max(1, totalPages)}
                </span>
              )}
            </div>
          </CardHeader>
          <CardContent className="p-0">
            {trialsEngineOffline ? (
              <div className="flex items-center justify-center py-16 text-xs text-muted-foreground font-sans">
                No trial data — engine offline
              </div>
            ) : trials.length > 0 ? (
              <>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow className="border-border hover:bg-transparent">
                        {(
                          [
                            { key: "number", label: "Trial #" },
                            { key: "status", label: "Status" },
                            { key: "value", label: "Value" },
                          ] as { key: SortKey; label: string }[]
                        ).map(({ key, label }) => (
                          <TableHead
                            key={key}
                            className="text-xs text-muted-foreground font-sans cursor-pointer select-none hover:text-foreground transition-colors py-2 px-3"
                            onClick={() => handleSort(key)}
                          >
                            <span className="flex items-center gap-1">
                              {label}
                              {sortKey === key ? (
                                sortDir === "asc" ? (
                                  <ChevronUp size={11} />
                                ) : (
                                  <ChevronDown size={11} />
                                )
                              ) : null}
                            </span>
                          </TableHead>
                        ))}
                        <TableHead className="text-xs text-muted-foreground font-sans py-2 px-3">LR</TableHead>
                        <TableHead className="text-xs text-muted-foreground font-sans py-2 px-3">N Steps</TableHead>
                        <TableHead className="text-xs text-muted-foreground font-sans py-2 px-3">Batch</TableHead>
                        <TableHead className="text-xs text-muted-foreground font-sans py-2 px-3">Gamma</TableHead>
                        <TableHead className="text-xs text-muted-foreground font-sans py-2 px-3">Net Arch</TableHead>
                        <TableHead
                          className="text-xs text-muted-foreground font-sans cursor-pointer select-none hover:text-foreground transition-colors py-2 px-3"
                          onClick={() => handleSort("duration_secs")}
                        >
                          <span className="flex items-center gap-1">
                            Duration
                            {sortKey === "duration_secs" ? (
                              sortDir === "asc" ? <ChevronUp size={11} /> : <ChevronDown size={11} />
                            ) : null}
                          </span>
                        </TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {pagedTrials.map((trial) => {
                        const isBest = bestTrial?.number === trial.number;
                        return (
                          <TableRow
                            key={trial.number}
                            className={cn(
                              "border-border/50 transition-colors hover:bg-accent/20 text-xs",
                              isBest
                                ? "border-yellow-500/50 bg-yellow-500/5 hover:bg-yellow-500/10"
                                : ""
                            )}
                            data-testid={`trial-row-${trial.number}`}
                          >
                            <TableCell className="px-3 py-2 font-mono text-foreground">
                              {isBest && <span className="mr-1 text-yellow-400">★</span>}
                              {trial.number}
                            </TableCell>
                            <TableCell className="px-3 py-2">
                              <span
                                className={cn(
                                  "px-1.5 py-0.5 rounded text-xs font-mono font-medium border",
                                  statusBadgeVariant(trial.status)
                                )}
                              >
                                {trial.status}
                              </span>
                            </TableCell>
                            <TableCell className="px-3 py-2 font-mono text-foreground">
                              {trial.value.toFixed(4)}
                            </TableCell>
                            <TableCell className="px-3 py-2 font-mono text-muted-foreground">
                              {trial.params.learning_rate.toExponential(2)}
                            </TableCell>
                            <TableCell className="px-3 py-2 font-mono text-muted-foreground">
                              {trial.params.n_steps}
                            </TableCell>
                            <TableCell className="px-3 py-2 font-mono text-muted-foreground">
                              {trial.params.batch_size}
                            </TableCell>
                            <TableCell className="px-3 py-2 font-mono text-muted-foreground">
                              {trial.params.gamma.toFixed(4)}
                            </TableCell>
                            <TableCell className="px-3 py-2 font-mono text-muted-foreground">
                              {trial.params.net_arch}
                            </TableCell>
                            <TableCell className="px-3 py-2 font-mono text-muted-foreground">
                              {formatDuration(trial.duration_secs)}
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>

                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="flex items-center justify-center gap-2 py-3 border-t border-border">
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 text-xs border-border"
                      onClick={() => setPage((p) => Math.max(0, p - 1))}
                      disabled={page === 0}
                    >
                      Prev
                    </Button>
                    <span className="text-xs text-muted-foreground font-mono">
                      {page + 1} / {totalPages}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 text-xs border-border"
                      onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                      disabled={page >= totalPages - 1}
                    >
                      Next
                    </Button>
                  </div>
                )}
              </>
            ) : (
              <div className="flex items-center justify-center py-16 text-xs text-muted-foreground font-sans">
                No trial data yet. Run HPO to populate this table.
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
