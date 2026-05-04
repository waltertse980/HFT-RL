import { useState, useEffect, useCallback, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
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
import { apiRequest } from "@/lib/queryClient";
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
} from "lucide-react";

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

interface StudySummary {
  study_id: string;
  study_name: string;
  n_trials: number;
  best_value: number | null;
  status: string;
  created_at: string;
}

// ─── Mock data ─────────────────────────────────────────────────────────────────

function mockTrials(n = 30): HpoTrial[] {
  return Array.from({ length: n }, (_, i) => ({
    number: i,
    status: (i < n - 3 ? "complete" : (["pruned", "complete", "failed"] as const)[i % 3]),
    value: 0.3 + Math.random() * 1.2 - (i < 5 ? 0.3 : 0),
    params: {
      learning_rate: parseFloat((1e-5 + Math.random() * 9e-4).toExponential(2)),
      n_steps: ([512, 1024, 2048, 4096] as const)[i % 4],
      batch_size: ([64, 128, 256, 512] as const)[i % 4],
      gamma: 0.95 + Math.random() * 0.049,
      net_arch: (["small", "medium", "large"] as const)[i % 3],
    },
    duration_secs: 30 + Math.random() * 120,
  }));
}

function mockStatus(studyId: string, trials: HpoTrial[], totalTrials: number): HpoStatus {
  const completed = trials.filter((t) => t.status === "complete");
  const bestTrial = completed.reduce<HpoTrial | null>(
    (best, t) => (best === null || t.value > best.value ? t : best),
    null
  );
  return {
    study_id: studyId,
    study_name: `optuna_study_${studyId.slice(0, 6)}`,
    n_trials_total: totalTrials,
    n_completed: completed.length,
    n_pruned: trials.filter((t) => t.status === "pruned").length,
    n_failed: trials.filter((t) => t.status === "failed").length,
    best_value: bestTrial?.value ?? null,
    best_params: bestTrial?.params ?? null,
    status: trials.length >= totalTrials ? "done" : "running",
    eta_secs:
      trials.length >= totalTrials
        ? 0
        : Math.max(0, (totalTrials - trials.length) * 15),
  };
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

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
  const qc = useQueryClient();

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
  const [mockTrialsList, setMockTrialsList] = useState<HpoTrial[]>([]);
  const [usingMock, setUsingMock] = useState(false);

  // ── Sort / pagination state ───────────────────────────────────────────────
  const [sortKey, setSortKey] = useState<SortKey>("number");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 10;

  // ── API: run HPO ──────────────────────────────────────────────────────────
  const runMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/hpo/run", {
        market,
        timescale,
        algo,
        n_trials: nTrials,
        timesteps_per_trial: timestepsPerTrial,
        n_jobs: parseInt(nJobs),
        storage: storage || undefined,
      });
      return res.json() as Promise<{ study_id: string; status: string }>;
    },
    onSuccess: (data) => {
      setActiveStudyId(data.study_id);
      setUsingMock(false);
      setMockTrialsList([]);
      setPage(0);
    },
    onError: () => {
      // Fall back to mock
      const id = `mock_${Date.now()}`;
      setActiveStudyId(id);
      setUsingMock(true);
      setMockTrialsList([]);
      setPage(0);
    },
  });

  // ── API: stop HPO ─────────────────────────────────────────────────────────
  const stopMutation = useMutation({
    mutationFn: async () => {
      if (!activeStudyId || usingMock) return;
      const res = await apiRequest("POST", `/api/hpo/stop/${activeStudyId}`);
      return res.json();
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["/api/hpo/status", activeStudyId] });
    },
    onSettled: () => {
      if (usingMock) {
        setMockTrialsList((prev) => {
          const t = mockTrials(nTrials);
          return t.slice(0, Math.max(prev.length, 1));
        });
      }
    },
  });

  // ── API: poll status ──────────────────────────────────────────────────────
  const { data: statusData } = useQuery<HpoStatus>({
    queryKey: ["/api/hpo/status", activeStudyId],
    enabled: !!activeStudyId && !usingMock,
    refetchInterval: (query) => {
      const data = query.state.data as HpoStatus | undefined;
      return data?.status === "running" ? 2000 : false;
    },
    retry: false,
  });

  // ── API: fetch trials ─────────────────────────────────────────────────────
  const { data: trialsData } = useQuery<{ trials: HpoTrial[] }>({
    queryKey: ["/api/hpo/trials", activeStudyId],
    enabled: !!activeStudyId && !usingMock,
    refetchInterval: (query) => {
      const status = qc.getQueryData<HpoStatus>(["/api/hpo/status", activeStudyId]);
      return status?.status === "running" ? 3000 : false;
    },
    retry: false,
  });

  // ── Mock simulation ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!usingMock || !activeStudyId) return;
    const fullSet = mockTrials(nTrials);
    let current = 0;
    const addTrial = () => {
      current++;
      setMockTrialsList(fullSet.slice(0, current));
    };
    // Add one immediately then every 400ms
    addTrial();
    const iv = setInterval(() => {
      if (current >= fullSet.length) {
        clearInterval(iv);
        return;
      }
      addTrial();
    }, 400);
    return () => clearInterval(iv);
  }, [usingMock, activeStudyId, nTrials]);

  // ── Resolved data ─────────────────────────────────────────────────────────
  const trials: HpoTrial[] = usingMock
    ? mockTrialsList
    : trialsData?.trials ?? [];

  const status: HpoStatus | null = usingMock && activeStudyId
    ? mockStatus(activeStudyId, mockTrialsList, nTrials)
    : statusData ?? null;

  const isRunning =
    (usingMock && activeStudyId !== null && mockTrialsList.length < nTrials) ||
    (!usingMock && status?.status === "running");

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
  const progressPct = status
    ? Math.round((status.n_completed / Math.max(status.n_trials_total, 1)) * 100)
    : 0;

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full overflow-hidden" data-testid="page-hpo">
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
          {status?.status === "done" && (
            <span className="flex items-center gap-1 text-green-400 text-xs font-mono">
              <Trophy size={10} /> Done
            </span>
          )}
        </h1>
        <span className="text-xs text-muted-foreground font-mono">
          {usingMock && activeStudyId ? "mock mode" : activeStudyId ? `study: ${status?.study_name ?? activeStudyId}` : "no active study"}
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
                    onClick={() => runMutation.mutate()}
                    disabled={isRunning || runMutation.isPending}
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
                      onClick={() => stopMutation.mutate()}
                      disabled={stopMutation.isPending}
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
                {status ? (
                  <>
                    {/* Study meta */}
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                      <div>
                        <p className="text-xs text-muted-foreground mb-0.5">Study</p>
                        <p className="text-xs font-mono text-foreground truncate">{status.study_name}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground mb-0.5">Algorithm</p>
                        <p className="text-xs font-mono text-foreground">{algo}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground mb-0.5">Best Value</p>
                        <p className="text-xs font-mono text-primary font-semibold">
                          {status.best_value !== null ? status.best_value.toFixed(4) : "—"}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground mb-0.5">Completed</p>
                        <p className="text-xs font-mono text-green-400">{status.n_completed}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground mb-0.5">Pruned</p>
                        <p className="text-xs font-mono text-amber-400">{status.n_pruned}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground mb-0.5">Failed</p>
                        <p className="text-xs font-mono text-destructive">{status.n_failed}</p>
                      </div>
                    </div>

                    {/* Progress bar */}
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground font-sans">
                          {status.n_completed} / {status.n_trials_total} trials
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
                        {status.eta_secs !== null ? formatEta(status.eta_secs) : "—"}
                      </span>
                      <span className="ml-2">
                        Status:{" "}
                        <Badge
                          className={cn(
                            "text-xs px-1.5 py-0 font-mono border",
                            status.status === "running"
                              ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                              : status.status === "done"
                              ? "bg-green-500/10 text-green-400 border-green-500/30"
                              : "bg-destructive/10 text-destructive border-destructive/30"
                          )}
                        >
                          {status.status}
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
                {bestTrial ? (
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
            {trials.length > 0 ? (
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
