import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { Play, Square, Download, Trash2, BarChart, Loader2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";
import { apiRequest } from "@/lib/queryClient";

// ---- Local Types ----
interface TrainingJobRecord {
  id: number;
  jobId: string;
  market: string;
  timescale: string;
  algo: string;
  totalTimesteps: number;
  status: string; // pending|running|done|error|stopped
  progressPct: number;
  currentReward: number | null;
  modelPath: string | null;
  errorMsg: string | null;
  startedAt: string | null;
  completedAt: string | null;
  rewardHistory?: { step: number; reward: number }[];
}

interface ModelsResponse {
  models: string[];
}

// ---- Constants ----
const TIMESCALES = ["10s", "1m", "5m", "1h"];
const ALGOS = ["PPO", "TD3"];
const MARKETS = ["US", "HK", "Both"];
const JOB_COLORS = ["#3b82f6", "#f97316", "#a855f7", "#22c55e", "#ef4444", "#06b6d4"];

// ---- Helpers ----
function formatTimesteps(steps: number): string {
  if (steps >= 1_000_000) return `${(steps / 1_000_000).toFixed(1)}M`;
  return `${(steps / 1_000).toFixed(0)}K`;
}

function logSlider(value: number): number {
  const minLog = Math.log10(100_000);
  const maxLog = Math.log10(5_000_000);
  return Math.round(Math.pow(10, minLog + (value / 100) * (maxLog - minLog)));
}

function sliderValue(steps: number): number {
  const minLog = Math.log10(100_000);
  const maxLog = Math.log10(5_000_000);
  return Math.round(((Math.log10(steps) - minLog) / (maxLog - minLog)) * 100);
}

// Parse algo/market/timescale from filename like "ppo_us_1m_abc123.zip"
function parseModelFilename(path: string): { algo: string; market: string; timescale: string } {
  const filename = path.split("/").pop() ?? path;
  const parts = filename.replace(/\.zip$/i, "").split("_");
  return {
    algo: (parts[0] ?? "?").toUpperCase(),
    market: (parts[1] ?? "?").toUpperCase(),
    timescale: parts[2] ?? "?",
  };
}

function isEngineOfflineError(error: unknown): boolean {
  if (!error) return false;
  const msg = String(error);
  return msg.includes("503") || msg.toLowerCase().includes("engine_offline");
}

// ---- Sub-components ----

function AlgoBadge({ algo }: { algo: string }) {
  const colors: Record<string, string> = {
    PPO: "bg-primary/10 text-primary border-primary/30",
    TD3: "bg-purple-500/10 text-purple-400 border-purple-500/30",
  };
  return (
    <span className={cn("px-2 py-0.5 rounded text-xs font-mono font-semibold border", colors[algo] ?? "bg-muted text-muted-foreground border-border")}>
      {algo}
    </span>
  );
}

function StatusBadge({ status }: { status: string }) {
  const config: Record<string, { color: string; label: string }> = {
    running: { color: "bg-success/10 text-success border-success/30", label: "RUNNING" },
    done: { color: "bg-muted text-muted-foreground border-border", label: "DONE" },
    error: { color: "bg-destructive/10 text-destructive border-destructive/30", label: "ERROR" },
    pending: { color: "bg-warning/10 text-warning border-warning/30", label: "PENDING" },
    stopped: { color: "bg-muted text-muted-foreground border-border", label: "STOPPED" },
  };
  const cfg = config[status] ?? config.pending;
  return (
    <span className={cn("px-2 py-0.5 rounded text-xs font-mono font-semibold border", cfg.color, status === "running" && "animate-pulse")}>
      {cfg.label}
    </span>
  );
}

// ---- Engine Offline Banner ----
function EngineOfflineBanner() {
  return (
    <div className="bg-destructive/10 border border-destructive/40 rounded-md px-4 py-3 mb-4 flex flex-col gap-1" data-testid="engine-offline-banner">
      <p className="text-sm font-semibold text-destructive">⚠ Python engine offline — Start it first:</p>
      <code className="text-xs font-mono text-destructive/80 bg-destructive/5 rounded px-2 py-1 mt-1">
        cd python_engine &amp;&amp; uvicorn api_server:app --port 8001 --reload
      </code>
    </div>
  );
}

// ---- Job Card ----
interface JobCardProps {
  job: TrainingJobRecord;
  engineOffline: boolean;
  onStop: (jobId: string) => void;
  isStopping: boolean;
  evalModelPath: (modelPath: string, market: string, timescale: string) => void;
  isEvaling: boolean;
  exportOnnx: (modelPath: string, algo: string) => void;
  isExporting: boolean;
  wentOffline?: boolean;
}

function JobCard({
  job,
  engineOffline,
  onStop,
  isStopping,
  evalModelPath,
  isEvaling,
  exportOnnx,
  isExporting,
  wentOffline,
}: JobCardProps) {
  const doneSteps = job.totalTimesteps > 0
    ? Math.floor((job.progressPct / 100) * job.totalTimesteps)
    : 0;

  return (
    <Card className="bg-card border-border" data-testid={`job-card-${job.jobId}`}>
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div className="space-y-1">
            <div className="flex items-center gap-2 flex-wrap">
              <AlgoBadge algo={job.algo} />
              <span className="text-xs font-mono text-muted-foreground">{job.market} · {job.timescale}</span>
              <StatusBadge status={job.status} />
            </div>
            <p className="text-xs text-muted-foreground font-mono">
              {formatTimesteps(doneSteps)} / {formatTimesteps(job.totalTimesteps)} steps
            </p>
            <p className="text-xs text-muted-foreground font-mono opacity-60">ID: {job.jobId}</p>
          </div>
          {(job.status === "running" || job.status === "pending") && (
            <Button
              size="sm"
              variant="outline"
              className="text-xs border-border h-7"
              onClick={() => onStop(job.jobId)}
              disabled={engineOffline || isStopping}
              data-testid={`btn-stop-job-${job.jobId}`}
            >
              {isStopping ? <Loader2 size={10} className="mr-1 animate-spin" /> : <Square size={10} className="mr-1" />}
              Stop
            </Button>
          )}
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-muted-foreground font-sans">Progress</span>
            <span className="font-mono text-foreground">{job.progressPct.toFixed(1)}%</span>
          </div>
          <Progress value={job.progressPct} className="h-1.5" />
        </div>

        <div className="grid grid-cols-2 gap-4 mt-3">
          <div>
            <p className="text-xs text-muted-foreground font-sans">Mean Reward</p>
            {job.currentReward !== null ? (
              <p className={cn("font-mono text-sm font-semibold", job.currentReward > 0 ? "text-success" : "text-destructive")}>
                {job.currentReward.toFixed(3)}
              </p>
            ) : (
              <p className="font-mono text-sm text-muted-foreground">—</p>
            )}
          </div>
          <div>
            <p className="text-xs text-muted-foreground font-sans">Status</p>
            <p className="font-mono text-sm text-foreground capitalize">{job.status}</p>
          </div>
        </div>

        {/* Done: model path + eval/onnx */}
        {job.status === "done" && job.modelPath && (
          <div className="mt-3 space-y-2">
            <p className="text-xs font-mono text-muted-foreground break-all">{job.modelPath}</p>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                className="h-6 text-xs px-2 border-border"
                disabled={engineOffline || isEvaling}
                onClick={() => evalModelPath(job.modelPath!, job.market, job.timescale)}
                data-testid={`btn-eval-job-${job.jobId}`}
              >
                {isEvaling ? <Loader2 size={10} className="mr-1 animate-spin" /> : <BarChart size={10} className="mr-1" />}
                Eval
              </Button>
              <Button
                size="sm"
                variant="outline"
                className="h-6 text-xs px-2 border-border"
                disabled={engineOffline || isExporting}
                onClick={() => exportOnnx(job.modelPath!, job.algo)}
                data-testid={`btn-onnx-job-${job.jobId}`}
              >
                {isExporting ? <Loader2 size={10} className="mr-1 animate-spin" /> : <Download size={10} className="mr-1" />}
                ONNX
              </Button>
            </div>
          </div>
        )}

        {/* Error message */}
        {job.status === "error" && job.errorMsg && (
          <p className="mt-2 text-xs text-destructive font-mono break-all">{job.errorMsg}</p>
        )}

        {/* Engine went offline warning */}
        {wentOffline && (
          <p className="mt-2 text-xs text-destructive font-mono">⚠ Engine went offline — polling stopped.</p>
        )}
      </CardContent>
    </Card>
  );
}

// ---- Reward Curves ----
interface RewardCurvesProps {
  jobs: TrainingJobRecord[];
}

function RewardCurves({ jobs }: RewardCurvesProps) {
  const jobsWithHistory = jobs.filter((j) => j.rewardHistory && j.rewardHistory.length > 0);

  if (jobsWithHistory.length === 0) {
    return (
      <Card className="bg-card border-border" data-testid="reward-curves-chart">
        <CardHeader className="py-3 px-4 border-b border-border">
          <CardTitle className="text-sm font-sans font-medium">Reward Curves</CardTitle>
        </CardHeader>
        <CardContent className="p-6 flex items-center justify-center h-40">
          <p className="text-xs text-muted-foreground font-mono">Waiting for training to start...</p>
        </CardContent>
      </Card>
    );
  }

  // Merge reward histories by step
  const merged: Record<number, Record<string, number>> = {};
  jobsWithHistory.forEach((job) => {
    (job.rewardHistory ?? []).forEach(({ step, reward }) => {
      if (!merged[step]) merged[step] = { step };
      merged[step][`${job.market}-${job.algo}-${job.jobId}`] = reward;
    });
  });
  const data = Object.values(merged).sort((a, b) => (a.step as number) - (b.step as number));

  return (
    <Card className="bg-card border-border" data-testid="reward-curves-chart">
      <CardHeader className="py-3 px-4 border-b border-border">
        <CardTitle className="text-sm font-sans font-medium">Reward Curves</CardTitle>
      </CardHeader>
      <CardContent className="p-3">
        <div className="h-52">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(222 35% 18%)" vertical={false} />
              <XAxis
                dataKey="step"
                tick={{ fill: "#64748b", fontSize: 10, fontFamily: "'JetBrains Mono'" }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v) => formatTimesteps(v)}
              />
              <YAxis
                tick={{ fill: "#64748b", fontSize: 10, fontFamily: "'JetBrains Mono'" }}
                tickLine={false}
                axisLine={false}
                domain={["auto", "auto"]}
                width={40}
              />
              <Tooltip
                contentStyle={{
                  background: "hsl(222 44% 12%)",
                  border: "1px solid hsl(222 35% 20%)",
                  borderRadius: "6px",
                  fontSize: "11px",
                  fontFamily: "'JetBrains Mono'",
                }}
                labelStyle={{ color: "#64748b" }}
              />
              <Legend wrapperStyle={{ fontSize: "11px", fontFamily: "'JetBrains Mono'" }} />
              {jobsWithHistory.map((job, idx) => (
                <Line
                  key={job.jobId}
                  type="monotone"
                  dataKey={`${job.market}-${job.algo}-${job.jobId}`}
                  stroke={JOB_COLORS[idx % JOB_COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                  name={`${job.market} ${job.algo}`}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

// ---- Checkpoints Table ----
interface CheckpointsTableProps {
  models: string[];
  engineOffline: boolean;
  onEval: (modelName: string, market: string, timescale: string) => void;
  isEvaling: boolean;
  onExport: (modelPath: string, algo: string) => void;
  isExporting: boolean;
  onDelete: (modelName: string) => void;
  isDeleting: boolean;
}

function CheckpointsTable({
  models,
  engineOffline,
  onEval,
  isEvaling,
  onExport,
  isExporting,
  onDelete,
  isDeleting,
}: CheckpointsTableProps) {
  const td = "px-3 py-2 text-xs";

  return (
    <Card className="bg-card border-border" data-testid="checkpoints-table">
      <CardHeader className="py-3 px-4 border-b border-border">
        <CardTitle className="text-sm font-sans font-medium">Model Checkpoints</CardTitle>
      </CardHeader>
      <div className="overflow-x-auto">
        {models.length === 0 ? (
          <p className="px-4 py-6 text-xs text-muted-foreground font-mono text-center">No model checkpoints found.</p>
        ) : (
          <table className="w-full">
            <thead className="border-b border-border">
              <tr>
                <th className="px-3 py-2 text-xs text-muted-foreground font-sans font-medium text-left">Model Path</th>
                <th className="px-3 py-2 text-xs text-muted-foreground font-sans font-medium text-left">Algorithm</th>
                <th className="px-3 py-2 text-xs text-muted-foreground font-sans font-medium text-left">Market</th>
                <th className="px-3 py-2 text-xs text-muted-foreground font-sans font-medium text-left">Timescale</th>
                <th className="px-3 py-2 text-xs text-muted-foreground font-sans font-medium text-left">Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((modelPath) => {
                const filename = modelPath.split("/").pop() ?? modelPath;
                const { algo, market, timescale } = parseModelFilename(filename);
                return (
                  <tr
                    key={modelPath}
                    className="border-b border-border/50 hover:bg-accent/20 transition-colors"
                    data-testid={`ckpt-row-${filename}`}
                  >
                    <td className={cn(td, "font-mono text-foreground max-w-xs truncate")} title={modelPath}>
                      {filename}
                    </td>
                    <td className={td}><AlgoBadge algo={algo} /></td>
                    <td className={cn(td, "font-mono text-muted-foreground")}>{market}</td>
                    <td className={cn(td, "font-mono text-muted-foreground")}>{timescale}</td>
                    <td className={td}>
                      <div className="flex items-center gap-1">
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-6 text-xs px-2 border-border"
                          disabled={engineOffline || isEvaling}
                          onClick={() => onEval(filename, market, timescale)}
                          data-testid={`btn-evaluate-${filename}`}
                        >
                          {isEvaling ? <Loader2 size={10} className="mr-1 animate-spin" /> : <BarChart size={10} className="mr-1" />}
                          Eval
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-6 text-xs px-2 border-border"
                          disabled={engineOffline || isExporting}
                          onClick={() => onExport(modelPath, algo)}
                          data-testid={`btn-export-${filename}`}
                        >
                          {isExporting ? <Loader2 size={10} className="mr-1 animate-spin" /> : <Download size={10} className="mr-1" />}
                          ONNX
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-6 text-xs px-2 border-border text-destructive hover:text-destructive"
                          disabled={isDeleting}
                          onClick={() => onDelete(filename)}
                          data-testid={`btn-delete-${filename}`}
                        >
                          {isDeleting ? <Loader2 size={10} className="animate-spin" /> : <Trash2 size={10} />}
                        </Button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </Card>
  );
}

// ---- Main Page ----
export default function TrainingConsole() {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Config state
  const [market, setMarket] = useState("US");
  const [timescale, setTimescale] = useState("1m");
  const [algo, setAlgo] = useState("PPO");
  const [timesteps, setTimesteps] = useState(1_000_000);
  const [tickers, setTickers] = useState("");

  // Track which jobs went offline (engine dropped during poll)
  const [offlineJobIds, setOfflineJobIds] = useState<Set<string>>(new Set());

  // Track engine offline state (from any query/mutation)
  const [engineOffline, setEngineOffline] = useState(false);

  // ---- Load all jobs on mount ----
  const {
    data: jobsData,
    error: jobsError,
    isLoading: jobsLoading,
  } = useQuery<TrainingJobRecord[]>({
    queryKey: ["/api/training/jobs"],
    refetchInterval: false,
    retry: false,
  });

  // Derive jobs list
  const allJobs: TrainingJobRecord[] = jobsData ?? [];
  const runningJobs = allJobs.filter((j) => j.status === "running" || j.status === "pending");

  // Handle jobs load error for engine offline
  if (jobsError && isEngineOfflineError(jobsError) && !engineOffline) {
    setEngineOffline(true);
  }

  // ---- Poll status for each running job ----
  // We poll the list endpoint at interval to refresh all jobs
  const { data: refreshedJobs } = useQuery<TrainingJobRecord[]>({
    queryKey: ["/api/training/jobs", "poll"],
    queryFn: async () => {
      const res = await fetch("/api/training/jobs");
      if (res.status === 503) {
        setEngineOffline(true);
        // Mark all running jobs as offline
        runningJobs.forEach((j) => {
          setOfflineJobIds((prev) => { const next = new Set(prev); next.add(j.jobId); return next; });
        });
        throw new Error("503: ENGINE_OFFLINE");
      }
      if (!res.ok) throw new Error(`${res.status}: ${res.statusText}`);
      setEngineOffline(false);
      return res.json();
    },
    refetchInterval: runningJobs.length > 0 ? 3000 : false,
    enabled: allJobs.length > 0 || !!jobsData,
    retry: false,
  });

  // Merged job list: prefer refreshed data
  const displayJobs: TrainingJobRecord[] = refreshedJobs ?? allJobs;
  const displayRunningJobs = displayJobs.filter(
    (j) => j.status === "running" || j.status === "pending"
  );

  // Invalidate models when a job completes
  const doneJobs = displayJobs.filter((j) => j.status === "done");
  if (doneJobs.length > 0) {
    queryClient.invalidateQueries({ queryKey: ["/api/models"] });
  }

  // ---- Load models ----
  const {
    data: modelsData,
    error: modelsError,
  } = useQuery<ModelsResponse>({
    queryKey: ["/api/models"],
    refetchInterval: false,
    retry: false,
  });

  const modelList: string[] = modelsData?.models ?? [];

  if (modelsError && isEngineOfflineError(modelsError) && !engineOffline) {
    setEngineOffline(true);
  }

  // ---- Start Training ----
  const startMutation = useMutation({
    mutationFn: async (params: {
      market: string;
      timescale: string;
      algo: string;
      timesteps: number;
      tickers: string;
    }) => {
      const res = await fetch("/api/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });
      if (res.status === 503) {
        setEngineOffline(true);
        throw new Error("503: ENGINE_OFFLINE");
      }
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status}: ${text}`);
      }
      return res.json();
    },
    onSuccess: (data) => {
      setEngineOffline(false);
      toast({
        title: "Training started",
        description: `Job ID: ${data.job_id ?? data.jobId ?? JSON.stringify(data)}`,
      });
      queryClient.invalidateQueries({ queryKey: ["/api/training/jobs"] });
      queryClient.invalidateQueries({ queryKey: ["/api/training/jobs", "poll"] });
    },
    onError: (err) => {
      if (isEngineOfflineError(err)) {
        toast({
          title: "Python engine is offline. Start it first.",
          variant: "destructive",
        });
      } else {
        toast({
          title: "Failed to start training",
          description: String(err),
          variant: "destructive",
        });
      }
    },
  });

  // ---- Stop Job ----
  const stopMutation = useMutation({
    mutationFn: async (jobId: string) => {
      const res = await fetch(`/api/training/stop/${jobId}`, { method: "POST" });
      if (res.status === 503) {
        setEngineOffline(true);
        throw new Error("503: ENGINE_OFFLINE");
      }
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status}: ${text}`);
      }
      return res.json();
    },
    onSuccess: (_data, jobId) => {
      toast({ title: "Job stopped", description: `Job ${jobId} stopped.` });
      queryClient.invalidateQueries({ queryKey: ["/api/training/jobs"] });
      queryClient.invalidateQueries({ queryKey: ["/api/training/jobs", "poll"] });
    },
    onError: (err) => {
      toast({
        title: "Failed to stop job",
        description: String(err),
        variant: "destructive",
      });
    },
  });

  // ---- Evaluate Model ----
  const evalMutation = useMutation({
    mutationFn: async ({ modelName, market, timescale }: { modelName: string; market: string; timescale: string }) => {
      const res = await fetch(`/api/training/evaluate/${encodeURIComponent(modelName)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ market, timescale }),
      });
      if (res.status === 503) {
        setEngineOffline(true);
        throw new Error("503: ENGINE_OFFLINE");
      }
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status}: ${text}`);
      }
      return res.json();
    },
    onSuccess: (data) => {
      const sharpe = typeof data.sharpe === "number" ? data.sharpe.toFixed(2) : "N/A";
      const maxdd = typeof data.max_drawdown === "number" ? `${(data.max_drawdown * 100).toFixed(1)}%` : "N/A";
      const winrate = typeof data.win_rate === "number" ? `${(data.win_rate * 100).toFixed(0)}%` : "N/A";
      toast({
        title: "Evaluation complete",
        description: `Sharpe: ${sharpe}, MaxDD: ${maxdd}, WinRate: ${winrate}`,
      });
    },
    onError: (err) => {
      toast({
        title: "Evaluation failed",
        description: String(err),
        variant: "destructive",
      });
    },
  });

  // ---- Export ONNX ----
  const exportMutation = useMutation({
    mutationFn: async ({ modelPath, algorithm }: { modelPath: string; algorithm: string }) => {
      const res = await fetch("/api/training/export-onnx", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_path: modelPath, algorithm }),
      });
      if (res.status === 503) {
        setEngineOffline(true);
        throw new Error("503: ENGINE_OFFLINE");
      }
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status}: ${text}`);
      }
      return res.json();
    },
    onSuccess: (data) => {
      toast({
        title: "ONNX export complete",
        description: `Saved to: ${data.onnx_path ?? "unknown"}`,
      });
    },
    onError: (err) => {
      toast({
        title: "ONNX export failed",
        description: String(err),
        variant: "destructive",
      });
    },
  });

  // ---- Delete Model ----
  const deleteMutation = useMutation({
    mutationFn: async (modelName: string) => {
      const res = await fetch(`/api/models/${encodeURIComponent(modelName)}`, { method: "DELETE" });
      if (res.status === 503) {
        setEngineOffline(true);
        throw new Error("503: ENGINE_OFFLINE");
      }
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status}: ${text}`);
      }
      return res.json();
    },
    onSuccess: (_data, modelName) => {
      toast({ title: "Model deleted", description: modelName });
      queryClient.invalidateQueries({ queryKey: ["/api/models"] });
    },
    onError: (err) => {
      toast({
        title: "Delete failed",
        description: String(err),
        variant: "destructive",
      });
    },
  });

  // ---- Handlers ----
  const handleStartTraining = () => {
    startMutation.mutate({ market, timescale, algo, timesteps, tickers });
  };

  const handleStop = (jobId: string) => {
    stopMutation.mutate(jobId);
  };

  const handleEval = (modelName: string, mkt: string, ts: string) => {
    evalMutation.mutate({ modelName, market: mkt, timescale: ts });
  };

  const handleExport = (modelPath: string, algoName: string) => {
    exportMutation.mutate({ modelPath, algorithm: algoName });
  };

  const handleDelete = (modelName: string) => {
    deleteMutation.mutate(modelName);
  };

  const buttonsDisabled = engineOffline;

  return (
    <div className="flex flex-col h-full overflow-hidden" data-testid="page-training-console">
      {/* Header */}
      <div className="border-b border-border px-6 py-3 shrink-0">
        <h1 className="text-sm font-sans font-semibold text-foreground">Training Console</h1>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Engine Offline Banner */}
        {engineOffline && <EngineOfflineBanner />}

        {/* Config Card */}
        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex flex-wrap items-end gap-4">
              {/* Market */}
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground font-sans">Market</label>
                <Select value={market} onValueChange={setMarket}>
                  <SelectTrigger className="h-8 w-28 text-xs font-mono bg-muted border-border" data-testid="train-market-select">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border">
                    {MARKETS.map((m) => (
                      <SelectItem key={m} value={m} className="text-xs font-mono">{m}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Timescale */}
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground font-sans">Timescale</label>
                <Select value={timescale} onValueChange={setTimescale}>
                  <SelectTrigger className="h-8 w-24 text-xs font-mono bg-muted border-border" data-testid="train-timescale-select">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border">
                    {TIMESCALES.map((t) => (
                      <SelectItem key={t} value={t} className="text-xs font-mono">{t}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Algorithm */}
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground font-sans">Algorithm</label>
                <Select value={algo} onValueChange={setAlgo}>
                  <SelectTrigger className="h-8 w-24 text-xs font-mono bg-muted border-border" data-testid="train-algo-select">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border">
                    {ALGOS.map((a) => (
                      <SelectItem key={a} value={a} className="text-xs font-mono">{a}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Timesteps slider */}
              <div className="flex-1 min-w-48 space-y-1">
                <label className="text-xs text-muted-foreground font-sans flex items-center justify-between">
                  <span>Timesteps</span>
                  <span className="font-mono text-foreground">{formatTimesteps(timesteps)}</span>
                </label>
                <Slider
                  data-testid="timesteps-slider"
                  value={[sliderValue(timesteps)]}
                  onValueChange={([v]) => setTimesteps(logSlider(v))}
                  min={0}
                  max={100}
                  step={1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground font-mono">
                  <span>100K</span>
                  <span>5M</span>
                </div>
              </div>

              {/* Tickers */}
              <div className="space-y-1 min-w-40">
                <label className="text-xs text-muted-foreground font-sans">Tickers (optional)</label>
                <Input
                  className="h-8 text-xs font-mono bg-muted border-border placeholder:text-muted-foreground/40"
                  placeholder="AAPL, TSLA, ..."
                  value={tickers}
                  onChange={(e) => setTickers(e.target.value)}
                  data-testid="train-tickers-input"
                />
              </div>

              <Button
                data-testid="btn-start-training"
                className="bg-primary hover:bg-primary/90 text-primary-foreground text-xs h-8 px-4"
                onClick={handleStartTraining}
                disabled={buttonsDisabled || startMutation.isPending}
              >
                {startMutation.isPending
                  ? <><Loader2 size={12} className="mr-1 animate-spin" /> Starting...</>
                  : <><Play size={12} className="mr-1" /> Start Training</>
                }
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Active Jobs */}
        {jobsLoading ? (
          <p className="text-xs text-muted-foreground font-mono px-1">Loading jobs...</p>
        ) : displayJobs.length > 0 ? (
          <div>
            <h2 className="text-xs font-sans font-semibold text-muted-foreground uppercase tracking-wider mb-2">
              Active Jobs ({displayRunningJobs.length} running)
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
              {displayJobs.map((job) => (
                <JobCard
                  key={job.jobId}
                  job={job}
                  engineOffline={engineOffline}
                  onStop={handleStop}
                  isStopping={stopMutation.isPending && stopMutation.variables === job.jobId}
                  evalModelPath={handleEval}
                  isEvaling={evalMutation.isPending}
                  exportOnnx={handleExport}
                  isExporting={exportMutation.isPending}
                  wentOffline={offlineJobIds.has(job.jobId)}
                />
              ))}
            </div>
          </div>
        ) : !jobsError ? (
          <p className="text-xs text-muted-foreground font-mono px-1">No training jobs yet. Configure and start one above.</p>
        ) : null}

        {/* Reward Curves */}
        {displayJobs.length > 0 && (
          <RewardCurves jobs={displayRunningJobs} />
        )}

        {/* Model Checkpoints */}
        <CheckpointsTable
          models={modelList}
          engineOffline={engineOffline}
          onEval={handleEval}
          isEvaling={evalMutation.isPending}
          onExport={handleExport}
          isExporting={exportMutation.isPending}
          onDelete={handleDelete}
          isDeleting={deleteMutation.isPending}
        />
      </div>
    </div>
  );
}
