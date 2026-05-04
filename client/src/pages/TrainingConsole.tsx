import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
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
import { Play, Square, Download, Trash2, BarChart, ChevronUp, ChevronDown } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";
import {
  generateTrainingJobs,
  generateModelCheckpoints,
  type TrainingJob,
  type ModelCheckpoint,
} from "@/lib/mockData";

const TIMESCALES = ["10s", "1m", "5m", "1h"];
const ALGOS = ["PPO", "TD3", "Both"];
const MARKETS = ["US", "HK", "Both"];

function formatElapsed(secs: number) {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  return h > 0 ? `${h}h ${m}m` : `${m}m ${s}s`;
}

function formatTimesteps(steps: number) {
  if (steps >= 1000000) return `${(steps / 1000000).toFixed(1)}M`;
  return `${(steps / 1000).toFixed(0)}K`;
}

function AlgoBadge({ algo }: { algo: string }) {
  const colors: Record<string, string> = {
    PPO: "bg-primary/10 text-primary border-primary/30",
    TD3: "bg-purple-500/10 text-purple-400 border-purple-500/30",
    Both: "bg-warning/10 text-warning border-warning/30",
  };
  return (
    <span className={cn("px-2 py-0.5 rounded text-xs font-mono font-semibold border", colors[algo] ?? colors.PPO)}>
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
  };
  const cfg = config[status] ?? config.pending;
  return (
    <span className={cn("px-2 py-0.5 rounded text-xs font-mono font-semibold border animate-pulse", cfg.color)}>
      {cfg.label}
    </span>
  );
}

// ---- Job Card ----
function JobCard({ job }: { job: TrainingJob }) {
  return (
    <Card className="bg-card border-border" data-testid={`job-card-${job.id}`}>
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div className="space-y-1">
            <div className="flex items-center gap-2 flex-wrap">
              <AlgoBadge algo={job.algo} />
              <span className="text-xs font-mono text-muted-foreground">{job.market} · {job.timescale}</span>
              <StatusBadge status={job.status} />
            </div>
            <p className="text-xs text-muted-foreground font-mono">
              {formatTimesteps(Math.floor(job.totalTimesteps * job.progressPct / 100))} / {formatTimesteps(job.totalTimesteps)} steps
            </p>
          </div>
          <Button size="sm" variant="outline" className="text-xs border-border h-7" data-testid={`btn-stop-job-${job.id}`}>
            <Square size={10} className="mr-1" />
            Stop
          </Button>
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
            <p className={cn("font-mono text-sm font-semibold", job.currentReward > 0 ? "text-success" : "text-destructive")}>
              {job.currentReward.toFixed(3)}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground font-sans">Elapsed</p>
            <p className="font-mono text-sm text-foreground">{formatElapsed(job.elapsedSeconds)}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ---- Reward Curves ----
function RewardCurves({ jobs }: { jobs: TrainingJob[] }) {
  const colors = ["#3b82f6", "#f97316", "#a855f7", "#22c55e"];

  // Merge reward histories by step index
  const merged: Record<number, Record<string, number>> = {};
  jobs.forEach((job, idx) => {
    job.rewardHistory.forEach(({ step, reward }) => {
      if (!merged[step]) merged[step] = { step };
      merged[step][`${job.market}-${job.algo}`] = reward;
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
                contentStyle={{ background: "hsl(222 44% 12%)", border: "1px solid hsl(222 35% 20%)", borderRadius: "6px", fontSize: "11px", fontFamily: "'JetBrains Mono'" }}
                labelStyle={{ color: "#64748b" }}
              />
              <Legend wrapperStyle={{ fontSize: "11px", fontFamily: "'JetBrains Mono'" }} />
              {jobs.map((job, idx) => (
                <Line
                  key={job.id}
                  type="monotone"
                  dataKey={`${job.market}-${job.algo}`}
                  stroke={colors[idx % colors.length]}
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
type SortKey = "name" | "market" | "timescale" | "algo" | "sharpe" | "createdAt";
type SortDir = "asc" | "desc";

function CheckpointsTable({ checkpoints }: { checkpoints: ModelCheckpoint[] }) {
  const { toast } = useToast();
  const [sortKey, setSortKey] = useState<SortKey>("createdAt");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(key); setSortDir("desc"); }
  };

  const sorted = [...checkpoints].sort((a, b) => {
    let av = a[sortKey] ?? "";
    let bv = b[sortKey] ?? "";
    if (typeof av === "number" && typeof bv === "number") return sortDir === "asc" ? av - bv : bv - av;
    return sortDir === "asc" ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
  });

  const SortIcon = ({ k }: { k: SortKey }) => {
    if (sortKey !== k) return <span className="text-muted-foreground/40 ml-1">↕</span>;
    return sortDir === "asc" ? <ChevronUp size={12} className="inline ml-1" /> : <ChevronDown size={12} className="inline ml-1" />;
  };

  const th = "px-3 py-2 text-xs text-muted-foreground font-sans font-medium text-left cursor-pointer select-none hover:text-foreground";
  const td = "px-3 py-2 text-xs";

  return (
    <Card className="bg-card border-border" data-testid="checkpoints-table">
      <CardHeader className="py-3 px-4 border-b border-border">
        <CardTitle className="text-sm font-sans font-medium">Model Checkpoints</CardTitle>
      </CardHeader>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="border-b border-border">
            <tr>
              {(["name", "market", "timescale", "algo", "sharpe", "createdAt"] as SortKey[]).map((k) => (
                <th key={k} className={th} onClick={() => handleSort(k)}>
                  {k === "sharpe" ? "Sharpe" : k === "createdAt" ? "Created" : k.charAt(0).toUpperCase() + k.slice(1)}
                  <SortIcon k={k} />
                </th>
              ))}
              <th className="px-3 py-2 text-xs text-muted-foreground font-sans font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((ckpt) => (
              <tr key={ckpt.id} className="border-b border-border/50 hover:bg-accent/20 transition-colors" data-testid={`ckpt-row-${ckpt.id}`}>
                <td className={cn(td, "font-mono text-foreground")}>{ckpt.name}</td>
                <td className={cn(td, "font-mono text-muted-foreground")}>{ckpt.market}</td>
                <td className={cn(td, "font-mono text-muted-foreground")}>{ckpt.timescale}</td>
                <td className={td}><AlgoBadge algo={ckpt.algo} /></td>
                <td className={cn(td, "font-mono", ckpt.sharpe !== null ? (ckpt.sharpe > 1.5 ? "text-success" : "text-warning") : "text-muted-foreground")}>
                  {ckpt.sharpe !== null ? ckpt.sharpe.toFixed(2) : "—"}
                </td>
                <td className={cn(td, "font-mono text-muted-foreground text-xs")}>{ckpt.createdAt}</td>
                <td className={cn(td)}>
                  <div className="flex items-center gap-1">
                    <Button size="sm" variant="outline" className="h-6 text-xs px-2 border-border"
                      onClick={() => toast({ title: "Evaluating model...", description: ckpt.name })}
                      data-testid={`btn-evaluate-${ckpt.id}`}>
                      <BarChart size={10} className="mr-1" />Eval
                    </Button>
                    <Button size="sm" variant="outline" className="h-6 text-xs px-2 border-border"
                      onClick={() => toast({ title: "Exporting ONNX...", description: ckpt.name })}
                      data-testid={`btn-export-${ckpt.id}`}>
                      <Download size={10} className="mr-1" />ONNX
                    </Button>
                    <Button size="sm" variant="outline" className="h-6 text-xs px-2 border-border text-destructive hover:text-destructive"
                      onClick={() => toast({ title: "Deleted", description: ckpt.name, variant: "destructive" })}
                      data-testid={`btn-delete-${ckpt.id}`}>
                      <Trash2 size={10} />
                    </Button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

export default function TrainingConsole() {
  const { toast } = useToast();
  const [market, setMarket] = useState("US");
  const [timescale, setTimescale] = useState("1m");
  const [algo, setAlgo] = useState("PPO");
  const [timesteps, setTimesteps] = useState(1000000);
  const [jobs, setJobs] = useState<TrainingJob[]>(generateTrainingJobs());
  const [checkpoints] = useState<ModelCheckpoint[]>(generateModelCheckpoints());
  const [isStarting, setIsStarting] = useState(false);

  // Simulate job progress
  useEffect(() => {
    const interval = setInterval(() => {
      setJobs((prev) =>
        prev.map((job) => {
          if (job.status !== "running") return job;
          const newProgress = Math.min(job.progressPct + 0.08, 100);
          const newReward = job.currentReward + (Math.random() - 0.45) * 0.02;
          return {
            ...job,
            progressPct: newProgress,
            currentReward: newReward,
            elapsedSeconds: job.elapsedSeconds + 3,
            status: newProgress >= 100 ? "done" : "running",
          };
        })
      );
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleStartTraining = async () => {
    setIsStarting(true);
    await new Promise((r) => setTimeout(r, 800));
    const newJob: TrainingJob = {
      id: `job-${Date.now()}`,
      market,
      timescale,
      algo,
      totalTimesteps: timesteps,
      progressPct: 0,
      currentReward: -0.5,
      elapsedSeconds: 0,
      status: "running",
      rewardHistory: [],
    };
    setJobs((prev) => [...prev, newJob]);
    setIsStarting(false);
    toast({ title: "Training started", description: `${algo} on ${market} ${timescale} · ${formatTimesteps(timesteps)} steps` });
  };

  const logSlider = (value: number) => {
    const minLog = Math.log10(100000);
    const maxLog = Math.log10(5000000);
    return Math.round(Math.pow(10, minLog + (value / 100) * (maxLog - minLog)));
  };
  const sliderValue = (steps: number) => {
    const minLog = Math.log10(100000);
    const maxLog = Math.log10(5000000);
    return Math.round(((Math.log10(steps) - minLog) / (maxLog - minLog)) * 100);
  };

  const runningJobs = jobs.filter((j) => j.status === "running");

  return (
    <div className="flex flex-col h-full overflow-hidden" data-testid="page-training-console">
      {/* Header */}
      <div className="border-b border-border px-6 py-3 shrink-0">
        <h1 className="text-sm font-sans font-semibold text-foreground">Training Console</h1>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Config Row */}
        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex flex-wrap items-end gap-4">
              {/* Market */}
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground font-sans">Market</label>
                <Select value={market} onValueChange={setMarket} data-testid="train-market-select">
                  <SelectTrigger className="h-8 w-28 text-xs font-mono bg-muted border-border">
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
                <Select value={timescale} onValueChange={setTimescale} data-testid="train-timescale-select">
                  <SelectTrigger className="h-8 w-24 text-xs font-mono bg-muted border-border">
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
                <Select value={algo} onValueChange={setAlgo} data-testid="train-algo-select">
                  <SelectTrigger className="h-8 w-24 text-xs font-mono bg-muted border-border">
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

              <Button
                data-testid="btn-start-training"
                className="bg-primary hover:bg-primary/90 text-primary-foreground text-xs h-8 px-4"
                onClick={handleStartTraining}
                disabled={isStarting}
              >
                <Play size={12} className="mr-1" />
                {isStarting ? "Starting..." : "Start Training"}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Active Jobs */}
        {jobs.length > 0 && (
          <div>
            <h2 className="text-xs font-sans font-semibold text-muted-foreground uppercase tracking-wider mb-2">
              Active Jobs ({runningJobs.length} running)
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
              {jobs.map((job) => (
                <JobCard key={job.id} job={job} />
              ))}
            </div>
          </div>
        )}

        {/* Reward Curves */}
        {runningJobs.length > 0 && <RewardCurves jobs={runningJobs} />}

        {/* Checkpoints */}
        <CheckpointsTable checkpoints={checkpoints} />
      </div>
    </div>
  );
}
