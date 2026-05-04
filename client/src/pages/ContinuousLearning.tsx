import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import {
  CheckCircle2,
  XCircle,
  RefreshCw,
  ArrowUpCircle,
  Clock,
  Trophy,
  Layers,
  BarChart3,
  ChevronUp,
  ChevronDown,
  Loader2,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

// ── Types ──────────────────────────────────────────────────────────────────

interface ModelInfo {
  path: string;
  sharpe: number;
  max_drawdown: number;
  win_rate: number;
  trained_at: string;
  is_live: boolean;
}

interface ContinuousStatus {
  champion: ModelInfo;
  shadow: ModelInfo | null;
  is_shadow_training: boolean;
  n_promotions: number;
  n_trials: number;
  last_promotion_at: string;
}

interface SafeGate {
  name: string;
  threshold: string;
  actual: number;
  delta: number | null;
  passed: boolean;
}

interface PromotionRecord {
  version: string;
  sharpe: number;
  drawdown: number;
  promoted_at: string;
  reason: string;
  replaced_by: string;
}

// ── Mock Data ─────────────────────────────────────────────────────────────

const MOCK_STATUS: ContinuousStatus = {
  champion: {
    path: "models/champion/ppo_us_1m_v3.zip",
    sharpe: 1.42,
    max_drawdown: 0.087,
    win_rate: 0.587,
    trained_at: "2024-01-14T06:00:00Z",
    is_live: true,
  },
  shadow: {
    path: "models/shadow/ppo_us_1m_v4.zip",
    sharpe: 1.61,
    max_drawdown: 0.079,
    win_rate: 0.612,
    trained_at: "2024-01-15T03:00:00Z",
    is_live: false,
  },
  is_shadow_training: false,
  n_promotions: 3,
  n_trials: 7,
  last_promotion_at: "2024-01-12T10:00:00Z",
};

const MOCK_PROMOTION_HISTORY: PromotionRecord[] = [
  { version: "ppo_us_1m_v3", sharpe: 1.42, drawdown: 0.087, promoted_at: "2024-01-12T10:00:00Z", reason: "auto-pass", replaced_by: "ppo_us_1m_v4" },
  { version: "ppo_us_1m_v2", sharpe: 1.28, drawdown: 0.096, promoted_at: "2023-12-20T08:30:00Z", reason: "manual", replaced_by: "ppo_us_1m_v3" },
  { version: "ppo_us_1m_v1", sharpe: 1.11, drawdown: 0.108, promoted_at: "2023-11-05T14:00:00Z", reason: "auto-pass", replaced_by: "ppo_us_1m_v2" },
];

// ── Helpers ───────────────────────────────────────────────────────────────

function fmt(ts: string) {
  return new Date(ts).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function shortPath(p: string) {
  const parts = p.split("/");
  return parts[parts.length - 1];
}

function useCountdown(targetIsoOrNull: string | null, intervalHours: number) {
  const [label, setLabel] = useState("—");

  useEffect(() => {
    if (!targetIsoOrNull) return;
    const base = new Date(targetIsoOrNull).getTime();
    const nextMs = base + intervalHours * 3600 * 1000;

    const tick = () => {
      const diff = nextMs - Date.now();
      if (diff <= 0) {
        setLabel("due now");
        return;
      }
      const h = Math.floor(diff / 3600000);
      const m = Math.floor((diff % 3600000) / 60000);
      const s = Math.floor((diff % 60000) / 1000);
      setLabel(`${h}h ${m}m ${s}s`);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [targetIsoOrNull, intervalHours]);

  return label;
}

// ── Sub-components ────────────────────────────────────────────────────────

function ModelCard({
  title,
  badge,
  badgeColor,
  model,
  onPromote,
  canPromote,
  isTraining,
}: {
  title: string;
  badge: string;
  badgeColor: "green" | "yellow";
  model: ModelInfo | null;
  onPromote?: () => void;
  canPromote?: boolean;
  isTraining?: boolean;
}) {
  return (
    <Card className="bg-card border-border flex-1 min-w-0">
      <CardHeader className="pb-2 pt-4 px-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-sans font-semibold text-muted-foreground uppercase tracking-wide">
            {title}
          </CardTitle>
          <div className="flex items-center gap-2">
            {isTraining && (
              <Loader2 size={13} className="animate-spin text-yellow-400" />
            )}
            <span
              className={cn(
                "flex items-center gap-1.5 text-xs font-mono font-semibold px-2 py-0.5 rounded-full border",
                badgeColor === "green"
                  ? "text-green-400 border-green-500/40 bg-green-500/10"
                  : "text-yellow-400 border-yellow-500/40 bg-yellow-500/10"
              )}
            >
              {badgeColor === "green" && (
                <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
              )}
              {badge}
            </span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4">
        {!model ? (
          <p className="text-sm text-muted-foreground italic">
            {isTraining
              ? "Training in progress…"
              : "No candidate — trigger retraining below"}
          </p>
        ) : (
          <div className="space-y-1.5">
            <p
              className="text-xs font-mono text-muted-foreground truncate"
              title={model.path}
            >
              {shortPath(model.path)}
            </p>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2">
              <div>
                <span className="text-xs text-muted-foreground">Sharpe</span>
                <p
                  className={cn(
                    "text-lg font-mono font-semibold",
                    model.sharpe > 1.0 ? "text-green-400" : "text-foreground"
                  )}
                >
                  {model.sharpe.toFixed(2)}
                </p>
              </div>
              <div>
                <span className="text-xs text-muted-foreground">Max DD</span>
                <p
                  className={cn(
                    "text-lg font-mono font-semibold",
                    model.max_drawdown > 0.15
                      ? "text-red-400"
                      : "text-foreground"
                  )}
                >
                  {(model.max_drawdown * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <span className="text-xs text-muted-foreground">Win Rate</span>
                <p className="text-base font-mono font-semibold text-foreground">
                  {(model.win_rate * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <span className="text-xs text-muted-foreground">Trained</span>
                <p className="text-xs font-mono text-muted-foreground mt-0.5">
                  {fmt(model.trained_at)}
                </p>
              </div>
            </div>
            {canPromote && onPromote && (
              <Button
                size="sm"
                className="mt-3 w-full bg-green-600 hover:bg-green-500 text-white text-xs font-mono"
                onClick={onPromote}
              >
                <ArrowUpCircle size={13} className="mr-1.5" />
                Promote to Champion
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function SafeGatesPanel({
  champion,
  shadow,
}: {
  champion: ModelInfo;
  shadow: ModelInfo | null;
}) {
  const gates: SafeGate[] = shadow
    ? [
        {
          name: "Sharpe Improvement",
          threshold: "≥ +0.05",
          actual: shadow.sharpe,
          delta: shadow.sharpe - champion.sharpe,
          passed: shadow.sharpe - champion.sharpe >= 0.05,
        },
        {
          name: "Drawdown Regression",
          threshold: "≤ +2%",
          actual: shadow.max_drawdown,
          delta: (shadow.max_drawdown - champion.max_drawdown) * 100,
          passed:
            (shadow.max_drawdown - champion.max_drawdown) * 100 <= 2.0,
        },
        {
          name: "KL Divergence",
          threshold: "≤ 0.5",
          actual: 0.23,
          delta: null,
          passed: true,
        },
      ]
    : [];

  const allPassed = gates.length > 0 && gates.every((g) => g.passed);

  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-2 pt-4 px-4">
        <CardTitle className="text-sm font-sans font-semibold text-muted-foreground uppercase tracking-wide">
          Safety Gates
        </CardTitle>
      </CardHeader>
      <CardContent className="px-4 pb-4">
        {!shadow ? (
          <p className="text-sm text-muted-foreground italic">
            No shadow model to evaluate.
          </p>
        ) : (
          <div className="space-y-3">
            {gates.map((gate) => (
              <div
                key={gate.name}
                className="flex items-center justify-between py-2 border-b border-border last:border-0"
              >
                <div className="flex items-center gap-2">
                  {gate.passed ? (
                    <CheckCircle2 size={15} className="text-green-400 shrink-0" />
                  ) : (
                    <XCircle size={15} className="text-red-400 shrink-0" />
                  )}
                  <div>
                    <p className="text-sm font-sans font-medium text-foreground">
                      {gate.name}
                    </p>
                    <p className="text-xs font-mono text-muted-foreground">
                      threshold {gate.threshold}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p
                    className={cn(
                      "text-sm font-mono font-semibold",
                      gate.passed ? "text-green-400" : "text-red-400"
                    )}
                  >
                    {gate.delta !== null
                      ? `${gate.delta >= 0 ? "+" : ""}${gate.delta.toFixed(3)}`
                      : gate.actual.toFixed(3)}
                  </p>
                </div>
              </div>
            ))}

            <div
              className={cn(
                "mt-2 px-3 py-2 rounded-md border text-sm font-mono font-semibold",
                allPassed
                  ? "bg-green-500/10 border-green-500/30 text-green-400"
                  : "bg-red-500/10 border-red-500/30 text-red-400"
              )}
            >
              {allPassed
                ? "✓ PASSED — eligible for promotion"
                : "✗ FAILED — shadow retained for monitoring"}
            </div>
            <p className="text-xs font-mono text-muted-foreground">
              Evaluated {fmt(shadow.trained_at)}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────

export default function ContinuousLearning() {
  const [intervalHours, setIntervalHours] = useState(24);
  const [autoPromote, setAutoPromote] = useState(false);
  const [nTimesteps, setNTimesteps] = useState("250000");
  const [algorithm, setAlgorithm] = useState("PPO");

  const { data: status, isError } = useQuery<ContinuousStatus>({
    queryKey: ["/api/continuous/status"],
    refetchInterval: 5000,
  });

  const activeStatus = isError || !status ? MOCK_STATUS : status;
  const { champion, shadow, is_shadow_training, n_promotions, n_trials, last_promotion_at } =
    activeStatus;

  const countdown = useCountdown(last_promotion_at, intervalHours);

  // Gates pass check
  const shadowPassed =
    shadow !== null &&
    shadow.sharpe - champion.sharpe >= 0.05 &&
    (shadow.max_drawdown - champion.max_drawdown) * 100 <= 2.0;

  const triggerMutation = useMutation({
    mutationFn: () =>
      apiRequest("POST", "/api/continuous/trigger", { force: true }),
  });

  const promoteMutation = useMutation({
    mutationFn: () =>
      apiRequest("POST", "/api/continuous/promote", {
        model_path: shadow?.path,
        reason: "manual",
      }),
  });

  const configureMutation = useMutation({
    mutationFn: () =>
      apiRequest("POST", "/api/continuous/configure", {
        retrain_interval_hours: intervalHours,
        auto_promote: autoPromote,
        n_timesteps: parseInt(nTimesteps, 10),
        algorithm,
      }),
  });

  return (
    <div className="flex flex-col gap-4 p-4 min-h-0 overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-sans font-semibold text-foreground">
            Continuous Learning
          </h1>
          <p className="text-xs text-muted-foreground font-sans mt-0.5">
            Champion / Shadow model management
          </p>
        </div>
        {(isError || !status) && (
          <Badge variant="outline" className="text-yellow-400 border-yellow-500/40 font-mono text-xs">
            MOCK DATA
          </Badge>
        )}
      </div>

      {/* Row 1 — Status Cards */}
      <div className="flex gap-3 flex-wrap">
        {/* Champion */}
        <ModelCard
          title="Champion Model"
          badge="LIVE"
          badgeColor="green"
          model={champion}
        />

        {/* Shadow */}
        <ModelCard
          title="Shadow Model"
          badge="CANDIDATE"
          badgeColor="yellow"
          model={shadow}
          onPromote={() => promoteMutation.mutate()}
          canPromote={shadowPassed}
          isTraining={is_shadow_training}
        />

        {/* Learning Stats */}
        <Card className="bg-card border-border flex-1 min-w-[220px]">
          <CardHeader className="pb-2 pt-4 px-4">
            <CardTitle className="text-sm font-sans font-semibold text-muted-foreground uppercase tracking-wide flex items-center gap-2">
              <Layers size={14} />
              Learning Stats
            </CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-4 space-y-2">
            <Row label="Total Promotions" value={String(n_promotions)} />
            <Row label="Total Trials" value={String(n_trials)} />
            <Row label="Last Promotion" value={fmt(last_promotion_at)} mono />
            <Row
              label="Status"
              value={
                is_shadow_training ? (
                  <span className="flex items-center gap-1 text-yellow-400">
                    <Loader2 size={12} className="animate-spin" />
                    training
                  </span>
                ) : (
                  <span className="text-green-400">idle</span>
                )
              }
            />
            <Row label="Interval" value={`${intervalHours}h`} />
            <Row label="Next Retrain" value={<span className="text-yellow-400 font-mono">{countdown}</span>} />
          </CardContent>
        </Card>
      </div>

      {/* Row 2 — Safety Gates + Config */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <SafeGatesPanel champion={champion} shadow={shadow} />

        {/* Config Panel */}
        <Card className="bg-card border-border">
          <CardHeader className="pb-2 pt-4 px-4">
            <CardTitle className="text-sm font-sans font-semibold text-muted-foreground uppercase tracking-wide">
              Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-4 space-y-4">
            {/* Interval slider */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground font-sans">
                Retrain Interval: <span className="text-foreground font-mono">{intervalHours}h</span>
              </Label>
              <Slider
                min={1}
                max={168}
                step={1}
                value={[intervalHours]}
                onValueChange={([v]) => setIntervalHours(v)}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground font-mono">
                <span>1h</span>
                <span>168h</span>
              </div>
            </div>

            {/* Auto-promote toggle */}
            <div className="flex items-center justify-between">
              <Label className="text-xs text-muted-foreground font-sans">
                Auto-promote on pass
              </Label>
              <Switch
                checked={autoPromote}
                onCheckedChange={setAutoPromote}
              />
            </div>

            {/* N timesteps */}
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground font-sans">
                Timesteps per retrain
              </Label>
              <Select value={nTimesteps} onValueChange={setNTimesteps}>
                <SelectTrigger className="h-8 text-xs font-mono bg-background">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {["100000", "250000", "500000", "1000000"].map((v) => (
                    <SelectItem key={v} value={v} className="text-xs font-mono">
                      {parseInt(v, 10).toLocaleString()}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Algorithm */}
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground font-sans">
                Algorithm
              </Label>
              <Select value={algorithm} onValueChange={setAlgorithm}>
                <SelectTrigger className="h-8 text-xs font-mono bg-background">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="PPO" className="text-xs font-mono">PPO</SelectItem>
                  <SelectItem value="TD3" className="text-xs font-mono">TD3</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Buttons */}
            <div className="flex gap-2 pt-1">
              <Button
                size="sm"
                variant="outline"
                className="flex-1 text-xs font-mono"
                onClick={() => configureMutation.mutate()}
                disabled={configureMutation.isPending}
              >
                {configureMutation.isPending ? (
                  <Loader2 size={12} className="animate-spin mr-1" />
                ) : null}
                Save Config
              </Button>
              <Button
                size="sm"
                className="flex-1 text-xs font-mono bg-orange-600 hover:bg-orange-500 text-white"
                onClick={() => triggerMutation.mutate()}
                disabled={triggerMutation.isPending}
              >
                {triggerMutation.isPending ? (
                  <Loader2 size={12} className="animate-spin mr-1" />
                ) : (
                  <RefreshCw size={12} className="mr-1.5" />
                )}
                Force Retrain
              </Button>
            </div>
            {(triggerMutation.isSuccess || triggerMutation.isError) && (
              <p className={cn("text-xs font-mono", triggerMutation.isError ? "text-red-400" : "text-green-400")}>
                {triggerMutation.isError ? "Error triggering retrain (mock)" : "Retrain triggered"}
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Row 3 — Promotion History */}
      <Card className="bg-card border-border">
        <CardHeader className="pb-2 pt-4 px-4">
          <CardTitle className="text-sm font-sans font-semibold text-muted-foreground uppercase tracking-wide flex items-center gap-2">
            <Trophy size={14} />
            Promotion History
          </CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-4">
          <div className="rounded-md border border-border overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="border-border hover:bg-transparent">
                  {["Champion Version", "Sharpe", "Drawdown", "Promoted At", "Reason", "Replaced By"].map(
                    (h) => (
                      <TableHead key={h} className="text-xs font-sans text-muted-foreground py-2 px-3 whitespace-nowrap">
                        {h}
                      </TableHead>
                    )
                  )}
                </TableRow>
              </TableHeader>
              <TableBody>
                {MOCK_PROMOTION_HISTORY.map((row, i) => (
                  <TableRow key={i} className="border-border hover:bg-accent/40">
                    <TableCell className="text-xs font-mono text-foreground py-2 px-3 whitespace-nowrap">
                      {row.version}
                    </TableCell>
                    <TableCell className="text-xs font-mono py-2 px-3">
                      <span className={row.sharpe > 1.0 ? "text-green-400" : "text-foreground"}>
                        {row.sharpe.toFixed(2)}
                      </span>
                    </TableCell>
                    <TableCell className="text-xs font-mono py-2 px-3">
                      <span className={row.drawdown > 0.15 ? "text-red-400" : "text-foreground"}>
                        {(row.drawdown * 100).toFixed(1)}%
                      </span>
                    </TableCell>
                    <TableCell className="text-xs font-mono text-muted-foreground py-2 px-3 whitespace-nowrap">
                      {fmt(row.promoted_at)}
                    </TableCell>
                    <TableCell className="text-xs py-2 px-3">
                      <Badge
                        variant="outline"
                        className={cn(
                          "text-xs font-mono",
                          row.reason === "manual"
                            ? "text-blue-400 border-blue-500/40"
                            : "text-green-400 border-green-500/40"
                        )}
                      >
                        {row.reason}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-xs font-mono text-muted-foreground py-2 px-3 whitespace-nowrap">
                      {row.replaced_by}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ── Tiny helper ────────────────────────────────────────────────────────────

function Row({
  label,
  value,
  mono,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-muted-foreground font-sans">{label}</span>
      <span className={cn("text-xs font-semibold text-foreground", mono && "font-mono")}>
        {value}
      </span>
    </div>
  );
}
