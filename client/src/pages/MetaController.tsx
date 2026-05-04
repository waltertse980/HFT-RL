import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Zap,
  Plus,
  Trash2,
  Loader2,
  Brain,
  BarChart3,
  Users,
  Activity,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Dot,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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

type Regime = "bull" | "bear" | "sideways" | "volatile";
type Action = "BUY" | "HOLD" | "SELL";

interface MetaStatus {
  regime: Regime;
  kelly_fraction: number;
  active_models: string[];
  signal_weights: Record<string, number>;
  n_models: number;
}

interface SignalRecord {
  timestamp: string;
  action: Action;
  confidence: number;
  regime: Regime;
  kelly_fraction: number;
  price: number;
}

interface RegisteredModel {
  model_id: string;
  algorithm: string;
  market: string;
  timescale: string;
  weight: number;
  sharpe: number;
  last_signal: Action;
}

// ── Mock Data ─────────────────────────────────────────────────────────────

const MOCK_META: MetaStatus = {
  regime: "bull",
  kelly_fraction: 0.18,
  active_models: ["ppo_us_1m", "ppo_us_5m", "td3_us_1m"],
  signal_weights: { ppo_us_1m: 0.5, ppo_us_5m: 0.3, td3_us_1m: 0.2 },
  n_models: 3,
};

const MOCK_SIGNALS: SignalRecord[] = Array.from({ length: 20 }, (_, i) => ({
  timestamp: new Date(Date.now() - i * 60000).toISOString(),
  action: (["BUY", "HOLD", "SELL", "HOLD", "BUY"] as Action[])[i % 5],
  confidence: 0.45 + Math.random() * 0.45,
  regime: "bull" as Regime,
  kelly_fraction: 0.15 + Math.random() * 0.08,
  price: 191 + Math.random() * 4,
}));

const MOCK_MODELS: RegisteredModel[] = [
  { model_id: "ppo_us_1m", algorithm: "PPO", market: "US", timescale: "1m", weight: 0.5, sharpe: 1.42, last_signal: "BUY" },
  { model_id: "ppo_us_5m", algorithm: "PPO", market: "US", timescale: "5m", weight: 0.3, sharpe: 1.28, last_signal: "HOLD" },
  { model_id: "td3_us_1m", algorithm: "TD3", market: "US", timescale: "1m", weight: 0.2, sharpe: 1.09, last_signal: "BUY" },
];

// ── Helpers ───────────────────────────────────────────────────────────────

function fmt(ts: string) {
  return new Date(ts).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

const REGIME_META: Record<
  Regime,
  { label: string; icon: React.ReactNode; color: string; bg: string; border: string }
> = {
  bull: {
    label: "BULL",
    icon: <TrendingUp size={18} />,
    color: "text-green-400",
    bg: "bg-green-500/10",
    border: "border-green-500/30",
  },
  bear: {
    label: "BEAR",
    icon: <TrendingDown size={18} />,
    color: "text-red-400",
    bg: "bg-red-500/10",
    border: "border-red-500/30",
  },
  sideways: {
    label: "SIDEWAYS",
    icon: <Minus size={18} />,
    color: "text-yellow-400",
    bg: "bg-yellow-500/10",
    border: "border-yellow-500/30",
  },
  volatile: {
    label: "VOLATILE",
    icon: <Zap size={18} />,
    color: "text-orange-400",
    bg: "bg-orange-500/10",
    border: "border-orange-500/30",
  },
};

const ACTION_COLOR: Record<Action, string> = {
  BUY: "#22c55e",
  HOLD: "#64748b",
  SELL: "#ef4444",
};

// ── KPI Cards ─────────────────────────────────────────────────────────────

function RegimeCard({ regime }: { regime: Regime }) {
  const meta = REGIME_META[regime];
  return (
    <Card className="bg-card border-border flex-1">
      <CardContent className="p-4">
        <p className="text-xs text-muted-foreground font-sans uppercase tracking-wide mb-2">
          Current Regime
        </p>
        <div
          className={cn(
            "flex items-center gap-2 px-3 py-2 rounded-md border",
            meta.bg,
            meta.border
          )}
        >
          <span className={meta.color}>{meta.icon}</span>
          <span className={cn("text-xl font-mono font-bold", meta.color)}>
            {meta.label}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

function KellyCard({ fraction }: { fraction: number }) {
  const pct = fraction * 100;
  const arcDeg = (pct / 25) * 180; // 0-25% maps to 0-180deg arc
  const r = 40;
  const cx = 60;
  const cy = 52;
  // SVG arc: start left (-180deg), sweep right
  const startX = cx - r;
  const startY = cy;
  const endX = cx + r;
  const endY = cy;
  // active arc endpoint
  const rad = ((arcDeg - 180) * Math.PI) / 180;
  const activeX = cx + r * Math.cos(rad);
  const activeY = cy + r * Math.sin(rad);
  const largeArc = arcDeg > 180 ? 1 : 0;

  return (
    <Card className="bg-card border-border flex-1">
      <CardContent className="p-4">
        <p className="text-xs text-muted-foreground font-sans uppercase tracking-wide mb-1">
          Kelly Fraction
        </p>
        <div className="flex items-center gap-3">
          <svg width="120" height="58" viewBox="0 0 120 58">
            {/* Background arc */}
            <path
              d={`M ${startX} ${cy} A ${r} ${r} 0 0 1 ${endX} ${cy}`}
              fill="none"
              stroke="#1e293b"
              strokeWidth="8"
              strokeLinecap="round"
            />
            {/* Active arc */}
            {arcDeg > 0 && (
              <path
                d={`M ${startX} ${cy} A ${r} ${r} 0 ${largeArc} 1 ${activeX} ${activeY}`}
                fill="none"
                stroke={pct < 10 ? "#22c55e" : pct < 18 ? "#f59e0b" : "#ef4444"}
                strokeWidth="8"
                strokeLinecap="round"
              />
            )}
            <text
              x={cx}
              y={cy - 6}
              textAnchor="middle"
              className="text-foreground"
              fill="currentColor"
              fontSize="13"
              fontWeight="700"
              fontFamily="monospace"
            >
              {pct.toFixed(1)}%
            </text>
            <text
              x={cx}
              y={cy + 10}
              textAnchor="middle"
              fill="#64748b"
              fontSize="8"
              fontFamily="sans-serif"
            >
              of capital
            </text>
          </svg>
          <div className="text-xs text-muted-foreground font-sans space-y-1">
            <p>
              <span className="text-green-400">0–10%</span> safe
            </p>
            <p>
              <span className="text-yellow-400">10–18%</span> moderate
            </p>
            <p>
              <span className="text-red-400">18–25%</span> aggressive
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function ActiveModelsCard({ count, models }: { count: number; models: string[] }) {
  return (
    <Card className="bg-card border-border flex-1">
      <CardContent className="p-4">
        <p className="text-xs text-muted-foreground font-sans uppercase tracking-wide mb-2">
          Active Models
        </p>
        <div className="flex items-center gap-3">
          <span className="text-3xl font-mono font-bold text-foreground">{count}</span>
          <div className="flex -space-x-2">
            {models.slice(0, 5).map((m, i) => (
              <div
                key={m}
                className="w-7 h-7 rounded-full bg-primary/20 border-2 border-background flex items-center justify-center"
                title={m}
                style={{ zIndex: 5 - i }}
              >
                <Brain size={12} className="text-primary" />
              </div>
            ))}
          </div>
        </div>
        <p className="text-xs text-muted-foreground font-mono mt-1">
          {models.join(", ")}
        </p>
      </CardContent>
    </Card>
  );
}

function AggregatedSignalCard({
  signals,
}: {
  signals: SignalRecord[];
}) {
  const latest = signals[0];
  if (!latest) return null;
  const pct = Math.round(latest.confidence * 100);

  return (
    <Card className="bg-card border-border flex-1">
      <CardContent className="p-4">
        <p className="text-xs text-muted-foreground font-sans uppercase tracking-wide mb-2">
          Aggregated Signal
        </p>
        <div className="flex items-center gap-2 mb-2">
          <span
            className={cn(
              "text-2xl font-mono font-bold",
              latest.action === "BUY"
                ? "text-green-400"
                : latest.action === "SELL"
                ? "text-red-400"
                : "text-muted-foreground"
            )}
          >
            {latest.action}
          </span>
          <span className="text-sm font-mono text-muted-foreground">
            {pct}% conf
          </span>
        </div>
        {/* Confidence bar */}
        <div className="h-2 rounded-full bg-muted overflow-hidden">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-500",
              latest.action === "BUY"
                ? "bg-green-500"
                : latest.action === "SELL"
                ? "bg-red-500"
                : "bg-slate-500"
            )}
            style={{ width: `${pct}%` }}
          />
        </div>
      </CardContent>
    </Card>
  );
}

// ── Register Form ──────────────────────────────────────────────────────────

function RegisterModelForm({
  onClose,
  onSubmit,
}: {
  onClose: () => void;
  onSubmit: (data: {
    model_id: string;
    model_path: string;
    vecnorm_path: string;
    weight: number;
    algorithm: string;
    market: string;
    timescale: string;
  }) => void;
}) {
  const [modelId, setModelId] = useState("");
  const [modelPath, setModelPath] = useState("");
  const [vecnormPath, setVecnormPath] = useState("");
  const [weight, setWeight] = useState(0.5);
  const [algorithm, setAlgorithm] = useState("PPO");
  const [market, setMarket] = useState("US");
  const [timescale, setTimescale] = useState("1m");

  return (
    <div className="mt-3 p-3 border border-border rounded-md bg-background space-y-2">
      <p className="text-xs font-sans font-semibold text-muted-foreground uppercase tracking-wide">
        Register New Model
      </p>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <Label className="text-xs text-muted-foreground">Model ID</Label>
          <Input
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="h-7 text-xs font-mono bg-card mt-1"
            placeholder="ppo_us_1m_v5"
          />
        </div>
        <div>
          <Label className="text-xs text-muted-foreground">Algorithm</Label>
          <Select value={algorithm} onValueChange={setAlgorithm}>
            <SelectTrigger className="h-7 text-xs font-mono bg-card mt-1">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="PPO" className="text-xs font-mono">PPO</SelectItem>
              <SelectItem value="TD3" className="text-xs font-mono">TD3</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="col-span-2">
          <Label className="text-xs text-muted-foreground">Model Path</Label>
          <Input
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
            className="h-7 text-xs font-mono bg-card mt-1"
            placeholder="models/champion/ppo_us_1m_v5.zip"
          />
        </div>
        <div className="col-span-2">
          <Label className="text-xs text-muted-foreground">VecNorm Path</Label>
          <Input
            value={vecnormPath}
            onChange={(e) => setVecnormPath(e.target.value)}
            className="h-7 text-xs font-mono bg-card mt-1"
            placeholder="models/vecnorm/ppo_us_1m_v5.pkl"
          />
        </div>
        <div>
          <Label className="text-xs text-muted-foreground">Market</Label>
          <Select value={market} onValueChange={setMarket}>
            <SelectTrigger className="h-7 text-xs font-mono bg-card mt-1">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="US" className="text-xs font-mono">US</SelectItem>
              <SelectItem value="HK" className="text-xs font-mono">HK</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label className="text-xs text-muted-foreground">Timescale</Label>
          <Select value={timescale} onValueChange={setTimescale}>
            <SelectTrigger className="h-7 text-xs font-mono bg-card mt-1">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {["10s", "1m", "5m", "1h"].map((v) => (
                <SelectItem key={v} value={v} className="text-xs font-mono">
                  {v}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="col-span-2 space-y-1">
          <Label className="text-xs text-muted-foreground">
            Weight: <span className="text-foreground font-mono">{weight.toFixed(2)}</span>
          </Label>
          <Slider
            min={0}
            max={1}
            step={0.05}
            value={[weight]}
            onValueChange={([v]) => setWeight(v)}
            className="w-full"
          />
        </div>
      </div>
      <div className="flex gap-2 pt-1">
        <Button
          size="sm"
          className="text-xs font-mono flex-1 bg-primary hover:bg-primary/90"
          onClick={() =>
            onSubmit({ model_id: modelId, model_path: modelPath, vecnorm_path: vecnormPath, weight, algorithm, market, timescale })
          }
        >
          Submit
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="text-xs font-mono flex-1"
          onClick={onClose}
        >
          Cancel
        </Button>
      </div>
    </div>
  );
}

// ── Signal Chart ──────────────────────────────────────────────────────────

function SignalChart({ signals }: { signals: SignalRecord[] }) {
  const data = [...signals].reverse().map((s, i) => ({
    i,
    time: fmt(s.timestamp),
    confidence: parseFloat((s.confidence * 100).toFixed(1)),
    action: s.action,
    color: ACTION_COLOR[s.action],
  }));

  return (
    <ResponsiveContainer width="100%" height={160}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -24 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis
          dataKey="i"
          tick={{ fontSize: 9, fill: "#64748b", fontFamily: "monospace" }}
          tickLine={false}
          axisLine={false}
          interval={4}
          tickFormatter={(i) => data[i]?.time ?? ""}
        />
        <YAxis
          domain={[0, 100]}
          tick={{ fontSize: 9, fill: "#64748b", fontFamily: "monospace" }}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `${v}%`}
        />
        <Tooltip
          contentStyle={{
            background: "#0f172a",
            border: "1px solid #1e293b",
            borderRadius: 4,
            fontSize: 11,
            fontFamily: "monospace",
            color: "#e2e8f0",
          }}
          formatter={(v: number, name: string) => [`${v}%`, "Confidence"]}
          labelFormatter={(i) => data[i]?.time ?? ""}
        />
        <Line
          type="monotone"
          dataKey="confidence"
          dot={(props) => {
            const { cx, cy, payload } = props;
            return (
              <circle
                key={payload.i}
                cx={cx}
                cy={cy}
                r={3}
                fill={payload.color}
                stroke="none"
              />
            );
          }}
          stroke="#3b82f6"
          strokeWidth={1.5}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────

export default function MetaController() {
  const [showRegisterForm, setShowRegisterForm] = useState(false);
  const [modelWeights, setModelWeights] = useState<Record<string, number>>({});
  const [regimeOverride, setRegimeOverride] = useState("auto");
  const [kellyOverride, setKellyOverride] = useState(0.18);
  const [minConfidence, setMinConfidence] = useState(0.4);

  const { data: metaStatus, isError: metaError } = useQuery<MetaStatus>({
    queryKey: ["/api/meta/status"],
    refetchInterval: 3000,
  });

  const { data: signalsData, isError: signalsError } = useQuery<{
    signals: SignalRecord[];
  }>({
    queryKey: ["/api/meta/signals"],
    refetchInterval: 3000,
  });

  const activeStatus = metaError || !metaStatus ? MOCK_META : metaStatus;
  const signals =
    signalsError || !signalsData ? MOCK_SIGNALS : signalsData.signals;

  const configureMutation = useMutation({
    mutationFn: () =>
      apiRequest("POST", "/api/meta/configure", {
        kelly_fraction: kellyOverride,
        regime_override: regimeOverride === "auto" ? null : regimeOverride,
        min_confidence: minConfidence,
      }),
  });

  const registerMutation = useMutation({
    mutationFn: (data: {
      model_id: string;
      model_path: string;
      vecnorm_path: string;
      weight: number;
      algorithm: string;
      market: string;
      timescale: string;
    }) => apiRequest("POST", "/api/meta/register", data),
    onSuccess: () => setShowRegisterForm(false),
  });

  const unregisterMutation = useMutation({
    mutationFn: (model_id: string) =>
      apiRequest("POST", "/api/meta/unregister", { model_id }),
  });

  // Merge model list: from mock + weights overrides
  const models = MOCK_MODELS.map((m) => ({
    ...m,
    weight: modelWeights[m.model_id] ?? m.weight,
  }));

  return (
    <div className="flex flex-col gap-4 p-4 min-h-0 overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-sans font-semibold text-foreground">
            Meta Controller
          </h1>
          <p className="text-xs text-muted-foreground font-sans mt-0.5">
            Multi-model signal aggregation · regime detection · Kelly sizing
          </p>
        </div>
        {(metaError || !metaStatus) && (
          <Badge
            variant="outline"
            className="text-yellow-400 border-yellow-500/40 font-mono text-xs"
          >
            MOCK DATA
          </Badge>
        )}
      </div>

      {/* Row 1 — KPI Cards */}
      <div className="flex gap-3 flex-wrap">
        <RegimeCard regime={activeStatus.regime as Regime} />
        <KellyCard fraction={activeStatus.kelly_fraction} />
        <ActiveModelsCard
          count={activeStatus.n_models}
          models={activeStatus.active_models}
        />
        <AggregatedSignalCard signals={signals} />
      </div>

      {/* Row 2 — Model Registry + Config */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        {/* Model Registry (2 cols) */}
        <Card className="bg-card border-border lg:col-span-2">
          <CardHeader className="pb-2 pt-4 px-4">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-sans font-semibold text-muted-foreground uppercase tracking-wide flex items-center gap-2">
                <Users size={14} />
                Model Registry
              </CardTitle>
              <Button
                size="sm"
                variant="outline"
                className="h-7 text-xs font-mono"
                onClick={() => setShowRegisterForm((v) => !v)}
              >
                <Plus size={12} className="mr-1" />
                Register Model
              </Button>
            </div>
          </CardHeader>
          <CardContent className="px-4 pb-4">
            {showRegisterForm && (
              <RegisterModelForm
                onClose={() => setShowRegisterForm(false)}
                onSubmit={(data) => registerMutation.mutate(data)}
              />
            )}
            <div className="rounded-md border border-border overflow-x-auto mt-2">
              <Table>
                <TableHeader>
                  <TableRow className="border-border hover:bg-transparent">
                    {[
                      "Model ID",
                      "Algo",
                      "Market",
                      "Timescale",
                      "Weight",
                      "Sharpe",
                      "Last Signal",
                      "",
                    ].map((h) => (
                      <TableHead
                        key={h}
                        className="text-xs font-sans text-muted-foreground py-2 px-3 whitespace-nowrap"
                      >
                        {h}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {models.map((m) => (
                    <TableRow
                      key={m.model_id}
                      className="border-border hover:bg-accent/40"
                    >
                      <TableCell className="text-xs font-mono text-foreground py-2 px-3 whitespace-nowrap">
                        {m.model_id}
                      </TableCell>
                      <TableCell className="text-xs font-mono py-2 px-3">
                        <Badge
                          variant="outline"
                          className={cn(
                            "text-xs font-mono",
                            m.algorithm === "PPO"
                              ? "text-blue-400 border-blue-500/40"
                              : "text-purple-400 border-purple-500/40"
                          )}
                        >
                          {m.algorithm}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-xs font-mono text-muted-foreground py-2 px-3">
                        {m.market}
                      </TableCell>
                      <TableCell className="text-xs font-mono text-muted-foreground py-2 px-3">
                        {m.timescale}
                      </TableCell>
                      <TableCell className="py-2 px-3 min-w-[120px]">
                        <div className="flex items-center gap-2">
                          <Slider
                            min={0}
                            max={1}
                            step={0.05}
                            value={[m.weight]}
                            onValueChange={([v]) =>
                              setModelWeights((prev) => ({
                                ...prev,
                                [m.model_id]: v,
                              }))
                            }
                            className="w-16"
                          />
                          <span className="text-xs font-mono text-foreground w-8">
                            {m.weight.toFixed(2)}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell className="text-xs font-mono py-2 px-3">
                        <span
                          className={
                            m.sharpe > 1.0 ? "text-green-400" : "text-foreground"
                          }
                        >
                          {m.sharpe.toFixed(2)}
                        </span>
                      </TableCell>
                      <TableCell className="text-xs py-2 px-3">
                        <span
                          className={cn(
                            "font-mono font-semibold text-xs",
                            m.last_signal === "BUY"
                              ? "text-green-400"
                              : m.last_signal === "SELL"
                              ? "text-red-400"
                              : "text-muted-foreground"
                          )}
                        >
                          {m.last_signal}
                        </span>
                      </TableCell>
                      <TableCell className="py-2 px-3">
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-6 w-6 p-0 text-muted-foreground hover:text-red-400"
                          onClick={() => unregisterMutation.mutate(m.model_id)}
                        >
                          <Trash2 size={12} />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>

        {/* Regime Config */}
        <Card className="bg-card border-border">
          <CardHeader className="pb-2 pt-4 px-4">
            <CardTitle className="text-sm font-sans font-semibold text-muted-foreground uppercase tracking-wide flex items-center gap-2">
              <Activity size={14} />
              Regime Config
            </CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-4 space-y-4">
            {/* Regime override */}
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground font-sans">
                Regime Override
              </Label>
              <Select value={regimeOverride} onValueChange={setRegimeOverride}>
                <SelectTrigger className="h-8 text-xs font-mono bg-background">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {["auto", "bull", "bear", "sideways", "volatile"].map((v) => (
                    <SelectItem key={v} value={v} className="text-xs font-mono capitalize">
                      {v === "auto" ? "Auto-Detect" : v.charAt(0).toUpperCase() + v.slice(1)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Kelly override slider */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground font-sans">
                Kelly Fraction:{" "}
                <span className="text-foreground font-mono">
                  {(kellyOverride * 100).toFixed(0)}%
                </span>
              </Label>
              <Slider
                min={0}
                max={0.25}
                step={0.01}
                value={[kellyOverride]}
                onValueChange={([v]) => setKellyOverride(v)}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground font-mono">
                <span>0%</span>
                <span>25%</span>
              </div>
            </div>

            {/* Min confidence */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground font-sans">
                Min Confidence:{" "}
                <span className="text-foreground font-mono">
                  {(minConfidence * 100).toFixed(0)}%
                </span>
              </Label>
              <Slider
                min={0.3}
                max={0.8}
                step={0.05}
                value={[minConfidence]}
                onValueChange={([v]) => setMinConfidence(v)}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground font-mono">
                <span>30%</span>
                <span>80%</span>
              </div>
            </div>

            <Button
              size="sm"
              className="w-full text-xs font-mono"
              onClick={() => configureMutation.mutate()}
              disabled={configureMutation.isPending}
            >
              {configureMutation.isPending ? (
                <Loader2 size={12} className="animate-spin mr-1.5" />
              ) : null}
              Apply Config
            </Button>
            {(configureMutation.isSuccess || configureMutation.isError) && (
              <p
                className={cn(
                  "text-xs font-mono",
                  configureMutation.isError ? "text-red-400" : "text-green-400"
                )}
              >
                {configureMutation.isError
                  ? "Error (mock mode)"
                  : "Config applied"}
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Row 3 — Signal History */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {/* Chart */}
        <Card className="bg-card border-border">
          <CardHeader className="pb-2 pt-4 px-4">
            <CardTitle className="text-sm font-sans font-semibold text-muted-foreground uppercase tracking-wide flex items-center gap-2">
              <BarChart3 size={14} />
              Signal Confidence (last 50)
            </CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-4">
            <div className="flex gap-4 text-xs font-mono mb-2">
              <span className="text-green-400">● BUY</span>
              <span className="text-slate-400">● HOLD</span>
              <span className="text-red-400">● SELL</span>
            </div>
            <SignalChart signals={signals} />
          </CardContent>
        </Card>

        {/* Signal Table */}
        <Card className="bg-card border-border">
          <CardHeader className="pb-2 pt-4 px-4">
            <CardTitle className="text-sm font-sans font-semibold text-muted-foreground uppercase tracking-wide">
              Recent Signals
            </CardTitle>
          </CardHeader>
          <CardContent className="px-0 pb-4">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow className="border-border hover:bg-transparent">
                    {["Time", "Action", "Conf", "Regime", "Kelly %", "Price"].map(
                      (h) => (
                        <TableHead
                          key={h}
                          className="text-xs font-sans text-muted-foreground py-2 px-3 whitespace-nowrap"
                        >
                          {h}
                        </TableHead>
                      )
                    )}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {signals.slice(0, 20).map((s, i) => (
                    <TableRow
                      key={i}
                      className="border-border hover:bg-accent/40"
                    >
                      <TableCell className="text-xs font-mono text-muted-foreground py-1.5 px-3 whitespace-nowrap">
                        {fmt(s.timestamp)}
                      </TableCell>
                      <TableCell className="py-1.5 px-3">
                        <span
                          className={cn(
                            "text-xs font-mono font-bold",
                            s.action === "BUY"
                              ? "text-green-400"
                              : s.action === "SELL"
                              ? "text-red-400"
                              : "text-muted-foreground"
                          )}
                        >
                          {s.action}
                        </span>
                      </TableCell>
                      <TableCell className="text-xs font-mono text-foreground py-1.5 px-3">
                        {(s.confidence * 100).toFixed(0)}%
                      </TableCell>
                      <TableCell className="text-xs font-mono text-muted-foreground py-1.5 px-3 capitalize">
                        {s.regime}
                      </TableCell>
                      <TableCell className="text-xs font-mono text-foreground py-1.5 px-3">
                        {(s.kelly_fraction * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell className="text-xs font-mono text-foreground py-1.5 px-3">
                        ${s.price.toFixed(2)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
