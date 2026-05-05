import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import {
  Shield,
  Play,
  CheckCircle2,
  XCircle,
  ChevronDown,
  ChevronUp,
  Zap,
  Droplets,
  Target,
  TrendingDown,
  FlaskConical,
  AlertTriangle,
  Loader2,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import { RED_TEAM_SCENARIOS, type RedTeamScenarioResult } from "@/lib/mockData";

// ---- Types (kept for interface compatibility; no mock data generated) ----
// RED_TEAM_SCENARIOS is only the scenario metadata (IDs, icons, descriptions)
// — no generated results are imported or used.

const MARKETS = ["US", "HK"];
const TIMESCALES = ["10s", "1m", "5m", "1h"];

// ---- Engine offline banner ----
function EngineOfflineBanner({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-3 rounded-md border border-destructive/40 bg-destructive/10 px-4 py-3 text-xs text-destructive font-mono">
      <AlertTriangle size={14} className="shrink-0" />
      <span className="font-sans">{message}</span>
      <code className="ml-auto bg-destructive/10 px-2 py-0.5 rounded text-[10px]">
        uvicorn api_server:app --port 8001 --reload
      </code>
    </div>
  );
}

// ---- Scenario icons ----
const SCENARIO_ICONS: Record<string, React.ReactNode> = {
  flash_crash: <Zap size={20} />,
  liquidity_drought: <Droplets size={20} />,
  adverse_selection: <Target size={20} />,
  regime_change: <TrendingDown size={20} />,
  overfitting: <FlaskConical size={20} />,
};

// ---- Scenario Card ----
function ScenarioCard({
  scenario,
  checked,
  onToggle,
  result,
}: {
  scenario: (typeof RED_TEAM_SCENARIOS)[0];
  checked: boolean;
  onToggle: (id: string) => void;
  result?: RedTeamScenarioResult;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <Card
      data-testid={`scenario-card-${scenario.id}`}
      className={cn(
        "bg-card border transition-all duration-200",
        result
          ? result.passed
            ? "border-green-500/40"
            : "border-destructive/40"
          : checked
          ? "border-primary/40"
          : "border-border"
      )}
    >
      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          {/* Checkbox / Result Icon */}
          <div className="mt-0.5 shrink-0">
            {result ? (
              result.passed ? (
                <CheckCircle2 size={20} className="text-success" />
              ) : (
                <XCircle size={20} className="text-destructive" />
              )
            ) : (
              <Checkbox
                data-testid={`checkbox-${scenario.id}`}
                checked={checked}
                onCheckedChange={() => onToggle(scenario.id)}
                className="border-border data-[state=checked]:bg-primary data-[state=checked]:border-primary"
              />
            )}
          </div>

          {/* Scenario icon */}
          <div
            className={cn(
              "shrink-0 p-2 rounded-md",
              result
                ? result.passed
                  ? "bg-success/10 text-success"
                  : "bg-destructive/10 text-destructive"
                : "bg-muted text-muted-foreground"
            )}
          >
            {SCENARIO_ICONS[scenario.id]}
          </div>

          {/* Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between gap-2">
              <h3 className="text-sm font-sans font-medium text-foreground">{scenario.name}</h3>
              {result && (
                <Badge
                  className={cn(
                    "text-xs font-mono shrink-0",
                    result.passed
                      ? "bg-success/10 text-success border-success/30"
                      : "bg-destructive/10 text-destructive border-destructive/30"
                  )}
                  variant="outline"
                >
                  {result.passed ? "PASSED" : "FAILED"}
                </Badge>
              )}
            </div>
            <p className="text-xs text-muted-foreground mt-0.5 font-sans">{scenario.description}</p>
            {result && (
              <p className="text-xs font-mono text-foreground mt-1.5 font-medium">{result.metric}</p>
            )}
          </div>
        </div>

        {/* Expandable detail */}
        {result && (
          <div className="mt-3">
            <button
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors font-sans"
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              {expanded ? "Hide detail" : "Show detail"}
            </button>
            {expanded && (
              <p className="mt-2 text-xs text-muted-foreground font-sans leading-relaxed border-l-2 border-border pl-3">
                {result.detail}
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ---- History Row ----
function HistoryRow({
  run,
}: {
  run: { date: string; model: string; passed: number; total: number };
}) {
  const pct = (run.passed / run.total) * 100;
  return (
    <tr className="border-b border-border/50 hover:bg-accent/20 transition-colors">
      <td className="px-3 py-2 text-xs font-mono text-muted-foreground">{run.date}</td>
      <td className="px-3 py-2 text-xs font-mono text-foreground">{run.model}</td>
      <td className="px-3 py-2">
        <div className="flex items-center gap-2">
          <Progress value={pct} className="h-1.5 w-20" />
          <span
            className={cn(
              "text-xs font-mono font-semibold",
              pct >= 60 ? "text-success" : "text-destructive"
            )}
          >
            {run.passed}/{run.total}
          </span>
        </div>
      </td>
      <td className="px-3 py-2">
        <Badge
          variant="outline"
          className={cn(
            "text-xs",
            pct >= 60
              ? "text-success border-success/30 bg-success/10"
              : "text-destructive border-destructive/30 bg-destructive/10"
          )}
        >
          {pct >= 60 ? "PASS" : "FAIL"}
        </Badge>
      </td>
    </tr>
  );
}

export default function RedTeamTesting() {
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedMarket, setSelectedMarket] = useState("US");
  const [selectedTimescale, setSelectedTimescale] = useState("1m");
  const [selectedScenarios, setSelectedScenarios] = useState<string[]>(
    RED_TEAM_SCENARIOS.map((s) => s.id)
  );
  const [results, setResults] = useState<RedTeamScenarioResult[] | null>(null);
  const [engineError, setEngineError] = useState<string | null>(null);

  // Fetch real model list from engine
  const { data: modelsData } = useQuery<{ models: string[] }>({
    queryKey: ["/api/models"],
    retry: false,
  });
  const modelList = modelsData?.models ?? [];
  // Auto-select first model once list loads
  useEffect(() => {
    if (modelList.length > 0 && !selectedModel) {
      setSelectedModel(modelList[0]);
    }
  }, [modelList.length, selectedModel]);
  const resolvedModel = selectedModel || modelList[0] || "— no models loaded —";

  // Fetch historical red team runs from DB
  const { data: historyData } = useQuery<{
    results: { date: string; model: string; passed: number; total: number }[];
  }>({
    queryKey: ["/api/redteam/results/all"],
    retry: false,
  });
  const historyRuns = historyData?.results ?? [];

  // Run red team mutation — real API, no silent mock fallback
  const runMutation = useMutation({
    mutationFn: async () => {
      // Use raw fetch to inspect ENGINE_OFFLINE before throwing
      const res = await fetch("/api/redteam/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          market: selectedMarket.toLowerCase(),
          timescale: selectedTimescale,
          modelPath: resolvedModel,
          scenarios: selectedScenarios,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        if (data?.error === "ENGINE_OFFLINE" || res.status === 503) throw new Error("ENGINE_OFFLINE");
        throw new Error(data?.message ?? `HTTP ${res.status}`);
      }
      return data as { results: RedTeamScenarioResult[] };
    },
    onSuccess: (data) => {
      if (data?.results && data.results.length > 0) {
        // Normalise: Python may return `scenario` or `scenarioId`
        const normalized = data.results.map((r: any) => ({
          ...r,
          scenarioId: r.scenarioId ?? r.scenario,
        })) as RedTeamScenarioResult[];
        setResults(normalized);
        setEngineError(null);
      } else {
        setEngineError("Engine returned empty results — check if the model file exists and the engine is running.");
      }
    },
    onError: (err: Error) => {
      if (err.message === "ENGINE_OFFLINE") {
        setEngineError(
          "Python engine is offline. Start it with the command on the right, then retry."
        );
      } else {
        setEngineError(`Red team run failed: ${err.message}`);
      }
    },
  });

  const toggleScenario = (id: string) => {
    setSelectedScenarios((prev) =>
      prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]
    );
  };

  const passedCount = results?.filter((r) => r.passed).length ?? 0;
  const totalCount = results?.length ?? 0;
  const passRate = totalCount > 0 ? (passedCount / totalCount) * 100 : 0;

  return (
    <div className="flex flex-col h-full overflow-hidden" data-testid="page-red-team">
      {/* Header */}
      <div className="border-b border-border px-6 py-3 flex items-center justify-between shrink-0">
        <h1 className="text-sm font-sans font-semibold text-foreground flex items-center gap-2">
          <Shield size={16} className="text-primary" />
          Red Team Testing
        </h1>
        <span className="text-xs text-muted-foreground font-mono">
          Adversarial stress tests for RL trading models
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Controls */}
        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex flex-wrap items-end gap-3">
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground font-sans">Model</label>
                <Select value={resolvedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger
                    data-testid="redteam-model-select"
                    className="h-8 text-xs font-mono bg-muted border-border w-44"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border">
                    {modelList.length === 0 ? (
                      <SelectItem value="— no models loaded —" className="text-xs font-mono text-muted-foreground">
                        — no models loaded —
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
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground font-sans">Market</label>
                <Select value={selectedMarket} onValueChange={setSelectedMarket}>
                  <SelectTrigger
                    data-testid="redteam-market-select"
                    className="h-8 text-xs font-mono bg-muted border-border w-24"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border">
                    {MARKETS.map((m) => (
                      <SelectItem key={m} value={m} className="text-xs font-mono">
                        {m}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground font-sans">Timescale</label>
                <Select value={selectedTimescale} onValueChange={setSelectedTimescale}>
                  <SelectTrigger
                    data-testid="redteam-timescale-select"
                    className="h-8 text-xs font-mono bg-muted border-border w-24"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border">
                    {TIMESCALES.map((t) => (
                      <SelectItem key={t} value={t} className="text-xs font-mono">
                        {t}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2 ml-auto">
                <Button
                  data-testid="btn-select-all"
                  variant="outline"
                  size="sm"
                  className="h-8 text-xs border-border bg-muted hover:bg-accent"
                  onClick={() =>
                    setSelectedScenarios(
                      selectedScenarios.length === RED_TEAM_SCENARIOS.length
                        ? []
                        : RED_TEAM_SCENARIOS.map((s) => s.id)
                    )
                  }
                >
                  {selectedScenarios.length === RED_TEAM_SCENARIOS.length
                    ? "Deselect All"
                    : "Select All"}
                </Button>
                <Button
                  data-testid="btn-run-red-team"
                  size="sm"
                  className="h-8 text-xs bg-primary hover:bg-primary/90 text-primary-foreground"
                  onClick={() => {
                    setResults(null);
                    setEngineError(null);
                    runMutation.mutate();
                  }}
                  disabled={runMutation.isPending || selectedScenarios.length === 0}
                >
                  {runMutation.isPending ? (
                    <Loader2 size={12} className="mr-1.5 animate-spin" />
                  ) : (
                    <Play size={12} className="mr-1.5" />
                  )}
                  {runMutation.isPending ? "Running..." : "Run Red Team Tests"}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Engine offline banner */}
        {engineError && <EngineOfflineBanner message={engineError} />}

        {/* Scenarios Grid */}
        <div>
          <h2 className="text-xs font-sans font-semibold text-muted-foreground uppercase tracking-widest mb-3">
            Scenarios
          </h2>
          {runMutation.isPending ? (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
              {RED_TEAM_SCENARIOS.map((s) => (
                <Skeleton key={s.id} className="h-28 rounded-lg" />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
              {RED_TEAM_SCENARIOS.map((scenario) => (
                <ScenarioCard
                  key={scenario.id}
                  scenario={scenario}
                  checked={selectedScenarios.includes(scenario.id)}
                  onToggle={toggleScenario}
                  result={results?.find((r) => r.scenarioId === scenario.id)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Summary Bar */}
        {results && (
          <Card className="bg-card border-border" data-testid="red-team-summary">
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h3 className="text-sm font-sans font-medium text-foreground">Test Summary</h3>
                  <p className="text-xs text-muted-foreground font-sans">
                    Model:{" "}
                    <span className="font-mono text-foreground">{resolvedModel}</span>
                    {" · "}Market:{" "}
                    <span className="font-mono text-foreground">{selectedMarket}</span>
                    {" · "}Timescale:{" "}
                    <span className="font-mono text-foreground">{selectedTimescale}</span>
                  </p>
                </div>
                <div className="text-right">
                  <span
                    className={cn(
                      "text-2xl font-mono font-bold",
                      passRate >= 60 ? "text-success" : "text-destructive"
                    )}
                  >
                    {passedCount}/{totalCount}
                  </span>
                  <p className="text-xs text-muted-foreground font-sans">scenarios passed</p>
                </div>
              </div>

              <div className="space-y-2">
                {RED_TEAM_SCENARIOS.filter((s) => results.find((r) => r.scenarioId === s.id)).map(
                  (scenario) => {
                    const result = results.find((r) => r.scenarioId === scenario.id)!;
                    return (
                      <div key={scenario.id} className="flex items-center gap-3 text-xs">
                        <span className="w-36 font-sans text-muted-foreground truncate">
                          {scenario.name}
                        </span>
                        <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                          <div
                            className={cn(
                              "h-full rounded-full",
                              result.passed ? "bg-success" : "bg-destructive"
                            )}
                            style={{ width: "100%" }}
                          />
                        </div>
                        <span
                          className={cn(
                            "font-mono font-semibold w-12 text-right",
                            result.passed ? "text-success" : "text-destructive"
                          )}
                        >
                          {result.passed ? "PASS" : "FAIL"}
                        </span>
                      </div>
                    );
                  }
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Historical Runs — from DB, not mock */}
        <div>
          <h2 className="text-xs font-sans font-semibold text-muted-foreground uppercase tracking-widest mb-3">
            Historical Red Team Runs
          </h2>
          <Card className="bg-card border-border">
            {historyRuns.length === 0 ? (
              <div className="flex flex-col items-center py-10 text-center">
                <Shield size={28} className="text-muted-foreground opacity-30 mb-2" />
                <p className="text-xs text-muted-foreground font-sans">
                  No historical runs yet — complete a red team test to see results here.
                </p>
              </div>
            ) : (
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left px-3 py-2 text-xs font-sans font-medium text-muted-foreground">
                      Date
                    </th>
                    <th className="text-left px-3 py-2 text-xs font-sans font-medium text-muted-foreground">
                      Model
                    </th>
                    <th className="text-left px-3 py-2 text-xs font-sans font-medium text-muted-foreground">
                      Score
                    </th>
                    <th className="text-left px-3 py-2 text-xs font-sans font-medium text-muted-foreground">
                      Result
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {historyRuns.map((run, i) => (
                    <HistoryRow key={i} run={run} />
                  ))}
                </tbody>
              </table>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
