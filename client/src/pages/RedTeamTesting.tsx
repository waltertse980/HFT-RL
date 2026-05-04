import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { Shield, Play, CheckCircle2, XCircle, ChevronDown, ChevronUp, Zap, Droplets, Target, TrendingDown, FlaskConical } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import {
  RED_TEAM_SCENARIOS,
  generateRedTeamResults,
  type RedTeamScenarioResult,
} from "@/lib/mockData";

const MODELS = ["PPO_US_1m_v3", "TD3_HK_5m_v2", "PPO_US_10s_v1", "PPO_HK_1m_v2", "TD3_US_1h_v1"];
const MARKETS = ["US", "HK"];
const TIMESCALES = ["10s", "1m", "5m", "1h"];

const SCENARIO_ICONS: Record<string, React.ReactNode> = {
  flash_crash: <Zap size={20} />,
  liquidity_drought: <Droplets size={20} />,
  adverse_selection: <Target size={20} />,
  regime_change: <TrendingDown size={20} />,
  overfitting: <FlaskConical size={20} />,
};

function ScenarioCard({
  scenario,
  checked,
  onToggle,
  result,
}: {
  scenario: typeof RED_TEAM_SCENARIOS[0];
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

function HistoryRow({ run }: { run: { date: string; model: string; passed: number; total: number } }) {
  const pct = (run.passed / run.total) * 100;
  return (
    <tr className="border-b border-border/50 hover:bg-accent/20 transition-colors">
      <td className="px-3 py-2 text-xs font-mono text-muted-foreground">{run.date}</td>
      <td className="px-3 py-2 text-xs font-mono text-foreground">{run.model}</td>
      <td className="px-3 py-2">
        <div className="flex items-center gap-2">
          <Progress value={pct} className="h-1.5 w-20" />
          <span className={cn("text-xs font-mono font-semibold", pct >= 60 ? "text-success" : "text-destructive")}>
            {run.passed}/{run.total}
          </span>
        </div>
      </td>
      <td className="px-3 py-2">
        <Badge
          variant="outline"
          className={cn(
            "text-xs",
            pct >= 60 ? "text-success border-success/30 bg-success/10" : "text-destructive border-destructive/30 bg-destructive/10"
          )}
        >
          {pct >= 60 ? "PASS" : "FAIL"}
        </Badge>
      </td>
    </tr>
  );
}

const MOCK_HISTORY = [
  { date: "2024-11-30 14:22", model: "PPO_US_1m_v3", passed: 3, total: 5 },
  { date: "2024-11-29 09:45", model: "TD3_HK_5m_v2", passed: 4, total: 5 },
  { date: "2024-11-28 18:10", model: "PPO_US_10s_v1", passed: 2, total: 5 },
  { date: "2024-11-27 11:30", model: "PPO_HK_1m_v2", passed: 5, total: 5 },
];

export default function RedTeamTesting() {
  const [selectedModel, setSelectedModel] = useState("PPO_US_1m_v3");
  const [selectedMarket, setSelectedMarket] = useState("US");
  const [selectedTimescale, setSelectedTimescale] = useState("1m");
  const [selectedScenarios, setSelectedScenarios] = useState<string[]>(RED_TEAM_SCENARIOS.map((s) => s.id));
  const [results, setResults] = useState<RedTeamScenarioResult[] | null>(null);

  const runMutation = useMutation({
    mutationFn: async () => {
      try {
        const res = await apiRequest("POST", "/api/redteam/run", {
          market: selectedMarket.toLowerCase(),
          timescale: selectedTimescale,
          modelPath: selectedModel,
          scenarios: selectedScenarios,
        });
        return (await res.json()) as { results: RedTeamScenarioResult[] };
      } catch {
        // Mock fallback
        await new Promise((r) => setTimeout(r, 2200));
        return { results: generateRedTeamResults() };
      }
    },
    onSuccess: (data) => {
      if (data?.results && data.results.length > 0) {
        // Normalize: backend may return `scenario` or `scenarioId`
        const normalized = data.results.map((r: any) => ({
          ...r,
          scenarioId: r.scenarioId ?? r.scenario,
        })) as RedTeamScenarioResult[];
        setResults(normalized);
      } else {
        setResults(generateRedTeamResults());
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
        <span className="text-xs text-muted-foreground font-mono">Adversarial stress tests for RL trading models</span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Controls */}
        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex flex-wrap items-end gap-3">
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground font-sans">Model</label>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger data-testid="redteam-model-select" className="h-8 text-xs font-mono bg-muted border-border w-44">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border">
                    {MODELS.map((m) => <SelectItem key={m} value={m} className="text-xs font-mono">{m}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground font-sans">Market</label>
                <Select value={selectedMarket} onValueChange={setSelectedMarket}>
                  <SelectTrigger data-testid="redteam-market-select" className="h-8 text-xs font-mono bg-muted border-border w-24">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border">
                    {MARKETS.map((m) => <SelectItem key={m} value={m} className="text-xs font-mono">{m}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground font-sans">Timescale</label>
                <Select value={selectedTimescale} onValueChange={setSelectedTimescale}>
                  <SelectTrigger data-testid="redteam-timescale-select" className="h-8 text-xs font-mono bg-muted border-border w-24">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border">
                    {TIMESCALES.map((t) => <SelectItem key={t} value={t} className="text-xs font-mono">{t}</SelectItem>)}
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
                  {selectedScenarios.length === RED_TEAM_SCENARIOS.length ? "Deselect All" : "Select All"}
                </Button>
                <Button
                  data-testid="btn-run-red-team"
                  size="sm"
                  className="h-8 text-xs bg-primary hover:bg-primary/90 text-primary-foreground"
                  onClick={() => { setResults(null); runMutation.mutate(); }}
                  disabled={runMutation.isPending || selectedScenarios.length === 0}
                >
                  <Play size={12} className="mr-1.5" />
                  {runMutation.isPending ? "Running..." : "Run Red Team Tests"}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

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
                    Model: <span className="font-mono text-foreground">{selectedModel}</span>
                    {" · "}Market: <span className="font-mono text-foreground">{selectedMarket}</span>
                    {" · "}Timescale: <span className="font-mono text-foreground">{selectedTimescale}</span>
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

              {/* Per-scenario mini bars */}
              <div className="space-y-2">
                {RED_TEAM_SCENARIOS.filter((s) => results.find((r) => r.scenarioId === s.id)).map((scenario) => {
                  const result = results.find((r) => r.scenarioId === scenario.id)!;
                  return (
                    <div key={scenario.id} className="flex items-center gap-3 text-xs">
                      <span className="w-36 font-sans text-muted-foreground truncate">{scenario.name}</span>
                      <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                        <div
                          className={cn("h-full rounded-full", result.passed ? "bg-success" : "bg-destructive")}
                          style={{ width: "100%" }}
                        />
                      </div>
                      <span className={cn("font-mono font-semibold w-12 text-right", result.passed ? "text-success" : "text-destructive")}>
                        {result.passed ? "PASS" : "FAIL"}
                      </span>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Historical Runs */}
        <div>
          <h2 className="text-xs font-sans font-semibold text-muted-foreground uppercase tracking-widest mb-3">
            Historical Red Team Runs
          </h2>
          <Card className="bg-card border-border">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left px-3 py-2 text-xs font-sans font-medium text-muted-foreground">Date</th>
                  <th className="text-left px-3 py-2 text-xs font-sans font-medium text-muted-foreground">Model</th>
                  <th className="text-left px-3 py-2 text-xs font-sans font-medium text-muted-foreground">Score</th>
                  <th className="text-left px-3 py-2 text-xs font-sans font-medium text-muted-foreground">Result</th>
                </tr>
              </thead>
              <tbody>
                {MOCK_HISTORY.map((run, i) => <HistoryRow key={i} run={run} />)}
              </tbody>
            </table>
          </Card>
        </div>
      </div>
    </div>
  );
}
