import { useState, useEffect, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Settings as SettingsIcon,
  CheckCircle2,
  AlertCircle,
  Eye,
  EyeOff,
  Save,
  TestTube,
  AlertTriangle,
  RefreshCw,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";
import type { TradingSettings } from "@shared/schema";

// ─── Constants ────────────────────────────────────────────────────────────────

const TIMESCALES = ["10s", "1m", "5m", "1h"];
const US_TICKERS = ["AAPL", "NVDA", "MSFT", "META", "GOOGL", "TSLA", "SPY", "QQQ"];
const HK_TICKERS = ["0700.HK", "9988.HK", "0005.HK", "2318.HK", "1299.HK"];

// ─── Types ────────────────────────────────────────────────────────────────────

interface MarketFormState {
  id: number | null;
  ticker: string;
  timescale: string;
  maxPositionPct: number; // stored as 0–100 in UI, 0–1 in DB
  stopLossPct: number;
  dailyLossLimitPct: number;
  alpacaApiKey: string;
  alpacaApiSecret: string;
}

type TestConnectionResult =
  | { status: "idle" }
  | { status: "testing" }
  | { status: "missing_creds" }
  | { status: "engine_offline" }
  | { status: "success"; accountStatus: string; buyingPower: number; portfolioValue: number }
  | { status: "error"; message: string };

// ─── Helpers ──────────────────────────────────────────────────────────────────

function fmtCurrency(n: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(n);
}

function defaultUsForm(): MarketFormState {
  return {
    id: null,
    ticker: "AAPL",
    timescale: "1m",
    maxPositionPct: 95,
    stopLossPct: 2,
    dailyLossLimitPct: 5,
    alpacaApiKey: "",
    alpacaApiSecret: "",
  };
}

function defaultHkForm(): MarketFormState {
  return {
    id: null,
    ticker: "0700.HK",
    timescale: "1m",
    maxPositionPct: 95,
    stopLossPct: 2,
    dailyLossLimitPct: 5,
    alpacaApiKey: "",
    alpacaApiSecret: "",
  };
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function SliderField({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
  testId,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  onChange: (v: number) => void;
  testId?: string;
}) {
  return (
    <div className="space-y-2" data-testid={testId}>
      <div className="flex items-center justify-between">
        <Label className="text-xs font-sans text-muted-foreground">{label}</Label>
        <span className="text-xs font-mono text-foreground font-semibold">{format(value)}</span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        className="w-full"
      />
      <div className="flex justify-between text-xs font-mono text-muted-foreground">
        <span>{format(min)}</span>
        <span>{format(max)}</span>
      </div>
    </div>
  );
}

function SecretInput({
  value,
  onChange,
  placeholder,
  testId,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  testId?: string;
}) {
  const [show, setShow] = useState(false);
  return (
    <div className="relative">
      <Input
        data-testid={testId}
        type={show ? "text" : "password"}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="h-8 text-xs font-mono bg-muted border-border pr-8"
      />
      <button
        type="button"
        className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
        onClick={() => setShow(!show)}
      >
        {show ? <EyeOff size={14} /> : <Eye size={14} />}
      </button>
    </div>
  );
}

function StatusRow({
  label,
  value,
  ok,
  mono = false,
}: {
  label: string;
  value: string;
  ok: boolean;
  mono?: boolean;
}) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-border/50 last:border-0">
      <span className="text-xs font-sans text-muted-foreground">{label}</span>
      <div className="flex items-center gap-2">
        <span className={cn("text-xs", mono ? "font-mono" : "font-sans", "text-foreground")}>
          {value}
        </span>
        {ok ? (
          <CheckCircle2 size={12} className="text-green-500" />
        ) : (
          <AlertCircle size={12} className="text-destructive" />
        )}
      </div>
    </div>
  );
}

function TestConnectionResultCard({ result }: { result: TestConnectionResult }) {
  if (result.status === "idle" || result.status === "testing") return null;

  if (result.status === "missing_creds") {
    return (
      <div className="mt-3 p-3 rounded-md border border-yellow-500/40 bg-yellow-500/10 flex items-start gap-2">
        <AlertTriangle size={13} className="text-yellow-400 mt-0.5 shrink-0" />
        <p className="text-xs font-sans text-yellow-300">
          Enter your API key and secret first.
        </p>
      </div>
    );
  }

  if (result.status === "engine_offline") {
    return (
      <div className="mt-3 p-3 rounded-md border border-destructive/40 bg-destructive/10 flex items-start gap-2">
        <AlertCircle size={13} className="text-destructive mt-0.5 shrink-0" />
        <div>
          <p className="text-xs font-sans text-destructive font-semibold">Python engine offline</p>
          <p className="text-xs font-sans text-muted-foreground mt-0.5">
            Cannot test connection. Start the engine first:{" "}
            <code className="font-mono text-foreground">
              uvicorn api_server:app --port 8001
            </code>
          </p>
        </div>
      </div>
    );
  }

  if (result.status === "error") {
    return (
      <div className="mt-3 p-3 rounded-md border border-destructive/40 bg-destructive/10 flex items-start gap-2">
        <AlertCircle size={13} className="text-destructive mt-0.5 shrink-0" />
        <p className="text-xs font-sans text-destructive">{result.message}</p>
      </div>
    );
  }

  // success
  return (
    <div className="mt-3 p-3 rounded-md border border-green-500/40 bg-green-500/10 flex items-start gap-2">
      <CheckCircle2 size={13} className="text-green-400 mt-0.5 shrink-0" />
      <div className="space-y-0.5">
        <p className="text-xs font-sans text-green-300 font-semibold">
          Connected — Account: {result.accountStatus}
        </p>
        <p className="text-xs font-mono text-muted-foreground">
          Buying Power: {fmtCurrency(result.buyingPower)} · Portfolio:{" "}
          {fmtCurrency(result.portfolioValue)}
        </p>
      </div>
    </div>
  );
}

// ─── US Market Tab ─────────────────────────────────────────────────────────────

function USMarketTab({
  form,
  setForm,
  onSave,
  isSaving,
}: {
  form: MarketFormState;
  setForm: React.Dispatch<React.SetStateAction<MarketFormState>>;
  onSave: () => void;
  isSaving: boolean;
}) {
  const [testResult, setTestResult] = useState<TestConnectionResult>({ status: "idle" });

  const testConnection = useCallback(async () => {
    if (!form.alpacaApiKey || !form.alpacaApiSecret) {
      setTestResult({ status: "missing_creds" });
      return;
    }
    setTestResult({ status: "testing" });
    try {
      const res = await fetch("/api/settings/test-connection", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          api_key: form.alpacaApiKey,
          api_secret: form.alpacaApiSecret,
        }),
      });

      if (res.status === 503) {
        setTestResult({ status: "engine_offline" });
        return;
      }

      if (res.status === 400) {
        setTestResult({ status: "missing_creds" });
        return;
      }

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setTestResult({
          status: "error",
          message: body?.message ?? `Connection failed (HTTP ${res.status})`,
        });
        return;
      }

      const data = await res.json();

      if (!data?.connected) {
        setTestResult({
          status: "error",
          message: data?.message ?? "Alpaca rejected the credentials.",
        });
        return;
      }

      setTestResult({
        status: "success",
        accountStatus: data.account_status ?? "Active",
        buyingPower: data.buying_power ?? 0,
        portfolioValue: data.portfolio_value ?? 0,
      });
    } catch {
      setTestResult({ status: "error", message: "Network error — could not reach the server." });
    }
  }, [form.alpacaApiKey, form.alpacaApiSecret]);

  return (
    <div className="space-y-5 pt-4">
      {/* Defaults */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1.5">
          <Label className="text-xs font-sans text-muted-foreground">Default Ticker</Label>
          <Select
            value={form.ticker}
            onValueChange={(v) => setForm((s) => ({ ...s, ticker: v }))}
          >
            <SelectTrigger
              data-testid="select-ticker-us"
              className="h-8 text-xs font-mono bg-muted border-border"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-popover border-border">
              {US_TICKERS.map((t) => (
                <SelectItem key={t} value={t} className="text-xs font-mono">
                  {t}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label className="text-xs font-sans text-muted-foreground">Default Timescale</Label>
          <Select
            value={form.timescale}
            onValueChange={(v) => setForm((s) => ({ ...s, timescale: v }))}
          >
            <SelectTrigger
              data-testid="select-timescale-us"
              className="h-8 text-xs font-mono bg-muted border-border"
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
      </div>

      {/* Risk sliders */}
      <div className="space-y-5 p-4 bg-muted/40 rounded-lg border border-border/50">
        <h3 className="text-xs font-sans font-semibold text-foreground uppercase tracking-widest">
          Risk Parameters
        </h3>
        <SliderField
          label="Max Position Size"
          value={form.maxPositionPct}
          min={10}
          max={100}
          step={5}
          format={(v) => `${v}%`}
          onChange={(v) => setForm((s) => ({ ...s, maxPositionPct: v }))}
          testId="slider-max-position-us"
        />
        <SliderField
          label="Stop Loss"
          value={form.stopLossPct}
          min={0.5}
          max={10}
          step={0.5}
          format={(v) => `${v}%`}
          onChange={(v) => setForm((s) => ({ ...s, stopLossPct: v }))}
          testId="slider-stop-loss-us"
        />
        <SliderField
          label="Daily Loss Limit"
          value={form.dailyLossLimitPct}
          min={1}
          max={20}
          step={1}
          format={(v) => `${v}%`}
          onChange={(v) => setForm((s) => ({ ...s, dailyLossLimitPct: v }))}
          testId="slider-daily-loss-us"
        />
      </div>

      {/* Alpaca credentials */}
      <div className="space-y-3 p-4 bg-muted/40 rounded-lg border border-border/50">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-sans font-semibold text-foreground uppercase tracking-widest">
            Alpaca API Credentials
          </h3>
          <Badge
            variant="outline"
            className="text-xs border-yellow-500/40 text-yellow-400 bg-yellow-500/10"
          >
            Paper Trading
          </Badge>
        </div>
        <p className="text-xs text-muted-foreground font-sans">
          Used for US market paper trading via Alpaca's WebSocket feed. Get free API keys at{" "}
          <span className="text-primary font-mono">alpaca.markets</span>.
        </p>
        <div className="space-y-2">
          <Label className="text-xs font-sans text-muted-foreground">API Key</Label>
          <SecretInput
            value={form.alpacaApiKey}
            onChange={(v) => setForm((s) => ({ ...s, alpacaApiKey: v }))}
            placeholder="PK••••••••••••••••••••"
            testId="input-alpaca-key"
          />
        </div>
        <div className="space-y-2">
          <Label className="text-xs font-sans text-muted-foreground">API Secret</Label>
          <SecretInput
            value={form.alpacaApiSecret}
            onChange={(v) => setForm((s) => ({ ...s, alpacaApiSecret: v }))}
            placeholder="••••••••••••••••••••••••••••••••••••••••"
            testId="input-alpaca-secret"
          />
        </div>
        <Button
          data-testid="btn-test-connection"
          variant="outline"
          size="sm"
          className="h-8 text-xs border-border bg-muted hover:bg-accent w-full"
          onClick={testConnection}
          disabled={testResult.status === "testing"}
        >
          {testResult.status === "testing" ? (
            <>
              <RefreshCw size={12} className="mr-1.5 animate-spin" />
              Testing...
            </>
          ) : (
            <>
              <TestTube size={12} className="mr-1.5" />
              Test Connection
            </>
          )}
        </Button>
        <TestConnectionResultCard result={testResult} />
      </div>

      {/* Save */}
      <Button
        data-testid="btn-save-us"
        className="w-full h-9 text-xs bg-primary hover:bg-primary/90 text-primary-foreground"
        onClick={onSave}
        disabled={isSaving}
      >
        <Save size={14} className="mr-1.5" />
        {isSaving ? "Saving..." : "Save US Settings"}
      </Button>
    </div>
  );
}

// ─── HK Market Tab ─────────────────────────────────────────────────────────────

function HKMarketTab({
  form,
  setForm,
  onSave,
  isSaving,
}: {
  form: MarketFormState;
  setForm: React.Dispatch<React.SetStateAction<MarketFormState>>;
  onSave: () => void;
  isSaving: boolean;
}) {
  return (
    <div className="space-y-5 pt-4">
      {/* Defaults */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1.5">
          <Label className="text-xs font-sans text-muted-foreground">Default Ticker</Label>
          <Select
            value={form.ticker}
            onValueChange={(v) => setForm((s) => ({ ...s, ticker: v }))}
          >
            <SelectTrigger
              data-testid="select-ticker-hk"
              className="h-8 text-xs font-mono bg-muted border-border"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-popover border-border">
              {HK_TICKERS.map((t) => (
                <SelectItem key={t} value={t} className="text-xs font-mono">
                  {t}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label className="text-xs font-sans text-muted-foreground">Default Timescale</Label>
          <Select
            value={form.timescale}
            onValueChange={(v) => setForm((s) => ({ ...s, timescale: v }))}
          >
            <SelectTrigger
              data-testid="select-timescale-hk"
              className="h-8 text-xs font-mono bg-muted border-border"
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
      </div>

      {/* Risk sliders */}
      <div className="space-y-5 p-4 bg-muted/40 rounded-lg border border-border/50">
        <h3 className="text-xs font-sans font-semibold text-foreground uppercase tracking-widest">
          Risk Parameters
        </h3>
        <SliderField
          label="Max Position Size"
          value={form.maxPositionPct}
          min={10}
          max={100}
          step={5}
          format={(v) => `${v}%`}
          onChange={(v) => setForm((s) => ({ ...s, maxPositionPct: v }))}
          testId="slider-max-position-hk"
        />
        <SliderField
          label="Stop Loss"
          value={form.stopLossPct}
          min={0.5}
          max={10}
          step={0.5}
          format={(v) => `${v}%`}
          onChange={(v) => setForm((s) => ({ ...s, stopLossPct: v }))}
          testId="slider-stop-loss-hk"
        />
        <SliderField
          label="Daily Loss Limit"
          value={form.dailyLossLimitPct}
          min={1}
          max={20}
          step={1}
          format={(v) => `${v}%`}
          onChange={(v) => setForm((s) => ({ ...s, dailyLossLimitPct: v }))}
          testId="slider-daily-loss-hk"
        />
      </div>

      {/* HK-specific fields */}
      <div className="space-y-3 p-4 bg-muted/40 rounded-lg border border-border/50">
        <h3 className="text-xs font-sans font-semibold text-foreground uppercase tracking-widest">
          HK Market Details
        </h3>
        <p className="text-xs text-muted-foreground font-sans leading-relaxed">
          HK market paper trading uses local data replay — no live broker API required. The system
          replays held-out HKEX test data at 10× real speed to simulate live conditions.
        </p>
        <div className="grid grid-cols-2 gap-4 pt-1">
          <div className="space-y-1.5">
            <Label className="text-xs font-sans text-muted-foreground">Board Lot Size</Label>
            <Input
              type="number"
              defaultValue={500}
              className="h-8 text-xs font-mono bg-muted border-border"
              data-testid="input-board-lot"
            />
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs font-sans text-muted-foreground">Stamp Duty</Label>
            <div className="h-8 flex items-center px-3 rounded-md border border-border/50 bg-muted/30">
              <span className="text-xs font-mono text-muted-foreground">
                0.13% per side (fixed)
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Save */}
      <Button
        data-testid="btn-save-hk"
        className="w-full h-9 text-xs bg-primary hover:bg-primary/90 text-primary-foreground"
        onClick={onSave}
        disabled={isSaving}
      >
        <Save size={14} className="mr-1.5" />
        {isSaving ? "Saving..." : "Save HK Settings"}
      </Button>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function Settings() {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // ── Form state ──────────────────────────────────────────────────────────────
  const [usForm, setUsForm] = useState<MarketFormState>(defaultUsForm());
  const [hkForm, setHkForm] = useState<MarketFormState>(defaultHkForm());
  const [initialCapital, setInitialCapital] = useState<number>(100000);
  const [engineOfflineGlobal, setEngineOfflineGlobal] = useState(false);

  // Databento (LOB-HFT v2)
  const [databentoKey, setDatabentoKey] = useState<string>("");
  const [databentoConfigured, setDatabentoConfigured] = useState<boolean>(false);
  const [savingDatabento, setSavingDatabento] = useState<boolean>(false);
  const [databentoStatus, setDatabentoStatus] = useState<{ ok: boolean; msg: string } | null>(null);

  const saveDatabentoKey = async () => {
    if (!databentoKey.trim()) {
      setDatabentoStatus({ ok: false, msg: "API key is empty." });
      return;
    }
    setSavingDatabento(true);
    setDatabentoStatus(null);
    try {
      const res = await fetch("/api/databento", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: databentoKey }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setDatabentoStatus({ ok: false, msg: data?.detail || `HTTP ${res.status}` });
        setDatabentoConfigured(false);
      } else {
        setDatabentoStatus({ ok: true, msg: "Databento API key saved." });
        setDatabentoConfigured(true);
        toast({ title: "Databento API key saved" });
      }
    } catch (err) {
      setDatabentoStatus({ ok: false, msg: err instanceof Error ? err.message : String(err) });
      setDatabentoConfigured(false);
    } finally {
      setSavingDatabento(false);
    }
  };

  // ── Queries ─────────────────────────────────────────────────────────────────

  const { data: settingsData, isLoading: settingsLoading } = useQuery({
    queryKey: ["/api/settings"],
    queryFn: async () => {
      const res = await fetch("/api/settings");
      if (!res.ok) throw new Error("Failed to load settings");
      return res.json() as Promise<TradingSettings[]>;
    },
  });

  const { data: globalData } = useQuery({
    queryKey: ["/api/settings/global"],
    queryFn: async () => {
      const res = await fetch("/api/settings/global");
      if (res.status === 503) {
        setEngineOfflineGlobal(true);
        return null;
      }
      if (!res.ok) throw new Error("Failed to load global settings");
      setEngineOfflineGlobal(false);
      return res.json() as Promise<Record<string, unknown>>;
    },
    retry: false,
  });

  const { data: healthData } = useQuery({
    queryKey: ["/api/health"],
    queryFn: async () => {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error("Health check failed");
      return res.json() as Promise<{ status: string; pythonApi: string; timestamp: string }>;
    },
    refetchInterval: 5000,
    retry: false,
  });

  const { data: modelsData, isLoading: modelsLoading } = useQuery<{ models: string[] }>({
    queryKey: ["/api/models"],
    retry: false,
  });

  const { data: jobsData } = useQuery<{ jobs: { completedAt?: string; startedAt?: string }[] }>({
    queryKey: ["/api/training/jobs"],
    retry: false,
  });

  // ── Populate forms from server data ─────────────────────────────────────────

  useEffect(() => {
    if (!settingsData) return;
    const usSettings = settingsData.find((s) => s.market === "us");
    const hkSettings = settingsData.find((s) => s.market === "hk");

    if (usSettings) {
      setUsForm({
        id: usSettings.id,
        ticker: usSettings.ticker,
        timescale: usSettings.timescale,
        maxPositionPct: usSettings.maxPositionPct * 100,
        stopLossPct: usSettings.stopLossPct * 100,
        dailyLossLimitPct: usSettings.dailyLossLimitPct * 100,
        alpacaApiKey: usSettings.alpacaApiKey ?? "",
        alpacaApiSecret: usSettings.alpacaApiSecret ?? "",
      });
    }

    if (hkSettings) {
      setHkForm({
        id: hkSettings.id,
        ticker: hkSettings.ticker,
        timescale: hkSettings.timescale,
        maxPositionPct: hkSettings.maxPositionPct * 100,
        stopLossPct: hkSettings.stopLossPct * 100,
        dailyLossLimitPct: hkSettings.dailyLossLimitPct * 100,
        alpacaApiKey: hkSettings.alpacaApiKey ?? "",
        alpacaApiSecret: hkSettings.alpacaApiSecret ?? "",
      });
    }
  }, [settingsData]);

  useEffect(() => {
    if (globalData && typeof globalData.initial_capital === "number") {
      setInitialCapital(globalData.initial_capital);
    }
  }, [globalData]);

  // ── Save mutations ──────────────────────────────────────────────────────────

  const saveGlobalMutation = useMutation({
    mutationFn: async (capital: number) => {
      const res = await fetch("/api/settings/global", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ initial_capital: capital }),
      });
      if (res.status === 503) throw new Error("ENGINE_OFFLINE");
      if (!res.ok) throw new Error("Failed to save global settings");
      return res.json();
    },
  });

  const saveMarketMutation = useMutation({
    mutationFn: async ({
      form,
      market,
    }: {
      form: MarketFormState;
      market: "us" | "hk";
    }) => {
      const payload = {
        market,
        ticker: form.ticker,
        timescale: form.timescale,
        maxPositionPct: form.maxPositionPct / 100,
        stopLossPct: form.stopLossPct / 100,
        dailyLossLimitPct: form.dailyLossLimitPct / 100,
        alpacaApiKey: form.alpacaApiKey || null,
        alpacaApiSecret: form.alpacaApiSecret || null,
        isActive: false,
      };

      let res: Response;
      if (form.id !== null) {
        res = await fetch(`/api/settings/${form.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      } else {
        res = await fetch("/api/settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      }

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.error ?? "Failed to save settings");
      }
      return res.json() as Promise<TradingSettings>;
    },
    onSuccess: (data, { market }) => {
      // Update local id so next save uses PATCH
      if (market === "us") {
        setUsForm((f) => ({ ...f, id: data.id }));
      } else {
        setHkForm((f) => ({ ...f, id: data.id }));
      }
      queryClient.invalidateQueries({ queryKey: ["/api/settings"] });
    },
  });

  const handleSaveUs = useCallback(async () => {
    try {
      await Promise.all([
        saveMarketMutation.mutateAsync({ form: usForm, market: "us" }),
        saveGlobalMutation.mutateAsync(initialCapital),
      ]);
      toast({
        title: "US Settings saved",
        description: `Ticker: ${usForm.ticker} · Timescale: ${usForm.timescale} · Initial Capital: ${fmtCurrency(initialCapital)}`,
      });
    } catch (err) {
      const msg = (err as Error).message;
      toast({
        title: "Save failed",
        description: msg === "ENGINE_OFFLINE"
          ? "Global settings could not be saved — Python engine is offline."
          : msg,
        variant: "destructive",
      });
    }
  }, [usForm, initialCapital, saveMarketMutation, saveGlobalMutation, toast]);

  const handleSaveHk = useCallback(async () => {
    try {
      await Promise.all([
        saveMarketMutation.mutateAsync({ form: hkForm, market: "hk" }),
        saveGlobalMutation.mutateAsync(initialCapital),
      ]);
      toast({
        title: "HK Settings saved",
        description: `Ticker: ${hkForm.ticker} · Timescale: ${hkForm.timescale} · Initial Capital: ${fmtCurrency(initialCapital)}`,
      });
    } catch (err) {
      const msg = (err as Error).message;
      toast({
        title: "Save failed",
        description: msg === "ENGINE_OFFLINE"
          ? "Global settings could not be saved — Python engine is offline."
          : msg,
        variant: "destructive",
      });
    }
  }, [hkForm, initialCapital, saveMarketMutation, saveGlobalMutation, toast]);

  // ── Derived status ──────────────────────────────────────────────────────────

  const pythonConnected = healthData?.pythonApi?.startsWith("connected") ?? false;
  const lastJob = jobsData?.jobs?.[0];
  const lastTrained = lastJob?.completedAt ?? lastJob?.startedAt ?? "—";
  const isSavingUs = saveMarketMutation.isPending && saveMarketMutation.variables?.market === "us";
  const isSavingHk = saveMarketMutation.isPending && saveMarketMutation.variables?.market === "hk";

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full overflow-hidden" data-testid="page-settings">
      {/* Header */}
      <div className="border-b border-border px-6 py-3 flex items-center gap-2 shrink-0">
        <SettingsIcon size={16} className="text-primary" />
        <h1 className="text-sm font-sans font-semibold text-foreground">Settings</h1>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <div className="max-w-2xl space-y-4">

          {/* ── Global Settings ───────────────────────────────────────────── */}
          <Card className="bg-card border-border">
            <CardHeader className="py-3 px-4 border-b border-border">
              <CardTitle className="text-sm font-sans font-medium text-foreground">
                Global Settings
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4 space-y-4">
              {engineOfflineGlobal && (
                <div className="flex items-start gap-2 p-3 rounded-md border border-yellow-500/40 bg-yellow-500/10">
                  <AlertTriangle size={13} className="text-yellow-400 mt-0.5 shrink-0" />
                  <p className="text-xs font-sans text-yellow-300">
                    Initial Capital setting requires the Python engine. Start it, then reload.
                  </p>
                </div>
              )}
              <div className="space-y-1.5">
                <Label className="text-xs font-sans text-muted-foreground">
                  Initial Capital
                </Label>
                <div className="relative">
                  <span className="absolute left-3 top-1/2 -translate-y-1/2 text-xs font-mono text-muted-foreground pointer-events-none">
                    $
                  </span>
                  <Input
                    data-testid="input-initial-capital"
                    type="number"
                    min={1000}
                    step={1000}
                    value={initialCapital}
                    onChange={(e) => setInitialCapital(Number(e.target.value))}
                    className="h-8 text-xs font-mono bg-muted border-border pl-6"
                    disabled={engineOfflineGlobal}
                  />
                </div>
                <p className="text-xs text-muted-foreground font-sans">
                  Starting portfolio value used for P&L calculations and position sizing.
                </p>
              </div>
            </CardContent>
          </Card>

          {/* ── Market Configuration ──────────────────────────────────────── */}
          <Card className="bg-card border-border">
            <CardHeader className="py-3 px-4 border-b border-border">
              <CardTitle className="text-sm font-sans font-medium text-foreground">
                Market Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4">
              {settingsLoading ? (
                <div className="space-y-3 pt-2">
                  {[...Array(6)].map((_, i) => (
                    <Skeleton key={i} className="h-8 rounded" />
                  ))}
                </div>
              ) : (
                <Tabs defaultValue="us">
                  <TabsList className="w-full bg-muted h-8">
                    <TabsTrigger
                      value="us"
                      className="flex-1 text-xs"
                      data-testid="tab-us"
                    >
                      US Market
                    </TabsTrigger>
                    <TabsTrigger
                      value="hk"
                      className="flex-1 text-xs"
                      data-testid="tab-hk"
                    >
                      HK Market
                    </TabsTrigger>
                  </TabsList>
                  <TabsContent value="us">
                    <USMarketTab
                      form={usForm}
                      setForm={setUsForm}
                      onSave={handleSaveUs}
                      isSaving={isSavingUs}
                    />
                  </TabsContent>
                  <TabsContent value="hk">
                    <HKMarketTab
                      form={hkForm}
                      setForm={setHkForm}
                      onSave={handleSaveHk}
                      isSaving={isSavingHk}
                    />
                  </TabsContent>
                </Tabs>
              )}
            </CardContent>
          </Card>

          {/* Databento API Key (LOB-HFT v2) */}
          <Card className="bg-card border-border" data-testid="card-databento">
            <CardHeader className="py-3 px-4 border-b border-border">
              <CardTitle className="text-sm font-sans font-medium text-foreground flex items-center gap-2">
                Databento API Key
                <Badge
                  variant="outline"
                  className="text-[10px] border-blue-500/40 text-blue-400 bg-blue-500/10"
                >
                  LOB-HFT v2
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4 space-y-3">
              <p className="text-xs text-muted-foreground font-sans">
                Required for downloading Level-3 (MBO) order-book data used by the LOB-HFT v2
                strategy. Get a key at{" "}
                <span className="text-primary font-mono">databento.com</span>.
              </p>
              <div className="space-y-2">
                <Label className="text-xs font-sans text-muted-foreground">API Key</Label>
                <SecretInput
                  value={databentoKey}
                  onChange={setDatabentoKey}
                  placeholder="db-••••••••••••••••••••••••••••"
                  testId="input-databento-key"
                />
              </div>
              <Button
                data-testid="btn-save-databento"
                variant="outline"
                size="sm"
                className="h-8 text-xs border-border bg-muted hover:bg-accent w-full"
                onClick={saveDatabentoKey}
                disabled={savingDatabento}
              >
                {savingDatabento ? (
                  <>
                    <RefreshCw size={12} className="mr-1.5 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save size={12} className="mr-1.5" />
                    Save & Validate
                  </>
                )}
              </Button>
              {databentoStatus && (
                <div
                  className={`p-2 rounded-md text-xs font-mono border ${
                    databentoStatus.ok
                      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                      : "border-red-500/40 bg-red-500/10 text-red-300"
                  }`}
                  data-testid="databento-status"
                >
                  {databentoStatus.msg}
                </div>
              )}
              <p className="text-xs text-muted-foreground font-sans">
                Status:{" "}
                <span className={databentoConfigured ? "text-emerald-400" : "text-muted-foreground"}>
                  {databentoConfigured ? "Configured" : "Not configured"}
                </span>
              </p>
            </CardContent>
          </Card>

          {/* ── System Status ─────────────────────────────────────────────── */}
          <Card className="bg-card border-border">
            <CardHeader className="py-3 px-4 border-b border-border">
              <CardTitle className="text-sm font-sans font-medium text-foreground">
                System Status
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4">
              {modelsLoading ? (
                <div className="space-y-2">
                  {[...Array(5)].map((_, i) => (
                    <Skeleton key={i} className="h-8 rounded" />
                  ))}
                </div>
              ) : (
                <div data-testid="system-status">
                  <StatusRow
                    label="Python Engine"
                    value={
                      pythonConnected
                        ? "Connected"
                        : "Offline — start: uvicorn api_server:app --port 8001"
                    }
                    ok={pythonConnected}
                  />
                  <StatusRow label="SQLite Database" value="Healthy" ok={true} />
                  <StatusRow
                    label="Available Models"
                    value={
                      !pythonConnected
                        ? "Engine offline"
                        : modelsData?.models?.length
                        ? `${modelsData.models.length} checkpoint(s)`
                        : "0 — run training first"
                    }
                    ok={!!(modelsData?.models?.length)}
                  />
                  <StatusRow
                    label="Last Training"
                    value={lastTrained}
                    ok={!!lastJob}
                    mono={true}
                  />
                  <StatusRow
                    label="WebSocket"
                    value={pythonConnected ? "Active (live)" : "Active (local mock)"}
                    ok={true}
                  />
                </div>
              )}
            </CardContent>
          </Card>

          {/* ── Quick Start ───────────────────────────────────────────────── */}
          <Card className="bg-card border-border">
            <CardHeader className="py-3 px-4 border-b border-border">
              <CardTitle className="text-sm font-sans font-medium text-foreground">
                Quick Start
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4 space-y-3">
              <p className="text-xs text-muted-foreground font-sans">
                To activate the Python engine and live trading:
              </p>
              <div className="space-y-2">
                {[
                  { step: "1", cmd: "cd hft-trader/python_engine", label: "Navigate to engine" },
                  { step: "2", cmd: "pip install -r requirements.txt", label: "Install dependencies" },
                  { step: "3", cmd: "python data_pipeline.py", label: "Download datasets" },
                  {
                    step: "4",
                    cmd: "python trainer.py --market us --timescale 1m --algo PPO",
                    label: "Train a model",
                  },
                  {
                    step: "5",
                    cmd: "uvicorn api_server:app --port 8001 --reload",
                    label: "Start the API server",
                  },
                ].map(({ step, cmd, label }) => (
                  <div
                    key={step}
                    className="flex items-start gap-3 p-2.5 bg-muted/50 rounded-md border border-border/50"
                  >
                    <span className="w-5 h-5 rounded-full bg-primary/20 text-primary text-xs font-mono font-bold flex items-center justify-center shrink-0 mt-0.5">
                      {step}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs text-muted-foreground font-sans mb-0.5">{label}</p>
                      <code className="text-xs font-mono text-foreground break-all">{cmd}</code>
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-muted-foreground font-sans mt-2">
                The dashboard works in mock mode without the Python engine — all charts, backtest
                results, and red team tests use realistic simulated data.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
