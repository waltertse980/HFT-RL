import { useState, useEffect } from "react";
import { useHashLocation } from "wouter/use-hash-location";
import { Link } from "wouter";
import { Activity, Brain, BarChart2, Shield, Settings, ChevronLeft, ChevronRight, Zap, Database, RefreshCw, Layers } from "lucide-react";
import { cn } from "@/lib/utils";

interface NavItem {
  icon: React.ReactNode;
  label: string;
  path: string;
}

const navItems: NavItem[] = [
  { icon: <Activity size={18} />, label: "Live Trading", path: "/" },
  { icon: <Brain size={18} />, label: "Training", path: "/training" },
  { icon: <Database size={18} />, label: "Data Manager", path: "/data" },
  { icon: <Zap size={18} />, label: "HPO", path: "/hpo" },
  { icon: <BarChart2 size={18} />, label: "Backtest", path: "/backtest" },
  { icon: <Shield size={18} />, label: "Red Team", path: "/redteam" },
  { icon: <RefreshCw size={18} />, label: "Continuous Learning", path: "/continuous" },
  { icon: <Layers size={18} />, label: "Meta Controller", path: "/meta" },
  { icon: <Settings size={18} />, label: "Settings", path: "/settings" },
];

function HFTLogo() {
  return (
    <svg
      width="120"
      height="32"
      viewBox="0 0 120 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="HFT Trader"
    >
      {/* H */}
      <rect x="0" y="4" width="3" height="24" fill="#3b82f6" />
      <rect x="0" y="14" width="14" height="3" fill="#3b82f6" />
      <rect x="11" y="4" width="3" height="24" fill="#3b82f6" />
      {/* F */}
      <rect x="20" y="4" width="3" height="24" fill="#3b82f6" />
      <rect x="20" y="4" width="14" height="3" fill="#3b82f6" />
      <rect x="20" y="14" width="11" height="3" fill="#3b82f6" />
      {/* T */}
      <rect x="40" y="4" width="16" height="3" fill="#3b82f6" />
      <rect x="46" y="4" width="3" height="24" fill="#3b82f6" />
      {/* Arrow */}
      <path
        d="M64 24 L76 12 M70 12 L76 12 L76 18"
        stroke="#22c55e"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* TRADER text */}
      <text
        x="82"
        y="26"
        fill="#64748b"
        fontSize="10"
        fontFamily="'JetBrains Mono', monospace"
        fontWeight="500"
        letterSpacing="2"
      >
        TRADER
      </text>
    </svg>
  );
}

interface SidebarProps {
  className?: string;
}

export default function Sidebar({ className }: SidebarProps) {
  const [location] = useHashLocation();
  const [collapsed, setCollapsed] = useState(false);
  const [engineConnected, setEngineConnected] = useState(false);

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch("/api/health", { signal: AbortSignal.timeout(2000) });
        setEngineConnected(res.ok);
      } catch {
        setEngineConnected(false);
      }
    };
    check();
    const id = setInterval(check, 5000);
    return () => clearInterval(id);
  }, []);

  return (
    <aside
      data-testid="sidebar"
      className={cn(
        "flex flex-col h-screen border-r border-border bg-sidebar transition-all duration-200 shrink-0",
        collapsed ? "w-14" : "w-60",
        className
      )}
    >
      {/* Logo */}
      <div className="flex items-center justify-between px-4 py-4 border-b border-border h-14">
        {!collapsed && (
          <div data-testid="sidebar-logo">
            <HFTLogo />
          </div>
        )}
        {collapsed && (
          <div className="flex items-center justify-center w-full">
            <Zap size={20} className="text-primary" />
          </div>
        )}
      </div>

      {/* Nav Items */}
      <nav className="flex-1 py-3 space-y-0.5 px-2 overflow-y-auto">
        {navItems.map((item) => {
          const isActive =
            item.path === "/"
              ? location === "/" || location === ""
              : location.startsWith(item.path);

          return (
            <Link key={item.path} href={item.path}>
              <a
                data-testid={`nav-${item.label.toLowerCase().replace(/\s+/g, "-")}`}
                className={cn(
                  "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-sans transition-all cursor-pointer select-none",
                  "hover:bg-accent hover:text-foreground",
                  isActive
                    ? "bg-primary/10 text-primary border-l-2 border-primary font-medium pl-[10px]"
                    : "text-muted-foreground border-l-2 border-transparent"
                )}
              >
                <span className="shrink-0">{item.icon}</span>
                {!collapsed && <span className="truncate">{item.label}</span>}
              </a>
            </Link>
          );
        })}
      </nav>

      {/* Connection Status */}
      {!collapsed && (
        <div className="px-4 py-3 border-t border-border">
          <div className="flex items-center gap-2 text-xs font-sans">
            <span
              data-testid="engine-status-dot"
              className={cn(
                "w-2 h-2 rounded-full shrink-0",
                engineConnected
                  ? "bg-success animate-pulse-dot"
                  : "bg-destructive"
              )}
            />
            <span className="text-muted-foreground">Python Engine:</span>
            <span
              className={engineConnected ? "text-success" : "text-destructive"}
            >
              {engineConnected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </div>
      )}

      {/* Collapse Toggle */}
      <button
        data-testid="sidebar-collapse-btn"
        onClick={() => setCollapsed(!collapsed)}
        className="flex items-center justify-center py-3 border-t border-border text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
        aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
      </button>
    </aside>
  );
}
