import { Switch, Route, Router } from "wouter";
import { useHashLocation } from "wouter/use-hash-location";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import Sidebar from "@/components/Sidebar";
import NotFound from "@/pages/not-found";

import LiveTrading from "@/pages/LiveTrading";
import TrainingConsole from "@/pages/TrainingConsole";
import BacktestResults from "@/pages/BacktestResults";
import RedTeamTesting from "@/pages/RedTeamTesting";
import Settings from "@/pages/Settings";
import DataManager from "@/pages/DataManager";
import HPO from "@/pages/HPO";
import ContinuousLearning from "@/pages/ContinuousLearning";
import MetaController from "@/pages/MetaController";

function AppRouter() {
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar />
      <main className="flex-1 overflow-y-auto min-w-0">
        <Switch>
          <Route path="/" component={LiveTrading} />
          <Route path="/training" component={TrainingConsole} />
          <Route path="/backtest" component={BacktestResults} />
          <Route path="/redteam" component={RedTeamTesting} />
          <Route path="/data" component={DataManager} />
          <Route path="/hpo" component={HPO} />
          <Route path="/continuous" component={ContinuousLearning} />
          <Route path="/meta" component={MetaController} />
          <Route path="/settings" component={Settings} />
          <Route component={NotFound} />
        </Switch>
      </main>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Router hook={useHashLocation}>
          <AppRouter />
        </Router>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
