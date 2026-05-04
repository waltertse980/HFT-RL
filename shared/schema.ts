import { sqliteTable, text, integer, real } from 'drizzle-orm/sqlite-core';
import { createInsertSchema } from 'drizzle-zod';
import { z } from 'zod';

// Trading settings per market
export const tradingSettings = sqliteTable('trading_settings', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  market: text('market').notNull(), // 'us' | 'hk'
  ticker: text('ticker').notNull(),
  timescale: text('timescale').notNull().default('10s'), // '10s'|'1m'|'5m'|'1h'
  maxPositionPct: real('max_position_pct').notNull().default(0.95),
  stopLossPct: real('stop_loss_pct').notNull().default(0.02),
  dailyLossLimitPct: real('daily_loss_limit_pct').notNull().default(0.05),
  alpacaApiKey: text('alpaca_api_key'),
  alpacaApiSecret: text('alpaca_api_secret'),
  isActive: integer('is_active', { mode: 'boolean' }).notNull().default(false),
});

// Training jobs
export const trainingJobs = sqliteTable('training_jobs', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  jobId: text('job_id').notNull().unique(),
  market: text('market').notNull(),
  timescale: text('timescale').notNull(),
  algo: text('algo').notNull().default('PPO'),
  totalTimesteps: integer('total_timesteps').notNull().default(1000000),
  status: text('status').notNull().default('pending'), // pending|running|done|error
  progressPct: real('progress_pct').notNull().default(0),
  currentReward: real('current_reward'),
  modelPath: text('model_path'),
  errorMsg: text('error_msg'),
  startedAt: text('started_at'),
  completedAt: text('completed_at'),
  createdAt: text('created_at').notNull(),
});

// Backtest results
export const backtestResults = sqliteTable('backtest_results', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  market: text('market').notNull(),
  timescale: text('timescale').notNull(),
  ticker: text('ticker').notNull(),
  modelPath: text('model_path').notNull(),
  startDate: text('start_date').notNull(),
  endDate: text('end_date').notNull(),
  sharpeRatio: real('sharpe_ratio'),
  sortinoRatio: real('sortino_ratio'),
  calmarRatio: real('calmar_ratio'),
  maxDrawdownPct: real('max_drawdown_pct'),
  totalReturnPct: real('total_return_pct'),
  winRate: real('win_rate'),
  nTrades: integer('n_trades'),
  avgPnlPerTrade: real('avg_pnl_per_trade'),
  profitFactor: real('profit_factor'),
  equityCurve: text('equity_curve'), // JSON array
  createdAt: text('created_at').notNull(),
});

// Red team test results
export const redTeamResults = sqliteTable('red_team_results', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  backtestId: integer('backtest_id'),
  scenario: text('scenario').notNull(),
  passed: integer('passed', { mode: 'boolean' }).notNull(),
  metrics: text('metrics').notNull(), // JSON object
  createdAt: text('created_at').notNull(),
});

// Paper trades log
export const paperTrades = sqliteTable('paper_trades', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  market: text('market').notNull(),
  ticker: text('ticker').notNull(),
  action: text('action').notNull(), // BUY|SELL|HOLD
  price: real('price').notNull(),
  quantity: real('quantity').notNull().default(0),
  portfolioValue: real('portfolio_value').notNull(),
  pnl: real('pnl').notNull().default(0),
  signal: text('signal'), // raw model signal
  inferenceMsec: real('inference_msec'),
  timestamp: text('timestamp').notNull(),
});

// Insert schemas
export const insertTradingSettingsSchema = createInsertSchema(tradingSettings).omit({ id: true });
export const insertTrainingJobSchema = createInsertSchema(trainingJobs).omit({ id: true });
export const insertBacktestResultSchema = createInsertSchema(backtestResults).omit({ id: true });
export const insertRedTeamResultSchema = createInsertSchema(redTeamResults).omit({ id: true });
export const insertPaperTradeSchema = createInsertSchema(paperTrades).omit({ id: true });

// Types
export type TradingSettings = typeof tradingSettings.$inferSelect;
export type InsertTradingSettings = z.infer<typeof insertTradingSettingsSchema>;
export type TrainingJob = typeof trainingJobs.$inferSelect;
export type InsertTrainingJob = z.infer<typeof insertTrainingJobSchema>;
export type BacktestResult = typeof backtestResults.$inferSelect;
export type InsertBacktestResult = z.infer<typeof insertBacktestResultSchema>;
export type RedTeamResult = typeof redTeamResults.$inferSelect;
export type InsertRedTeamResult = z.infer<typeof insertRedTeamResultSchema>;
export type PaperTrade = typeof paperTrades.$inferSelect;
export type InsertPaperTrade = z.infer<typeof insertPaperTradeSchema>;
