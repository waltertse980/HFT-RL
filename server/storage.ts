import { drizzle } from 'drizzle-orm/better-sqlite3';
import Database from 'better-sqlite3';
import { eq, desc, lt, sql } from 'drizzle-orm';
import {
  tradingSettings,
  trainingJobs,
  backtestResults,
  redTeamResults,
  paperTrades,
  type TradingSettings,
  type InsertTradingSettings,
  type TrainingJob,
  type InsertTrainingJob,
  type BacktestResult,
  type InsertBacktestResult,
  type RedTeamResult,
  type InsertRedTeamResult,
  type PaperTrade,
  type InsertPaperTrade,
} from '@shared/schema';

const sqlite = new Database('data.db');
sqlite.pragma('journal_mode = WAL');

export const db = drizzle(sqlite);

// ─── IStorage Interface ──────────────────────────────────────────────────────

export interface IStorage {
  // TradingSettings
  getAllSettings(): Promise<TradingSettings[]>;
  getSettingsByMarket(market: string): Promise<TradingSettings[]>;
  createSettings(data: InsertTradingSettings): Promise<TradingSettings>;
  updateSettings(id: number, data: Partial<InsertTradingSettings>): Promise<TradingSettings | undefined>;
  deleteSettings(id: number): Promise<void>;

  // TrainingJobs
  getAllJobs(): Promise<TrainingJob[]>;
  getJobById(id: number): Promise<TrainingJob | undefined>;
  getJobByJobId(jobId: string): Promise<TrainingJob | undefined>;
  createJob(data: InsertTrainingJob): Promise<TrainingJob>;
  updateJob(jobId: string, data: Partial<InsertTrainingJob>): Promise<TrainingJob | undefined>;
  deleteJob(jobId: string): Promise<void>;

  // BacktestResults
  getAllBacktestResults(): Promise<BacktestResult[]>;
  getBacktestResultsByMarket(market: string): Promise<BacktestResult[]>;
  getBacktestResultById(id: number): Promise<BacktestResult | undefined>;
  createBacktestResult(data: InsertBacktestResult): Promise<BacktestResult>;
  deleteBacktestResult(id: number): Promise<void>;

  // RedTeamResults
  getAllRedTeamResults(): Promise<RedTeamResult[]>;
  getRedTeamResultsByBacktestId(backtestId: number): Promise<RedTeamResult[]>;
  createRedTeamResult(data: InsertRedTeamResult): Promise<RedTeamResult>;

  // PaperTrades
  getRecentPaperTrades(limit?: number): Promise<PaperTrade[]>;
  getPaperTradesByMarket(market: string, limit?: number): Promise<PaperTrade[]>;
  createPaperTrade(data: InsertPaperTrade): Promise<PaperTrade>;
  deleteOlderThanDays(days: number): Promise<void>;
}

// ─── DatabaseStorage Implementation ─────────────────────────────────────────

export class DatabaseStorage implements IStorage {
  // ── TradingSettings ──────────────────────────────────────────────────────

  async getAllSettings(): Promise<TradingSettings[]> {
    return db.select().from(tradingSettings).all();
  }

  async getSettingsByMarket(market: string): Promise<TradingSettings[]> {
    return db.select().from(tradingSettings).where(eq(tradingSettings.market, market)).all();
  }

  async createSettings(data: InsertTradingSettings): Promise<TradingSettings> {
    return db.insert(tradingSettings).values(data).returning().get();
  }

  async updateSettings(
    id: number,
    data: Partial<InsertTradingSettings>,
  ): Promise<TradingSettings | undefined> {
    return db
      .update(tradingSettings)
      .set(data)
      .where(eq(tradingSettings.id, id))
      .returning()
      .get();
  }

  async deleteSettings(id: number): Promise<void> {
    db.delete(tradingSettings).where(eq(tradingSettings.id, id)).run();
  }

  // ── TrainingJobs ─────────────────────────────────────────────────────────

  async getAllJobs(): Promise<TrainingJob[]> {
    return db.select().from(trainingJobs).orderBy(desc(trainingJobs.createdAt)).all();
  }

  async getJobById(id: number): Promise<TrainingJob | undefined> {
    return db.select().from(trainingJobs).where(eq(trainingJobs.id, id)).get();
  }

  async getJobByJobId(jobId: string): Promise<TrainingJob | undefined> {
    return db.select().from(trainingJobs).where(eq(trainingJobs.jobId, jobId)).get();
  }

  async createJob(data: InsertTrainingJob): Promise<TrainingJob> {
    return db.insert(trainingJobs).values(data).returning().get();
  }

  async updateJob(
    jobId: string,
    data: Partial<InsertTrainingJob>,
  ): Promise<TrainingJob | undefined> {
    return db
      .update(trainingJobs)
      .set(data)
      .where(eq(trainingJobs.jobId, jobId))
      .returning()
      .get();
  }

  async deleteJob(jobId: string): Promise<void> {
    db.delete(trainingJobs).where(eq(trainingJobs.jobId, jobId)).run();
  }

  // ── BacktestResults ──────────────────────────────────────────────────────

  async getAllBacktestResults(): Promise<BacktestResult[]> {
    return db.select().from(backtestResults).orderBy(desc(backtestResults.createdAt)).all();
  }

  async getBacktestResultsByMarket(market: string): Promise<BacktestResult[]> {
    return db
      .select()
      .from(backtestResults)
      .where(eq(backtestResults.market, market))
      .orderBy(desc(backtestResults.createdAt))
      .all();
  }

  async getBacktestResultById(id: number): Promise<BacktestResult | undefined> {
    return db.select().from(backtestResults).where(eq(backtestResults.id, id)).get();
  }

  async createBacktestResult(data: InsertBacktestResult): Promise<BacktestResult> {
    return db.insert(backtestResults).values(data).returning().get();
  }

  async deleteBacktestResult(id: number): Promise<void> {
    db.delete(backtestResults).where(eq(backtestResults.id, id)).run();
  }

  // ── RedTeamResults ───────────────────────────────────────────────────────

  async getAllRedTeamResults(): Promise<RedTeamResult[]> {
    return db.select().from(redTeamResults).all();
  }

  async getRedTeamResultsByBacktestId(backtestId: number): Promise<RedTeamResult[]> {
    return db
      .select()
      .from(redTeamResults)
      .where(eq(redTeamResults.backtestId, backtestId))
      .all();
  }

  async createRedTeamResult(data: InsertRedTeamResult): Promise<RedTeamResult> {
    return db.insert(redTeamResults).values(data).returning().get();
  }

  // ── PaperTrades ──────────────────────────────────────────────────────────

  async getRecentPaperTrades(limit = 500): Promise<PaperTrade[]> {
    return db
      .select()
      .from(paperTrades)
      .orderBy(desc(paperTrades.timestamp))
      .limit(limit)
      .all();
  }

  async getPaperTradesByMarket(market: string, limit = 500): Promise<PaperTrade[]> {
    return db
      .select()
      .from(paperTrades)
      .where(eq(paperTrades.market, market))
      .orderBy(desc(paperTrades.timestamp))
      .limit(limit)
      .all();
  }

  async createPaperTrade(data: InsertPaperTrade): Promise<PaperTrade> {
    return db.insert(paperTrades).values(data).returning().get();
  }

  async deleteOlderThanDays(days: number): Promise<void> {
    const cutoff = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
    db.delete(paperTrades).where(lt(paperTrades.timestamp, cutoff)).run();
  }
}

export const storage = new DatabaseStorage();
