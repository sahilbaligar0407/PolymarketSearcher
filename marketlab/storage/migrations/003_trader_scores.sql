-- TraderScore: each wallet's latest realized-track-record analysis
-- (marketlab/signals/trader_analytics.py). One row per wallet, replaced on rescore.
CREATE TABLE IF NOT EXISTS trader_scores (
    wallet TEXT PRIMARY KEY,
    username TEXT NOT NULL DEFAULT '',
    computed_at TEXT NOT NULL,
    resolved_positions INTEGER NOT NULL DEFAULT 0,
    realized_pnl REAL NOT NULL DEFAULT 0,
    total_staked REAL NOT NULL DEFAULT 0,
    roi REAL NOT NULL DEFAULT 0,
    win_rate REAL NOT NULL DEFAULT 0,
    profit_factor REAL,
    max_drawdown REAL NOT NULL DEFAULT 0,
    sharpe_like REAL NOT NULL DEFAULT 0,
    trades_per_day REAL NOT NULL DEFAULT 0,
    avg_entry_price REAL NOT NULL DEFAULT 0,
    favorite_share REAL NOT NULL DEFAULT 0,
    largest_win_share REAL NOT NULL DEFAULT 0,
    recent_roi REAL,
    recent_positions INTEGER NOT NULL DEFAULT 0,
    top_category TEXT NOT NULL DEFAULT '',
    category_share REAL NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'DISCOVERED',
    reasons TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_trader_scores_status ON trader_scores(status, score);
