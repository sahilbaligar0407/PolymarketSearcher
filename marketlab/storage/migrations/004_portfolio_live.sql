-- Exact crash-recovery checkpoint: one row per sleeve, overwritten every time the broker
-- applies a fill or a settlement. Balances are only written every snapshot (5 min), so a
-- hard kill used to lose every fill since the last one (299 fills / ~$286 of cash across
-- 11 restarts, FINDINGS 59). load_portfolio prefers this row when it is newer.
CREATE TABLE IF NOT EXISTS portfolio_live (
    experiment_id TEXT PRIMARY KEY,
    updated_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
