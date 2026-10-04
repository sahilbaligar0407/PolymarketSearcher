-- The human-readable audit record behind every filled paper trade.
--
-- `orders` records what happened to an order; it never stored WHY the strategy wanted
-- it. This table keeps the intent's rationale and features (model ids, probabilities,
-- evidence headlines, second-opinion verdicts, ...) for every intent that filled, so the
-- dashboard can answer "what triggered this trade, who supported it, what was the edge,
-- what evidence existed, what happened afterwards" for each one.
--
-- Only filled intents are kept: rejected intents run to ~200k/day and their reason is
-- already in orders.reject_detail.
CREATE TABLE IF NOT EXISTS decisions (
    intent_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL DEFAULT '',
    canonical_id TEXT NOT NULL,
    side TEXT NOT NULL,
    action TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    filled_quantity INTEGER NOT NULL,
    average_fill_price TEXT,
    fees_paid TEXT NOT NULL DEFAULT '0',
    model_probability TEXT,
    expected_edge TEXT,
    rationale TEXT NOT NULL DEFAULT '',
    features_json TEXT NOT NULL DEFAULT '{}',
    decided_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_experiment ON decisions(experiment_id);
CREATE INDEX IF NOT EXISTS idx_decisions_decided_at ON decisions(decided_at);
CREATE INDEX IF NOT EXISTS idx_decisions_canonical ON decisions(canonical_id);
