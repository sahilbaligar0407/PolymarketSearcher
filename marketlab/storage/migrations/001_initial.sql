-- Initial MarketLab storage schema.
-- All statements are defensive (IF NOT EXISTS) so this file may be re-executed safely;
-- run_migrations() also guards against re-applying an already-recorded version.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- experiments: immutable identity for one strategy/parameter/data-vintage run.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    strategy_name TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    git_commit TEXT NOT NULL DEFAULT '',
    parameter_hash TEXT NOT NULL DEFAULT '',
    parameters_json TEXT NOT NULL DEFAULT '{}',
    market_universe TEXT NOT NULL DEFAULT '',
    venue TEXT NOT NULL DEFAULT '',
    data_version TEXT NOT NULL DEFAULT '',
    execution_model_version TEXT NOT NULL DEFAULT '',
    feature_version TEXT NOT NULL DEFAULT '',
    llm_model_id TEXT NOT NULL DEFAULT '',
    prompt_hash TEXT NOT NULL DEFAULT '',
    start_timestamp TEXT,
    starting_bankroll TEXT NOT NULL DEFAULT '50.00',
    status TEXT NOT NULL DEFAULT 'IDEA',
    cohort TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_strategy ON experiments(strategy_name);
CREATE INDEX IF NOT EXISTS idx_experiments_cohort ON experiments(cohort);

-- ---------------------------------------------------------------------------
-- orders / fills: full broker-side audit trail, one row per Order/Fill.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    intent_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    canonical_id TEXT NOT NULL,
    venue TEXT NOT NULL,
    side TEXT NOT NULL,
    action TEXT NOT NULL,
    order_type TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    limit_price TEXT,
    time_in_force TEXT NOT NULL,
    status TEXT NOT NULL,
    filled_quantity INTEGER NOT NULL DEFAULT 0,
    average_fill_price TEXT,
    worst_fill_price TEXT,
    fees_paid TEXT NOT NULL DEFAULT '0',
    reject_reason TEXT,
    reject_detail TEXT NOT NULL DEFAULT '',
    decision_timestamp TEXT NOT NULL,
    simulated_network_send_timestamp TEXT,
    simulated_exchange_arrival_timestamp TEXT,
    book_timestamp_used TEXT,
    reference_price TEXT,
    venue_order_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_orders_experiment ON orders(experiment_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_canonical ON orders(canonical_id);
CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders(strategy_id);

CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(order_id),
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    canonical_id TEXT NOT NULL,
    venue TEXT NOT NULL,
    side TEXT NOT NULL,
    action TEXT NOT NULL,
    price TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    fee TEXT NOT NULL DEFAULT '0',
    timestamp TEXT NOT NULL,
    is_maker INTEGER NOT NULL DEFAULT 0,
    book_timestamp_used TEXT,
    level_breakdown TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(order_id);
CREATE INDEX IF NOT EXISTS idx_fills_experiment ON fills(experiment_id);
CREATE INDEX IF NOT EXISTS idx_fills_canonical ON fills(canonical_id);

-- ---------------------------------------------------------------------------
-- positions / balances: crash-recovery state for a Portfolio sleeve.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS positions (
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    canonical_id TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 0,
    average_price TEXT NOT NULL DEFAULT '0',
    realized_pnl TEXT NOT NULL DEFAULT '0',
    fees_paid TEXT NOT NULL DEFAULT '0',
    opened_at TEXT,
    last_update TEXT,
    gross_bought INTEGER NOT NULL DEFAULT 0,
    gross_sold INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (experiment_id, canonical_id, side)
);
CREATE INDEX IF NOT EXISTS idx_positions_canonical ON positions(canonical_id);

-- Append-only equity snapshots. Also carries every Portfolio-level field that isn't
-- per-position, so the latest row for an experiment plus its positions rows are
-- together sufficient to reconstruct the Portfolio exactly (see StateStore.load_portfolio).
CREATE TABLE IF NOT EXISTS balances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    strategy_id TEXT NOT NULL DEFAULT '',
    timestamp TEXT NOT NULL,
    cash TEXT NOT NULL,
    equity TEXT NOT NULL,
    realized_pnl TEXT NOT NULL,
    unrealized_pnl TEXT NOT NULL,
    exposure TEXT NOT NULL,
    high_water_mark TEXT NOT NULL,
    max_drawdown TEXT NOT NULL,
    trade_count INTEGER NOT NULL DEFAULT 0,
    resolved_trade_count INTEGER NOT NULL DEFAULT 0,
    fees_paid TEXT NOT NULL DEFAULT '0',
    initial_capital TEXT NOT NULL DEFAULT '50.00',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT,
    died_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_balances_experiment_ts ON balances(experiment_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_balances_experiment_id ON balances(experiment_id, id);

-- ---------------------------------------------------------------------------
-- strategy_state: opaque per-experiment key/value store for crash recovery of
-- whatever internal state a Strategy subclass keeps that isn't an order/fill.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS strategy_state (
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (experiment_id, key)
);

-- ---------------------------------------------------------------------------
-- copy-trading intelligence.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trader_registry (
    wallet TEXT PRIMARY KEY,
    username TEXT NOT NULL DEFAULT '',
    x_username TEXT NOT NULL DEFAULT '',
    discovery_date TEXT,
    rank_at_discovery INTEGER,
    category TEXT NOT NULL DEFAULT '',
    day_pnl TEXT,
    week_pnl TEXT,
    month_pnl TEXT,
    all_time_pnl TEXT,
    reported_volume TEXT,
    number_of_markets INTEGER,
    number_of_observed_trades INTEGER,
    median_trade_size TEXT,
    position_concentration TEXT,
    resolved_win_rate TEXT,
    realized_pnl TEXT,
    estimated_roi TEXT,
    largest_loss TEXT,
    largest_win TEXT,
    max_observed_drawdown TEXT,
    category_specialization TEXT NOT NULL DEFAULT '',
    median_holding_time TEXT,
    turnover TEXT,
    recent_performance_slope TEXT,
    status TEXT NOT NULL DEFAULT 'DISCOVERED',
    last_updated TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trader_registry_status ON trader_registry(status);
CREATE INDEX IF NOT EXISTS idx_trader_registry_category ON trader_registry(category);

CREATE TABLE IF NOT EXISTS trader_leaderboard_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_time TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    period TEXT NOT NULL DEFAULT '',
    metric TEXT NOT NULL DEFAULT '',
    rank INTEGER,
    wallet TEXT NOT NULL,
    username TEXT NOT NULL DEFAULT '',
    pnl TEXT,
    volume TEXT
);
CREATE INDEX IF NOT EXISTS idx_leaderboard_snap_time ON trader_leaderboard_snapshots(snapshot_time);
CREATE INDEX IF NOT EXISTS idx_leaderboard_snap_wallet ON trader_leaderboard_snapshots(wallet);

CREATE TABLE IF NOT EXISTS trader_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet TEXT NOT NULL,
    username TEXT NOT NULL DEFAULT '',
    canonical_id TEXT NOT NULL DEFAULT '',
    poly_market_id TEXT NOT NULL DEFAULT '',
    poly_condition_id TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL DEFAULT '',
    side TEXT,
    action TEXT NOT NULL DEFAULT '',
    price TEXT,
    size TEXT,
    usd_size TEXT,
    category TEXT NOT NULL DEFAULT '',
    transaction_hash TEXT NOT NULL DEFAULT '',
    event_time TEXT NOT NULL,
    first_seen_time TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trader_actions_wallet ON trader_actions(wallet);
CREATE INDEX IF NOT EXISTS idx_trader_actions_first_seen ON trader_actions(first_seen_time);

-- ---------------------------------------------------------------------------
-- market registry / cross-venue matching.
-- ---------------------------------------------------------------------------
-- raw_json stores the full serialized NormalizedMarket (model_dump_json()), so
-- get_market() can reconstruct the model exactly; the other columns are denormalized
-- copies kept only for cheap filtering/indexing.
CREATE TABLE IF NOT EXISTS market_registry (
    canonical_id TEXT PRIMARY KEY,
    venue TEXT NOT NULL,
    venue_market_id TEXT NOT NULL,
    event_id TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'unknown',
    open_time TEXT,
    close_time TEXT,
    tick_size TEXT NOT NULL DEFAULT '0.01',
    min_order INTEGER NOT NULL DEFAULT 1,
    fees_json TEXT NOT NULL DEFAULT '{}',
    last_seen TEXT NOT NULL,
    raw_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_market_registry_venue ON market_registry(venue);
CREATE INDEX IF NOT EXISTS idx_market_registry_status ON market_registry(status);
CREATE INDEX IF NOT EXISTS idx_market_registry_category ON market_registry(category);

CREATE TABLE IF NOT EXISTS market_matches (
    match_id TEXT PRIMARY KEY,
    canonical_id_a TEXT NOT NULL,
    canonical_id_b TEXT NOT NULL,
    match_confidence TEXT NOT NULL DEFAULT '0',
    same_outcome_boolean INTEGER,
    rule_diff TEXT NOT NULL DEFAULT '',
    time_diff TEXT NOT NULL DEFAULT '',
    resolution_source_diff TEXT NOT NULL DEFAULT '',
    human_review_required INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    validator_version TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_market_matches_a ON market_matches(canonical_id_a);
CREATE INDEX IF NOT EXISTS idx_market_matches_b ON market_matches(canonical_id_b);

-- ---------------------------------------------------------------------------
-- forecasts / settlements.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    strategy_id TEXT NOT NULL DEFAULT '',
    canonical_id TEXT NOT NULL,
    as_of TEXT NOT NULL,
    p_yes TEXT NOT NULL,
    confidence TEXT NOT NULL DEFAULT '0.5',
    market_probability TEXT,
    abstain INTEGER NOT NULL DEFAULT 0,
    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
    rationale TEXT NOT NULL DEFAULT '',
    features_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_forecasts_experiment ON forecasts(experiment_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_canonical ON forecasts(canonical_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_asof ON forecasts(as_of);

CREATE TABLE IF NOT EXISTS settlements (
    canonical_id TEXT PRIMARY KEY,
    venue TEXT NOT NULL,
    winning_side TEXT,
    settlement_value TEXT,
    voided INTEGER NOT NULL DEFAULT 0,
    settled_at TEXT NOT NULL,
    first_seen_time TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_settlements_settled_at ON settlements(settled_at);

-- ---------------------------------------------------------------------------
-- operations.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    severity TEXT NOT NULL,
    component TEXT NOT NULL,
    message TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_component ON alerts(component);

CREATE TABLE IF NOT EXISTS service_checkpoints (
    component TEXT NOT NULL,
    checkpoint_key TEXT NOT NULL,
    checkpoint_value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (component, checkpoint_key)
);

CREATE TABLE IF NOT EXISTS social_challenge_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim TEXT NOT NULL,
    account TEXT NOT NULL DEFAULT '',
    platform TEXT NOT NULL DEFAULT '',
    claimed_starting_balance TEXT,
    claimed_current_balance TEXT,
    claim_date TEXT,
    wallet_if_public TEXT NOT NULL DEFAULT '',
    verified_by_market_data INTEGER NOT NULL DEFAULT 0,
    verification_status TEXT NOT NULL DEFAULT 'UNVERIFIED',
    notes TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_social_candidates_status
    ON social_challenge_candidates(verification_status);
