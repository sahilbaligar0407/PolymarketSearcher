# Data dictionary

## Units — the rule that prevents the worst bugs

**Every price in MarketLab is a probability: a `Decimal` in `[0.0000, 1.0000]`, quantized to
`PROB_QUANTUM = 0.0001`.**

Adapters convert at the boundary and nothing downstream ever sees a venue's native unit:

| Venue | Native unit | Conversion |
|---|---|---|
| Kalshi | integer cents, 0–100 | `cents_to_probability(52)` → `Decimal("0.5200")` |
| Polymarket | decimal dollars, 0–1 | validated through `to_probability()` |
| Sportsbook | American odds | `american_to_probability()` then `remove_vig()` |

`to_probability` **raises** outside `[0,1]` rather than clamping, because a price outside the
range means an adapter mixed up its units — which should fail loudly, not quietly.

Money is `Decimal` everywhere. A binary contract pays exactly `$1.00` on the winning side.
SQLite stores money and probabilities as `TEXT` and converts on read, so no precision is lost
to float. Never compare prices with `==` on floats.

## Order book convention

**Both sides of an `OrderBook` are expressed in YES-probability terms.** `bids` are sorted
descending, `asks` ascending.

This matters because venues do not present books this way:

- **Kalshi** returns `{"orderbook": {"yes": [[price_cents, size]...], "no": [[...]]}}` where
  *both* arrays are bids. The `yes` array is bids to buy YES; the `no` array is bids to buy
  NO. The adapter keeps YES bids as bids and folds each NO bid at price `p` into a **YES ask
  at `1 - p/100`**.
- **Polymarket** books are per-token — the YES token and NO token have separate books. A YES
  token's bids and asks map directly; a NO token's book is inverted (`price → 1-price`,
  bids ↔ asks).

Selling YES and buying NO are economically the same trade. The fill models handle the NO-side
arithmetic explicitly rather than leaving it to each strategy to get right.

## The four timestamps

Every external record carries all four. They are not interchangeable.

| Field | Meaning | Example |
|---|---|---|
| `event_time` | when the thing happened in the world | insider sold shares on July 1 |
| `published_time` | when the source released it | Form 4 accepted August 10 |
| `first_seen_time` | when MarketLab observed it | our poller saw it August 10, 14:32 |
| `ingested_time` | when it was written to storage | August 10, 14:32:01 |

**`first_seen_time` is the only field a strategy may gate on.** `BaseEvent.visible_at(T)`
returns `first_seen_time <= T`. A July backtest may not use that filing; an August one may.

`first_seen_time` always comes from the injected `Clock` — never from a timestamp inside the
content, which the source controls and may back-date.

The same discipline applies to economic revisions (use the ALFRED `vintage_date`), edited
social posts, updated injury reports, final poll averages, and **the discovery timestamp of a
public trader**.

## Core objects

### `NormalizedMarket`
The universal market object. `canonical_id` is the primary key across the whole system:
`kalshi:<ticker>` or `poly:<condition_id>`, lowercase, deterministic.

Key fields: `venue`, `venue_market_id`, `event_id`, `title`, `resolution_rules`,
`resolution_source`, `category`, `outcome_type`, `open_time`, `close_time`,
`expected_resolution_time`, `tick_size`, `min_order`, `status`, `fees`, `liquidity`,
`volume`, and `raw` (the untouched venue payload, kept for audit and never read by strategies).

`resolution_rules` and `resolution_source` are load-bearing, not documentation: the matching
engine reads them to decide whether two markets are the same bet.

### `OrderIntent` → `Order` → `Fill`
A strategy emits an `OrderIntent` carrying `rationale`, `features`, `evidence_ids`,
`model_probability` and `expected_edge`. Under `strict_audit` an intent with an empty
rationale is rejected — there is no `BUY because AI said so` record anywhere.

An `Order` carries the full latency audit trail:

| Timestamp | Meaning |
|---|---|
| `decision_timestamp` | when the strategy decided, from its injected clock |
| `simulated_network_send_timestamp` | decision + signal-to-order latency |
| `simulated_exchange_arrival_timestamp` | + network + processing latency |
| `book_timestamp_used` | the snapshot the simulator actually consumed |

`book_timestamp_used <= simulated_exchange_arrival_timestamp` always. A violation is
look-ahead, and it is asserted in the paper-broker tests.

`reference_price` (mid at decision time) makes `Order.slippage` meaningful.
`Fill.level_breakdown` records which book levels were consumed, so slippage can be attributed
rather than estimated.

### `Portfolio`
One virtual $50 sleeve, or the single real bankroll. YES and NO positions are tracked
**separately** per market — holding both is a real, meaningful state (a parity pair), not
something to net away. Settlement pays `$1.00` to the winning side; a voided market refunds
cost basis.

`SleeveStatus`: `ACTIVE` / `DEAD` / `RETIRED`. A sleeve dies below `death_floor` (default
$1.00) and is **never deleted**.

## Experiment identity

An experiment id is derived from all of:

`strategy_name`, `strategy_version`, `git_commit`, `parameter_hash`, `market_universe`,
`venue`, `data_version`, `execution_model_version`, `feature_version`, `llm_model_id`,
`prompt_hash`, `start_timestamp`, `starting_bankroll`.

Changing any of them produces a **new** experiment. A strategy cannot be quietly modified
under an existing identity, and a restart after a dead sleeve creates a new cohort rather
than a continuation that hides the earlier failure.

## Storage layout

**SQLite (WAL)** — transactional state that gets re-read:
`experiments`, `orders`, `fills`, `positions`, `balances`, `strategy_state`,
`trader_registry`, `trader_leaderboard_snapshots`, `trader_actions`, `market_registry`,
`market_matches`, `forecasts`, `settlements`, `alerts`, `service_checkpoints`,
`social_challenge_candidates`, `schema_migrations`.

**Parquet (append-only, date-partitioned)** — high-volume immutable history:
`books`, `trades`, `prices`, `market_metadata`, `news`, `social`, `external_prices`,
`forecasts`, `trader_actions`. Path: `data/parquet/<dataset>/date=YYYY-MM-DD/part-<n>.parquet`.
Decimal columns are stored as `float64` plus an exact string twin (`<field>_exact`).

**DuckDB** — the query engine over the Parquet lake, with the SQLite state attached, for
research queries and leaderboard reports.

## Source classes

Evidentiary weight is explicit. An SEC filing is not a tweet.

`OFFICIAL_PRIMARY` → `MAJOR_WIRE` → `MAJOR_NEWS` → `SPECIALIST` → `SOCIAL_VERIFIED` →
`SOCIAL_UNVERIFIED` → `UNKNOWN`

News deduplication marks republications via `duplicate_of` rather than dropping them, so ten
copies of one wire story count as one source while the fan-out remains visible.

## Trader records

Discovered from the official Polymarket leaderboard, stored with `discovery_date` and
`rank_at_discovery`, then tracked **forward**. Status progresses
`DISCOVERED` → `TRACKING` → `QUALIFIED` / `REJECTED`.

Statistics computed from public activity: `median_trade_size`, `position_concentration` (HHI),
`resolved_win_rate`, `estimated_roi`, `largest_loss`, `largest_win`, `max_observed_drawdown`,
`category_specialization`, `median_holding_time`, `turnover`, `recent_performance_slope`.

Traders are never ranked by raw win rate. A 90% win rate with asymmetric losses is
unprofitable, and the scoring function is tested against exactly that case.

Follower execution never uses the source trader's price. It uses the **Kalshi** book available
after `first_seen_time` plus the configured follower delay.

## Social challenge candidates

Viral claims are quarantined: `UNVERIFIED` → `PARTIALLY_VERIFIED` →
`ONCHAIN_OR_API_CONFIRMED` / `DISPROVEN` / `STALE`. Only `ONCHAIN_OR_API_CONFIRMED` may
influence a strategy, and that restriction is enforced in code rather than by policy.
