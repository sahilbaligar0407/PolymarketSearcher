# MarketLab architecture

## Shape of the system

A single Python 3.12 asyncio process. No GUI, no web server, no message broker, no
container runtime. Everything is local because nothing about this workload needs more, and
the PRD is explicit that infrastructure gets added when the machine demands it, not before.

```
                  ┌──────────────── adapters ────────────────┐
  Kalshi REST/WS ─┤                                          │
  Polymarket ─────┤   normalize to NormalizedMarket /        │
  GDELT / SEC ────┤   OrderBook / Trade / *Event             │
  NWS / FRED ─────┤   (all prices -> Decimal in [0,1])       │
  Coinbase / X ───┤                                          │
  Bluesky / Odds ─┘                                          │
                  └──────────────────┬───────────────────────┘
                                     │ events, each stamped with
                                     │ event/published/first_seen/ingested
                                     ▼
                            ┌────────────────┐
                            │   event bus    │  asyncio.Queue
                            └───────┬────────┘
                     ┌──────────────┼──────────────┐
                     ▼              ▼              ▼
              ┌───────────┐  ┌────────────┐  ┌──────────┐
              │  storage  │  │  matching  │  │ signals  │
              │ sqlite +  │  │ cross-venue│  │ features │
              │ parquet   │  │ rule gate  │  │          │
              └───────────┘  └────────────┘  └────┬─────┘
                                                  ▼
                                        ┌───────────────────┐
                                        │  strategy runner  │  N strategies
                                        │  (one $50 sleeve  │  each isolated
                                        │   per variant)    │
                                        └─────────┬─────────┘
                                                  │ OrderIntent
                                                  ▼
                                        ┌───────────────────┐
                                        │   risk gateway    │
                                        └─────────┬─────────┘
                                                  ▼
                                        ┌───────────────────┐
                                        │      broker       │
                                        │  PaperBroker  or  │
                                        │  KalshiLiveBroker │
                                        └─────────┬─────────┘
                                                  ▼
                                        ┌───────────────────┐
                                        │ portfolio + fills │
                                        └─────────┬─────────┘
                                                  ▼
                                        ┌───────────────────┐
                                        │     analytics     │
                                        │  metrics, calib., │
                                        │  bootstrap, league│
                                        └───────────────────┘
```

The AI layer hangs off the side of the event bus: it is triggered by events (high-impact
news, a filing, a large move, a tracked-trader action) or by a scheduled refresh, and its
output re-enters the system as a validated `ProbabilityForecast`, never as an action.

## Layer responsibilities

**`core/`** — frozen domain contracts. No I/O, no config, no logging side effects. Every
other package depends on it and it depends on nothing but the standard library and
pydantic. This is what makes the same strategy code run under backtest, paper and live.

**`adapters/`** — one package per external source. Each owns its authentication, its rate
limiting, its pagination quirks, and its normalization into core types. Adapters never
decide anything; they translate. A missing credential yields `NO_CREDENTIALS` health and a
single warning, never an exception at boot.

**`storage/`** — SQLite in WAL mode for state that must be transactional and re-read
(experiments, orders, fills, positions, balances, trader registry, checkpoints); append-only
Parquet for high-volume immutable ticks (books, trades, news, social, forecasts); DuckDB as
the query engine across the Parquet lake. Money is `Decimal` in Python and `TEXT` in SQLite
so no precision is lost.

**`matching/`** — turns "these two markets look similar" into a defensible boolean. A
semantic or LLM proposer may *suggest* candidates; only the deterministic resolution-rule
validator can *approve* one. The cross-venue strategy may act automatically only where
`same_outcome_boolean` is true and confidence clears a stringent threshold.

**`signals/`** — pure feature functions with no I/O and no wall clock. Every time-dependent
function takes an explicit `now`, which is what makes replay reproducible.

**`strategies/`** — small independent plugins. A strategy receives a read-only
`StrategyContext`, reacts to events, and emits `OrderIntent` / `ProbabilityForecast`. It
gets no credentials, no broker handle, no network, and no `datetime.now()`.

**`execution/`** — the only place that turns an intent into an order. `PaperBroker` walks
the real visible book at the simulated arrival time; `KalshiLiveBroker` exists but is
structurally unreachable without every gate in `docs/live_safety.md` satisfied.

**`experiments/`** — immutable identity, variant generation from YAML, the runner that owns
the per-variant sleeves, and the champion/challenger promotion state machine.

**`analytics/`** — the scorekeeper. Separates `IN_SAMPLE` / `VALIDATION` / `FORWARD_PAPER` /
`LIVE` and refuses to blend them.

**`daemon/`** — supervisor, health watchdog, crash recovery. Restart reloads open orders,
positions and experiment state; it never creates a fresh bankroll because the process died.

## Why the broker boundary is where it is

The most common way a research system lies is that the backtest and the live path are two
different implementations. MarketLab makes that structurally impossible: a strategy cannot
observe its own mode. It emits an intent; the broker decides. Swapping `PaperBroker` for
`KalshiLiveBroker` changes nothing a strategy can see.

The corollary is that the PaperBroker must be pessimistic enough to be believable. It walks
real depth, charges real fees, applies configured latency before choosing which book
snapshot to consume, and models limit fills under three assumptions (`TOUCH`,
`TRADE_THROUGH`, `QUEUE`) whose results are reported side by side.

## Time and the anti-look-ahead rule

Two mechanisms enforce point-in-time discipline:

1. **Clock injection.** `Clock` is a protocol; `LiveClock` in production, `SimulatedClock`
   in replay. No component outside `clock.py` calls `datetime.now()`. A `SimulatedClock`
   refuses to move backwards.

2. **Four timestamps on every external record.** `event_time` (when it happened),
   `published_time` (when the source released it), `first_seen_time` (when MarketLab
   observed it), `ingested_time` (when it was stored). A strategy deciding at `T` may only
   see records with `first_seen_time <= T`, checked by `BaseEvent.visible_at(T)`.

The second is what stops the subtle failures: a politician's July transaction disclosed in
August, an economic series revised after the fact, an edited social post, a news article
back-dated to the event it describes, and — most importantly — the discovery timestamp of a
public trader. Selecting today's leaderboard winner and backtesting their past is the single
easiest way to manufacture an edge that does not exist.

## Determinism

Given the same data, strategy version, parameters, random seed and execution model, replay
must produce identical orders and P&L. That requires: injected clocks, seeded randomness,
seeded latency jitter, ordered event processing, and `Decimal` arithmetic rather than float
comparison. The one honest exception is the local LLM, whose output is not bit-reproducible
across model or runtime versions — which is exactly why `llm_model_id` and `prompt_hash` are
part of the immutable experiment identity rather than incidental metadata.

## Concurrency model

One event loop. Adapters are async tasks feeding a shared queue. The strategy runner drains
the queue in order and fans each event to subscribed strategies synchronously — strategies
are cheap, pure-CPU, and running them concurrently would make ordering non-deterministic for
no benefit. Blocking work (SQLite writes, Parquet flushes, LLM inference) goes through
`asyncio.to_thread` or a bounded semaphore so it cannot stall ingestion.

## Storage growth

Parquet files are date-partitioned and append-only. SQLite holds state, not history. If the
Parquet lake or the tick rate eventually outgrows a single machine, migrating to
Postgres/NATS/Kafka becomes its own experiment — but that is a decision for measured
evidence, not for anticipation.

## Failure behaviour

Every feed carries `last_message_at`, latency, error count and reconnect count, and reports
`HEALTHY` / `DEGRADED` / `STALE` / `DOWN`. When a required feed goes `STALE`, new intents
are refused, resting live orders are cancelled where safe, and an alert is written. The
system stops trading rather than trading blind.

Optional sources are genuinely optional. Missing X credentials produce a warning and a
disabled adapter. Missing Kalshi credentials still allow public market-data ingestion and
full paper trading — only authenticated websocket channels and live orders are lost.

## Runtime additions (2026-10)

```
ingest ──► event bus ──► runner.dispatch ──┬─► EvidenceCache (news/social/filings/trader, in memory)
  │  Google News / GDELT (targeted)        ├─► TraderActionEvent translation (poly: -> matched Kalshi twin)
  │  sports matcher ──► market_matches     └─► sleeves ──► RiskGateway (+ kill switch) ──► PaperBroker
  │                         │                                                              │
  │                         └──► MatchBook (refreshed every 5 min) ──► MatchView per sleeve │
  │                                                                                         ▼
  │                     AI sleeves ──(background)──► AIStack: local ─► [OpenAI 2nd opinion]   decisions table
  ▼                                         (AssessmentCache shares one inference per market)
dashboard (read-only, 127.0.0.1:8765) ◄── SQLite (mode=ro) + daemon_status.json
```

* **AIStack** (`marketlab/ai/stack.py`): tiers `local` (Ollama), `jev` (OpenAI-compatible
  URL), `openai` (budget ledger in `data/openai_spend.json`). Arms: `local`, `hybrid`,
  `jev`, `jev_hybrid`; missing tiers mean the arm is not created.
* **Kill switch**: the file `data/KILL_SWITCH`, checked by `RiskGateway.evaluate` (stat
  cached for 1 s). Blocks every risk-increasing order; reducing orders still pass.
* **Match book** (`marketlab/matching/book.py`): approved pairs from storage, given to
  `cross_venue`/`copy_trader`/`copy_basket` as a universe-filtered view.
* **Replay** (`marketlab/experiments/replay.py`): a recorded day's Kalshi books and
  settlements through the same runner/broker on a `SimulatedClock`, into `data/replay/`.
* **Watchdog**: `START_TRADING.bat` restarts the daemon after any exit unless the kill
  switch is engaged; the daemon itself starts Ollama if it is installed but not running.
