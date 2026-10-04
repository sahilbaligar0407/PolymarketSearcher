# MarketLab

An autonomous, terminal-only, backend-only **prediction-market research laboratory**.

It consumes real production market data, runs many independent strategies against that
same data, measures them honestly, kills the weak ones, freezes the durable ones, and only
then considers combining survivors into a tightly risk-controlled live strategy.

MarketLab is not a trading bot. It is a strategy tournament with a trustworthy scorekeeper.

```
REAL MARKET DATA
     ├── market microstructure      ├── news
     ├── cross-venue discrepancies  ├── social / public statements
     ├── sports consensus           ├── SEC / company events
     ├── BTC quantitative models    ├── public winning traders
     └── local AI probability estimates
                    │
                    ▼
        MANY INDEPENDENT STRATEGIES
                    │
                    ▼
        REALISTIC LOCAL PAPER BROKER      ← walks the real book, charges real fees,
                    │                       models latency and queue position
                    ▼
        $50 EXPERIMENT SLEEVES
                    │
                    ▼
        FORWARD PERFORMANCE DATABASE
                    │
        ├── winners  ├── losers  ├── latency failures
        ├── overfit strategies   └── regime specialists
                    │
                    ▼
        CHAMPION / CHALLENGER FILTER
                    │
                    ▼
        DIVERSIFIED QUALIFIED ENSEMBLE
                    │
                    ▼
        $50 LIVE-SMALL GATE  (hard-disabled by default)
```

> **This repository also contains PolymarketSearcher**, the earlier Node.js top-trader
> position collector and AI budget allocator (`collect.js`, `server.js`, `lib/`, `public/`).
> It is independent of MarketLab; see [`docs/PolymarketSearcher.md`](docs/PolymarketSearcher.md).

## Venue policy

**Kalshi is the only execution venue.** Polymarket global is a read-only intelligence
source — prices for cross-venue comparison, the public leaderboard, and public trader
activity. The operator is in the US, where global Polymarket is close-only for new
positions, so MarketLab never submits a Polymarket order and never attempts to bypass the
geoblock. A Polymarket discrepancy is a *signal to trade the equivalent Kalshi contract*,
and only after the matching engine certifies the two markets are the same bet.

See [`docs/live_safety.md`](docs/live_safety.md) for every gate that stands between this
repository and real money.

## Quick start

```bash
uv sync --extra dev
cp .env.example .env          # optional: fill in what you have; nothing is required for PAPER
uv run marketlab doctor       # what is reachable, what is missing, what is disabled
uv run marketlab ingest --once
uv run marketlab markets --venue kalshi --limit 10
uv run marketlab trader-discover
uv run marketlab ai doctor
uv run pytest -m "not integration"
uv run marketlab paper start --bankroll-per-strategy 50
```

Then watch it:

```bash
uv run marketlab paper status
uv run marketlab paper leaderboard
uv run marketlab report strategies
```

No credentials are needed to run in PAPER mode. Kalshi's market data, Polymarket's public
APIs, GDELT, SEC EDGAR, NWS and Coinbase spot are all keyless. Optional sources (X, FRED,
The Odds API, Alpaca) degrade to `NO_CREDENTIALS` with a single warning and the engine
carries on.

## What makes the numbers trustworthy

The whole design exists to stop the system from fooling itself.

**Orders walk the real book.** A market buy consumes the actual visible asks, level by
level, at the simulated arrival time — never `if last_price <= my_limit: fill me`. Partial
fills happen. Liquidity runs out.

**Three fill models, and the reports show all of them.** `TOUCH` is an optimistic
diagnostic. `TRADE_THROUGH` is the conservative default. `QUEUE` estimates how much size
sat ahead of you. A strategy that is profitable only under `TOUCH` is not a live candidate.

**Latency is real and swept.** Every order carries four timestamps: decision, simulated
network send, simulated exchange arrival, and the book timestamp actually consumed. Copy
trading and apparent arbitrage are run across latency arms from 100 ms to 30 s, because
that is where most fantasy edges die.

**Point-in-time discipline is enforced, not assumed.** Every external record stores
`event_time`, `published_time`, `first_seen_time` and `ingested_time`. A strategy deciding
at time T may only see records with `first_seen_time <= T`. A politician's July transaction
disclosed in August enters a backtest in August. Economic data uses ALFRED vintages, not
today's revised series.

**Settlement comes from the venue.** Never from a model, never from a tweet.

**Probability quality is scored separately from P&L.** Brier score, log loss, calibration
curves, and improvement versus the market midpoint — because a model can forecast well and
trade badly, or forecast badly and get lucky.

**Everything is benchmarked against cheap controls**: do nothing, market midpoint, random
direction at the same frequency, and a deliberate *fade* of each strategy. AI earns its
place only by beating those out of sample.

**Losses are preserved.** No experiment is ever deleted because it makes the aggregate
statistics look worse. A dead $50 sleeve stays in the database with its final equity.

## The $50 rule

Every strategy variant gets a virtual $50 sleeve. Within a run there is no replenishment,
the sleeve can reach zero, and a restart creates a **new experiment id** — a new cohort,
not a continuation that hides the earlier failure.

The real bankroll is stricter. It may decline, but the system will not martingale, average
down automatically, borrow, use leverage, or refill without recording a new capital epoch.

The target is not maximum backtest return. It is *maximum repeatable out-of-sample expected
value after fees, spread, latency, failed fills, stale information and correlated risk.* A
strategy with a 500% backtest and no forward edge is a loser. One making 5% with low
drawdown and repeatable execution may be a winner.

## Immutable experiments

An experiment id is derived from strategy name, version, git commit, parameter hash,
market universe, venue, data version, execution-model version, feature version, LLM model
id, prompt hash, start timestamp and starting bankroll. A strategy cannot be quietly edited
under the same identity — a change is a new experiment.

## Evidence classes

Ideas enter as hypotheses with an explicit provenance grade, recorded in
[`docs/research_registry.yaml`](docs/research_registry.yaml):

| Class | Meaning |
|---|---|
| A | credible academic empirical evidence (time-series momentum, pairs trading) |
| B | established quantitative theory (inventory-aware market making, probability arbitrage) |
| C | prominent open-source implementation pattern |
| D | directly observable public trader behaviour |
| E | plausible microstructure hypothesis (book imbalance, short-horizon reversal) |
| F | AI / social hypothesis |
| G | viral or unverified claim |

Class G may be investigated but can never skip validation. Popularity is not evidence of
profitability, and neither is a large star count.

## The local model is an analyst, not the broker

```
DATA → RETRIEVAL → LOCAL MODEL → STRICT JSON → DETERMINISTIC VALIDATOR
     → STRATEGY → RISK GATEWAY → BROKER
```

The model runs locally (Ollama autodetected). It cannot read secrets, invoke a shell,
construct an order, change a risk limit, or activate live mode. Malformed or unverifiable
output causes an **abstain** — it is never silently repaired into something tradeable.
There is no `BUY because AI said so` record anywhere in the audit trail.

## Layout

```
configs/     layered YAML: universes, strategies, risk, sources, copy traders
docs/        PRD, architecture, CONTRACTS.md, live_safety.md, research_registry.yaml
marketlab/
  core/        frozen domain contracts: instruments, orders, events, portfolio,
               strategy, broker, probability
  adapters/    kalshi, polymarket_global (read-only), polymarket_us, gdelt, sec,
               crypto, weather, fred, x, bluesky, alpaca, sports_odds
  storage/     SQLite WAL (state) + append-only Parquet (ticks) + DuckDB (analytics)
  execution/   paper broker, fill models, risk gateway, gated live broker
  matching/    resolution-rule validator and cross-venue market matching
  signals/     orderbook, technical, news, social, filings, copy trader, sports
  strategies/  the tournament entrants
  ai/          provider autodetection, prompts, retrieval, validator, calibration
  experiments/ immutable registry, runner, parameter sweep, promotion
  analytics/   metrics, calibration, attribution, bootstrap, reports
  daemon/      supervisor, health watchdog, crash recovery
tests/       unit, integration, replay
data/        sqlite db, parquet lake, reports, logs  (gitignored)
```

## Promotion rules

Nothing is promoted because it sounds smart, because GitHub likes it, because somebody
posted a huge return, because an LLM is confident, or because the backtest is beautiful.

A strategy is promoted because it was defined before the forward test, saw only information
available at the time, traded against realistic executable prices, survived fees, spread,
latency and poor fills, repeated its edge across enough independent observations, did not
depend on one lucky outcome, and fits inside the $50 risk budget.

If it keeps working, it is frozen and challengers try to beat it. If it breaks, it is
demoted. The loss is never erased.

## Disclaimer

Research software. Not financial advice. Live trading is disabled by default and every
strategy here is an unproven hypothesis until this system's own forward data says otherwise.
