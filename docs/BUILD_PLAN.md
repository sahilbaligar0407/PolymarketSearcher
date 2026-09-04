# MarketLab build plan

Tracks progress against the PRD's "Definition of done". Updated as waves land.

## Deployment decision (locks several design choices)

Polymarket global is **read-only intelligence**: markets, books, leaderboard, public trader
activity. **Kalshi is the only execution venue.** A Polymarket discrepancy is a signal to
trade the equivalent *Kalshi* contract. No Polymarket order code exists anywhere in the
tree, and the geoblock is never bypassed.

## Wave 0 — foundation (DONE)

- [x] Environment: uv 0.12.9, Python 3.12.14 pinned, venv synced
- [x] `pyproject.toml`, `.gitignore`, `.env.example`, `.python-version`
- [x] `marketlab/clock.py` — `Clock` protocol, `LiveClock`, `SimulatedClock`
- [x] `marketlab/core/instruments.py` — `NormalizedMarket`, `OrderBook`, `Venue`, price units
- [x] `marketlab/core/orders.py` — `OrderIntent`, `Order`, `Fill`, latency timestamps
- [x] `marketlab/core/events.py` — 14 event types with the 4-timestamp discipline
- [x] `marketlab/core/portfolio.py` — `Position`, `Portfolio`, $50 sleeve accounting
- [x] `marketlab/core/strategy.py` — `Strategy` ABC, `StrategyContext`, `ProbabilityForecast`
- [x] `marketlab/core/broker.py` — `Mode`, `Broker` ABC
- [x] `marketlab/core/probability.py` — Brier, vig removal, GBM, Kelly, `expected_edge`
- [x] `marketlab/settings.py` — layered YAML + `.env`, live arm token
- [x] `marketlab/logging.py` — structlog with secret redaction
- [x] `configs/default.yaml`, `paper.yaml`, `live.example.yaml`, `risk.yaml`
- [x] `docs/CONTRACTS.md` — the frozen interface every team builds against

## Wave 1 — sources, storage, execution (IN PROGRESS)

| Team | Owns | Status |
|---|---|---|
| STORAGE | `storage/` (SQLite WAL, Parquet, DuckDB) | running |
| KALSHI | `adapters/base.py`, `ratelimit.py`, `adapters/kalshi/` | running |
| POLYMARKET | `adapters/polymarket_global/`, `polymarket_us/`, `signals/copy_trader.py` | running |
| EXECUTION | `execution/` (PaperBroker, fill models, risk gateway) | running |
| INFOSOURCES | `adapters/{gdelt,sec,crypto,weather,fred,x,bluesky,alpaca,sports_odds}/`, `signals/{news,filings,social}.py` | running |
| AI | `ai/` (provider, ollama, schemas, prompts, retrieval, validator, calibration) | running |

## Wave 2 — matching, strategies, experiments (QUEUED)

| Team | Owns |
|---|---|
| MATCHING | `matching/` — resolution-rule validator, semantic candidate proposal, cross-venue links |
| SIGNALS | `signals/{orderbook,technical,sports}.py` |
| STRATEGIES | the 12 mandatory strategies in `strategies/` |
| EXPERIMENTS | `experiments/` — immutable registry, runner, sweep, promotion |
| ANALYTICS | `analytics/` — metrics, calibration, attribution, bootstrap, reports |

## Wave 3 — daemon, CLI, integration (QUEUED)

| Team | Owns |
|---|---|
| DAEMON | `daemon/` — supervisor, health watchdog, crash recovery |
| CLI | `cli.py` — every command in the PRD's list |
| INTEGRATION | end-to-end wiring, `scripts/`, replay tests, no-look-ahead tests |

## Definition of done (from the PRD)

Verified by running the system, not by inspecting the code.

| Requirement | Status |
|---|---|
| Repository/environment initialized | done |
| Terminal CLI works | done - every PRD command implemented |
| Kalshi production market ingestion | done - 400 live markets, all volume > 0 |
| Local Kalshi PaperBroker | done - validated against a live NFL book |
| Polymarket global read-only ingestion | done - 100 markets mirrored |
| Global Polymarket geoblock enforced | done - confirmed blocked, US, execution False |
| Official Polymarket leaderboard ingestion | done - 492 unique wallets |
| Public trader forward tracker | done - discovery writes DISCOVERED only |
| Polymarket US public adapter | done - reports DOWN (401 on all paths) |
| Local AI autodetection | done - Ollama, gpt-oss:20b |
| GDELT news ingestion | done - rate-limited to 1 req/15s |
| SEC ingestion | needs SEC_USER_AGENT with an email; reports NO_CREDENTIALS until then |
| BTC strategy variants | done - 27 sleeves (15m/1h/daily) |
| Sports strategy framework | done - 40 sleeves |
| Politics/news strategy framework | done - 24 sleeves |
| Trump/public-statement tracker | done - event-study mode, emits no trades by design |
| Momentum baseline | done - 60 sleeves |
| Mean-reversion baseline | done - 24 sleeves |
| Book imbalance | done - 32 sleeves |
| Market maker | done - 27 sleeves |
| Cross-market comparator | done - 30 sleeves |
| Copy trader | done - 60 + 24 basket sleeves |
| $50 virtual sleeves | done - 388 sleeves, $19,400 total |
| Fee/slippage/latency-aware execution | done - fee math verified against Kalshi's formula |
| Immutable experiment IDs | done - plus a cohort key so restarts resume |
| Persistent statistics | done - SQLite + Parquet + DuckDB |
| Terminal leaderboard | done - `paper leaderboard [--traded]` |
| Restart recovery | done - three consecutive boots added zero experiments |
| Live trading disabled by default | done - HARD DISABLED, every gate documented |
| Test suite green | done - 427 passing, ruff clean, mypy clean (118 files) |
| Continuous paper process successfully started | done - running, heartbeat every 30s |

## Verified running state

```
uptime 514s | events 17,659 | markets 400 | sleeves_alive 388 | orders 4,217 | trading_allowed true
fills: 161 on Kalshi, 0 on Polymarket
```
