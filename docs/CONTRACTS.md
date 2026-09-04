# MarketLab shared contracts

**Read this before writing any code.** These modules are already implemented and are
frozen. Build against them; do not modify them without telling the orchestrator.

## Environment

- Windows 11, `uv` at `C:\Python314\Scripts\uv.exe` (add `/c/Python314/Scripts` to PATH in bash).
- Repo root: `C:\Users\sahil\Documents\Code\Kalshi`
- Python 3.12.14 pinned via `.python-version`; venv at `.venv`.
- Run anything with `uv run python ...`, `uv run pytest`, `uv run ruff check .`, `uv run mypy marketlab`.
- `pyproject.toml` sets `pythonpath = ["."]`, so `uv run pytest` resolves `marketlab` without
  an editable install. Registered markers: `integration` (hits a real external API) and
  `slow`. Run the fast suite with `uv run pytest -m "not integration"`.
- Deps already installed: httpx, websockets, pydantic v2, pydantic-settings, typer, rich,
  orjson, polars, pyarrow, duckdb, numpy, scipy, statsmodels, scikit-learn, tenacity,
  feedparser, trafilatura, pyyaml, cryptography, structlog, pytest(+asyncio,cov), ruff, mypy.
  **If you need another dependency, add it to `pyproject.toml` `[project].dependencies` and
  run `uv sync --extra dev`.** Prefer stdlib/existing deps.

## The one non-negotiable rule

Polymarket is **read-only** for this deployment (US geoblock — never bypassed).
**Kalshi is the only execution venue.** A Polymarket price discrepancy is a *signal to
trade the equivalent Kalshi contract*, never a Polymarket order leg. Any code path that
could POST an order to Polymarket global must not exist.

## Modules that exist (frozen contracts)

### `marketlab/clock.py`
`Clock` protocol: `.now() -> datetime` (tz-aware UTC), `await .sleep(seconds)`.
`LiveClock`, `SimulatedClock(start)` with `.set()` / `.advance()`.
**Never call `datetime.now()` outside `clock.py`.** Always take an injected `Clock`.

### `marketlab/core/instruments.py`
- `Venue` StrEnum: `KALSHI`, `POLY_GLOBAL`, `POLY_US`, `ALPACA`, `SYNTHETIC`.
  `EXECUTION_VENUES = {KALSHI, POLY_US}`, `READ_ONLY_VENUES = {POLY_GLOBAL, ALPACA}`.
- `MarketStatus`, `OutcomeType`, `Category`, `Side` (`YES`/`NO`, `.opposite`).
- `to_probability(v)` -> `Decimal` quantized to `PROB_QUANTUM = 0.0001`, raises outside `[0,1]`.
  `cents_to_probability(int)`, `probability_to_cents(Decimal)`.
- `Fees(formula, taker_rate, maker_rate, settlement_rate, min_fee_cents)`.
- `NormalizedMarket` — frozen pydantic model, the universal market object. Fields:
  `canonical_id, venue, venue_market_id, event_id, title, description, resolution_rules,
  resolution_source, category, subcategory, outcome_type, yes_symbol, no_symbol,
  open_time, close_time, expected_resolution_time, timezone, tick_size, min_order,
  max_payout_per_contract, status, fees, liquidity, volume, open_interest, raw`.
- `BookLevel(price: Decimal, size: int)`, `OrderBook(canonical_id, venue, timestamp,
  bids, asks, venue_timestamp, sequence)` with `.best_bid/.best_ask/.mid/.spread/
  .microprice/.depth(levels, side)/.is_stale(now, max_age)`.
  **Convention: both sides are expressed in YES-probability terms.** A Kalshi NO bid at
  30c is folded into a YES ask at 0.70 by the adapter.
- `Trade(canonical_id, venue, timestamp, price, size, aggressor, trade_id)`.

### `marketlab/core/orders.py`
- `OrderType`, `TimeInForce`, `Action` (BUY/SELL), `OrderStatus`, `RejectReason`.
- `OrderIntent` — what a strategy emits. Frozen. Key fields: `intent_id, strategy_id,
  experiment_id, canonical_id, venue, side, action, quantity, order_type, limit_price,
  time_in_force, decision_time, rationale, features, evidence_ids, model_probability,
  expected_edge, replaces_order_id`.
- `Fill` — `fill_id, order_id, canonical_id, venue, side, action, price, quantity, fee,
  timestamp, is_maker, book_timestamp_used, level_breakdown`.
- `Order` — broker-side state incl. the four latency timestamps
  (`decision_timestamp, simulated_network_send_timestamp,
  simulated_exchange_arrival_timestamp, book_timestamp_used`), `reference_price`,
  `.slippage`, `.unfilled_quantity`, `.is_terminal`.

### `marketlab/core/events.py`
`EventType`, `SourceClass`, and `BaseEvent` with the four timestamps:
`event_time`, `published_time`, `first_seen_time`, `ingested_time`, plus
`.visible_at(decision_time)` → `first_seen_time <= decision_time`.
**`first_seen_time` is the only field a strategy may gate on.**
Concrete events: `MarketUpdateEvent, BookUpdateEvent, TradeEvent, MarketStatusEvent,
SettlementEvent, NewsEvent, SocialEvent, FilingEvent, ExternalPriceEvent,
TraderActionEvent, WeatherEvent, EconomicEvent, SportsStateEvent, TimerEvent`. Plus `Alert`.

### `marketlab/core/portfolio.py`
`Position` (per canonical_id+side, `.apply(fill)`, `.settle(won)`, `.unrealized_pnl(mark)`)
and `Portfolio` (one $50 sleeve): `.apply_fill(fill)`, `.settle(canonical_id, winning_side)`,
`.equity(marks)`, `.exposure()`, `.mark(marks)`, `.is_dead(floor)`, `SleeveStatus`.

### `marketlab/core/strategy.py`
- `ProbabilityForecast` — recorded whether or not it produces a trade.
- `StrategyContext(clock, books, markets, marks, params)` — read-only world view.
  `.now()`, `.book(id)`, `.market(id)`, `.markets()`, `.mid(id)`.
- `Strategy` ABC — class attrs `name`, `version`, `evidence_class`, `universes`.
  Handlers: `on_market_update, on_book_update, on_trade, on_market_status, on_settlement,
  on_news, on_social, on_filing, on_external_price, on_trader_action, on_weather,
  on_economic, on_sports_state, on_timer, on_order_update, on_fill`.
  Output: `generate_intents()`, `drain_forecasts()`. Helpers: `self.emit(intent)`,
  `self.forecast(f)`, `self.now()`, `self.param(k, default)`.

### `marketlab/core/broker.py`
`Mode` StrEnum: `DATA_ONLY, BACKTEST, PAPER, LIVE` with `.allows_orders`, `.is_real_money`.
`Broker` ABC: `async submit(intent) -> Order`, `async cancel(order_id)`,
`async open_orders(strategy_id=None)`, `async get_order(order_id)`.

### `marketlab/core/probability.py`
`brier_score`, `log_loss`, `american_to_probability`, `decimal_odds_to_probability`,
`remove_vig`, `normal_cdf`, `gbm_touch_probability`, `gbm_barrier_touch_probability`,
`kelly_fraction`, `expected_edge`.

### `marketlab/settings.py`
`load_settings(profile=None)` → `Settings` with `.mode, .secrets, .execution, .risk,
.paper, .sources, .universes, .strategies, .copy_traders, .source_toggles,
.polymarket_global_execution, .db_path, .parquet_dir, .reports_dir, .log_dir`,
`.live_allowed`, `.ensure_dirs()`. `Secrets` reads `.env`; `.redacted()` gives a
presence-only map for `doctor`. `LIVE_ARM_TOKEN = "YES_I_ACCEPT_REAL_LOSS"`.
`SourcesConfig` holds every base URL — **use these, don't hardcode hosts.**

### `marketlab/logging.py`
`configure_logging(log_dir, level, json_console)` and `get_logger(name)`.
Structlog; secret keys and PEM blocks are auto-redacted. Use `log.info("event", k=v)`.

## Verified-working endpoints (probed 2026-09-04)

| Purpose | URL | Status |
|---|---|---|
| Kalshi markets | `https://api.elections.kalshi.com/trade-api/v2/markets?limit=N` | 200 |
| Kalshi markets (alt host) | `https://external-api.kalshi.com/trade-api/v2/markets` | 200 |
| Polymarket Gamma | `https://gamma-api.polymarket.com/markets?limit=N` | 200 |
| Polymarket CLOB | `https://clob.polymarket.com/markets` | 200 |
| Polymarket leaderboard (official) | `https://data-api.polymarket.com/v1/leaderboard?category=overall&period=week&metric=pnl&limit=N` | 200, returns `rank, proxyWallet, userName, xUsername, vol, pnl` |
| Polymarket leaderboard (legacy) | `https://lb-api.polymarket.com/profit?window=30d&limit=100` | 200, returns `proxyWallet, amount, pseudonym, name` |
| Polymarket positions | `https://data-api.polymarket.com/positions?user=<wallet>` | public, no auth |

### Kalshi live API reality (verified 2026-09-04 — differs from the public docs)

The production host has moved to dollar-denominated decimal **strings** and fractional sizes.
Anything you write against Kalshi must assume this shape, not integer cents:

- Prices: `yes_bid_dollars`, `yes_ask_dollars`, `no_bid_dollars`, `no_ask_dollars`,
  `last_price_dollars`, `liquidity_dollars` — decimal strings, e.g. `"0.4100"`.
  There is no plain integer `yes_bid` on the live host.
- Sizes/volumes: `_fp`-suffixed decimal strings (`volume_fp`, `open_interest_fp`,
  `yes_bid_size_fp`) — Kalshi now supports **fractional** contract counts.
  `BookLevel.size` is `int`, so the adapter rounds half-up; a documented lossy step.
- Order book: `{"orderbook_fp": {"yes_dollars": [[price, size], ...], "no_dollars": [...]}}`.
  Both arrays are **bids**. The adapter folds each NO bid at `p` into a YES ask at `1 - p`.
  (The older `{"orderbook": {"yes": [[cents, size]]}}` shape is also handled.)
- Settled markets report `status: "finalized"`, not `"settled"`.
- No flat `tick_size`. Markets expose `price_ranges`: `[{start, end, step}]` dollar bands
  (penny ticks near 0/1, coarser in the middle). `tick_size` is derived as the finest step.
- **Fees live on the series, not the market.** `/markets` rows carry no fee fields;
  `/series/<ticker>` exposes `fee_type` ("quadratic") and `fee_multiplier`. Default to the
  standard `ceil(0.07 * C * P * (1-P))` formula when the series has not been fetched.

`marketlab/adapters/kalshi/normalize.py` handles all of the above — go through it rather than
parsing raw Kalshi payloads yourself.

Local Ollama is running with models: `gpt-oss:20b`, `glm-4.7-flash:latest`,
`qwen2.5vl:7b`, `qwen3:0.6b`. **Do not pull new models.**

## House style

- Type-annotate public functions. `from __future__ import annotations` at the top.
- `Decimal` for all money/probability arithmetic. Never float equality on prices.
- Async everywhere for I/O; use `httpx.AsyncClient` and `websockets`.
- Every network client: timeout, retry with `tenacity`, respect 429 with backoff.
- Comment *why*, not *what*. Match the density of the existing core modules.
- Every module you own gets at least one test under `tests/unit/` or `tests/integration/`.
- `uv run ruff check .` and `uv run mypy marketlab` must stay clean for files you own.
- Missing credentials → log a WARN and degrade. Never raise at import or boot.
