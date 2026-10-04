# Findings from building against the live APIs

Things that turned out to be different from what the PRD and the vendors' documentation
assumed. Recorded because a wrong assumption that goes unnoticed is how a research system
starts lying to itself.

All verified against the live APIs on **2026-09-04**.

## Kalshi

**1. Prices are dollar-denominated decimal strings, not integer cents.**
The live production host returns `yes_bid_dollars`, `yes_ask_dollars`, `no_bid_dollars`,
`last_price_dollars`, `liquidity_dollars` — strings like `"0.4100"`. The integer-cents
fields the docs describe are not present. Every tutorial that multiplies by 100 is wrong
against this host.

**2. Contract sizes are fractional.**
Volumes and sizes arrive as `_fp`-suffixed decimal strings (`volume_fp: "5000.00"`,
`yes_bid_size_fp`, `open_interest_fp`). Kalshi now supports fractional contract counts.
`BookLevel.size` is an `int`, so the adapter rounds half-up — a documented lossy step that
is harmless at our order sizes but would matter to a larger operator.

**3. The order book endpoint has a different shape than documented.**
Real: `{"orderbook_fp": {"yes_dollars": [[price, size], ...], "no_dollars": [...]}}`.
Both arrays are **bids**, not a bid/ask pair. The adapter folds each NO bid at price `p`
into a YES ask at `1 - p`. The older `{"orderbook": {"yes": [[cents, size]]}}` shape is
still handled for compatibility.

**4. Settled markets report `status: "finalized"`, not `"settled"`.**

**5. There is no flat `tick_size` field.**
Markets expose `price_ranges`: a list of `{start, end, step}` dollar bands — penny ticks
near 0 and 1, coarser in the middle. `tick_size` is derived as the finest step.

**6. Fee configuration lives on the series, not the market.**
`/markets` rows carry no fee fields at all. `/series/<ticker>` exposes `fee_type`
("quadratic") and `fee_multiplier`. Without fetching the series, the standard
`ceil(0.07 · C · P · (1−P))` formula is the fallback.

**7. Kalshi lists no 1-minute or 5-minute BTC contract.**
The finest crypto granularity is **fifteen minutes** (`KXBTC15M`); hourly is `KXBTCD`,
daily is `KXBTCMAXD`. The PRD asked for BTC_1M and BTC_5M universes. Rather than silently
dropping them, `configs/universes.yaml` declares them `available: false` with the reason.
A horizon that does not exist is itself a research finding.

**8. ~12,000 zero-volume parlay markets flood the market list.**
The `Exotics` category contains multi-variate event (`KXMVE*`) contracts — cross-category
parlays. Paginating `/markets?status=open` returns almost nothing else: 11,886 of the first
12,000 rows were `KXMVECROSSCATEGORY`, all with volume 0. They are globally excluded in
`configs/universes.yaml`; without that filter every universe would be swamped by contracts
nobody trades.

**9. The real catalogue is large and well-structured.**
13,798 series across 18 categories: Sports 3,612 · Entertainment 2,535 · Politics 2,287 ·
Elections 1,651 · Financials 955 · Economics 763 · Mentions 435 · Climate and Weather 367 ·
Science and Technology 320 · Crypto 273 · Companies 176 · World 143 · Health 96 ·
Commodities 81 · Social 52 · Transportation 38 · Exotics 13 · Education 1.

Kalshi has an entire **Mentions** category — markets on what a public figure will say
(`KXELONMENTION`, `KXCOMMENTTRUMP`, `KXSECPRESSMENTION`). These are a cleaner target for
the public-statement tracker than trading a stock reaction, because they resolve on the
statement itself rather than on a contested price move.

## Polymarket

**10. The geoblock is real and confirmed from this machine.**
`GET /api/geoblock` → `{"blocked": true, "country": "US", "region": "IN"}`.
`settings.polymarket_global_execution` is `False` and no code path can set it True. There
is no `close_only` field on the wire; the adapter derives `close_only = blocked` as the
conservative reading.

**11. The official leaderboard silently ignores `period` and `metric`.**
`data-api.polymarket.com/v1/leaderboard` validates `category` (400 on an unknown value) and
genuinely filters on it. But `period` and `metric` are accepted and ignored — identical rows
come back for every value, verified by comparing outputs across all of them.

This mattered. The PRD's board sweep (overall/day/pnl, overall/week/pnl, …) would have
issued 15 queries that returned **the same board 15 times**, and stored them as though they
were 15 independent snapshots. That is a manufactured sample size — precisely the
self-deception this project exists to prevent.

The time dimension does work on the legacy host: `lb-api.polymarket.com/{profit,volume}`
honours `window` ∈ {1d, 7d, 30d, all} and the rankings genuinely change. So
`configs/copy_traders.yaml` now takes the **category** dimension from the official host and
the **period** dimension from the legacy host, and dedupes by wallet.

**12. `/fee-rate-bps` does not exist** (404). Fee information is on the market object
(`feeSchedule.rate`, or `makerBaseFee`/`takerBaseFee`). `/spreads` (plural) is POST-only;
`/spread` (singular) works on GET.

**13. Gamma returns stringified JSON inside JSON.**
`outcomes`, `outcomePrices` and `clobTokenIds` arrive as JSON-encoded *strings*, not arrays.
They must be parsed defensively.

**14. `/activity` mixes non-trade rows.**
It returns `MAKER_REBATE`, `TAKER_REBATE`, `YIELD` and `REWARD` entries alongside `TRADE`.
Treating those as trades would corrupt every copy-trading statistic. They are filtered.

**15. Polymarket US requires credentials for everything.**
`api.polymarket.us` returns `401 "Missing required API key headers"` on every path,
including nominally public ones (`/`, `/markets`, `/events`, `/health`). There is no public
data tier. The adapter is implemented and reports `DOWN` with that detail. This confirms the
operating decision: Kalshi is the only execution venue.

**16. A base-URL-with-path bug that failed silently.**
`poly_leaderboard` and `poly_geoblock` in `SourcesConfig` are full URLs including a path.
Passing an empty path to `HttpAdapter` made httpx issue a 301/308 that the client does not
follow, returning `{}` — a *silent* empty result rather than an error. Fixed by splitting
origin from path. Worth noting as a class of bug: a redirect that returns empty data is far
more dangerous than one that raises.

## Environment

**17. `uv run pytest` could not import `marketlab`.**
The editable install was registered in `direct_url.json` but had no `.pth` wiring, so
pytest's console script could not resolve the package while `python -c` could (it inherits
cwd on `sys.path`). This would have blocked every team. Fixed by adding
`pythonpath = ["."]` to `[tool.pytest.ini_options]`.

**18. Local models available, none downloaded.**
Ollama was already running with `gpt-oss:20b`, `glm-4.7-flash:latest`, `qwen2.5vl:7b` and
`qwen3:0.6b`. Per the PRD, nothing new was pulled; the provider autodetects and prefers
`gpt-oss:20b` for analysis and `qwen3:0.6b` for fast classification and tests.

## Local AI

**19. httpx's default 5-second timeout was silently truncating every slow inference.**
`httpx.AsyncClient` defaults to a 5s timeout regardless of any outer
`asyncio.wait_for(...)` wrapper. Every call to a model slower than 5s was being cut off and
returning an abstain — which looks exactly like "the model declined to answer" rather than
like a bug. Fixed by constructing the client with `timeout=None` and letting `wait_for` be
the sole timeout authority. A failure mode that degrades into a plausible-looking result is
far more dangerous than one that raises.

**20. Ollama's structured-output `format` enforces JSON types but not string patterns.**
Pydantic renders a `Decimal` field as `anyOf: [number, pattern-string]`. Both `gpt-oss:20b`
and `qwen3:0.6b` exploited the string branch and wrote qualitative words — `"moderate"` for
a confidence, `"abstain"` for `p_yes` — which are schema-valid and semantically useless.
Fixed by collapsing `Decimal` fields to a bare `{"type": "number"}` in the emitted schema.
Reproducible across models, not a one-off.

**21. Measured inference latency, full structured assessment:**

| model | latency |
|---|---|
| `qwen3:0.6b` | ~2.5 s |
| `qwen2.5vl:7b` | ~9.5 s |
| `glm-4.7-flash` | ~61 s |
| `gpt-oss:20b` | ~100–124 s |

`gpt-oss:20b` is the strongest reasoner but at ~2 minutes per assessment with 2 concurrent
slots it caps out near one market per minute. That is fine for a political or economic
contract resolving in weeks and useless for anything intraday. `configs/default.yaml` now
selects the analyst model by the market's horizon and puts a cheap `qwen3:0.6b` triage pass
in front, so the slow model is only woken when the fast one says a market is worth the time.

**22. Even the 20B model does not reliably cite `evidence_ids`** despite explicit prompt
instructions. Empty citations pass the validator (they are not a correctness failure), but
they mean that call produced no traceability. Worth watching during prompt iteration — and
a reason the deterministic validator, not the prompt, is where the guarantees live.

## Runtime bugs found by actually running the daemon

None of these were visible from unit tests. Every one was found by starting the system and
watching what it really did.

**23. A supervised loop that returned normally was restarted instantly, forever.**
`run_supervised` wrapped each ingest loop in `while not stop: await fn()`. Six sources
without credentials (FRED, X, Alpaca, odds, …) return immediately by design, so the wrapper
re-entered them in a tight loop with no await in it. The event loop was saturated: no
heartbeat, no orders, no events processed, 100% CPU — a daemon that looked hung while
insisting it was fine. A normal return now ends supervision; only an exception earns a
retry.

**24. Strategies were trading Polymarket.** The single most serious bug in the build.
`BaseStrategy` stamps a constant `VENUE = Venue.KALSHI` on every intent, and the risk
gateway checked only `intent.venue`. A strategy iterating `ctx.markets()` therefore emitted
orders against `poly:` contracts that arrived at the gateway *declaring themselves Kalshi
orders*, and sailed through. 166 fills landed on markets this deployment can never trade
before it was caught. Two independent guards now exist — the gateway checks the market's
own venue, and `should_skip` refuses a non-execution venue — plus
`tests/unit/test_venue_isolation.py` to make sure it cannot return silently.

**25. Every filled order was lost; only rejections persisted.**
`PaperBroker` writes each fill inside its execution helper but persists the *order* after
that helper returns, so on the filling path the fill reached storage first and violated
`fills.order_id`'s foreign key. The `IntegrityError` propagated into the strategy and got
it counted as failing. The tables therefore read "nothing ever fills" rather than "the
writes are failing". The store bridge now buffers early fills until their order row exists,
and a persistence failure can never reach strategy code.

**26. Per-row commits on the forecast path stalled the event loop.**
Every `ProbabilityForecast` opened its own SQLite transaction. At ~290k forecasts the main
loop spent minutes inside `save_forecast` and the daemon appeared hung. Forecasts are now
buffered and written with one `executemany`; a failed flush keeps the buffer rather than
dropping the calibration record.

**27. Normalising 32,788 markets blocked everything for seconds.**
The undirected `/markets` sweep is pure CPU with no await, and ran every 60s. It starved
the orderbook loop down to roughly one cycle where it should manage a dozen, so strategies
had almost no books to trade against. Now it yields every 1,000 rows, and the sweep is
shallow (3 pages) because targeted per-series discovery already returns what the universes
need.

**28. Sequential orderbook fetches made the venue look dead.**
~200ms per book × 120 markets exceeded the health monitor's staleness window, so Kalshi was
declared STALE between passes and the supervisor correctly — but needlessly — halted
trading. Books are now fetched with bounded concurrency (8) and health is recorded per call
rather than per pass.

**29. `CopyBasketStrategy` raised on an empty roster.**
Zero validated traders is the *normal* day-one state — wallets are discovered and then
forward-tracked for weeks before any qualifies. Raising in `__init__` aborted sleeve
creation for the whole tournament: 19 of 388 sleeves came up. The basket now constructs and
stays dormant until the roster fills, and the runner isolates any strategy that fails to
construct so one bad variant costs only its own sleeve.

**30. Every restart minted 388 brand-new $50 sleeves.**
`start_timestamp` is part of the immutable experiment identity, so a restart produced
entirely new experiment ids and orphaned the previous run — 1,506 experiments accumulated
across a few test boots. A `cohort_key` (the identity minus `start_timestamp`) now lets a
restart resume its live sleeves, while a DEAD or DISABLED sleeve is deliberately *not*
resumed because that genuinely is a new cohort.

**31. Ingested data evaporated.** `marketlab ingest --once` built its `IngestService`
without a store, so it fetched several hundred markets, reported them, and dropped every
one. `marketlab markets` then showed an empty database immediately after an apparently
successful ingest.

**32. A wiring TypeError was reported as "adapter not implemented".**
The GDELT adapter was constructed with the wrong keyword and the failure was rendered in
`doctor` as a missing feature rather than a bug. Construction errors are now recorded and
surfaced verbatim.

**33. Unthrottled forecast recording would have filled the disk in days.**
Every strategy emits a `ProbabilityForecast` per subscribed market per 1s tick. Across ~400
sleeves that measured at **623,000 rows in nine minutes — 358 MB of SQLite, projecting ~90M
rows and tens of gigabytes per day.** The run would have died of disk exhaustion long
before any strategy accumulated a statistically meaningful sample.

Recording is now throttled per (experiment, market): a forecast is kept when at least 60s
have passed **or** the probability moved by >=0.01. Calibration needs a representative
sample of forecasts, not every redundant restatement of an unchanged number, and every
genuine move is still captured.

Measured after the fix: 9 MB and ~1.56M rows/day — a 40x reduction in size and 57x in row
count, with fill behaviour unchanged.

## Second tournament: what a week of live running hid (2026-10-03/04)

**34. The AI layer never ran.** `news_probability` read an `llm_provider` param that
nothing injected, and its `process_pending()` - the only path that calls a model - was
never invoked. For the whole first tournament all 72 AI sleeves ran against
`DisabledProvider` and abstained, so "AI vs non-AI" had no AI arm at all. The runner now
injects a tiered stack (local / Jev / OpenAI) and drains each AI sleeve's queue in the
background on every tick.

**35. News was never ingested.** GDELT was only health-probed: no `NewsEvent` ever reached
the event bus, and the AI evidence layer was empty. Worse, GDELT's DOC API throttled our
targeted queries into uselessness (three 429s, then an empty 200 for "federal reserve
rates"), while Google News RSS returned 47 stories for the same query in 0.8 s. News now
comes from keyword queries built from each AI-universe market's own title.

**36. A 3B model's "I don't know" looked like a 42-point edge.** `qwen2.5:3b` answered
exactly `p_yes=0.50, confidence=0.9` on a Nasdaq range bucket priced at 0.075. Without a
market price in the prompt (deliberately, to avoid anchoring) a small model defaults to a
coin flip, and the edge arithmetic happily reads that as massive mispricing. A bare 0.50
is now an abstain.

**37. Copy trading and cross-venue were structurally dead - three separate breaks.**
(a) The free-text matcher approved 0 of 2,372 pairs: every one lacked a stated resolution
authority, and most compared a Kalshi *event* title against a Polymarket threshold
contract. (b) Nothing injected `params["matches"]`, so an approved pair would have been
invisible anyway. (c) Wallet trades arrived keyed by the `poly:` market with `side=None` for
any outcome not literally "Yes"/"No" - i.e. every sports trade. Game-winner pairs are now
proven structurally (178 on the first pass), delivered through a universe-filtered match
book, and matched Polymarket buys are re-pointed at the Kalshi twin with the correct side.

**38. One malformed title disabled the matcher entirely.** "13PM" parsed to hour 25 and
`datetime()` raised, aborting the *whole* matching pass every 15 minutes since. Unparseable
times are now "unknown". Same lesson as #16 and #32: an isolated bad record must cost only
itself.

**39. A doubleheader is two bets with one Polymarket name.** `KXMLBGAME-...DETCLEG2` (game
2) matched the single Polymarket Tigers-Guardians market because a lenient prefix rule
read `cleg2` as `cle`. That would have traded game 2 against game 1's price. G-suffixed and
duplicate-slot games are now never matched.

**40. The variant cap fails closed - all the way closed.** Adding sports universes pushed
the sweep to 448 variants against `max_total_variants: 400`, and the sweep *raises* rather
than truncating, so the daemon would have started with zero sleeves. Universes where no
pair had ever been approved were dropped to make room instead.

**41. Every trade's "why" was being discarded.** `orders` stored what happened, never the
intent's rationale or features, so no filled trade could be explained after the fact. A
`decisions` table now keeps them for every fill (rejections already carry their reason).

**42. One strategy's housekeeping froze the whole daemon.** Twenty-five minutes after a
restart, trading halted with `kalshi_rest` "down" and HTTP calls reporting 975-second
latencies - against a 10 s client timeout and a network that answered `curl` in 0.1 s.
Heartbeats (every 30 s) arrived 211 s, 289 s, then 587 s apart. `py-spy dump` on the live
process showed the cause in one frame: `binary_parity._refresh_pairs` running regex claim
extraction over every market in the shared context and comparing all n^2 pairs, on the
event loop, in each of 12 sleeves, whenever the market count changed. It had been latent
all along; growing the tracked set from 115 to 505 markets made it fatal. Lesson worth
keeping: when every source "hangs" at once, profile the process before blaming the network.

**43. `ctx.markets()` was every market, for every sleeve, every second.** Strategies that
loop over it on each tick paid for thousands of markets they never trade, and loops that
don't check universe membership - the random control among them - traded markets outside
their sleeve's declared universe. Each sleeve now gets a context whose `markets()` is its
own universe; lookups by id stay global.

**44. A green test run was not a gate.** One commit in this session was pushed with two
failing tests because the shell chain checked `tail`'s exit status rather than
pytest's. The gate now reads pytest's own exit code.

**45. The weather strategy had its station map withheld, and would have mispriced
without it.** `configs/universes.yaml` maps each Kalshi weather series to an NWS station;
the runner used the map to *route* weather events but never handed it to the strategy, so
in a week it produced zero forecasts. The second bug was hiding behind the first: most
Kalshi daily-high markets are two-degree range buckets (`-B68.5` = 68-69 F), and the
strategy only modelled thresholds, so a bucket would have been priced as ">= 69". Same
lesson as #37 and #39: wiring a dead path back up is when its untested logic first meets
real data.

**46. A P&L leaderboard is a survivorship filter.** Half the scored leaderboard (155 of
300) qualified, at a median ROI of 16% and a Sharpe-like 2.6. That is not 155 skilled
traders; the leaderboard surfaces wallets *because* their recent record is good. A
backward-looking score cannot separate skill from a streak among survivors, so QUALIFIED
now also requires the older and newer halves of a record to be profitable independently.
The copy arms remain the only real forward test.
