"""The ingestion pipeline: every adapter, one event queue.

``IngestService`` owns every venue/data-source adapter and is the only thing in the
daemon allowed to call them.  Every normalized fact it produces goes two places:

1. the shared :class:`~marketlab.daemon.registry.MarketRegistry` /
   :class:`~marketlab.daemon.registry.BookRegistry` (so ``marketlab markets`` and the
   ``PaperBroker``'s ``book_provider``/``market_provider`` see it immediately), and
2. the single ``asyncio.Queue[Event]`` the :class:`~marketlab.daemon.supervisor.Supervisor`
   drains to feed the broker and the strategy tournament.

``run_once`` performs exactly one pass of every enabled REST source and returns an
:class:`IngestOnceResult` with real counts -- this is what ``marketlab ingest --once``
renders, and per the PRD, zero markets ingested must never look like success (see
``IngestOnceResult.ok``).  ``start``/``stop`` launch one independently-supervised
``asyncio.Task`` per source at the cadence configured in ``configs/default.yaml``'s
``ingest:`` block, so one failing source (a dead websocket, a rate-limited REST host)
never stops any other.

Every event's ``first_seen_time`` is stamped from the injected :class:`Clock` -- never
from a venue payload's own timestamp.  That is enforced at the two or three narrow
choke points below (``_market_update_event`` / ``_book_update_event`` /
``_trade_event_from_raw``), not by convention.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import random
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from marketlab.adapters.base import Adapter, SourceStatus
from marketlab.adapters.kalshi.normalize import (
    normalize_market as normalize_kalshi_market,
)
from marketlab.adapters.kalshi.normalize import (
    normalize_orderbook,
    normalize_settlement,
    normalize_trade,
)
from marketlab.adapters.kalshi.rest import KalshiRestAdapter
from marketlab.adapters.polymarket_global import geoblock as poly_geoblock
from marketlab.adapters.polymarket_global.clob import ClobAdapter
from marketlab.adapters.polymarket_global.data_api import DataApiAdapter
from marketlab.adapters.polymarket_global.gamma import GammaAdapter
from marketlab.adapters.polymarket_global.leaderboard import LeaderboardAdapter
from marketlab.adapters.polymarket_global.normalize import (
    normalize_book as normalize_poly_book,
)
from marketlab.adapters.polymarket_global.normalize import (
    normalize_market as normalize_poly_market,
)
from marketlab.adapters.polymarket_us.public import PolymarketUsAdapter
from marketlab.clock import Clock
from marketlab.core.events import (
    BookUpdateEvent,
    Event,
    ExternalPriceEvent,
    MarketStatusEvent,
    MarketUpdateEvent,
    SettlementEvent,
    TradeEvent,
)
from marketlab.core.instruments import (
    Category,
    MarketStatus,
    NormalizedMarket,
    OrderBook,
    Venue,
)
from marketlab.daemon.registry import series_token
from marketlab.logging import get_logger
from marketlab.settings import CONFIG_DIR, Settings
from marketlab.storage.state import LeaderboardRow as StoreLeaderboardRow
from marketlab.storage.state import Settlement as StoreSettlement

log = get_logger(__name__)

#: Concurrent Kalshi orderbook fetches. The RateLimiter still governs request rate;
#: this only stops per-request latency from being serialised across a whole pass.
_KALSHI_BOOK_CONCURRENCY = 8

#: Markets resolved per settlement pass. Bounded so a large backlog is worked through
#: over several cycles instead of stalling one pass.
_SETTLEMENT_BATCH = 60


# ---------------------------------------------------------------------------
# configs/default.yaml's `ingest:` block is loaded into `settings.py`'s merge dict but
# never surfaced on the returned Settings object (no declared field, and pydantic drops
# unknown kwargs by default) -- see the bug note in the module docstring below and the
# final report. We read it directly here rather than duplicate/patch that module.
# ---------------------------------------------------------------------------

_DEFAULT_INGEST_CFG: dict[str, Any] = {
    "kalshi_market_refresh_seconds": 60,
    "kalshi_book_refresh_seconds": 5,
    "kalshi_trades_refresh_seconds": 10,
    "poly_market_refresh_seconds": 120,
    "poly_book_refresh_seconds": 15,
    "poly_leaderboard_refresh_seconds": 3600,
    "poly_activity_refresh_seconds": 30,
    "gdelt_refresh_seconds": 900,
    "sec_refresh_seconds": 300,
    "crypto_spot_refresh_seconds": 5,
    "weather_refresh_seconds": 1800,
    "fred_refresh_seconds": 3600,
    "odds_refresh_seconds": 600,
    "social_refresh_seconds": 300,
    "max_tracked_markets": 400,
    "kalshi_book_sample": 120,
    "kalshi_broad_sweep_pages": 3,
    "kalshi_settlement_refresh_seconds": 120,
    "market_match_refresh_seconds": 900,
    "poly_market_limit": 500,
    "kalshi_category_series_per_category": 30,
    "min_market_volume": 0,
}


def load_raw_config_block(block: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Reload a top-level block from ``default.yaml``/``<profile>.yaml`` that
    ``Settings`` does not surface (``ingest``, ``ai``, ``logging``, ...).

    Duplicates ``settings.load_settings``'s tiny YAML-merge exactly (see that module)
    rather than importing its private helpers, so this keeps working even if that
    module's internals change shape.
    """
    import yaml

    cdir = config_dir or CONFIG_DIR

    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        out = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    merged = _load_yaml(cdir / "default.yaml")
    profile = os.getenv("MARKETLAB_PROFILE", "paper")
    merged = _deep_merge(merged, _load_yaml(cdir / f"{profile}.yaml"))
    return dict(merged.get(block) or {})


def load_ingest_config(config_dir: Path | None = None) -> dict[str, Any]:
    cfg = dict(_DEFAULT_INGEST_CFG)
    cfg.update(load_raw_config_block("ingest", config_dir))
    return cfg


# ---------------------------------------------------------------------------
# Universe expansion (pure functions -- no I/O, fully unit-testable)
# ---------------------------------------------------------------------------

#: Kalshi's own category labels (configs/universes.yaml `kalshi_series_categories`) ->
#: the shared Category enum. Table-driven and documented as a judgment call: Kalshi's
#: 18-category taxonomy (docs/FINDINGS.md #9) is finer than ours.
KALSHI_CATEGORY_LABEL_MAP: dict[str, Category] = {
    "crypto": Category.CRYPTO,
    "sports": Category.SPORTS,
    "politics": Category.POLITICS,
    "elections": Category.POLITICS,
    "economics": Category.ECONOMICS,
    "financials": Category.FINANCE,
    "companies": Category.FINANCE,
    "commodities": Category.FINANCE,
    "mentions": Category.OTHER,
    "climate and weather": Category.WEATHER,
    "science and technology": Category.TECH,
    "entertainment": Category.ENTERTAINMENT,
    "world": Category.OTHER,
    "health": Category.OTHER,
    "social": Category.OTHER,
    "transportation": Category.OTHER,
    "exotics": Category.OTHER,
    "education": Category.OTHER,
}


def collect_universe_allowlists(
    universes_cfg: Mapping[str, Any],
) -> tuple[set[str], set[Category]]:
    """Every ``kalshi_series`` prefix and ``kalshi_series_categories`` label declared by
    an ``available: true`` universe, unioned across the whole file."""
    series: set[str] = set()
    categories: set[Category] = set()
    for uni in (universes_cfg.get("universes") or {}).values():
        if not isinstance(uni, dict) or uni.get("available") is False:
            continue
        series.update(str(s).upper() for s in (uni.get("kalshi_series") or []))
        for label in uni.get("kalshi_series_categories") or []:
            cat = KALSHI_CATEGORY_LABEL_MAP.get(str(label).strip().lower())
            if cat is not None:
                categories.add(cat)
    return series, categories


def select_tracked_markets(
    markets: Iterable[NormalizedMarket],
    universes_cfg: Mapping[str, Any] | None,
    max_tracked_markets: int,
    quotas: Mapping[str, int] | None = None,
    now: datetime | None = None,
) -> list[NormalizedMarket]:
    """Kalshi's ~14k-series catalogue -> the subset a paper-trading run should track.

    Applies (in order): ``exclude_series_prefixes`` (drops the ~12k zero-volume
    ``KXMVE*`` parlays, docs/FINDINGS.md #8), ``min_volume``/``min_liquidity``,
    ``require_status``, and universe membership (a market must match a declared
    ``kalshi_series`` prefix or ``kalshi_series_categories`` label -- unless no universe
    declares either, in which case every market that survives the filters above is
    eligible, so a minimal test config doesn't accidentally track nothing). Markets are
    then ranked by (volume + open_interest) descending, nearest-close-time first, and
    capped at ``max_tracked_markets``.
    """
    cfg = universes_cfg or {}
    defaults = cfg.get("defaults") or {}
    exclude_prefixes = tuple(str(p).upper() for p in (defaults.get("exclude_series_prefixes") or []))
    min_volume = Decimal(str(defaults.get("min_volume", 0)))
    min_liquidity = Decimal(str(defaults.get("min_liquidity", 0)))
    require_status = defaults.get("require_status")
    allowed_series, allowed_categories = collect_universe_allowlists(cfg)
    has_allowlist = bool(allowed_series or allowed_categories)

    selected: list[NormalizedMarket] = []
    for m in markets:
        ticker = m.venue_market_id.upper()
        series = (m.subcategory or ticker).upper()
        if any(series.startswith(p) or ticker.startswith(p) for p in exclude_prefixes):
            continue
        if require_status and m.status.value != str(require_status):
            continue
        if m.volume < min_volume:
            continue
        if m.liquidity < min_liquidity:
            continue
        if has_allowlist:
            in_series = series_token(m) in allowed_series
            if not in_series and m.category not in allowed_categories:
                continue
        selected.append(m)

    def _sort_key(m: NormalizedMarket) -> tuple[Decimal, datetime]:
        close = m.close_time or datetime.max.replace(tzinfo=UTC)
        return (-(m.volume + m.open_interest), close)

    selected.sort(key=_sort_key)
    if not (max_tracked_markets and max_tracked_markets > 0):
        return selected

    # Per-universe quotas first. Ranking everything by volume + open interest hands the
    # whole tracked set to season-long markets (NFL wins, spreads) with enormous open
    # interest, and squeezes out short-lived contracts such as hourly BTC strikes, which
    # can never build that much interest before they settle. A quota universe gets its
    # most active markets that settle within a day.
    reserved: list[NormalizedMarket] = []
    if quotas:
        universes = cfg.get("universes") or {}
        horizon = (now or datetime.now(UTC)) + timedelta(days=1)
        for name, quota in quotas.items():
            series = tuple(str(x).upper() for x in ((universes.get(name) or {}).get("kalshi_series") or []))
            if not series:
                continue
            current = now or datetime.now(UTC)
            members = [
                m for m in selected
                if series_token(m) in series
                and m.close_time is not None and current < m.close_time <= horizon
            ]
            reserved.extend(members[: int(quota)])
    chosen = list({m.canonical_id: m for m in reserved}.values())
    seen = {m.canonical_id for m in chosen}
    chosen.extend(m for m in selected if m.canonical_id not in seen)
    return chosen[:max_tracked_markets]


# ---------------------------------------------------------------------------
# Event construction -- the only place `first_seen_time` is stamped.
# ---------------------------------------------------------------------------


def market_update_event(market: NormalizedMarket, clock: Clock, source: str) -> MarketUpdateEvent:
    now = clock.now()
    return MarketUpdateEvent(event_time=now, first_seen_time=now, source=source, market=market)


def book_update_event(book: OrderBook, clock: Clock, source: str) -> BookUpdateEvent:
    """``event_time`` is the book's own timestamp (when the snapshot was taken);
    ``first_seen_time`` is always this process's clock, never the payload's."""
    return BookUpdateEvent(event_time=book.timestamp, first_seen_time=clock.now(), source=source, book=book)


def trade_event_from_raw(raw: dict[str, Any], clock: Clock, source: str) -> TradeEvent:
    trade = normalize_trade(raw)
    return TradeEvent(event_time=trade.timestamp, first_seen_time=clock.now(), source=source, trade=trade)


def market_status_event(
    canonical_id: str, venue: Venue, status: MarketStatus, clock: Clock, source: str
) -> MarketStatusEvent:
    now = clock.now()
    return MarketStatusEvent(
        event_time=now, first_seen_time=now, source=source, canonical_id=canonical_id, venue=venue, status=status
    )


def settlement_event_from_raw(
    raw: dict[str, Any], canonical_id: str, venue: Venue, clock: Clock, source: str
) -> SettlementEvent | None:
    winning_side, voided = normalize_settlement(raw)
    if winning_side is None and not voided:
        return None
    now = clock.now()
    return SettlementEvent(
        event_time=now,
        first_seen_time=now,
        source=source,
        canonical_id=canonical_id,
        venue=venue,
        winning_side=winning_side,
        voided=voided,
    )


# ---------------------------------------------------------------------------
# Parquet row flattening (only for datasets storage/parquet.py actually defines)
# ---------------------------------------------------------------------------


def market_to_parquet_row(market: NormalizedMarket, first_seen: datetime) -> dict[str, Any]:
    return {
        "canonical_id": market.canonical_id,
        "venue": market.venue.value,
        "venue_market_id": market.venue_market_id,
        "title": market.title,
        "category": market.category.value,
        "status": market.status.value,
        "open_time": market.open_time,
        "close_time": market.close_time,
        "tick_size": market.tick_size,
        "first_seen_time": first_seen,
        "date": first_seen.date(),
    }


def book_to_parquet_rows(book: OrderBook) -> list[dict[str, Any]]:
    """One row per depth level, pairing bid level *i* with ask level *i*."""
    rows: list[dict[str, Any]] = []
    depth = max(len(book.bids), len(book.asks))
    for i in range(depth):
        bid = book.bids[i] if i < len(book.bids) else None
        ask = book.asks[i] if i < len(book.asks) else None
        rows.append(
            {
                "canonical_id": book.canonical_id,
                "venue": book.venue.value,
                "timestamp": book.timestamp,
                "venue_timestamp": book.venue_timestamp,
                "sequence": book.sequence,
                "side": "both",
                "level": i,
                "bid_price": bid.price if bid else None,
                "ask_price": ask.price if ask else None,
                "bid_size": bid.size if bid else None,
                "ask_size": ask.size if ask else None,
                "date": book.timestamp.date(),
            }
        )
    return rows or [
        {
            "canonical_id": book.canonical_id,
            "venue": book.venue.value,
            "timestamp": book.timestamp,
            "venue_timestamp": book.venue_timestamp,
            "sequence": book.sequence,
            "side": "both",
            "level": 0,
            "bid_price": None,
            "ask_price": None,
            "bid_size": None,
            "ask_size": None,
            "date": book.timestamp.date(),
        }
    ]


def trade_to_parquet_row(trade: Any) -> dict[str, Any]:
    return {
        "canonical_id": trade.canonical_id,
        "venue": trade.venue.value,
        "timestamp": trade.timestamp,
        "trade_id": trade.trade_id,
        "aggressor": trade.aggressor.value if trade.aggressor else None,
        "size": trade.size,
        "price": trade.price,
        "date": trade.timestamp.date(),
    }


def trader_action_to_parquet_row(event: Any) -> dict[str, Any]:
    return {
        "wallet": event.wallet,
        "canonical_id": event.canonical_id,
        "title": event.title,
        "side": event.side.value if event.side else None,
        "action": event.action,
        "price": event.price,
        "size": event.size,
        "usd_size": event.usd_size,
        "category": event.category.value,
        "event_time": event.event_time,
        "first_seen_time": event.first_seen_time,
        "date": event.first_seen_time.date(),
    }


# ---------------------------------------------------------------------------
# Result of a single pass
# ---------------------------------------------------------------------------


@dataclass
class IngestOnceResult:
    kalshi_markets: int = 0
    kalshi_books: int = 0
    kalshi_trades: int = 0
    kalshi_settlements: int = 0
    market_matches: int = 0
    poly_markets: int = 0
    poly_books: int = 0
    poly_activity: int = 0
    leaderboard_rows: int = 0
    geoblock_confirmed: bool | None = None
    misc: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def total_markets(self) -> int:
        return self.kalshi_markets + self.poly_markets

    @property
    def ok(self) -> bool:
        """PRD-mandated: zero markets ingested must never present as success."""
        return self.total_markets > 0


# ---------------------------------------------------------------------------
# Supervised-task helper (used by both IngestService and the Supervisor)
# ---------------------------------------------------------------------------


async def run_supervised(
    name: str,
    fn: Callable[[], Awaitable[None]],
    stop_event: asyncio.Event,
    clock: Clock,
    *,
    health: Any = None,
    max_backoff: float = 60.0,
) -> None:
    """Run ``fn`` forever; if it raises, log, record a reconnect, back off, retry.

    This is the "one failing source must never stop the others" primitive: each
    per-source loop is wrapped in this, so an unhandled exception inside one adapter's
    polling coroutine never propagates out of its own ``asyncio.Task``.

    **A normal return ends supervision.** Every real source loop runs
    ``while not self._stop.is_set()`` internally and only returns when it has opted out -
    typically an adapter with no credentials returning immediately. Restarting that is a
    hot loop with no await in it: six unconfigured sources doing this saturated the event
    loop, starved every other task, and the daemon ran at 100% CPU while processing no
    events and emitting no heartbeat. Only an *exception* earns a backoff-and-retry.
    """
    backoff = 1.0
    while not stop_event.is_set():
        try:
            await fn()
            log.info(
                "ingest_task_completed",
                task=name,
                detail="loop returned normally; not restarting (source opted out or finished)",
            )
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a supervised loop must never die
            log.error("ingest_task_failed", task=name, error=str(exc), exc_info=True)
            if health is not None:
                health.record_error(name, exc)
                health.record_reconnect(name)
            jitter = random.uniform(0, backoff * 0.25)
            sleep_for = min(backoff + jitter, max_backoff)
            with contextlib.suppress(asyncio.CancelledError):
                await clock.sleep(sleep_for)
            backoff = min(backoff * 2, max_backoff)


# ---------------------------------------------------------------------------
# IngestService
# ---------------------------------------------------------------------------


class IngestService:
    """Owns every adapter; feeds one ``asyncio.Queue[Event]`` and the shared registries."""

    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        *,
        store: Any | None = None,
        parquet: Any | None = None,
        health: Any | None = None,
        market_registry: Any | None = None,
        book_registry: Any | None = None,
        queue: asyncio.Queue | None = None,
        kalshi_rest: KalshiRestAdapter | None = None,
        ingest_config: dict[str, Any] | None = None,
    ) -> None:
        from marketlab.daemon.health import HealthMonitor
        from marketlab.daemon.registry import BookRegistry, MarketRegistry

        self.settings = settings
        self.clock = clock
        self.store = store
        self.parquet = parquet
        self._cfg = ingest_config or load_ingest_config()
        self._max_tracked = int(self._cfg.get("max_tracked_markets", 400))
        #: Polymarket ids with an approved Kalshi twin; their books are fetched first.
        self.matched_poly_ids: set[str] = set()
        self.matched_poly_markets: dict[str, NormalizedMarket] = {}
        self.matched_kalshi_markets: dict[str, NormalizedMarket] = {}

        self.health = health or HealthMonitor(clock)
        self.markets = market_registry or MarketRegistry(
            max_tracked=self._max_tracked, universes_cfg=settings.universes
        )
        self.books = book_registry or BookRegistry()
        self.queue: asyncio.Queue[Event] = queue or asyncio.Queue(maxsize=50_000)

        self.kalshi_rest = kalshi_rest or KalshiRestAdapter(settings, clock=clock)
        self._kalshi_ws: Any | None = None

        self.poly_gamma = GammaAdapter(settings.sources.poly_gamma, clock)
        self.poly_clob = ClobAdapter(settings.sources.poly_clob, clock)
        self.poly_data = DataApiAdapter(settings.sources.poly_data, clock)
        self.poly_leaderboard = LeaderboardAdapter(
            settings.sources.poly_leaderboard, settings.sources.poly_lb_legacy, clock
        )
        self.poly_us = PolymarketUsAdapter(settings.sources.poly_us_rest, clock)
        self._poly_us_seen: set[str] = set()
        #: Shared with the runner (set by the supervisor); filled by _loop_poly_holdings.
        self.holdings_book: Any | None = None
        #: Match discovery state: the Kalshi catalogue index (rebuilt hourly), the
        #: Polymarket targets seen so far, and the twins it approved (book priority).
        self._jev: Any | None = None
        self._kalshi_catalog: Any | None = None
        self._kalshi_catalog_at = datetime.min.replace(tzinfo=UTC)
        self._poly_targets: dict[str, Any] = {}
        self.discovered_kalshi: dict[str, NormalizedMarket] = {}
        self.discovered_poly: dict[str, NormalizedMarket] = {}

        #: name -> why that adapter could not be constructed. Populated by
        #: _build_misc_adapters so `doctor` can report a wiring bug as a bug
        #: rather than as a missing feature.
        self._adapter_build_errors: dict[str, str] = {}
        #: Kalshi category label -> series tickers, cached for the process lifetime.
        self._category_series_cache: dict[str, list[str]] = {}
        self._misc: dict[str, Adapter] = self._build_misc_adapters()

        self.health.register("kalshi_rest", required=True, stale_after_seconds=180.0)
        self.health.register("poly_gamma", required=False, stale_after_seconds=600.0)
        self.health.register("poly_clob", required=False, stale_after_seconds=600.0)
        self.health.register("poly_data", required=False, stale_after_seconds=1800.0)
        self.health.register("poly_leaderboard", required=False, stale_after_seconds=7200.0)
        self.health.register("poly_us_rest", required=False, stale_after_seconds=1800.0)
        for name in self._misc:
            self.health.register(name, required=False, stale_after_seconds=3600.0)

        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stop = asyncio.Event()
        self.geoblock_result: poly_geoblock.GeoblockResult | None = None
        self.events_processed = 0
        self._tracked_wallets: set[str] = set()
        #: TraderScore-ordered wallets (QUALIFIED first) and the QUALIFIED roster.
        self._wallet_priority: list[str] = []
        self.qualified_wallets: dict[str, float] = {}
        self._activity_cursor = 0

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def _build_misc_adapters(self) -> dict[str, Adapter]:
        """Best-effort construction of every optional info-source adapter.

        Never raises: a source whose module is missing (e.g. ``marketlab.adapters.gdelt``
        is an empty package as of this build) or whose constructor fails is logged and
        simply absent from the returned dict -- `doctor`/`ingest` degrade to reporting it
        unavailable rather than crashing the daemon.
        """
        s = self.settings
        out: dict[str, Adapter] = {}

        def _try(name: str, factory: Callable[[], Adapter]) -> None:
            try:
                out[name] = factory()
            except Exception as exc:  # noqa: BLE001
                # Record the reason. A construction failure is a real bug (a wiring
                # mismatch, a renamed argument) and must not be reported downstream as
                # "not implemented" - that hid a live TypeError in the GDELT wiring.
                self._adapter_build_errors[name] = f"{type(exc).__name__}: {exc}"
                log.warning("misc_adapter_unavailable", source=name, error=str(exc), exc_info=True)

        def _sec() -> Adapter:
            from marketlab.adapters.sec.client import SecAdapter

            return SecAdapter(
                sec_base=s.sources.sec_base,
                edgar_base=s.sources.sec_edgar,
                user_agent=s.secrets.sec_user_agent,
                clock=self.clock,
            )

        def _gdelt() -> Adapter:
            from marketlab.adapters.gdelt.client import GdeltAdapter

            # GDELT exposes two distinct endpoints (article search and sentence-level
            # context) rather than one base URL, so it takes both explicitly.
            return GdeltAdapter(
                doc_base=s.sources.gdelt_doc,
                context_base=s.sources.gdelt_context,
                clock=self.clock,
            )

        def _crypto() -> Adapter:
            from marketlab.adapters.crypto.spot import CryptoSpotAdapter

            return CryptoSpotAdapter(
                coinbase_base=s.sources.coinbase_spot, binance_base=s.sources.binance_spot, clock=self.clock
            )

        def _weather() -> Adapter:
            from marketlab.adapters.weather.nws import NwsAdapter

            return NwsAdapter(base_url=s.sources.nws_base, user_agent=s.secrets.sec_user_agent, clock=self.clock)

        def _fred() -> Adapter:
            from marketlab.adapters.fred.client import FredAdapter

            return FredAdapter(base_url=s.sources.fred_base, api_key=s.secrets.fred_api_key, clock=self.clock)

        def _x() -> Adapter:
            from marketlab.adapters.x.client import XAdapter

            return XAdapter(bearer_token=s.secrets.x_bearer_token, clock=self.clock)

        def _bluesky() -> Adapter:
            from marketlab.adapters.bluesky.client import BlueskyAdapter

            return BlueskyAdapter(
                firehose_url=s.sources.bluesky_firehose,
                handle=s.secrets.bluesky_handle,
                app_password=s.secrets.bluesky_app_password,
                clock=self.clock,
            )

        def _alpaca() -> Adapter:
            from marketlab.adapters.alpaca.client import AlpacaAdapter

            return AlpacaAdapter(
                base_url=s.sources.alpaca_data,
                api_key=s.secrets.alpaca_api_key,
                secret_key=s.secrets.alpaca_secret_key,
                clock=self.clock,
            )

        def _odds() -> Adapter:
            from marketlab.adapters.sports_odds.client import TheOddsApiAdapter

            return TheOddsApiAdapter(base_url=s.sources.odds_base, api_key=s.secrets.the_odds_api_key, clock=self.clock)

        for name, factory in (
            ("sec", _sec),
            ("gdelt", _gdelt),
            ("crypto_spot", _crypto),
            ("weather_nws", _weather),
            ("fred", _fred),
            ("x", _x),
            ("bluesky", _bluesky),
            ("alpaca", _alpaca),
            ("sports_odds", _odds),
        ):
            _try(name, factory)
        return out

    def misc_adapters(self) -> dict[str, Adapter]:
        return dict(self._misc)

    def adapter_build_errors(self) -> dict[str, str]:
        """Why a given adapter is absent. Empty string means "simply not configured"."""
        return dict(self._adapter_build_errors)

    def all_adapters(self) -> dict[str, Adapter]:
        """Every adapter this service owns, for `doctor`'s concurrent probe sweep."""
        out: dict[str, Adapter] = {
            "kalshi_rest": self.kalshi_rest,
            "poly_gamma": self.poly_gamma,
            "poly_clob": self.poly_clob,
            "poly_data": self.poly_data,
            "poly_leaderboard": self.poly_leaderboard,
            "poly_us_rest": self.poly_us,
        }
        out.update(self._misc)
        return out

    def kalshi_ws_adapter(self, tickers: Iterable[str] = ()) -> Any:
        """Lazily construct (and cache) the Kalshi websocket adapter, wired to this
        service's own event queue so its output lands exactly where REST-sourced events
        do."""
        from marketlab.adapters.kalshi.ws import KalshiWebSocketAdapter

        if self._kalshi_ws is None:
            self._kalshi_ws = KalshiWebSocketAdapter(
                self.settings, self.queue, clock=self.clock, tickers=tickers,
                min_book_interval=float(self._cfg.get("kalshi_ws_book_interval_seconds", 5.0)),
            )
        return self._kalshi_ws

    def _enqueue(self, event: Event) -> None:
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            log.warning("ingest_queue_full", event_type=event.event_type)

    async def _write_parquet(self, dataset: str, rows: list[dict[str, Any]]) -> None:
        if self.parquet is None or not rows:
            return
        try:
            write = self.parquet.write
            if asyncio.iscoroutinefunction(write):
                await write(dataset, rows)
            else:
                write(dataset, rows)
        except Exception as exc:  # noqa: BLE001 - never let storage kill ingestion
            log.warning("parquet_write_failed", dataset=dataset, error=str(exc))

    # ------------------------------------------------------------------
    # geoblock (mandatory before any Polymarket work)
    # ------------------------------------------------------------------

    async def geoblock_check(self) -> poly_geoblock.GeoblockResult:
        result = await poly_geoblock.enforce(self.settings)
        self.geoblock_result = result
        if result.blocked:
            log.warning(
                "polymarket_geoblock_confirmed",
                country=result.country,
                detail="Polymarket global execution is confirmed BLOCKED for this deployment; "
                "read-only intelligence only, Kalshi is the sole execution venue.",
            )
        else:
            log.warning(
                "polymarket_geoblock_unexpected_unblocked",
                country=result.country,
                detail="geoblock probe reported NOT blocked; execution stays hard-disabled regardless.",
            )
        return result

    # ------------------------------------------------------------------
    # one-shot passes (also the building blocks of the continuous loops)
    # ------------------------------------------------------------------

    async def _fetch_kalshi_series_markets(self, series: Iterable[str]) -> list[dict[str, Any]]:
        """Fetch open markets for specific series tickers.

        This is the primary discovery path and it exists because the undirected scan is
        useless here: ``/markets?status=open`` is dominated by ~12,000 zero-volume
        ``KXMVE*`` cross-category parlays (docs/FINDINGS.md #8), so paginating 25,000 rows
        surfaced only a handful of markets any universe actually wanted. Asking for the
        series we care about by name returns exactly them, in one page each.
        """
        rows: list[dict[str, Any]] = []
        for ticker in series:
            try:
                async for page in self.kalshi_rest.iter_all_markets(
                    status="open", limit=1000, max_pages=2, series_ticker=ticker
                ):
                    got = page.get("markets") or []
                    if not got:
                        break
                    rows.extend(got)
            except Exception as exc:  # noqa: BLE001 - one dead series must not stop the rest
                log.warning("kalshi_series_fetch_failed", series=ticker, error=str(exc))
        return rows

    async def _ingest_kalshi_markets_once(self, result: IngestOnceResult, *, max_pages: int | None = None) -> None:
        try:
            series_allowlist, _categories = collect_universe_allowlists(self.settings.universes)
            raw_rows: list[dict[str, Any]] = []

            # 1. Targeted: every series a universe named explicitly.
            if series_allowlist:
                raw_rows.extend(await self._fetch_kalshi_series_markets(sorted(series_allowlist)))

            # 2. Category-driven universes (politics, elections, companies, mentions)
            #    name no explicit series, so their series are discovered from the series
            #    catalogue by Kalshi's own category label. Without this they resolved to
            #    ZERO markets - `/markets` has no category filter and its default ordering
            #    is ~97% KXMVE parlays, so the shallow sweep below never reaches them and
            #    news_probability / public_statement had nothing to trade.
            if _categories:
                cat_series = await self._discover_category_series(_categories)
                if cat_series:
                    raw_rows.extend(await self._fetch_kalshi_series_markets(cat_series))

            # 3. Broad sweep, for the category-driven universes (politics, elections,
            #    companies, mentions) that name no explicit series. Deliberately shallow:
            #    `/markets?status=open` is ~97% zero-volume KXMVE parlays
            #    (docs/FINDINGS.md #8), so paging deep costs a lot of CPU and returns
            #    almost nothing the targeted pass above missed.
            if max_pages is None:
                max_pages = int(self._cfg.get("kalshi_broad_sweep_pages", 3))
            async for page in self.kalshi_rest.iter_all_markets(status="open", limit=1000, max_pages=max_pages):
                rows = page.get("markets") or []
                if not rows:
                    break
                raw_rows.extend(rows)

            # De-dupe by canonical_id: the same market can legitimately be returned by
            # more than one per-series fetch above (a ticker can match more than one
            # universe's `kalshi_series` list), and counting it twice would be exactly
            # the "manufactured sample size" this project exists to avoid.
            normalized_by_id: dict[str, NormalizedMarket] = {}
            for i, raw in enumerate(raw_rows):
                try:
                    market = normalize_kalshi_market(raw)
                except Exception as exc:  # noqa: BLE001
                    log.warning("kalshi_market_normalize_failed", error=str(exc))
                    continue
                normalized_by_id[market.canonical_id] = market
                # Normalising tens of thousands of rows is pure CPU with no await in it.
                # Unyielded it stalled the event loop for seconds on every refresh, which
                # starved the orderbook loop down to roughly one cycle where it should
                # manage a dozen - so the tournament saw almost no books to trade against.
                if i % 1000 == 999:
                    await asyncio.sleep(0)
            normalized = list(normalized_by_id.values())

            selected = select_tracked_markets(
                normalized, self.settings.universes, self._max_tracked,
                quotas=self._cfg.get("book_quota_per_universe"), now=self.clock.now(),
            )
            now = self.clock.now()
            rows_out: list[dict[str, Any]] = []
            for m in selected:
                self.markets.upsert(m)
                if self.store is not None:
                    with contextlib.suppress(Exception):
                        self.store.upsert_market(m)
                self._enqueue(market_update_event(m, self.clock, "kalshi_rest"))
                rows_out.append(market_to_parquet_row(m, now))
            await self._write_parquet("market_metadata", rows_out)

            self.health.record_message("kalshi_rest")
            result.kalshi_markets = len(selected)
            log.info(
                "ingest_kalshi_markets",
                pages_seen=len(raw_rows),
                normalized=len(normalized),
                tracked=len(selected),
                max_tracked=self._max_tracked,
            )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"kalshi_markets: {exc}")
            self.health.record_error("kalshi_rest", exc)
            log.error("ingest_kalshi_markets_failed", error=str(exc))

    async def _ingest_kalshi_books_and_trades_once(self, result: IngestOnceResult, *, sample: int = 25) -> None:
        # Prioritise by liquidity. Registry order is arbitrary, and a book for a market
        # nobody trades is worth far less to the tournament than a book for one that is
        # actively quoting - open interest is the better proxy here than volume, since a
        # freshly listed hourly contract can show heavy volume and no resting depth.
        # Kalshi only. The registry deliberately holds the Polymarket mirror alongside
        # the Kalshi tracked set, and without this filter the most liquid Polymarket
        # markets crowd out the sample and are then requested from Kalshi's orderbook
        # endpoint under a `poly:` id, which fails for every one of them.
        ranked = sorted(
            (m for m in self.markets.all() if m.venue is Venue.KALSHI),
            key=lambda m: (m.open_interest, m.volume),
            reverse=True,
        )
        # Kalshi legs of approved cross-venue pairs always get a book; without one the
        # pair can never trade, however good the match.
        matched = list(dict.fromkeys(
            m.venue_market_id
            for m in [*self.matched_kalshi_markets.values(), *self.discovered_kalshi.values()]
        ))[: sample * 2]
        # Per-universe quotas: a global open-interest ranking always hands the book budget
        # to sports and BTC, so thin-but-real universes (weather: 4 of 325 open markets had
        # a book) could never be tested at all. Each quota universe gets its most active
        # markets closing within two days.
        reserved: list[str] = []
        quotas: dict[str, int] = dict(self._cfg.get("book_quota_per_universe") or {"weather_daily_high": 24})
        horizon = self.clock.now() + timedelta(days=2)
        for universe, quota in quotas.items():
            members = [
                m for m in ranked
                if universe in self.markets.universes_for(m.canonical_id)
                and (m.close_time is None or m.close_time <= horizon)
            ]
            members.sort(key=lambda m: (m.volume, m.open_interest), reverse=True)
            reserved.extend(m.venue_market_id for m in members[: int(quota)])
        priority = list(dict.fromkeys(matched + reserved))
        seen = set(priority)
        tickers = priority + [m.venue_market_id for m in ranked if m.venue_market_id not in seen]
        tickers = tickers[: sample + min(len(priority) // 2, 20)]
        if not tickers:
            return
        # Trades are a secondary signal and cost a second request per market, so they are
        # sampled more thinly than books to stay inside the venue's request budget.
        trade_tickers = set(tickers[: max(1, sample // 3)])
        books_ok = 0
        trades_ok = 0
        book_rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []

        # Fetched concurrently, bounded. Sequentially, ~200ms per orderbook meant a
        # 120-market pass took longer than the health monitor's staleness window, so the
        # Kalshi feed was declared STALE between passes and the supervisor correctly - but
        # needlessly - halted all trading. The rate limiter still enforces the venue's
        # request budget; this only stops us serialising round-trip latency.
        semaphore = asyncio.Semaphore(_KALSHI_BOOK_CONCURRENCY)

        async def fetch_one(ticker: str) -> None:
            nonlocal books_ok, trades_ok
            async with semaphore:
                try:
                    raw_ob = await self.kalshi_rest.get_orderbook(ticker)
                    book = normalize_orderbook(ticker, raw_ob, self.clock.now())
                    self.books.upsert(book)
                    self._enqueue(book_update_event(book, self.clock, "kalshi_rest"))
                    book_rows.extend(book_to_parquet_rows(book))
                    books_ok += 1
                    # Per-call, not per-pass: a long pass must not look like a dead feed.
                    self.health.record_message("kalshi_rest")
                except Exception as exc:  # noqa: BLE001
                    log.warning("kalshi_orderbook_failed", ticker=ticker, error=str(exc))
                if ticker not in trade_tickers:
                    return
                try:
                    raw_trades = await self.kalshi_rest.get_trades(ticker=ticker, limit=20)
                    for raw_trade in raw_trades.get("trades") or []:
                        try:
                            event = trade_event_from_raw(raw_trade, self.clock, "kalshi_rest")
                        except (KeyError, ValueError):
                            continue
                        self._enqueue(event)
                        trade_rows.append(trade_to_parquet_row(event.trade))
                        trades_ok += 1
                    self.health.record_message("kalshi_rest")
                except Exception as exc:  # noqa: BLE001
                    log.warning("kalshi_trades_failed", ticker=ticker, error=str(exc))

        await asyncio.gather(*(fetch_one(t) for t in tickers))
        await self._write_parquet("books", book_rows)
        await self._write_parquet("trades", trade_rows)
        result.kalshi_books = books_ok
        result.kalshi_trades = trades_ok

    async def _ingest_poly_markets_once(self, result: IngestOnceResult, *, limit: int | None = None) -> None:
        if limit is None:
            limit = int(self._cfg.get("poly_market_limit", 500))
        try:
            raw_list = await self.poly_gamma.get_markets(limit=limit, closed=False, order="volume24hr", ascending=False)
            normalized: list[NormalizedMarket] = []
            for raw in raw_list:
                try:
                    normalized.append(normalize_poly_market(raw))
                except Exception as exc:  # noqa: BLE001
                    log.warning("poly_market_normalize_failed", error=str(exc))

            mirror_cfg = (self.settings.universes or {}).get("polymarket_mirror") or {}
            cap = int(mirror_cfg.get("max_tracked_markets", 300))
            selected = normalized[:cap]
            now = self.clock.now()
            rows_out: list[dict[str, Any]] = []
            for m in selected:
                self.markets.upsert(m)
                if self.store is not None:
                    with contextlib.suppress(Exception):
                        self.store.upsert_market(m)
                self._enqueue(market_update_event(m, self.clock, "poly_gamma"))
                rows_out.append(market_to_parquet_row(m, now))
            await self._write_parquet("market_metadata", rows_out)

            self.health.record_message("poly_gamma")
            result.poly_markets = len(selected)
            log.info("ingest_poly_markets", seen=len(normalized), tracked=len(selected))
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"poly_markets: {exc}")
            self.health.record_error("poly_gamma", exc)
            log.error("ingest_poly_markets_failed", error=str(exc))

    async def _ingest_poly_books_once(self, result: IngestOnceResult, *, sample: int = 15) -> None:
        all_poly = [m for m in self.markets.all() if m.venue is Venue.POLY_GLOBAL]
        # Markets with an approved Kalshi twin first: those are the books cross-venue and
        # copy trading actually read. The rest fill whatever sample remains.
        matched = list({**self.matched_poly_markets, **self.discovered_poly}.values())
        rest = [m for m in all_poly if m.canonical_id not in self.matched_poly_ids]
        poly_markets = (matched[: sample * 2] + rest)[: max(sample, min(len(matched), sample * 2))]
        if not poly_markets:
            return
        books_ok = 0
        book_rows: list[dict[str, Any]] = []
        for m in poly_markets:
            token_ids = _parse_clob_token_ids(m.raw)
            if not token_ids:
                continue
            try:
                raw_book = await self.poly_clob.get_book(token_ids[0])
                if not raw_book:
                    continue
                book = normalize_poly_book(token_ids[0], raw_book, self.clock.now(), canonical_id=m.canonical_id)
                self.books.upsert(book)
                self._enqueue(book_update_event(book, self.clock, "poly_clob"))
                book_rows.extend(book_to_parquet_rows(book))
                books_ok += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("poly_book_failed", canonical_id=m.canonical_id, error=str(exc))
        await self._write_parquet("books", book_rows)
        if books_ok:
            self.health.record_message("poly_clob")
        result.poly_books = books_ok

    async def _ingest_leaderboard_once(self, result: IngestOnceResult, *, quick: bool = True) -> None:
        try:
            if quick:
                official = await self.poly_leaderboard.get_official(category="overall", period="week", metric="pnl", limit=25)
                now = self.clock.now()
                rows = self.poly_leaderboard._rows_from_official(  # noqa: SLF001 - same-team adapter internals, avoids re-sweeping every combo on a quick pass
                    official, category="overall", period="week", metric="pnl", now=now
                )
            else:
                rows = await self.poly_leaderboard.snapshot_all(limit_per_board=25)
            if self.store is not None and rows:
                store_rows = [
                    StoreLeaderboardRow(
                        snapshot_time=r.snapshot_time,
                        category=r.category_raw or r.category.value,
                        period=r.period,
                        metric=r.metric,
                        rank=r.rank,
                        wallet=r.wallet,
                        username=r.username,
                        pnl=r.pnl,
                        volume=r.volume,
                    )
                    for r in rows
                ]
                with contextlib.suppress(Exception):
                    self.store.save_leaderboard_snapshot(store_rows)
            self._tracked_wallets.update(r.wallet for r in rows if r.wallet)
            self.health.record_message("poly_leaderboard")
            result.leaderboard_rows = len(rows)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"leaderboard: {exc}")
            self.health.record_error("poly_leaderboard", exc)
            log.error("ingest_leaderboard_failed", error=str(exc))

    async def _ingest_poly_activity_once(self, result: IngestOnceResult, *, n_wallets: int = 5) -> None:
        if not self._tracked_wallets:
            return
        activity_ok = 0
        rows_out: list[dict[str, Any]] = []
        # Rotate through every tracked wallet instead of re-polling the same first few,
        # always including the top QUALIFIED wallets so their trades are seen quickly.
        ordered = self._wallet_priority + [w for w in self._tracked_wallets if w not in set(self._wallet_priority)]
        top = [w for w in ordered if w in self.qualified_wallets][: max(1, n_wallets // 2)]
        rest = [w for w in ordered if w not in set(top)]
        start = self._activity_cursor % max(len(rest), 1)
        picked = top + (rest[start:] + rest[:start])[: n_wallets - len(top)]
        self._activity_cursor += n_wallets - len(top)
        for wallet in picked:
            try:
                async for event in self.poly_data.iter_trader_actions(wallet, page_size=10, max_pages=1):
                    self._enqueue(event)
                    if self.store is not None:
                        from marketlab.storage.state import TraderActionRecord

                        with contextlib.suppress(Exception):
                            self.store.save_trader_action(
                                TraderActionRecord(
                                    wallet=event.wallet,
                                    username=event.username,
                                    canonical_id=event.canonical_id,
                                    poly_market_id=event.poly_market_id,
                                    poly_condition_id=event.poly_condition_id,
                                    title=event.title,
                                    outcome=event.outcome,
                                    side=event.side,
                                    action=event.action,
                                    price=event.price,
                                    size=event.size,
                                    usd_size=event.usd_size,
                                    category=event.category,
                                    transaction_hash=event.transaction_hash,
                                    event_time=event.event_time,
                                    first_seen_time=event.first_seen_time,
                                )
                            )
                    rows_out.append(trader_action_to_parquet_row(event))
                    activity_ok += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("poly_activity_failed", wallet=wallet, error=str(exc))
        await self._write_parquet("trader_actions", rows_out)
        if activity_ok:
            self.health.record_message("poly_data")
        result.poly_activity = activity_ok

    async def run_once(self) -> IngestOnceResult:
        """One pass of every enabled REST source. Never raises."""
        result = IngestOnceResult()
        geo = await self.geoblock_check()
        result.geoblock_confirmed = geo.blocked
        await self._ingest_kalshi_markets_once(result)
        await self._ingest_kalshi_books_and_trades_once(result)
        await self._ingest_poly_markets_once(result)
        await self._ingest_poly_books_once(result)
        await self._ingest_leaderboard_once(result, quick=True)
        await self._ingest_poly_activity_once(result)
        if not result.ok:
            log.error(
                "ingest_once_zero_markets",
                detail="run_once ingested zero markets across every venue -- this is a failure, not a quiet no-op",
                errors=result.errors,
            )
        return result

    # ------------------------------------------------------------------
    # continuous supervised loops
    # ------------------------------------------------------------------

    def _seconds(self, key: str, default: float) -> float:
        return float(self._cfg.get(key, default))

    async def _loop_kalshi_markets(self) -> None:
        interval = self._seconds("kalshi_market_refresh_seconds", 60)
        while not self._stop.is_set():
            r = IngestOnceResult()
            await self._ingest_kalshi_markets_once(r, max_pages=25)
            await self.clock.sleep(interval)

    async def _discover_category_series(self, categories: set[Category]) -> list[str]:
        """Series tickers for the Kalshi categories our universes ask for, bounded.

        Kalshi lists thousands of series per category (2,303 Politics alone), so this
        takes a bounded slice per category; the volume/liquidity filters in
        ``select_tracked_markets`` decide which of the resulting markets are worth
        tracking. Cached for the process lifetime - the catalogue changes slowly and this
        runs on the market-refresh cadence.
        """
        per = int(self._cfg.get("kalshi_category_series_per_category", 30))
        wanted = {
            label
            for label, cat in KALSHI_CATEGORY_LABEL_MAP.items()
            if cat in categories
        }
        out: list[str] = []
        for label in sorted(wanted):
            cached = self._category_series_cache.get(label)
            if cached is None:
                try:
                    payload = await self.kalshi_rest.list_series(
                        category=label.title(), limit=200
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("category_series_failed", category=label, error=str(exc))
                    self._category_series_cache[label] = []
                    continue
                cached = [
                    str(x.get("ticker"))
                    for x in (payload.get("series") or [])
                    if x.get("ticker")
                ]
                self._category_series_cache[label] = cached
            out.extend(cached[:per])
        return out

    async def _ingest_kalshi_settlements_once(self, result: IngestOnceResult) -> None:
        """Resolve markets we still hold positions in.

        Settlement is the ONLY thing that turns an open position into realized P&L, and
        without it the whole tournament is scientifically inert: no realized returns, no
        Brier scores, no win rates, no calibration. It also deadlocks trading - positions
        never clear, so strategy exposure ratchets up to its cap and every subsequent
        order is refused.

        This works from open positions rather than the market registry on purpose. The
        registry only holds *open* markets, so a contract vanishes from it the instant it
        closes, which is exactly when its resolution becomes knowable.

        The venue's own ``result`` field is the sole authority here - never a model,
        never a news report, never an inferred score.
        """
        if self.store is None:
            return
        try:
            held = set(self.store.open_position_market_ids())
            already = self.store.settled_market_ids()
        except Exception as exc:  # noqa: BLE001
            log.warning("settlement_candidates_failed", error=str(exc))
            return

        pending = [cid for cid in held if cid not in already and cid.startswith("kalshi:")]
        if not pending:
            return

        settled = 0
        semaphore = asyncio.Semaphore(_KALSHI_BOOK_CONCURRENCY)
        store = self.store  # narrowed for the closure below

        async def resolve(canonical_id: str) -> None:
            nonlocal settled
            ticker = canonical_id.split(":", 1)[1].upper()
            async with semaphore:
                try:
                    raw = await self.kalshi_rest.get_market(ticker)
                except Exception as exc:  # noqa: BLE001
                    log.debug("settlement_fetch_failed", ticker=ticker, error=str(exc))
                    return
            nested = raw.get("market")
            market_raw: dict[str, Any] = nested if isinstance(nested, dict) else raw
            event = settlement_event_from_raw(
                market_raw, canonical_id, Venue.KALSHI, self.clock, "kalshi_rest"
            )
            if event is None:
                return  # not resolved yet; try again next pass
            self._enqueue(event)
            try:
                store.save_settlement(
                    StoreSettlement(
                        canonical_id=canonical_id,
                        venue=Venue.KALSHI,
                        winning_side=event.winning_side,
                        settlement_value=event.settlement_value,
                        voided=event.voided,
                        settled_at=event.event_time,
                        first_seen_time=event.first_seen_time,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("settlement_persist_failed", canonical_id=canonical_id, error=str(exc))
            settled += 1

        await asyncio.gather(*(resolve(cid) for cid in pending[:_SETTLEMENT_BATCH]))
        if settled:
            self.health.record_message("kalshi_rest")
            log.info("ingest_kalshi_settlements", resolved=settled, pending=len(pending))
        result.kalshi_settlements = settled

    async def _ingest_market_matches_once(self, result: IngestOnceResult) -> None:
        """Link Kalshi contracts to equivalent Polymarket markets.

        Without this the copy-trading and cross-venue families are structurally unable to
        trade: both require an APPROVED match before acting, because a Polymarket signal
        is only actionable once we know which Kalshi contract is the identical bet. The
        matcher existed and was tested but was never invoked by the daemon, which left
        114 of 388 sleeves permanently idle.

        Approval remains the deterministic resolution-rule validator's decision alone -
        this loop only feeds it candidates and persists what it certifies.
        """
        if self.store is None:
            return
        try:
            from marketlab.matching.cross_venue import CrossVenueMatcher, approved_for_automation
        except Exception as exc:  # noqa: BLE001
            log.warning("matcher_unavailable", error=str(exc))
            return

        # Read the full catalogue from the store, not the registry. The registry is a
        # bounded LRU hot-cache shared by both venues, so whichever venue refreshed last
        # evicts the other; matching needs to see every market we know about, and it runs
        # rarely enough (15 min) that a store read is cheap.
        try:
            kalshi = self.store.list_markets(venue=Venue.KALSHI.value, status="open")
            poly = self.store.list_markets(venue=Venue.POLY_GLOBAL.value, status="open")
        except Exception as exc:  # noqa: BLE001
            log.warning("match_catalogue_read_failed", error=str(exc))
            return
        if not kalshi or not poly:
            return

        # Structural game-winner matching first, in its own guard: it is the matcher
        # that actually produces approvable pairs, and a failure in the general
        # free-text matcher below must never take it down (or vice versa).
        matches: list[Any] = []
        try:
            from marketlab.matching.sports import match_games

            now = self.clock.now()
            sports = match_games(kalshi, poly, now)
            matches.extend(sports)
            by_id = {m.canonical_id: m for m in [*kalshi, *poly]}
            # Upcoming games only: a settled game's pair is history, not an opportunity.
            horizon = now + timedelta(days=4)
            live_pairs = [
                (by_id[m.canonical_id_a], by_id[m.canonical_id_b]) for m in sports
                if (c := by_id[m.canonical_id_a].close_time) is not None and now < c <= horizon
            ]
            self.matched_kalshi_markets = {k.canonical_id: k for k, _ in live_pairs}
            self.matched_poly_markets = {p.canonical_id: p for _, p in live_pairs}
            self.matched_poly_ids = set(self.matched_poly_markets)
            # Make both legs visible to strategies even if the bounded registry evicted
            # them: a matched pair whose Kalshi market the runner never saw cannot trade.
            for k_market, p_market in live_pairs:
                for market in (k_market, p_market):
                    if self.markets.get(market.canonical_id) is None:
                        self.markets.upsert(market)
                        self._enqueue(market_update_event(market, self.clock, "matcher"))
            log.info("sports_matches", total=len(sports), upcoming=len(live_pairs))
        except Exception as exc:  # noqa: BLE001
            log.warning("sports_matching_failed", error=str(exc), exc_info=True)
        try:
            matcher = CrossVenueMatcher(clock=self.clock)
            matches.extend(await matcher.match(kalshi, poly))
        except Exception as exc:  # noqa: BLE001
            log.warning("market_matching_failed", error=str(exc))

        saved = approved = 0
        for m in matches:
            try:
                self.store.save_match(m)
                saved += 1
                if approved_for_automation(m):
                    approved += 1
            except Exception as exc:  # noqa: BLE001
                log.debug("match_persist_failed", error=str(exc))
        if saved:
            log.info(
                "ingest_market_matches",
                kalshi=len(kalshi), poly=len(poly), saved=saved, approved=approved,
            )
        result.market_matches = saved

    async def _discover_matches_once(self) -> dict[str, int]:
        """Find Kalshi twins for the Polymarket markets that matter (marketlab.matching.discovery).

        Targets, most important first: every market the top leaderboard wallets hold
        (by holder count), then the high-volume Polymarket markets already catalogued.
        Each (target, candidate) pair is judged by Jev once and persisted either way;
        an approved pair makes both legs tradable (registry + store + book priority).
        """
        from marketlab.ai.typesafe import build_jev_client
        from marketlab.matching import discovery as disc

        if self.store is None:
            return {}
        if self._jev is None:
            self._jev = build_jev_client(self.settings)
        if self._jev is None:
            log.info("match_discovery_skipped", reason="no TYPESAFE_API_KEY")
            return {}

        now = self.clock.now()
        self._restore_discovered_twins(disc.VALIDATOR_VERSION)
        if self._kalshi_catalog is None or (now - self._kalshi_catalog_at).total_seconds() > 3600:
            events: list[dict[str, Any]] = []
            cursor: str | None = None
            for _ in range(200):
                page = await self.kalshi_rest.get_events(
                    limit=200, cursor=cursor, status="open", with_nested_markets=True
                )
                batch = page.get("events") or []
                events.extend(batch)
                cursor = page.get("cursor")
                if not cursor or not batch:
                    break
            entries = disc.kalshi_entries_from_events(events)
            self._kalshi_catalog = disc.KalshiCatalogIndex(entries)
            self._kalshi_catalog_at = now
            log.info("kalshi_catalog_built", events=len(events), markets=len(entries))
        index = self._kalshi_catalog

        held: dict[str, int] = {}
        if self.holdings_book is not None:
            for row in self.holdings_book.view("union", 100, False):
                held[row.condition_id] = max(held.get(row.condition_id, 0), row.n_holders)
        catalogued = {
            m.venue_market_id: m
            for m in self.store.list_markets(venue=Venue.POLY_GLOBAL.value, status="open")
        }
        missing = [cid for cid in held if cid not in self._poly_targets]
        for i in range(0, len(missing), 40):
            try:
                raws = await self.poly_gamma.get_markets(
                    limit=100, extra_params={"condition_ids": missing[i : i + 40]}
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("discovery_gamma_failed", error=str(exc))
                continue
            for raw in raws:
                target = disc.poly_target_from_gamma(raw)
                if target is not None:
                    self._poly_targets[target.condition_id] = target
        for cid, market in catalogued.items():
            if cid not in self._poly_targets and isinstance(market.raw, dict):
                target = disc.poly_target_from_gamma(market.raw)
                if target is not None:
                    self._poly_targets[cid] = target

        done = self.store.evaluated_pairs(disc.VALIDATOR_VERSION)
        judged_targets = {b for _, b in done}
        order = sorted(self._poly_targets.values(), key=lambda t: -held.get(t.condition_id, 0))
        budget = int(self._cfg.get("match_discovery_targets_per_pass", 400))
        todo = [t for t in order if t.canonical_id not in judged_targets][:budget]
        k = int(self._cfg.get("match_discovery_candidates", 8))
        gate = asyncio.Semaphore(16)

        async def judge(target: Any) -> list[Any]:
            cands = [
                (e, sc) for e, sc in index.candidates(target, k=k)
                if (e.canonical_id, target.canonical_id) not in done
            ]
            if not cands:
                return []
            async with gate:
                return await disc.verify(self._jev, target, cands)

        results = await asyncio.gather(*(judge(t) for t in todo))
        approved = 0
        for verdicts in results:
            for v in verdicts:
                with contextlib.suppress(Exception):
                    self.store.save_match(disc.verdict_to_match(v, now))
                if v.approved:
                    approved += 1
                    self._adopt_twin(v)
        log.info(
            "match_discovery", targets=len(todo), judged=sum(len(r) for r in results),
            approved=approved, held_targets=len(held), jev_calls=self._jev.calls,
        )
        return {"targets": len(todo), "approved": approved}

    def _restore_discovered_twins(self, validator_version: str) -> None:
        """Re-register both legs of every approved discovery pair (after a restart, or
        when the bounded registry evicted them), so approved twins keep their books."""
        try:
            rows = [m for m in self.store.approved_matches() if m.validator_version == validator_version]
        except Exception:  # noqa: BLE001
            return
        for match in rows:
            for cid, bucket in ((match.canonical_id_a, self.discovered_kalshi), (match.canonical_id_b, self.discovered_poly)):
                if self.markets.get(cid) is not None and cid in bucket:
                    continue
                market = self.store.get_market(cid)
                if market is None or str(market.status.value) != "open":
                    continue
                bucket[cid] = market
                if self.markets.get(cid) is None:
                    self.markets.upsert(market)
                    self._enqueue(market_update_event(market, self.clock, "match_discovery"))

    def _adopt_twin(self, verdict: Any) -> None:
        """Make both legs of a newly approved pair tradable and give them books."""
        try:
            kalshi = normalize_kalshi_market(verdict.entry.raw)
            poly = normalize_poly_market(verdict.target.raw)
        except Exception as exc:  # noqa: BLE001
            log.warning("discovery_twin_normalize_failed", ticker=verdict.entry.ticker, error=str(exc))
            return
        for market in (kalshi, poly):
            self.markets.upsert(market)
            if self.store is not None:
                with contextlib.suppress(Exception):
                    self.store.upsert_market(market)
            self._enqueue(market_update_event(market, self.clock, "match_discovery"))
        self.discovered_kalshi[kalshi.canonical_id] = kalshi
        self.discovered_poly[poly.canonical_id] = poly

    async def _loop_match_discovery(self) -> None:
        interval = self._seconds("match_discovery_refresh_seconds", 1800)
        await self.clock.sleep(120)  # let the first holdings snapshot land
        while not self._stop.is_set():
            try:
                await self._discover_matches_once()
            except Exception as exc:  # noqa: BLE001
                log.warning("match_discovery_failed", error=str(exc), exc_info=True)
            await self.clock.sleep(interval)

    async def _loop_market_matches(self) -> None:
        interval = self._seconds("market_match_refresh_seconds", 900)
        while not self._stop.is_set():
            r = IngestOnceResult()
            await self._ingest_market_matches_once(r)
            await self.clock.sleep(interval)

    async def _loop_kalshi_settlements(self) -> None:
        interval = self._seconds("kalshi_settlement_refresh_seconds", 120)
        while not self._stop.is_set():
            r = IngestOnceResult()
            await self._ingest_kalshi_settlements_once(r)
            await self.clock.sleep(interval)

    async def _loop_kalshi_books_trades(self) -> None:
        interval = self._seconds("kalshi_book_refresh_seconds", 5)
        while not self._stop.is_set():
            r = IngestOnceResult()
            # Book breadth is the binding constraint on how many strategies can act:
            # a market with no book is refused NO_LIQUIDITY. But one orderbook call per
            # tracked market per cycle is far past Kalshi's public budget, so we take the
            # most liquid `kalshi_book_sample` markets rather than all of them.
            await self._ingest_kalshi_books_and_trades_once(
                r, sample=int(self._cfg.get("kalshi_book_sample", 120))
            )
            await self.clock.sleep(interval)

    async def _loop_kalshi_ws(self) -> None:
        ws = self.kalshi_ws_adapter(self.markets.canonical_ids())
        ws.start()
        try:
            while not self._stop.is_set():
                await ws.subscribe([m.venue_market_id for m in self.markets.all()])
                self.health.apply_probe(ws.health())
                await self.clock.sleep(30.0)
        finally:
            await ws.close()

    async def _loop_poly_markets(self) -> None:
        interval = self._seconds("poly_market_refresh_seconds", 120)
        while not self._stop.is_set():
            r = IngestOnceResult()
            await self._ingest_poly_markets_once(r)
            await self.clock.sleep(interval)

    async def _loop_poly_books(self) -> None:
        interval = self._seconds("poly_book_refresh_seconds", 15)
        while not self._stop.is_set():
            r = IngestOnceResult()
            await self._ingest_poly_books_once(r, sample=30)
            await self.clock.sleep(interval)

    async def _ingest_poly_us_once(self) -> int:
        """Snapshot every open Polymarket US market's best bid/ask from the keyless public
        gateway into parquet. Collection only: these markets are never put in the market
        registry, so no strategy can see or trade them."""
        from marketlab.adapters.polymarket_us import public as poly_us_public

        cap = int(self._cfg.get("poly_us_max_markets", 3000))
        raw_markets = await self.poly_us.iter_open_markets(cap)
        if not raw_markets:
            self.health.record_error("poly_us_rest", RuntimeError("no markets returned"))
            return 0
        now = self.clock.now()
        meta_rows: list[dict[str, Any]] = []
        book_rows: list[dict[str, Any]] = []
        for raw in raw_markets:
            slug = raw.get("slug")
            if not slug:
                continue
            if slug not in self._poly_us_seen:
                self._poly_us_seen.add(slug)
                meta_rows.append(poly_us_public.market_metadata_row(raw, now))
            row = poly_us_public.top_of_book_row(raw, now)
            if row is not None:
                book_rows.append(row)
        await self._write_parquet("market_metadata", meta_rows)
        await self._write_parquet("books", book_rows)
        self.health.record_message("poly_us_rest")
        log.info("ingest_poly_us", markets=len(raw_markets), quoted=len(book_rows), new=len(meta_rows))
        return len(raw_markets)

    async def _ingest_poly_holdings_once(self) -> int:
        """Snapshot the open positions of the top wallets on the all-time, monthly and
        weekly overall leaderboards into the shared :class:`HoldingsBook`."""
        from marketlab.signals.holdings import parse_position

        if self.holdings_book is None:
            return 0
        top_n = int(self._cfg.get("poly_holdings_top_n", 100))
        boards: dict[str, list[str]] = {}
        for period in ("all", "month", "week"):
            rows = await self.poly_leaderboard.get_top(category="overall", period=period, n=top_n)
            boards[period] = [str(r.get("proxyWallet")) for r in rows if r.get("proxyWallet")]
        wallets = list(dict.fromkeys(w for board in boards.values() for w in board))
        membership = {w: ",".join(b for b, ws in boards.items() if w in ws) for w in wallets}
        gate = asyncio.Semaphore(8)

        async def fetch(wallet: str) -> list[Any]:
            async with gate:
                try:
                    raw = await self.poly_data.get_positions(wallet, limit=500, size_threshold=Decimal("1"))
                except Exception as exc:  # noqa: BLE001 - one wallet must not sink the snapshot
                    log.warning("poly_holdings_wallet_failed", wallet=wallet, error=str(exc))
                    return []
            return [h for h in (parse_position(wallet, r) for r in raw) if h is not None]

        per_wallet = await asyncio.gather(*(fetch(w) for w in wallets))
        holdings = [h for hs in per_wallet for h in hs]
        if not holdings:
            self.health.record_error("poly_data", RuntimeError("holdings snapshot returned no positions"))
            return 0
        now = self.clock.now()
        self.holdings_book.replace(boards, holdings, now)
        await self._write_parquet("holdings", [
            {
                "snapshot_time": now, "wallet": h.wallet, "boards": membership.get(h.wallet, ""),
                "condition_id": h.condition_id, "outcome_index": h.outcome_index, "outcome": h.outcome,
                "title": h.title, "size": h.size, "avg_price": h.avg_price, "cur_price": h.cur_price,
                "usd_value": h.usd_value, "end_date": h.end_date, "date": now.date(),
            }
            for h in holdings
        ])
        self.health.record_message("poly_data")
        log.info("ingest_poly_holdings", wallets=len(wallets), positions=len(holdings),
                 boards={k: len(v) for k, v in boards.items()})
        return len(holdings)

    async def _loop_poly_holdings(self) -> None:
        interval = self._seconds("poly_holdings_refresh_seconds", 900)
        while not self._stop.is_set():
            try:
                await self._ingest_poly_holdings_once()
            except Exception as exc:  # noqa: BLE001
                log.warning("ingest_poly_holdings_failed", error=str(exc))
            await self.clock.sleep(interval)

    async def _loop_poly_us(self) -> None:
        interval = self._seconds("poly_us_refresh_seconds", 300)
        while not self._stop.is_set():
            try:
                await self._ingest_poly_us_once()
            except Exception as exc:  # noqa: BLE001 - an optional source must never stop
                self.health.record_error("poly_us_rest", exc)
                log.warning("ingest_poly_us_failed", error=str(exc))
            await self.clock.sleep(interval)

    async def _loop_poly_leaderboard(self) -> None:
        interval = self._seconds("poly_leaderboard_refresh_seconds", 3600)
        while not self._stop.is_set():
            r = IngestOnceResult()
            await self._ingest_leaderboard_once(r, quick=False)
            await self.clock.sleep(interval)

    async def _loop_poly_activity(self) -> None:
        interval = self._seconds("poly_activity_refresh_seconds", 30)
        while not self._stop.is_set():
            r = IngestOnceResult()
            await self._ingest_poly_activity_once(r, n_wallets=10)
            await self.clock.sleep(interval)

    async def _loop_sec(self) -> None:
        from marketlab.adapters.sec.client import SecAdapter

        raw_adapter = self._misc.get("sec")
        if not isinstance(raw_adapter, SecAdapter):
            return
        adapter = raw_adapter
        interval = self._seconds("sec_refresh_seconds", 300)
        forms = ((self.settings.source_toggles or {}).get("sources", {}).get("sec", {}) or {}).get("forms")
        since_checkpoint = "sec_filings"
        while not self._stop.is_set():
            try:
                since = None
                if self.store is not None:
                    cp = self.store.get_checkpoint("ingest", since_checkpoint)
                    if cp:
                        since = datetime.fromisoformat(cp)
                events = await adapter.iter_filing_events(forms=forms, since=since)
                for event in events:
                    self._enqueue(event)
                self.health.record_message("sec")
                if events and self.store is not None:
                    self.store.set_checkpoint("ingest", since_checkpoint, self.clock.now().isoformat())
                log.info("ingest_sec_filings", count=len(events))
            except Exception as exc:  # noqa: BLE001
                self.health.record_error("sec", exc)
                log.warning("ingest_sec_failed", error=str(exc))
            await self.clock.sleep(interval)

    async def _loop_crypto_spot(self) -> None:
        """BTC (etc.) spot for the crypto strategies. Primary: the Coinbase websocket
        ticker, passed through at most once a second per symbol. Fallback: REST polling
        every ``crypto_spot_refresh_seconds`` whenever the stream has been quiet for 15s."""
        from marketlab.adapters.crypto.spot import CryptoSpotAdapter

        raw_adapter = self._misc.get("crypto_spot")
        if not isinstance(raw_adapter, CryptoSpotAdapter):
            return
        adapter = raw_adapter
        interval = self._seconds("crypto_spot_refresh_seconds", 5)
        symbols = ((self.settings.universes or {}).get("reference_feeds") or {}).get("crypto_spot") or ["BTC-USD"]
        stream_interval = float(self._cfg.get("crypto_spot_stream_interval_seconds", 1.0))
        relay: asyncio.Queue[ExternalPriceEvent] = asyncio.Queue(maxsize=1000)
        last_stream_at = [0.0]

        async def consume() -> None:
            while True:
                event = await relay.get()
                last_stream_at[0] = time.monotonic()
                await self._publish_spot(event)

        stream_task = asyncio.ensure_future(
            adapter.stream(list(symbols), relay, self.clock, min_interval_seconds=stream_interval)
        )
        consume_task = asyncio.ensure_future(consume())
        try:
            while not self._stop.is_set():
                if time.monotonic() - last_stream_at[0] > 15.0:
                    for symbol in symbols:
                        try:
                            await self._publish_spot(await adapter.get_spot(symbol))
                        except Exception as exc:  # noqa: BLE001
                            self.health.record_error("crypto_spot", exc)
                            log.warning("ingest_crypto_spot_failed", symbol=symbol, error=str(exc))
                await self.clock.sleep(interval)
        finally:
            stream_task.cancel()
            consume_task.cancel()

    async def _publish_spot(self, event: ExternalPriceEvent) -> None:
        self._enqueue(event)
        await self._write_parquet(
            "prices",
            [
                {
                    "symbol": event.symbol,
                    "venue": event.venue,
                    "timestamp": event.event_time,
                    "first_seen_time": event.first_seen_time,
                    "price": event.price,
                    "date": event.first_seen_time.date(),
                }
            ],
        )
        self.health.record_message("crypto_spot")

    async def _loop_weather(self) -> None:
        from marketlab.adapters.weather.nws import CITY_STATIONS, NwsAdapter

        raw_adapter = self._misc.get("weather_nws")
        if not isinstance(raw_adapter, NwsAdapter):
            return
        adapter = raw_adapter

        interval = self._seconds("weather_refresh_seconds", 1800)
        while not self._stop.is_set():
            for city_key in CITY_STATIONS:
                try:
                    events = await adapter.forecast_events(city_key)
                    for event in events:
                        self._enqueue(event)
                    self.health.record_message("weather_nws")
                except Exception as exc:  # noqa: BLE001
                    self.health.record_error("weather_nws", exc)
                    log.warning("ingest_weather_failed", city=city_key, error=str(exc))
            await self.clock.sleep(interval)

    async def _loop_fred(self) -> None:
        from marketlab.adapters.fred.client import FredAdapter

        raw_adapter = self._misc.get("fred")
        if not isinstance(raw_adapter, FredAdapter) or not raw_adapter.has_credentials:
            return
        adapter = raw_adapter
        interval = self._seconds("fred_refresh_seconds", 3600)
        series_ids = ((self.settings.universes or {}).get("reference_feeds") or {}).get("macro_fred") or []
        while not self._stop.is_set():
            for series_id in series_ids:
                try:
                    obs = await adapter.get_series(series_id)
                    if obs:
                        self.health.record_message("fred")
                except Exception as exc:  # noqa: BLE001
                    self.health.record_error("fred", exc)
                    log.warning("ingest_fred_failed", series_id=series_id, error=str(exc))
            await self.clock.sleep(interval)

    async def _loop_trader_scoring(self) -> None:
        """Score leaderboard wallets from their realized track records (TraderScore).

        Wallets come from the recorded leaderboard snapshots, best rank first; each pass
        rescores the stalest ``trader_scoring_batch`` of them. The QUALIFIED set then
        drives which wallets' live activity is polled first, and who the copy-basket arms
        may follow. Nothing here can place an order.
        """
        from marketlab.signals.trader_analytics import analyze_closed_positions

        if self.store is None:
            return
        interval = self._seconds("trader_scoring_seconds", 600.0)
        batch = int(self._cfg.get("trader_scoring_batch", 50))
        pool_size = int(self._cfg.get("trader_scoring_pool", 300))
        await self.clock.sleep(30.0)  # let the leaderboard loop record a snapshot first
        while not self._stop.is_set():
            try:
                pool = self.store.leaderboard_wallets(limit=pool_size)
                scored = {r["wallet"]: r["computed_at"] for r in self.store.trader_scores(limit=10_000)}
                # Never-scored wallets first, then the stalest.
                todo = sorted(pool, key=lambda r: (r[0] in scored, scored.get(r[0], "")))[:batch]
                done = 0
                for wallet, username, _best in todo:
                    rows: list[dict[str, Any]] = []
                    for offset in range(0, 200, 50):
                        page = await self.poly_data.get_closed_positions(wallet, limit=50, offset=offset)
                        rows.extend(page)
                        await self.clock.sleep(0.5)
                        if len(page) < 50:
                            break
                    analysis = analyze_closed_positions(wallet, rows, self.clock.now(), username or "")
                    self.store.save_trader_score(analysis.as_row())
                    done += 1
                self._refresh_wallet_priority()
                log.info("trader_scoring", scored=done, pool=len(pool), qualified=len(self.qualified_wallets))
            except Exception as exc:  # noqa: BLE001
                log.warning("trader_scoring_failed", error=str(exc), exc_info=True)
            await self.clock.sleep(interval)

    def _refresh_wallet_priority(self) -> None:
        if self.store is None:
            return
        rows = self.store.trader_scores(limit=10_000)
        self.qualified_wallets = {r["wallet"]: float(r["score"]) for r in rows if r["status"] == "QUALIFIED"}
        rejected = {r["wallet"] for r in rows if r["status"] == "REJECTED"}
        ranked = [r["wallet"] for r in rows if r["wallet"] not in rejected]
        # QUALIFIED first (by score), then the remaining leaderboard wallets.
        self._wallet_priority = [w for w in ranked if w in self.qualified_wallets] + [
            w for w in ranked if w not in self.qualified_wallets
        ]
        self._tracked_wallets.update(self._wallet_priority)
        self._tracked_wallets.difference_update(rejected)

    async def _loop_gdelt_news(self) -> None:
        """Pull news targeted at the markets the AI arms actually assess.

        GDELT used to be health-probed only, so no NewsEvent ever reached the event bus
        and the AI evidence layer was empty for the whole first tournament. Queries are
        built from the titles of open Kalshi markets in the AI universes, one event at a
        time in rotation, because GDELT throttles bursts and a firehose query ("news")
        returns nothing that is relevant to any contract.
        """
        from marketlab.adapters.gdelt.client import GdeltAdapter
        from marketlab.adapters.news.google_rss import GoogleNewsSearch
        from marketlab.ai.stack import market_keywords

        gdelt = self._misc.get("gdelt")
        gnews = GoogleNewsSearch(self.clock)
        interval = self._seconds("news_refresh_seconds", 600.0)
        per_cycle = int(self._cfg.get("news_queries_per_cycle", 12))
        ai_universes = set(self._cfg.get("gdelt_universes") or (
            "politics_general", "politics_elections", "economics_inflation", "economics_fed",
            "economics_jobs", "economics_growth", "companies_events", "mentions",
        ))
        seen: set[str] = set()
        cursor = 0
        while not self._stop.is_set():
            try:
                queries: dict[str, str] = {}
                for market in self.markets.all():
                    if str(market.venue) != "kalshi" or str(market.status) != "open":
                        continue
                    if not ai_universes.intersection(self.markets.universes_for(market.canonical_id)):
                        continue
                    kws = [k for k in market_keywords(market) if len(k) >= 4][:3]
                    if len(kws) >= 2:
                        queries.setdefault(market.event_id or market.canonical_id, " ".join(kws))
                ordered = [queries[k] for k in sorted(queries)]
                batch = ordered[cursor:cursor + per_cycle] or ordered[:per_cycle]
                cursor = cursor + per_cycle if cursor + per_cycle < len(ordered) else 0
                fresh = 0
                for query in batch:
                    # Google News first (fast, keyless); GDELT only as a fallback.
                    found = await gnews.search(query)
                    if not found and isinstance(gdelt, GdeltAdapter):
                        found = await gdelt.search(query, timespan="24h", maxrecords=25)
                    for event in found:
                        if event.news_id in seen:
                            continue
                        seen.add(event.news_id)
                        self._enqueue(event)
                        fresh += 1
                    await self.clock.sleep(3.0)  # be polite to both services
                if len(seen) > 50_000:
                    seen.clear()
                self.health.record_message("gdelt")
                log.info("ingest_news", queries=len(batch), new_articles=fresh, candidates=len(ordered))
            except Exception as exc:  # noqa: BLE001
                self.health.record_error("gdelt", exc)
                log.warning("ingest_news_failed", error=str(exc))
                ordered = []
            # At boot the market registry is still empty; retry soon rather than leaving
            # the AI arms without news for a full interval.
            await self.clock.sleep(interval if ordered else 60.0)

    async def _loop_probe_only(self, name: str, interval_key: str, default_seconds: float) -> None:
        """A source we only keep a live health signal for (X, Bluesky, sports odds,
        Alpaca, GDELT): periodic ``probe()`` calls, no deeper event pipeline yet."""
        adapter = self._misc.get(name)
        if adapter is None:
            return
        interval = self._seconds(interval_key, default_seconds)
        while not self._stop.is_set():
            try:
                health = await adapter.probe()
                self.health.apply_probe(health)
                if health.status == SourceStatus.HEALTHY:
                    self.health.record_message(name, health.latency_ms)
            except Exception as exc:  # noqa: BLE001
                self.health.record_error(name, exc)
            await self.clock.sleep(interval)

    #: task name -> configs/sources.yaml toggle key. A task with no entry here (the core
    #: Kalshi loops) is always considered enabled -- Kalshi is the mandatory execution
    #: venue and has no off switch.
    _TOGGLE_KEY: dict[str, str] = {
        "poly_markets": "polymarket_global",
        "poly_books": "polymarket_global",
        "poly_leaderboard": "polymarket_global",
        "poly_activity": "polymarket_global",
        "poly_us": "polymarket_us",
        "poly_holdings": "polymarket_global",
        "match_discovery": "polymarket_global",
        "sec": "sec",
        "crypto_spot": "crypto_spot",
        "weather_nws": "weather_nws",
        "fred": "fred",
        "x": "x",
        "bluesky": "bluesky",
        "sports_odds": "sports_odds",
        "alpaca": "alpaca",
        "gdelt": "gdelt",
    }

    def _source_enabled(self, task_name: str) -> bool:
        toggle_key = self._TOGGLE_KEY.get(task_name)
        if toggle_key is None:
            return True
        toggles = (self.settings.source_toggles or {}).get("sources", {})
        cfg = toggles.get(toggle_key)
        if cfg is None:
            return True
        return bool(cfg.get("enabled", True))

    async def start(self) -> None:
        """Launch one independently-supervised task per source."""
        self._stop.clear()
        await self.geoblock_check()

        loops: list[tuple[str, Callable[[], Awaitable[None]]]] = [
            ("kalshi_markets", self._loop_kalshi_markets),
            ("kalshi_books_trades", self._loop_kalshi_books_trades),
            ("kalshi_settlements", self._loop_kalshi_settlements),
            ("market_matches", self._loop_market_matches),
            ("kalshi_ws", self._loop_kalshi_ws),
            ("poly_markets", self._loop_poly_markets),
            ("poly_books", self._loop_poly_books),
            ("poly_leaderboard", self._loop_poly_leaderboard),
            ("poly_activity", self._loop_poly_activity),
            ("poly_us", self._loop_poly_us),
            ("poly_holdings", self._loop_poly_holdings),
            ("match_discovery", self._loop_match_discovery),
            ("sec", self._loop_sec),
            ("crypto_spot", self._loop_crypto_spot),
            ("weather_nws", self._loop_weather),
            ("fred", self._loop_fred),
            ("gdelt", self._loop_gdelt_news),
            ("trader_scoring", self._loop_trader_scoring),
        ]
        def _make_probe_loop(n: str, k: str, d: float) -> Callable[[], Awaitable[None]]:
            async def _loop() -> None:
                await self._loop_probe_only(n, k, d)

            return _loop

        for probe_name, interval_key, default_seconds in (
            ("x", "odds_refresh_seconds", 300.0),
            ("bluesky", "social_refresh_seconds", 300.0),
            ("sports_odds", "odds_refresh_seconds", 600.0),
            ("alpaca", "kalshi_market_refresh_seconds", 300.0),
        ):
            loops.append((probe_name, _make_probe_loop(probe_name, interval_key, default_seconds)))

        for name, fn in loops:
            if not self._source_enabled(name):
                continue
            self._tasks[name] = asyncio.ensure_future(
                run_supervised(name, fn, self._stop, self.clock, health=self.health)
            )
        log.info("ingest_service_started", tasks=list(self._tasks))

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks.values():
            task.cancel()
        for task in self._tasks.values():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        if self._kalshi_ws is not None:
            await self._kalshi_ws.close()
        for adapter in self.all_adapters().values():
            with contextlib.suppress(Exception):
                await adapter.close()
        log.info("ingest_service_stopped")


def _parse_clob_token_ids(raw: Mapping[str, Any]) -> list[str]:
    value = raw.get("clobTokenIds")
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return []
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    return []
