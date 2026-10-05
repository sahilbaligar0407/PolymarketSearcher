"""In-memory market/book/portfolio registries shared by the ingest pipeline and the
strategy runner.

These are the daemon's shared world-state: :class:`MarketRegistry` is what "the set of
markets we currently track" means at runtime, :class:`BookRegistry` is the point-in-time
book history the :class:`~marketlab.execution.paper_broker.PaperBroker` reads through its
injected ``book_provider`` callable, and :class:`PortfolioRegistry` is the thing that
actually holds each experiment sleeve's :class:`~marketlab.core.portfolio.Portfolio` so a
single long-lived ``PaperBroker`` can serve every sleeve via ``portfolio_provider``.

Everything here is synchronous and lock-guarded: the daemon's event loop calls these
directly (never through ``asyncio.to_thread``) because every operation is in-memory and
O(log n) at worst, so a lock is simpler and cheaper than an async wrapper.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from marketlab.core.instruments import (
    Category,
    NormalizedMarket,
    OrderBook,
    Side,
    Venue,
)
from marketlab.core.portfolio import Portfolio, liquidation_mark

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

#: Default bound on per-market book history, mirroring PaperBroker's own "keep enough to
#: answer book_as_of, not the whole day" philosophy.
DEFAULT_MAX_SNAPSHOTS = 200
#: Default bound on how far back in time a per-market history is kept, in seconds.
DEFAULT_MAX_HISTORY_SECONDS = 600.0

#: Kalshi's own category labels (configs/universes.yaml `kalshi_series_categories`) ->
#: the shared Category enum. Kept in sync by hand with the identical table in
#: ``daemon/ingest.py`` (``KALSHI_CATEGORY_LABEL_MAP``) -- duplicated rather than
#: imported to avoid a registry-depends-on-ingest cycle (ingest already imports this
#: module for MarketRegistry/BookRegistry).
_CATEGORY_LABEL_MAP: dict[str, Category] = {
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


def series_token(market: NormalizedMarket) -> str:
    """A Kalshi market's series: the ticker up to its first ``-`` (KXBTC15M-26OCT04... ->
    KXBTC15M).

    Universe membership used to test ``startswith``, so the series ``KXBTC`` (hourly BTC
    ranges) also claimed every ``KXBTC15M`` market - the 15-minute horizon that had been
    disabled on evidence - and ``KXETH`` claimed ``KXETH15M``. A series is a whole token.
    """
    raw = (market.subcategory or market.venue_market_id or "").upper()
    return raw.split("-", 1)[0]


def _market_matches_universe(market: NormalizedMarket, udef: Mapping[str, Any]) -> bool:
    if udef.get("available") is False:
        return False
    token = series_token(market)
    for s in udef.get("kalshi_series") or ():
        if token == str(s).upper():
            return True
    for label in udef.get("kalshi_series_categories") or ():
        cat = _CATEGORY_LABEL_MAP.get(str(label).strip().lower())
        if cat is not None and market.category is cat:
            return True
    return False


#: The registry holds the Kalshi tracked set plus the Polymarket mirror alongside it.
_VENUE_HEADROOM_FACTOR = 2


class MarketRegistry:
    """canonical_id -> latest :class:`NormalizedMarket`, bounded to the most recently
    touched ``max_tracked`` markets (an LRU by ``upsert`` recency).

    Bounding matters: Kalshi alone lists tens of thousands of markets (see
    ``docs/FINDINGS.md`` #8/#9); the daemon is only ever supposed to hold live state for
    the subset ``ingest.max_tracked_markets`` selected, not the whole catalogue. The
    catalogue itself lives in :class:`~marketlab.storage.state.StateStore`.
    """

    def __init__(self, max_tracked: int = 400, universes_cfg: Mapping[str, Any] | None = None) -> None:
        # Headroom for the read-only Polymarket mirror. `max_tracked` is the KALSHI
        # tracked-set size, but Polymarket markets are upserted into this same registry;
        # sizing the LRU to exactly max_tracked meant each venue's refresh evicted the
        # other's markets, so a strategy could look up a market that had just vanished.
        self._max_tracked = max(1, int(max_tracked * _VENUE_HEADROOM_FACTOR))
        self._kalshi_budget = max(1, max_tracked)
        self._markets: OrderedDict[str, NormalizedMarket] = OrderedDict()
        self._lock = threading.RLock()
        self._universes_cfg: Mapping[str, Any] = universes_cfg or {}

    def __len__(self) -> int:
        return len(self._markets)

    @property
    def max_tracked(self) -> int:
        return self._max_tracked

    def universes_for(self, canonical_id: str) -> list[str]:
        """Which configured universes (``configs/universes.yaml``) this market belongs
        to, by ``kalshi_series`` prefix or ``kalshi_series_categories`` label.

        Required by ``marketlab.experiments.runner.ExperimentRunner``'s
        ``MarketRegistryLike`` protocol so events route only to the sleeves whose
        universe actually covers this market, rather than broadcasting to every sleeve.
        """
        market = self.get(canonical_id)
        if market is None:
            return []
        universes = (self._universes_cfg or {}).get("universes") or {}
        return [
            name
            for name, udef in universes.items()
            if isinstance(udef, dict) and _market_matches_universe(market, udef)
        ]

    def upsert(self, market: NormalizedMarket) -> None:
        with self._lock:
            self._markets.pop(market.canonical_id, None)
            self._markets[market.canonical_id] = market
            self._markets.move_to_end(market.canonical_id)
            while len(self._markets) > self._max_tracked:
                self._markets.popitem(last=False)

    def upsert_many(self, markets: Iterable[NormalizedMarket]) -> int:
        n = 0
        for m in markets:
            self.upsert(m)
            n += 1
        return n

    def get(self, canonical_id: str) -> NormalizedMarket | None:
        with self._lock:
            return self._markets.get(canonical_id)

    def all(self) -> list[NormalizedMarket]:
        with self._lock:
            return list(self._markets.values())

    def by_venue(self, venue: Venue | str) -> list[NormalizedMarket]:
        v = Venue(venue) if not isinstance(venue, Venue) else venue
        with self._lock:
            return [m for m in self._markets.values() if m.venue is v]

    def by_category(self, category: Category | str) -> list[NormalizedMarket]:
        c = Category(category) if not isinstance(category, Category) else category
        with self._lock:
            return [m for m in self._markets.values() if m.category is c]

    def by_series_prefix(self, prefix: str) -> list[NormalizedMarket]:
        p = prefix.upper()
        with self._lock:
            return [
                m
                for m in self._markets.values()
                if m.venue_market_id.upper().startswith(p)
                or (m.subcategory or "").upper().startswith(p)
            ]

    def canonical_ids(self) -> list[str]:
        with self._lock:
            return list(self._markets.keys())


class _BookHistory:
    __slots__ = ("history", "latest")

    def __init__(self) -> None:
        self.history: list[OrderBook] = []
        self.latest: OrderBook | None = None


class BookRegistry:
    """canonical_id -> latest :class:`OrderBook`, plus a bounded per-market history ring
    buffer so ``book_as_of`` can answer "the most recent book with timestamp <= T".

    History is bounded both by count (``max_snapshots``) and by wall-clock span
    (``max_history_seconds``) so memory stays flat regardless of feed volume.
    """

    def __init__(
        self,
        max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
        max_history_seconds: float = DEFAULT_MAX_HISTORY_SECONDS,
    ) -> None:
        self._max_snapshots = max(1, max_snapshots)
        self._max_history_seconds = max_history_seconds
        self._books: dict[str, _BookHistory] = {}
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._books)

    def upsert(self, book: OrderBook) -> None:
        with self._lock:
            entry = self._books.setdefault(book.canonical_id, _BookHistory())
            hist = entry.history
            # Insertion-sorted by timestamp: live feeds are monotonic in practice, but a
            # REST fallback racing a websocket update is not guaranteed to be, and
            # book_as_of's correctness depends on the history staying sorted.
            idx = len(hist)
            while idx > 0 and hist[idx - 1].timestamp > book.timestamp:
                idx -= 1
            hist.insert(idx, book)
            if entry.latest is None or book.timestamp >= entry.latest.timestamp:
                entry.latest = book
            while len(hist) > self._max_snapshots:
                hist.pop(0)
            if hist:
                newest = hist[-1].timestamp
                while len(hist) > 1 and (newest - hist[0].timestamp).total_seconds() > (
                    self._max_history_seconds
                ):
                    hist.pop(0)

    def get(self, canonical_id: str) -> OrderBook | None:
        with self._lock:
            entry = self._books.get(canonical_id)
            return entry.latest if entry else None

    def book_as_of(self, canonical_id: str, ts: datetime) -> OrderBook | None:
        """The most recent recorded book with ``timestamp <= ts``, or ``None``.

        Mirrors ``PaperBroker._book_as_of``'s no-look-ahead discipline: a book stamped
        after ``ts`` is never returned even if it is the "latest" one on file.
        """
        with self._lock:
            entry = self._books.get(canonical_id)
            if entry is None:
                return None
            candidate: OrderBook | None = None
            for b in entry.history:
                if b.timestamp <= ts:
                    candidate = b
                else:
                    break
            return candidate

    def all_latest(self) -> dict[str, OrderBook]:
        with self._lock:
            return {cid: e.latest for cid, e in self._books.items() if e.latest is not None}

    def canonical_ids(self) -> list[str]:
        with self._lock:
            return list(self._books.keys())

    def mark_prices(self) -> dict[str, Decimal]:
        """``{"<canonical_id>|<side>": probability}`` for portfolio marking.

        Uses the book mid where available; a market with no two-sided book yet
        contributes no mark (callers fall back to a position's average price, per
        ``Portfolio.equity``'s own documented behaviour).
        """
        marks: dict[str, Decimal] = {}
        with self._lock:
            for cid, entry in self._books.items():
                book = entry.latest
                if book is None:
                    continue
                for side in (Side.YES, Side.NO):
                    mark = liquidation_mark(book, side)
                    if mark is not None:
                        marks[Portfolio.key(cid, side)] = mark
        return marks


class PortfolioRegistry:
    """experiment_id -> :class:`Portfolio`, the state backing ``PaperBroker``'s
    ``portfolio_provider`` callable.

    ``PaperBroker`` is constructed once, before the strategy tournament exists, and looks
    up a sleeve's portfolio purely by ``experiment_id`` on every ``submit()`` call (see
    ``execution/paper_broker.py``: ``portfolio = self.portfolio_provider(intent.experiment_id)``).
    This registry is the concrete object behind that callable: ``get_or_create`` is
    idempotent, so whichever caller (recovery, the experiment runner, or the broker
    itself on first sight of a new experiment id) asks first materializes the sleeve's
    bankroll and every later caller converges on the same object.
    """

    def __init__(self, default_bankroll: Decimal = Decimal("50.00")) -> None:
        self._default_bankroll = default_bankroll
        self._portfolios: dict[str, Portfolio] = {}
        self._lock = threading.RLock()

    def get(self, experiment_id: str) -> Portfolio | None:
        with self._lock:
            return self._portfolios.get(experiment_id)

    def get_or_create(
        self,
        experiment_id: str,
        *,
        strategy_id: str = "",
        initial_capital: Decimal | None = None,
        created_at: datetime | None = None,
    ) -> Portfolio:
        with self._lock:
            existing = self._portfolios.get(experiment_id)
            if existing is not None:
                return existing
            cap = initial_capital if initial_capital is not None else self._default_bankroll
            portfolio = Portfolio(
                experiment_id=experiment_id,
                strategy_id=strategy_id or experiment_id,
                initial_capital=cap,
                cash=cap,
                high_water_mark=cap,
                created_at=created_at,
            )
            self._portfolios[experiment_id] = portfolio
            return portfolio

    def register(self, portfolio: Portfolio) -> None:
        """Install an already-constructed (e.g. restored from the store) Portfolio."""
        with self._lock:
            self._portfolios[portfolio.experiment_id] = portfolio

    def all(self) -> dict[str, Portfolio]:
        with self._lock:
            return dict(self._portfolios)

    def __len__(self) -> int:
        return len(self._portfolios)

    def equities(self, marks: dict[str, Decimal] | None = None) -> dict[str, Decimal]:
        with self._lock:
            return {eid: p.equity(marks) for eid, p in self._portfolios.items()}

    def total_equity(self, marks: dict[str, Decimal] | None = None) -> Decimal:
        return sum(self.equities(marks).values(), Decimal("0"))

    def alive_count(self, floor: Decimal = Decimal("1.00")) -> tuple[int, int]:
        """``(alive, dead)`` sleeve counts, per ``Portfolio.is_dead``."""
        with self._lock:
            portfolios = list(self._portfolios.values())
        dead = sum(1 for p in portfolios if p.is_dead(floor))
        return (len(portfolios) - dead, dead)
