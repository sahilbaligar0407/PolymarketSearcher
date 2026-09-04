"""Shared test harness for strategy unit tests.

Used by both Team STRATEGIES-A (baselines/controls/microstructure) and Team
STRATEGIES-B (event-driven/copy-trading strategies). Builds a :class:`StrategyContext`
over plain in-memory dicts, drives a strategy with scripted events against a
:class:`~marketlab.clock.SimulatedClock`, and collects whatever it emits.

Not a pytest plugin - just plain importable helpers. Nothing here reaches the network,
storage, or a broker; it only calls the strategy's own public handler methods, exactly as
the real runner would.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from marketlab.clock import SimulatedClock
from marketlab.core.events import (
    BookUpdateEvent,
    EconomicEvent,
    ExternalPriceEvent,
    FilingEvent,
    MarketStatusEvent,
    NewsEvent,
    SocialEvent,
    SourceClass,
    SportsStateEvent,
    TimerEvent,
    TradeEvent,
    TraderActionEvent,
    WeatherEvent,
)
from marketlab.core.instruments import (
    BookLevel,
    Fees,
    MarketStatus,
    NormalizedMarket,
    OrderBook,
    OutcomeType,
    Trade,
    Venue,
)
from marketlab.core.orders import Fill, Order, OrderIntent
from marketlab.core.strategy import ProbabilityForecast, Strategy, StrategyContext

#: Default epoch used by every harness unless a test supplies its own clock/start time.
DEFAULT_START = datetime(2026, 1, 1, tzinfo=UTC)


def make_book(
    canonical_id: str,
    bids: list[tuple[str, int]],
    asks: list[tuple[str, int]],
    ts: datetime,
    venue: Venue = Venue.KALSHI,
    sequence: int | None = None,
) -> OrderBook:
    """Build an :class:`OrderBook` from plain ``(price_str, size)`` tuples.

    Bids are sorted descending and asks ascending regardless of input order, matching the
    contract every real adapter guarantees.
    """
    bid_levels = tuple(
        sorted(
            (BookLevel(price=Decimal(p), size=s) for p, s in bids),
            key=lambda lvl: lvl.price,
            reverse=True,
        )
    )
    ask_levels = tuple(
        sorted((BookLevel(price=Decimal(p), size=s) for p, s in asks), key=lambda lvl: lvl.price)
    )
    return OrderBook(
        canonical_id=canonical_id,
        venue=venue,
        timestamp=ts,
        bids=bid_levels,
        asks=ask_levels,
        sequence=sequence,
    )


def make_market(canonical_id: str = "TEST-MKT", **overrides: Any) -> NormalizedMarket:
    """Build a :class:`NormalizedMarket` with sane test defaults, overridable per-field."""
    defaults: dict[str, Any] = dict(
        canonical_id=canonical_id,
        venue=Venue.KALSHI,
        venue_market_id=canonical_id,
        event_id=f"{canonical_id}-EVT",
        title=f"Test market {canonical_id}",
        outcome_type=OutcomeType.BINARY,
        status=MarketStatus.OPEN,
        tick_size=Decimal("0.01"),
        min_order=1,
        fees=Fees(),
    )
    defaults.update(overrides)
    return NormalizedMarket(**defaults)


def make_trade(
    canonical_id: str,
    price: str,
    size: int,
    ts: datetime,
    aggressor: Any = None,
    venue: Venue = Venue.KALSHI,
) -> Trade:
    return Trade(
        canonical_id=canonical_id,
        venue=venue,
        timestamp=ts,
        price=Decimal(price),
        size=size,
        aggressor=aggressor,
    )


@dataclass
class StrategyHarness:
    """Drives one strategy instance against scripted events and collects its output."""

    strategy_cls: type[Strategy]
    params: dict[str, Any] = field(default_factory=dict)
    clock: SimulatedClock = field(default_factory=lambda: SimulatedClock(DEFAULT_START))
    strategy_id: str = "test-strategy"
    experiment_id: str = "test-experiment"

    def __post_init__(self) -> None:
        self.books: dict[str, OrderBook] = {}
        self.markets: dict[str, NormalizedMarket] = {}
        self.marks: dict[str, Decimal] = {}
        self.ctx = StrategyContext(self.clock, self.books, self.markets, self.marks, dict(self.params))
        self.strategy: Strategy = self.strategy_cls(
            self.strategy_id, self.experiment_id, self.ctx, dict(self.params)
        )
        self.intents: list[OrderIntent] = []
        self.forecasts: list[ProbabilityForecast] = []
        self.cancels: list[str] = []

    # ------------------------------------------------------------------ world setup

    def set_market(self, market: NormalizedMarket) -> None:
        self.markets[market.canonical_id] = market

    def set_book(self, book: OrderBook) -> None:
        self.books[book.canonical_id] = book

    def set_mark(self, canonical_id: str, price: Decimal) -> None:
        self.marks[canonical_id] = price

    def advance(self, seconds: float) -> None:
        self.clock.advance(seconds)

    def now(self) -> datetime:
        return self.clock.now()

    # ------------------------------------------------------------------ draining output

    def drain(self) -> None:
        self.intents.extend(self.strategy.generate_intents())
        self.forecasts.extend(self.strategy.drain_forecasts())
        cancel_fn = getattr(self.strategy, "cancel_requests", None)
        if callable(cancel_fn):
            self.cancels.extend(cancel_fn())

    # ------------------------------------------------------------------ feeding events

    def feed_book(self, book: OrderBook, first_seen: datetime | None = None) -> None:
        self.set_book(book)
        event = BookUpdateEvent(book=book, event_time=book.timestamp, first_seen_time=first_seen or book.timestamp)
        self.strategy.on_book_update(event)
        self.drain()

    def feed_book_series(
        self,
        canonical_id: str,
        series: list[tuple[list[tuple[str, int]], list[tuple[str, int]]]],
        step_seconds: float = 1.0,
    ) -> None:
        """Feed a sequence of ``(bids, asks)`` books, advancing the clock between each."""
        for bids, asks in series:
            book = make_book(canonical_id, bids, asks, self.now())
            self.feed_book(book)
            self.advance(step_seconds)

    def feed_trade(self, trade: Trade, first_seen: datetime | None = None) -> None:
        event = TradeEvent(trade=trade, event_time=trade.timestamp, first_seen_time=first_seen or trade.timestamp)
        self.strategy.on_trade(event)
        self.drain()

    def feed_trades(self, trades: list[Trade]) -> None:
        for trade in trades:
            self.feed_trade(trade)

    def feed_market_status(self, canonical_id: str, status: MarketStatus, venue: Venue = Venue.KALSHI) -> None:
        now = self.now()
        event = MarketStatusEvent(canonical_id=canonical_id, venue=venue, status=status, event_time=now, first_seen_time=now)
        self.strategy.on_market_status(event)
        self.drain()

    def feed_news(
        self,
        news_id: str = "news-1",
        title: str = "Test headline",
        source_class: SourceClass = SourceClass.UNKNOWN,
        tone: float | None = None,
        **overrides: Any,
    ) -> None:
        now = self.now()
        event = NewsEvent(
            news_id=news_id,
            title=title,
            source_class=source_class,
            tone=tone,
            event_time=overrides.pop("event_time", now),
            first_seen_time=overrides.pop("first_seen_time", now),
            **overrides,
        )
        self.strategy.on_news(event)
        self.drain()

    def feed_external_price(
        self, symbol: str, price: Decimal, implied_probability: Decimal | None = None, **overrides: Any
    ) -> None:
        now = self.now()
        event = ExternalPriceEvent(
            symbol=symbol,
            price=price,
            implied_probability=implied_probability,
            event_time=overrides.pop("event_time", now),
            first_seen_time=overrides.pop("first_seen_time", now),
            **overrides,
        )
        self.strategy.on_external_price(event)
        self.drain()

    def feed_trader_action(self, **kwargs: Any) -> None:
        now = self.now()
        kwargs.setdefault("event_time", now)
        kwargs.setdefault("first_seen_time", now)
        kwargs.setdefault("wallet", "0xTEST")
        event = TraderActionEvent(**kwargs)
        self.strategy.on_trader_action(event)
        self.drain()

    def feed_social(self, **kwargs: Any) -> None:
        now = self.now()
        kwargs.setdefault("event_time", now)
        kwargs.setdefault("first_seen_time", now)
        kwargs.setdefault("post_id", "post-1")
        kwargs.setdefault("platform", "test")
        kwargs.setdefault("author", "tester")
        kwargs.setdefault("text", "")
        event = SocialEvent(**kwargs)
        self.strategy.on_social(event)
        self.drain()

    def feed_filing(self, **kwargs: Any) -> None:
        now = self.now()
        kwargs.setdefault("event_time", now)
        kwargs.setdefault("first_seen_time", now)
        kwargs.setdefault("accession", "0000000000-00-000000")
        kwargs.setdefault("cik", "0000000000")
        event = FilingEvent(**kwargs)
        self.strategy.on_filing(event)
        self.drain()

    def feed_weather(self, **kwargs: Any) -> None:
        now = self.now()
        kwargs.setdefault("event_time", now)
        kwargs.setdefault("first_seen_time", now)
        kwargs.setdefault("station", "TEST")
        event = WeatherEvent(**kwargs)
        self.strategy.on_weather(event)
        self.drain()

    def feed_economic(self, **kwargs: Any) -> None:
        now = self.now()
        kwargs.setdefault("event_time", now)
        kwargs.setdefault("first_seen_time", now)
        kwargs.setdefault("series_id", "TEST")
        event = EconomicEvent(**kwargs)
        self.strategy.on_economic(event)
        self.drain()

    def feed_sports_state(self, **kwargs: Any) -> None:
        now = self.now()
        kwargs.setdefault("event_time", now)
        kwargs.setdefault("first_seen_time", now)
        kwargs.setdefault("game_id", "GAME-1")
        event = SportsStateEvent(**kwargs)
        self.strategy.on_sports_state(event)
        self.drain()

    def feed_timer(self, interval_seconds: float = 1.0, tag: str = "") -> None:
        now = self.now()
        event = TimerEvent(interval_seconds=interval_seconds, tag=tag, event_time=now, first_seen_time=now)
        self.strategy.on_timer(event)
        self.drain()

    def feed_fill(self, fill: Fill) -> None:
        self.strategy.on_fill(fill)
        self.drain()

    def feed_order_update(self, order: Order) -> None:
        self.strategy.on_order_update(order)
        self.drain()
