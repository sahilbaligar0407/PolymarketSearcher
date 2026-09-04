"""Unit tests for marketlab/strategies/cross_venue.py."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from marketlab.clock import SimulatedClock
from marketlab.core.events import TimerEvent
from marketlab.core.instruments import (
    BookLevel,
    Category,
    MarketStatus,
    NormalizedMarket,
    OrderBook,
    Venue,
)
from marketlab.core.strategy import StrategyContext
from marketlab.matching import MarketMatch
from marketlab.strategies.cross_venue import CrossVenueRelativeValueStrategy

TS = datetime(2026, 1, 1, tzinfo=UTC)
KALSHI_ID = "kalshi:btc-above-100k"
POLY_ID = "poly:btc-above-100k"


def _kalshi_market(open_interest: Decimal = Decimal("100")) -> NormalizedMarket:
    return NormalizedMarket(
        canonical_id=KALSHI_ID,
        venue=Venue.KALSHI,
        venue_market_id="KXBTCD-TEST",
        event_id="btc-event-1",
        title="Will BTC be above $100k?",
        category=Category.CRYPTO,
        status=MarketStatus.OPEN,
        open_interest=open_interest,
        min_order=1,
    )


def _kalshi_book(bid: str = "0.40", ask: str = "0.42", timestamp: datetime = TS) -> OrderBook:
    return OrderBook(
        canonical_id=KALSHI_ID,
        venue=Venue.KALSHI,
        timestamp=timestamp,
        bids=(BookLevel(price=Decimal(bid), size=100),),
        asks=(BookLevel(price=Decimal(ask), size=100),),
    )


def _poly_book(bid: str = "0.68", ask: str = "0.70", timestamp: datetime = TS) -> OrderBook:
    return OrderBook(
        canonical_id=POLY_ID,
        venue=Venue.POLY_GLOBAL,
        timestamp=timestamp,
        bids=(BookLevel(price=Decimal(bid), size=100),),
        asks=(BookLevel(price=Decimal(ask), size=100),),
    )


def _match(confidence: str = "0.95", same_outcome: bool | None = True, human_review: bool = False) -> MarketMatch:
    return MarketMatch(
        match_id="m1",
        canonical_id_a=KALSHI_ID,
        canonical_id_b=POLY_ID,
        match_confidence=Decimal(confidence),
        same_outcome_boolean=same_outcome,
        rule_diff="",
        time_diff="",
        resolution_source_diff="",
        human_review_required=human_review,
        created_at=TS,
    )


def _ctx(kalshi_book: OrderBook, poly_book: OrderBook | None, now: datetime = TS) -> StrategyContext:
    clock = SimulatedClock(now)
    market = _kalshi_market()
    books = {KALSHI_ID: kalshi_book}
    if poly_book is not None:
        books[POLY_ID] = poly_book
    return StrategyContext(clock=clock, books=books, markets={KALSHI_ID: market}, marks={})


def test_unapproved_match_produces_no_intent_even_with_huge_price_gap() -> None:
    # Kalshi mid ~0.41, Polymarket mid ~0.69 - a huge 28-point gap - but the match is
    # explicitly not approved (same_outcome_boolean False).
    match = _match(confidence="0.99", same_outcome=False)
    ctx = _ctx(_kalshi_book(), _poly_book())
    strat = CrossVenueRelativeValueStrategy("s1", "e1", ctx, params={"matches": [match], "min_edge": "0.02"})
    strat.on_timer(TimerEvent(event_time=TS, first_seen_time=TS))

    assert strat.generate_intents() == []
    assert strat.refusal_counts.get("unapproved_match", 0) == 1


def test_stale_polymarket_book_produces_no_intent() -> None:
    match = _match()
    stale_poly_book = _poly_book(timestamp=TS - timedelta(seconds=120))
    ctx = _ctx(_kalshi_book(), stale_poly_book)
    strat = CrossVenueRelativeValueStrategy(
        "s1",
        "e1",
        ctx,
        params={"matches": [match], "min_edge": "0.02", "max_book_staleness_seconds": 10.0},
    )
    strat.on_timer(TimerEvent(event_time=TS, first_seen_time=TS))

    assert strat.generate_intents() == []
    assert strat.refusal_counts.get("stale_poly_book", 0) == 1


def test_approved_match_emits_kalshi_only_intent() -> None:
    match = _match(confidence="0.95")
    ctx = _ctx(_kalshi_book(), _poly_book())
    strat = CrossVenueRelativeValueStrategy("s1", "e1", ctx, params={"matches": [match], "min_edge": "0.02"})
    strat.on_timer(TimerEvent(event_time=TS, first_seen_time=TS))

    intents = strat.generate_intents()
    assert len(intents) == 1
    intent = intents[0]
    assert intent.venue is Venue.KALSHI
    assert intent.canonical_id.startswith("kalshi:")
    assert intent.features["poly_canonical_id"] == POLY_ID
    assert intent.expected_edge is not None and intent.expected_edge > 0
