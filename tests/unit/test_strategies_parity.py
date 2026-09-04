"""Unit tests for marketlab/strategies/binary_parity.py."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from marketlab.clock import SimulatedClock
from marketlab.core.events import TimerEvent
from marketlab.core.instruments import (
    BookLevel,
    Category,
    Fees,
    MarketStatus,
    NormalizedMarket,
    OrderBook,
    Venue,
)
from marketlab.core.strategy import StrategyContext
from marketlab.strategies.binary_parity import BinaryParityStrategy

TS = datetime(2026, 1, 1, tzinfo=UTC)
CLOSE = datetime(2026, 1, 2, tzinfo=UTC)


def _market(canonical_id: str, title: str, fees: Fees | None = None) -> NormalizedMarket:
    return NormalizedMarket(
        canonical_id=canonical_id,
        venue=Venue.KALSHI,
        venue_market_id=canonical_id.upper(),
        event_id="game-1",
        title=title,
        category=Category.SPORTS,
        status=MarketStatus.OPEN,
        close_time=CLOSE,
        fees=fees or Fees(),
        min_order=1,
    )


def _book(canonical_id: str, ask_levels: list[tuple[str, int]]) -> OrderBook:
    return OrderBook(
        canonical_id=canonical_id,
        venue=Venue.KALSHI,
        timestamp=TS,
        bids=(BookLevel(price=Decimal("0.01"), size=1000),),
        asks=tuple(BookLevel(price=Decimal(p), size=s) for p, s in ask_levels),
    )


def _ctx(markets: list[NormalizedMarket], books: dict[str, OrderBook]) -> StrategyContext:
    clock = SimulatedClock(TS)
    return StrategyContext(
        clock=clock,
        books=books,
        markets={m.canonical_id: m for m in markets},
        marks={},
    )


def _complementary_markets(fees: Fees | None = None) -> tuple[NormalizedMarket, NormalizedMarket]:
    a = _market("kalshi:chiefs-win", "Will the Chiefs win their game?", fees)
    b = _market("kalshi:chiefs-lose", "Will the Chiefs lose their game?", fees)
    return a, b


def test_touch_looks_profitable_but_no_size_behind_it_is_rejected() -> None:
    a, b = _complementary_markets()
    # Touch prices sum to 0.60 (looks very profitable), but only 1 contract is quoted at
    # that price on each leg and there is nothing behind it - the requested probe quantity
    # (5) cannot be filled.
    books = {
        a.canonical_id: _book(a.canonical_id, [("0.30", 1)]),
        b.canonical_id: _book(b.canonical_id, [("0.30", 1)]),
    }
    ctx = _ctx([a, b], books)
    strat = BinaryParityStrategy("s1", "e1", ctx, params={"quantity": 5, "min_edge": "0.01"})
    strat.on_timer(TimerEvent(event_time=TS, first_seen_time=TS))

    assert strat.generate_intents() == []
    assert "leg_a_insufficient_depth" in strat.rejection_counts or "leg_b_insufficient_depth" in strat.rejection_counts


def test_genuinely_executable_pair_passes() -> None:
    a, b = _complementary_markets()
    books = {
        a.canonical_id: _book(a.canonical_id, [("0.45", 20)]),
        b.canonical_id: _book(b.canonical_id, [("0.45", 20)]),
    }
    ctx = _ctx([a, b], books)
    strat = BinaryParityStrategy("s1", "e1", ctx, params={"quantity": 5, "min_edge": "0.01"})
    strat.on_timer(TimerEvent(event_time=TS, first_seen_time=TS))

    intents = strat.generate_intents()
    assert len(intents) == 2
    canonical_ids = {i.canonical_id for i in intents}
    assert canonical_ids == {a.canonical_id, b.canonical_id}
    for intent in intents:
        assert intent.venue is Venue.KALSHI
        assert intent.expected_edge is not None and intent.expected_edge > 0
        assert intent.features["parity_pair_id"] == intents[0].features["parity_pair_id"]


def test_fees_flip_a_marginal_case_from_accept_to_reject() -> None:
    a, b = _complementary_markets(fees=Fees(taker_rate=Decimal("0")))
    a_fee, b_fee = _complementary_markets(fees=Fees(taker_rate=Decimal("0.07")))
    # Prices sum to 0.98: a razor-thin 0.02 raw edge before fees.
    ask_levels = [("0.49", 20)]
    books = {
        a.canonical_id: _book(a.canonical_id, ask_levels),
        b.canonical_id: _book(b.canonical_id, ask_levels),
    }
    params = {"quantity": 5, "min_edge": "0.01", "slippage_buffer": "0.0"}

    ctx_no_fees = _ctx([a, b], books)
    strat_no_fees = BinaryParityStrategy("s1", "e1", ctx_no_fees, params=params)
    strat_no_fees.on_timer(TimerEvent(event_time=TS, first_seen_time=TS))
    assert len(strat_no_fees.generate_intents()) == 2

    books_with_fee_markets = {
        a_fee.canonical_id: _book(a_fee.canonical_id, ask_levels),
        b_fee.canonical_id: _book(b_fee.canonical_id, ask_levels),
    }
    ctx_with_fees = _ctx([a_fee, b_fee], books_with_fee_markets)
    strat_with_fees = BinaryParityStrategy("s2", "e1", ctx_with_fees, params=params)
    strat_with_fees.on_timer(TimerEvent(event_time=TS, first_seen_time=TS))
    assert strat_with_fees.generate_intents() == []
    assert strat_with_fees.rejection_counts.get("edge_below_min", 0) >= 1
