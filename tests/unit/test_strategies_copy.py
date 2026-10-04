"""Unit tests for marketlab/strategies/copy_trader.py (highest priority: evidence D)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from marketlab.clock import SimulatedClock
from marketlab.core.events import BookUpdateEvent, TimerEvent, TraderActionEvent
from marketlab.core.instruments import (
    BookLevel,
    Category,
    MarketStatus,
    NormalizedMarket,
    OrderBook,
    Side,
    Venue,
)
from marketlab.core.strategy import StrategyContext
from marketlab.strategies.copy_trader import CopyBasketStrategy, CopyTraderStrategy

TS = datetime(2026, 1, 1, tzinfo=UTC)
CANONICAL_ID = "kalshi:matched-market-1"
WALLET = "0xWALLET"


def _market(open_interest: Decimal = Decimal("50")) -> NormalizedMarket:
    return NormalizedMarket(
        canonical_id=CANONICAL_ID,
        venue=Venue.KALSHI,
        venue_market_id="KXTEST-1",
        event_id="evt-1",
        title="Will X happen?",
        category=Category.POLITICS,
        status=MarketStatus.OPEN,
        open_interest=open_interest,
        min_order=1,
    )


def _book(bid: str, ask: str, timestamp: datetime = TS) -> OrderBook:
    return OrderBook(
        canonical_id=CANONICAL_ID,
        venue=Venue.KALSHI,
        timestamp=timestamp,
        bids=(BookLevel(price=Decimal(bid), size=1000),),
        asks=(BookLevel(price=Decimal(ask), size=1000),),
    )


class MutableCtx(StrategyContext):
    """A StrategyContext whose book can be swapped mid-test, to simulate time passing."""

    def set_book(self, book: OrderBook) -> None:
        self._books[CANONICAL_ID] = book


def _ctx(book: OrderBook, now: datetime = TS) -> MutableCtx:
    clock = SimulatedClock(now)
    return MutableCtx(clock=clock, books={CANONICAL_ID: book}, markets={CANONICAL_ID: _market()}, marks={})


def _trader_action(
    canonical_id: str = CANONICAL_ID,
    side: Side | None = Side.YES,
    price: Decimal = Decimal("0.10"),
    usd_size: Decimal = Decimal("500"),
    first_seen_time: datetime = TS,
    wallet: str = WALLET,
) -> TraderActionEvent:
    return TraderActionEvent(
        wallet=wallet,
        canonical_id=canonical_id,
        side=side,
        action="BUY",
        price=price,
        size=usd_size / price if price else None,
        usd_size=usd_size,
        category=Category.POLITICS,
        event_time=first_seen_time,
        first_seen_time=first_seen_time,
        transaction_hash="0xabc",
    )


# ---------------------------------------------------------------------------
# follower_delay_seconds=0 is not offered as a strategy arm
# ---------------------------------------------------------------------------


def test_zero_delay_is_rejected_at_construction() -> None:
    ctx = _ctx(_book("0.09", "0.11"))
    with pytest.raises(ValueError):
        CopyTraderStrategy("s1", "e1", ctx, params={"follower_delay_seconds": 0, "mode": "raw"})


# ---------------------------------------------------------------------------
# Headline test: execution uses the DELAYED book, never the source price
# ---------------------------------------------------------------------------


def test_execution_uses_delayed_book_not_source_price() -> None:
    ctx = _ctx(_book("0.09", "0.11"))  # book at signal time - source got a great price
    strat = CopyTraderStrategy("s1", "e1", ctx, params={"follower_delay_seconds": 5, "mode": "raw"})

    # Source trader paid 0.05 - a far better price than anything available later.
    event = _trader_action(side=Side.YES, price=Decimal("0.05"), first_seen_time=TS)
    strat.on_trader_action(event)
    assert strat.generate_intents() == []  # not due yet

    # Time passes. By the time the delay elapses, the book has moved much worse.
    later = TS + timedelta(seconds=10)
    ctx.clock.set(later)
    worse_book = _book("0.79", "0.81", timestamp=later)
    ctx.set_book(worse_book)
    strat.on_book_update(BookUpdateEvent(book=worse_book, event_time=later, first_seen_time=later))

    intents = strat.generate_intents()
    assert len(intents) == 1
    intent = intents[0]
    assert intent.venue is Venue.KALSHI
    assert intent.canonical_id.startswith("kalshi:")
    # The executed price must be the worse, later price - never the source's 0.05.
    assert intent.limit_price == Decimal("0.81")
    assert intent.features["source_price"] == 0.05
    assert intent.features["executed_price"] == 0.81


# ---------------------------------------------------------------------------
# No approved match -> no trade, counted
# ---------------------------------------------------------------------------


def test_no_market_match_produces_no_trade_and_is_counted() -> None:
    ctx = _ctx(_book("0.09", "0.11"))
    strat = CopyTraderStrategy("s1", "e1", ctx, params={"follower_delay_seconds": 5, "mode": "raw"})

    unmatched = _trader_action(canonical_id="")  # matcher never resolved a Kalshi contract
    strat.on_trader_action(unmatched)

    ctx.clock.advance(10)
    strat.on_timer(TimerEvent(event_time=ctx.now(), first_seen_time=ctx.now()))
    assert strat.generate_intents() == []
    assert strat.refusal_counts.get("no_market_match", 0) == 1


def test_match_confidence_below_threshold_produces_no_trade() -> None:
    ctx = _ctx(_book("0.09", "0.11"))
    matches = {"poly-cond-1": {"match_confidence": "0.5", "approved": True}}
    strat = CopyTraderStrategy(
        "s1",
        "e1",
        ctx,
        params={"follower_delay_seconds": 5, "mode": "raw", "matches": matches, "min_match_confidence": "0.9"},
    )
    event = _trader_action()
    event = event.model_copy(update={"poly_condition_id": "poly-cond-1"})
    strat.on_trader_action(event)

    assert strat.refusal_counts.get("match_confidence_below_min", 0) == 1
    ctx.clock.advance(10)
    strat.on_timer(TimerEvent(event_time=ctx.now(), first_seen_time=ctx.now()))
    assert strat.generate_intents() == []


# ---------------------------------------------------------------------------
# fade emits the opposite side of raw on identical input
# ---------------------------------------------------------------------------


def test_fade_emits_opposite_side_of_raw_on_identical_input() -> None:
    def run(mode: str) -> Decimal:
        ctx = _ctx(_book("0.09", "0.11"))
        strat = CopyTraderStrategy("s1", "e1", ctx, params={"follower_delay_seconds": 5, "mode": mode})
        event = _trader_action(side=Side.YES, price=Decimal("0.10"))
        strat.on_trader_action(event)

        later = TS + timedelta(seconds=10)
        ctx.clock.set(later)
        book = _book("0.39", "0.41", timestamp=later)
        ctx.set_book(book)
        strat.on_book_update(BookUpdateEvent(book=book, event_time=later, first_seen_time=later))
        intents = strat.generate_intents()
        assert len(intents) == 1
        return intents[0].side

    raw_side = run("raw")
    fade_side = run("fade")
    assert raw_side is Side.YES
    assert fade_side is Side.NO
    assert fade_side is raw_side.opposite


# ---------------------------------------------------------------------------
# An UNVERIFIED wallet is never followed
# ---------------------------------------------------------------------------


def test_unverified_wallet_is_never_followed() -> None:
    ctx = _ctx(_book("0.09", "0.11"))
    strat = CopyTraderStrategy(
        "s1",
        "e1",
        ctx,
        params={
            "follower_delay_seconds": 5,
            "mode": "raw",
            "wallet_status": {WALLET: "UNVERIFIED"},
            "allow_trading_status": ("ONCHAIN_OR_API_CONFIRMED",),
        },
    )
    strat.on_trader_action(_trader_action())

    ctx.clock.advance(10)
    strat.on_timer(TimerEvent(event_time=ctx.now(), first_seen_time=ctx.now()))
    assert strat.generate_intents() == []
    assert strat.refusal_counts.get("wallet_status_not_allowed", 0) == 1


def test_confirmed_wallet_is_followed() -> None:
    ctx = _ctx(_book("0.09", "0.11"))
    strat = CopyTraderStrategy(
        "s1",
        "e1",
        ctx,
        params={
            "follower_delay_seconds": 5,
            "mode": "raw",
            "wallet_status": {WALLET: "ONCHAIN_OR_API_CONFIRMED"},
        },
    )
    strat.on_trader_action(_trader_action())

    later = TS + timedelta(seconds=10)
    ctx.clock.set(later)
    book = _book("0.39", "0.41", timestamp=later)
    ctx.set_book(book)
    strat.on_book_update(BookUpdateEvent(book=book, event_time=later, first_seen_time=later))
    assert len(strat.generate_intents()) == 1


# ---------------------------------------------------------------------------
# CopyBasketStrategy: basic sanity
# ---------------------------------------------------------------------------


def test_copy_basket_is_dormant_below_five_traders_and_never_blocks_startup() -> None:
    """An under-filled basket constructs and stays dormant rather than raising.

    Zero validated wallets is the NORMAL day-one state: traders are recorded as
    DISCOVERED and forward-tracked for weeks before any of them qualifies. Raising here
    previously aborted sleeve creation for the entire tournament on a fresh database.
    """
    ctx = _ctx(_book("0.09", "0.11"))

    # Empty roster - a brand new deployment.
    empty = CopyBasketStrategy("s1", "e1", ctx, params={"follower_delay_seconds": 5})
    empty.on_trader_action(_trader_action(wallet="0xANY"))
    ctx.clock.set(TS + timedelta(seconds=10))
    empty.on_timer(TimerEvent(event_time=ctx.clock.now(), first_seen_time=ctx.clock.now()))
    assert empty.generate_intents() == []

    # Partially filled - still not a basket, so still dormant.
    partial = CopyBasketStrategy(
        "s2", "e2", ctx, params={"follower_delay_seconds": 5, "traders": {"0x1": "1.0", "0x2": "1.0"}}
    )
    partial.on_trader_action(_trader_action(wallet="0x1"))
    ctx.clock.set(TS + timedelta(seconds=20))
    partial.on_timer(TimerEvent(event_time=ctx.clock.now(), first_seen_time=ctx.clock.now()))
    assert partial.generate_intents() == []


def test_copy_basket_rejects_an_oversized_roster() -> None:
    """Too many traders is a real misconfiguration and still raises."""
    ctx = _ctx(_book("0.09", "0.11"))
    with pytest.raises(ValueError):
        CopyBasketStrategy(
            "s1",
            "e1",
            ctx,
            params={"follower_delay_seconds": 5, "traders": {f"0x{i}": "1.0" for i in range(25)}},
        )


def test_copy_basket_only_follows_member_wallets() -> None:
    ctx = _ctx(_book("0.09", "0.11"))
    traders = {f"0x{i}": "1.0" for i in range(5)}
    strat = CopyBasketStrategy(
        "s1", "e1", ctx, params={"follower_delay_seconds": 5, "traders": traders, "weighting": "equal"}
    )
    non_member_event = _trader_action(wallet="0xNOT_A_MEMBER")
    strat.on_trader_action(non_member_event)
    assert strat.refusal_counts.get("not_a_basket_member", 0) == 1

    member_event = _trader_action(wallet="0x0")
    strat.on_trader_action(member_event)

    later = TS + timedelta(seconds=10)
    ctx.clock.set(later)
    book = _book("0.39", "0.41", timestamp=later)
    ctx.set_book(book)
    strat.on_book_update(BookUpdateEvent(book=book, event_time=later, first_seen_time=later))
    intents = strat.generate_intents()
    assert len(intents) == 1
    assert intents[0].venue is Venue.KALSHI



# ---------------------------------------------------------------------------
# TraderScore-driven modes
# ---------------------------------------------------------------------------


def test_qualified_mode_copies_only_roster_wallets() -> None:
    roster = {WALLET: 80.0}
    strat = CopyTraderStrategy("s", "e", _ctx(_book("0.09", "0.11")),
                               params={"mode": "qualified", "roster": roster, "follower_delay_seconds": 5})
    strat.on_trader_action(_trader_action(wallet="0xSTRANGER"))
    assert strat.refusal_counts.get("wallet_not_qualified") == 1
    strat.on_trader_action(_trader_action())
    assert len(strat._pending_copies) == 1


def test_qualified_consensus_needs_two_qualified_wallets() -> None:
    roster = {WALLET: 80.0, "0xSECOND": 70.0}
    strat = CopyTraderStrategy("s", "e", _ctx(_book("0.09", "0.11")),
                               params={"mode": "qualified_consensus", "roster": roster, "follower_delay_seconds": 5})
    strat.on_trader_action(_trader_action())
    assert strat._pending_copies == []
    strat.on_trader_action(_trader_action(wallet="0xNOTQUALIFIED"))
    assert strat._pending_copies == []
    strat.on_trader_action(_trader_action(wallet="0xSECOND"))
    assert len(strat._pending_copies) == 1


def test_basket_follows_the_live_roster() -> None:
    roster: dict[str, float] = {}
    ctx = _ctx(_book("0.09", "0.11"))
    basket = CopyBasketStrategy("s", "e", ctx, params={"roster": roster, "basket_size": 5, "follower_delay_seconds": 5})
    basket.on_timer(TimerEvent(event_time=TS, first_seen_time=TS))
    assert basket._traders == {}
    roster.update({f"0xW{i}": 60.0 + i for i in range(8)})
    ctx.clock.advance(601)
    basket.on_timer(TimerEvent(event_time=TS, first_seen_time=TS))
    assert set(basket._traders) == {f"0xW{i}" for i in range(3, 8)}  # top 5 by score
