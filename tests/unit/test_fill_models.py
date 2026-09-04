"""Acceptance tests for marketlab.execution.fill_models.

These are the PRD's canonical cases. The book-walk test in particular exists to catch the
single most dangerous shortcut a fill simulator can take: pricing a whole order at the
touch price instead of actually walking the book.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from marketlab.core.instruments import BookLevel, OrderBook, Side, Trade, Venue
from marketlab.core.orders import Action
from marketlab.execution.fill_models import (
    OrderState,
    QueueFillModel,
    TouchFillModel,
    TradeThroughFillModel,
    build_limit_fill_model,
    walk_book,
)

TS = datetime(2026, 1, 1, tzinfo=UTC)


def _book(bids=(), asks=(), ts=TS, canonical_id="mkt-1"):
    return OrderBook(canonical_id=canonical_id, venue=Venue.KALSHI, timestamp=ts, bids=bids, asks=asks)


def _state(**kwargs):
    defaults = dict(
        order_id="ord-1",
        canonical_id="mkt-1",
        side=Side.YES,
        action=Action.BUY,
        limit_price=Decimal("0.50"),
        remaining_quantity=100,
        placed_at=TS,
    )
    defaults.update(kwargs)
    return OrderState(**defaults)


# ---------------------------------------------------------------------------
# walk_book: market orders
# ---------------------------------------------------------------------------


def test_canonical_book_walk_is_not_priced_at_touch():
    """The canonical case: never `if last_price <= my_limit: fill me`."""
    book = _book(
        asks=(
            BookLevel(price=Decimal("0.51"), size=10),
            BookLevel(price=Decimal("0.52"), size=20),
            BookLevel(price=Decimal("0.55"), size=100),
        )
    )
    result = walk_book(book, Action.BUY, Side.YES, 25)

    assert result.fills == ((Decimal("0.51"), 10), (Decimal("0.52"), 15))
    assert result.filled_qty == 25
    assert result.unfilled_qty == 0
    # 10 @ 0.51 + 15 @ 0.52 = 12.90 / 25 = 0.516
    assert result.avg_price == Decimal("0.516")
    assert result.avg_price != Decimal("0.51"), "must not be priced as if all 25 filled at touch"
    assert result.worst_price == Decimal("0.52")


def test_partial_fill_when_liquidity_exhausted():
    book = _book(
        asks=(
            BookLevel(price=Decimal("0.51"), size=10),
            BookLevel(price=Decimal("0.52"), size=20),
            BookLevel(price=Decimal("0.55"), size=100),
        )
    )
    result = walk_book(book, Action.BUY, Side.YES, 200)
    assert result.filled_qty == 130
    assert result.unfilled_qty == 70


def test_empty_book_yields_zero_fill_no_exception():
    book = _book()
    result = walk_book(book, Action.BUY, Side.YES, 10)
    assert result.fills == ()
    assert result.filled_qty == 0
    assert result.unfilled_qty == 10
    assert result.avg_price is None
    assert result.worst_price is None
    assert result.book_timestamp_used == TS


def test_no_side_buy_arithmetic_against_yes_terms_book():
    """BUY NO consumes bids at effective price (1 - bid_price)."""
    book = _book(
        bids=(
            BookLevel(price=Decimal("0.60"), size=10),
            BookLevel(price=Decimal("0.55"), size=20),
        )
    )
    result = walk_book(book, Action.BUY, Side.NO, 20)
    assert result.fills == ((Decimal("0.40"), 10), (Decimal("0.45"), 10))
    assert result.filled_qty == 20
    assert result.avg_price == Decimal("0.425")


def test_sell_no_consumes_asks_at_effective_price():
    """SELL NO == BUY YES: consumes asks ascending at effective price (1 - ask_price)."""
    book = _book(
        asks=(
            BookLevel(price=Decimal("0.30"), size=5),
            BookLevel(price=Decimal("0.40"), size=5),
        )
    )
    result = walk_book(book, Action.SELL, Side.NO, 10)
    assert result.fills == ((Decimal("0.70"), 5), (Decimal("0.60"), 5))
    assert result.filled_qty == 10


def test_sell_yes_consumes_bids_descending():
    book = _book(
        bids=(
            BookLevel(price=Decimal("0.60"), size=5),
            BookLevel(price=Decimal("0.55"), size=5),
        )
    )
    result = walk_book(book, Action.SELL, Side.YES, 10)
    assert result.fills == ((Decimal("0.60"), 5), (Decimal("0.55"), 5))


def test_walk_book_rejects_nonpositive_quantity():
    book = _book(asks=(BookLevel(price=Decimal("0.5"), size=10),))
    try:
        walk_book(book, Action.BUY, Side.YES, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-positive quantity")


# ---------------------------------------------------------------------------
# Limit fill models: TOUCH vs TRADE_THROUGH vs QUEUE ordering
# ---------------------------------------------------------------------------


def test_touch_ge_trade_through_ge_queue_on_same_tape():
    """The whole point: TOUCH is most permissive, QUEUE is most conservative."""
    touch_state = _state()
    trade_through_state = _state()
    queue_state = _state()

    touch = TouchFillModel()
    trade_through = TradeThroughFillModel()
    queue = QueueFillModel()

    # Event 1: a book update where the market touches the limit, with 500 already
    # resting ahead of us at that price level.
    book1 = _book(
        bids=(BookLevel(price=Decimal("0.50"), size=500),),
        asks=(BookLevel(price=Decimal("0.50"), size=50),),
    )

    def apply(model, state, fills):
        got = sum(q for _, q in fills)
        state.remaining_quantity -= got
        return got

    touch_filled = apply(touch, touch_state, touch.on_book_update(touch_state, book1))
    trade_through_filled = apply(
        trade_through, trade_through_state, trade_through.on_book_update(trade_through_state, book1)
    )
    queue_filled = apply(queue, queue_state, queue.on_book_update(queue_state, book1))

    # Event 2: a single print of 150 contracts prints exactly at the limit.
    trade = Trade(canonical_id="mkt-1", venue=Venue.KALSHI, timestamp=TS, price=Decimal("0.50"), size=150)

    touch_filled += apply(touch, touch_state, touch.on_trade(touch_state, trade))
    trade_through_filled += apply(
        trade_through, trade_through_state, trade_through.on_trade(trade_through_state, trade)
    )
    queue_filled += apply(queue, queue_state, queue.on_trade(queue_state, trade))

    assert touch_filled == 100  # touched immediately on the book update; fully filled
    assert trade_through_filled == 100  # 150 printed at the limit >= 100 remaining
    assert queue_filled == 0  # the 500 resting ahead of us absorbed the whole 150 print

    assert touch_filled >= trade_through_filled
    assert queue_filled <= trade_through_filled


def test_queue_model_fills_only_remainder_after_queue_ahead_clears():
    """500 resting ahead; prints of 300 then 300 -> fills only the last 100."""
    state = _state(remaining_quantity=100, limit_price=Decimal("0.50"))
    model = QueueFillModel()

    book = _book(bids=(BookLevel(price=Decimal("0.50"), size=500),))
    assert model.on_book_update(state, book) == []
    assert state.model_state["queue_ahead"] == 500

    trade1 = Trade(canonical_id="mkt-1", venue=Venue.KALSHI, timestamp=TS, price=Decimal("0.50"), size=300)
    fills1 = model.on_trade(state, trade1)
    assert fills1 == []
    assert state.model_state["queue_ahead"] == 200

    trade2 = Trade(canonical_id="mkt-1", venue=Venue.KALSHI, timestamp=TS, price=Decimal("0.50"), size=300)
    fills2 = model.on_trade(state, trade2)
    assert fills2 == [(Decimal("0.50"), 100)]
    assert state.model_state["queue_ahead"] == 0


def test_queue_model_never_decays_from_a_book_update_alone():
    """Cancellations ahead are unobservable: a shrinking level must not free up queue."""
    state = _state(remaining_quantity=50, limit_price=Decimal("0.50"))
    model = QueueFillModel()
    model.on_book_update(state, _book(bids=(BookLevel(price=Decimal("0.50"), size=1000),)))
    assert state.model_state["queue_ahead"] == 1000
    # The level shrinks with no trade print - could be a cancel, must not be trusted.
    model.on_book_update(state, _book(bids=(BookLevel(price=Decimal("0.50"), size=10),)))
    assert state.model_state["queue_ahead"] == 1000


def test_touch_fill_model_docstring_flags_it_as_optimistic_diagnostic_only():
    assert TouchFillModel.__doc__ is not None
    assert "Optimistic; research diagnostic only." in TouchFillModel.__doc__


def test_touch_model_fills_sell_side_on_bid_touch():
    state = _state(action=Action.SELL, limit_price=Decimal("0.60"), remaining_quantity=10)
    model = TouchFillModel()
    book = _book(bids=(BookLevel(price=Decimal("0.60"), size=5),))
    fills = model.on_book_update(state, book)
    assert fills == [(Decimal("0.60"), 10)]


def test_trade_through_strict_print_fills_in_full():
    state = _state(remaining_quantity=10, limit_price=Decimal("0.50"))
    model = TradeThroughFillModel()
    # A print strictly better (lower) than our buy limit proves the market walked
    # through our level.
    trade = Trade(canonical_id="mkt-1", venue=Venue.KALSHI, timestamp=TS, price=Decimal("0.48"), size=1)
    assert model.on_trade(state, trade) == [(Decimal("0.50"), 10)]


def test_build_limit_fill_model_factory():
    assert isinstance(build_limit_fill_model("touch"), TouchFillModel)
    assert isinstance(build_limit_fill_model("TRADE_THROUGH"), TradeThroughFillModel)
    assert isinstance(build_limit_fill_model("Queue"), QueueFillModel)
    try:
        build_limit_fill_model("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown model name")
