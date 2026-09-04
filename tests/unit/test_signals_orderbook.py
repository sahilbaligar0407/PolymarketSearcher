"""Unit tests for marketlab/signals/orderbook.py."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from marketlab.core.instruments import BookLevel, OrderBook, Side, Trade, Venue
from marketlab.core.orders import Action
from marketlab.signals.orderbook import (
    TradeFlowTracker,
    available_liquidity,
    book_pressure,
    book_shape,
    depth_imbalance,
    effective_spread,
    is_book_crossed,
    is_book_locked,
    microprice,
    relative_spread,
    slippage_estimate,
    spread,
    top_level_imbalance,
    weighted_depth_imbalance,
    weighted_microprice,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _book(
    bids: list[tuple[str, int]], asks: list[tuple[str, int]], ts: datetime = T0
) -> OrderBook:
    return OrderBook(
        canonical_id="TEST-MKT",
        venue=Venue.KALSHI,
        timestamp=ts,
        bids=tuple(BookLevel(price=Decimal(p), size=s) for p, s in bids),
        asks=tuple(BookLevel(price=Decimal(p), size=s) for p, s in asks),
    )


EMPTY_BOOK = _book([], [])
BID_ONLY_BOOK = _book([("0.49", 100)], [])
ASK_ONLY_BOOK = _book([], [("0.51", 50)])
SINGLE_LEVEL_BOOK = _book([("0.49", 100)], [("0.51", 50)])

FIVE_LEVEL_BOOK = _book(
    bids=[("0.49", 100), ("0.48", 80), ("0.47", 60), ("0.46", 40), ("0.45", 20)],
    asks=[("0.51", 50), ("0.52", 40), ("0.53", 30), ("0.54", 20), ("0.55", 10)],
)

CROSSED_BOOK = _book([("0.60", 10)], [("0.55", 10)])
LOCKED_BOOK = _book([("0.50", 10)], [("0.50", 10)])

DEEP_BOOK = _book(
    bids=[("0.49", 200), ("0.48", 300), ("0.47", 300), ("0.46", 300), ("0.45", 300)],
    asks=[("0.51", 200), ("0.52", 300), ("0.53", 300), ("0.54", 300), ("0.55", 300)],
)

# Deliberately asymmetric around mid (0.455): walking the ask side vs. the bid side for
# the same quantity produces genuinely different average prices, unlike DEEP_BOOK's
# mirror-image shape where top-of-book symmetry makes YES-buy and NO-buy slippage equal
# by construction (mid is defined as their average, so ask-mid always equals mid-bid at
# the touch).
ASYMMETRIC_BOOK = _book(
    bids=[("0.40", 100), ("0.35", 400)],
    asks=[("0.51", 50), ("0.60", 500)],
)


# ---------------------------------------------------------------------------
# Imbalance
# ---------------------------------------------------------------------------


def test_top_level_imbalance_known_book() -> None:
    b = _book([("0.49", 100)], [("0.51", 50)])
    imb = top_level_imbalance(b)
    assert imb is not None
    assert abs(imb - (100 / 150)) < 1e-12


def test_depth_imbalance_hand_computed_1_3_5_levels() -> None:
    # Every bid level is exactly double the corresponding ask level, so the ratio is 2/3
    # at every depth - a clean hand computation that also proves depth actually sums.
    assert abs(depth_imbalance(FIVE_LEVEL_BOOK, 1) - (2 / 3)) < 1e-12
    assert abs(depth_imbalance(FIVE_LEVEL_BOOK, 3) - (2 / 3)) < 1e-12
    assert abs(depth_imbalance(FIVE_LEVEL_BOOK, 5) - (2 / 3)) < 1e-12
    # Sanity: the raw sums behind the level-3 and level-5 ratios really do differ.
    assert FIVE_LEVEL_BOOK.depth(3, "bid") == 240
    assert FIVE_LEVEL_BOOK.depth(5, "bid") == 300


# ---------------------------------------------------------------------------
# Degenerate books: empty / one-sided / single-level never raise
# ---------------------------------------------------------------------------


def test_empty_and_one_sided_books_return_none_without_raising() -> None:
    for b in (EMPTY_BOOK, BID_ONLY_BOOK, ASK_ONLY_BOOK):
        assert top_level_imbalance(b) is None
        assert depth_imbalance(b, 3) is None
        assert weighted_depth_imbalance(b, 3, decay=5.0) is None
        assert microprice(b) is None
        assert weighted_microprice(b, 3) is None
        assert spread(b) is None
        assert relative_spread(b) is None
        assert book_pressure(b) is None
        assert effective_spread(b, 10) is None
        assert slippage_estimate(b, Action.BUY, Side.YES, 10) is None
        assert available_liquidity(b, Decimal("0.50"), "bid") == 0
        assert available_liquidity(b, Decimal("0.50"), "ask") == 0
        # is_book_crossed/is_book_locked degrade to False rather than raising - a book
        # missing a side isn't "crossed", it's just unpriceable.
        assert is_book_crossed(b) is False
        assert is_book_locked(b) is False
        shape = book_shape(b)
        assert shape.mid is None
        assert shape.is_crossed is False


def test_single_level_book_computes_without_raising() -> None:
    # A single resting level on each side is a perfectly valid (if thin) two-sided book -
    # every function should produce a real number here, not None.
    b = SINGLE_LEVEL_BOOK
    assert top_level_imbalance(b) is not None
    assert depth_imbalance(b, 5) is not None  # fewer levels than requested is fine
    assert microprice(b) is not None
    assert spread(b) == Decimal("0.02")
    assert relative_spread(b) is not None
    assert effective_spread(b, 1) is not None
    assert slippage_estimate(b, Action.BUY, Side.YES, 1) is not None
    assert is_book_crossed(b) is False
    assert is_book_locked(b) is False


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------


def test_crossed_book_detected() -> None:
    assert is_book_crossed(CROSSED_BOOK) is True
    assert is_book_crossed(FIVE_LEVEL_BOOK) is False


def test_locked_book_detected() -> None:
    assert is_book_locked(LOCKED_BOOK) is True
    assert is_book_locked(FIVE_LEVEL_BOOK) is False


# ---------------------------------------------------------------------------
# Execution cost
# ---------------------------------------------------------------------------


def test_effective_spread_larger_for_bigger_size() -> None:
    small = effective_spread(DEEP_BOOK, 10)
    large = effective_spread(DEEP_BOOK, 1000)
    assert small == Decimal("0.02")  # both walks stay in the touch level
    assert large == Decimal("0.05")  # hand-computed below
    assert large > small


def test_effective_spread_hand_computation_for_size_1000() -> None:
    # Buy-walk 1000 against asks: 200@0.51 + 300@0.52 + 300@0.53 + 200@0.54
    buy_cost = 200 * Decimal("0.51") + 300 * Decimal("0.52") + 300 * Decimal("0.53") + 200 * Decimal(
        "0.54"
    )
    buy_avg = buy_cost / 1000
    # Sell-walk 1000 against bids: 200@0.49 + 300@0.48 + 300@0.47 + 200@0.46
    sell_cost = 200 * Decimal("0.49") + 300 * Decimal("0.48") + 300 * Decimal("0.47") + 200 * Decimal(
        "0.46"
    )
    sell_avg = sell_cost / 1000
    expected = buy_avg - sell_avg
    assert effective_spread(DEEP_BOOK, 1000) == expected


def test_effective_spread_none_when_book_lacks_depth() -> None:
    assert effective_spread(DEEP_BOOK, 1_000_000) is None


def test_slippage_estimate_matches_hand_walked_book() -> None:
    # Buy-walk 500 against asks: 200@0.51 + 300@0.52
    cost = 200 * Decimal("0.51") + 300 * Decimal("0.52")
    avg_price = cost / 500
    mid = DEEP_BOOK.mid
    assert mid == Decimal("0.5000")
    expected = avg_price - mid
    assert slippage_estimate(DEEP_BOOK, Action.BUY, Side.YES, 500) == expected


def test_slippage_estimate_no_side_differs_from_yes_side_on_an_asymmetric_book() -> None:
    # Buying NO consumes the bid side (buying NO == selling YES) and is expressed in NO
    # terms. On a book that isn't a mirror image around mid, this must genuinely differ
    # from the YES-buy slippage.
    yes_buy = slippage_estimate(ASYMMETRIC_BOOK, Action.BUY, Side.YES, 300)
    no_buy = slippage_estimate(ASYMMETRIC_BOOK, Action.BUY, Side.NO, 300)
    assert yes_buy is not None and no_buy is not None
    assert yes_buy != no_buy

    mid = ASYMMETRIC_BOOK.mid
    assert mid is not None
    # Hand computation: buy-walk 300 against asks = 50@0.51 + 250@0.60.
    buy_avg = (50 * Decimal("0.51") + 250 * Decimal("0.60")) / 300
    assert yes_buy == buy_avg - mid
    # NO-buy walks bids instead: 100@0.40 + 200@0.35, then re-expressed in NO terms.
    sell_avg = (100 * Decimal("0.40") + 200 * Decimal("0.35")) / 300
    assert no_buy == (Decimal(1) - sell_avg) - (Decimal(1) - mid)


def test_available_liquidity() -> None:
    # Bids priced at or above 0.47: the 0.49/0.48/0.47 levels of FIVE_LEVEL_BOOK.
    assert available_liquidity(FIVE_LEVEL_BOOK, Decimal("0.47"), "bid") == 100 + 80 + 60
    # Asks priced at or below 0.53: the 0.51/0.52/0.53 levels.
    assert available_liquidity(FIVE_LEVEL_BOOK, Decimal("0.53"), "ask") == 50 + 40 + 30
    assert available_liquidity(EMPTY_BOOK, Decimal("0.50"), "bid") == 0


def test_weighted_microprice_matches_microprice_at_one_level() -> None:
    assert weighted_microprice(FIVE_LEVEL_BOOK, 1) == microprice(FIVE_LEVEL_BOOK)


def test_book_shape_single_pass_snapshot() -> None:
    shape = book_shape(FIVE_LEVEL_BOOK)
    assert shape.mid == FIVE_LEVEL_BOOK.mid
    assert shape.top_imbalance == top_level_imbalance(FIVE_LEVEL_BOOK)
    assert shape.is_crossed is False


# ---------------------------------------------------------------------------
# Trade flow
# ---------------------------------------------------------------------------


def _trade(ts: datetime, price: str, size: int, aggressor: Side | None) -> Trade:
    return Trade(
        canonical_id="TEST-MKT",
        venue=Venue.KALSHI,
        timestamp=ts,
        price=Decimal(price),
        size=size,
        aggressor=aggressor,
    )


def test_trade_flow_tracker_evicts_old_trades_and_computes_aggressor_imbalance() -> None:
    tracker = TradeFlowTracker(window_seconds=10)
    tracker.add_trade(_trade(T0, "0.50", 100, Side.YES))  # will be evicted
    tracker.add_trade(_trade(T0 + timedelta(seconds=5), "0.51", 50, Side.YES))
    tracker.add_trade(_trade(T0 + timedelta(seconds=8), "0.49", 30, Side.NO))

    now = T0 + timedelta(seconds=12)  # first trade (t=0) is now 12s old, outside window=10
    trades = tracker._trades(now)
    assert len(trades) == 2
    assert all(t.size != 100 for t in trades)

    imbalance = tracker.aggressor_imbalance(now)
    assert imbalance is not None
    # buy=50, sell=30, total=80 -> (50-30)/80 = 0.25
    assert abs(imbalance - 0.25) < 1e-12

    signed = tracker.signed_volume(now)
    assert signed == 50 - 30

    vwap = tracker.vwap(now)
    assert vwap == (Decimal("0.51") * 50 + Decimal("0.49") * 30) / 80

    largest = tracker.largest_trade(now)
    assert largest is not None
    assert largest.size == 50

    intensity = tracker.trade_intensity(now)
    assert intensity == 2 / 10


def test_trade_flow_tracker_empty_returns_none() -> None:
    tracker = TradeFlowTracker(window_seconds=10)
    assert tracker.aggressor_imbalance(T0) is None
    assert tracker.signed_volume(T0) is None
    assert tracker.vwap(T0) is None
    assert tracker.largest_trade(T0) is None
    assert tracker.trade_intensity(T0) is None
