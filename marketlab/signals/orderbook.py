"""Microstructure features computed from :class:`marketlab.core.instruments.OrderBook`.

Both sides of ``OrderBook`` are expressed in YES-probability terms (a Kalshi NO bid at
30c is folded into a YES ask at 0.70 by the adapter) - every function here inherits that
convention and never re-derives it.

The hard rule: every function returns ``None`` on an empty or one-sided book. It never
raises and never fabricates a number (e.g. falling back to 0.5) just because one side of
the book is missing - a strategy that gets ``None`` must explicitly decide what to do
about a book it cannot price, rather than silently trading on a made-up value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal

from marketlab.core.instruments import PROB_QUANTUM, BookLevel, OrderBook, Side, Trade
from marketlab.core.orders import Action
from marketlab.signals.rolling import TimeWindow

# ---------------------------------------------------------------------------
# Imbalance / shape
# ---------------------------------------------------------------------------


def top_level_imbalance(book: OrderBook) -> float | None:
    """``bid_size / (bid_size + ask_size)`` at the touch. ``None`` if either side is empty."""
    if not book.bids or not book.asks:
        return None
    bid_size = book.bids[0].size
    ask_size = book.asks[0].size
    total = bid_size + ask_size
    if total == 0:
        return None
    return bid_size / total


def depth_imbalance(book: OrderBook, levels: int) -> float | None:
    """Same as :func:`top_level_imbalance` but summed over the top ``levels`` levels."""
    if not book.bids or not book.asks:
        return None
    bid_depth = book.depth(levels, "bid")
    ask_depth = book.depth(levels, "ask")
    total = bid_depth + ask_depth
    if total == 0:
        return None
    return bid_depth / total


def weighted_depth_imbalance(book: OrderBook, levels: int, decay: float) -> float | None:
    """Depth imbalance where each level is weighted by its price distance from mid.

    A size wall three ticks away should count for less than the same size sitting at the
    touch, so each level's size is scaled by ``exp(-decay * |price - mid|)`` before being
    summed. ``decay`` is in probability units (per the same ``[0, 1]`` scale as the book's
    prices); larger ``decay`` concentrates weight closer to the touch.
    """
    if not book.bids or not book.asks:
        return None
    mid = book.mid
    if mid is None:
        return None
    mid_f = float(mid)

    def _weighted_sum(levels_seq: tuple[BookLevel, ...]) -> float:
        total = 0.0
        for lvl in levels_seq[:levels]:
            distance = abs(float(lvl.price) - mid_f)
            total += lvl.size * math.exp(-decay * distance)
        return total

    weighted_bid = _weighted_sum(book.bids)
    weighted_ask = _weighted_sum(book.asks)
    total = weighted_bid + weighted_ask
    if total == 0:
        return None
    return weighted_bid / total


def microprice(book: OrderBook) -> Decimal | None:
    """Size-weighted mid at the touch. Delegates to ``OrderBook.microprice``."""
    return book.microprice


def weighted_microprice(book: OrderBook, levels: int) -> Decimal | None:
    """Microprice generalized to ``levels`` levels of depth on each side.

    The standard microprice leans the mid toward the side with less resting size at the
    touch: ``(bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)``. This
    extends that idea by first computing each side's size-weighted average price over the
    top ``levels`` levels, then combining them the same way. With ``levels=1`` this reduces
    exactly to ``book.microprice``.
    """
    if not book.bids or not book.asks:
        return None
    bid_levels = book.bids[:levels]
    ask_levels = book.asks[:levels]
    bid_depth = sum(lvl.size for lvl in bid_levels)
    ask_depth = sum(lvl.size for lvl in ask_levels)
    if bid_depth == 0 or ask_depth == 0:
        return None
    avg_bid_price = sum((lvl.price * lvl.size for lvl in bid_levels), Decimal(0)) / bid_depth
    avg_ask_price = sum((lvl.price * lvl.size for lvl in ask_levels), Decimal(0)) / ask_depth
    total = bid_depth + ask_depth
    mp = (avg_bid_price * ask_depth + avg_ask_price * bid_depth) / total
    # Quantize to the same PROB_QUANTUM grid as every other price on the book, matching
    # OrderBook.microprice (levels=1 must reduce to exactly the same Decimal, not just the
    # same value to arbitrary precision).
    return mp.quantize(PROB_QUANTUM, rounding=ROUND_HALF_EVEN)


def spread(book: OrderBook) -> Decimal | None:
    return book.spread


def relative_spread(book: OrderBook) -> float | None:
    """Spread as a fraction of mid. ``None`` if either is unavailable or mid is zero."""
    sp = book.spread
    mid = book.mid
    if sp is None or mid is None or mid == 0:
        return None
    return float(sp / mid)


def book_pressure(book: OrderBook) -> float | None:
    """Signed touch imbalance in ``[-1, 1]``: positive means bid-heavy, negative ask-heavy."""
    if not book.bids or not book.asks:
        return None
    bid_size = book.bids[0].size
    ask_size = book.asks[0].size
    total = bid_size + ask_size
    if total == 0:
        return None
    return (bid_size - ask_size) / total


# ---------------------------------------------------------------------------
# Execution cost
# ---------------------------------------------------------------------------


def _walk_levels(levels: tuple[BookLevel, ...], quantity: int) -> tuple[Decimal, int] | None:
    """Walk ``levels`` in order to fill ``quantity`` contracts.

    Returns ``(size_weighted_average_price, filled)`` where ``filled`` may be less than
    ``quantity`` if the book runs out of depth. Returns ``None`` for a non-positive
    quantity or a book side with nothing on it.
    """
    if quantity <= 0 or not levels:
        return None
    remaining = quantity
    cost = Decimal(0)
    filled = 0
    for lvl in levels:
        take = min(remaining, lvl.size)
        if take <= 0:
            continue
        cost += lvl.price * take
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    if filled == 0:
        return None
    return cost / filled, filled


def effective_spread(book: OrderBook, quantity: int) -> Decimal | None:
    """The honest round-trip cost of trading ``quantity``: buy-walk cost minus sell-walk cost.

    The touch spread only prices the first contract; anything bigger has to walk through
    resting size at worse prices. ``None`` if the book can't fully absorb ``quantity`` on
    both sides (an honest cost can't be quoted for size the book doesn't have) or is empty
    / one-sided.
    """
    if quantity <= 0 or not book.bids or not book.asks:
        return None
    buy = _walk_levels(book.asks, quantity)
    sell = _walk_levels(book.bids, quantity)
    if buy is None or sell is None:
        return None
    buy_avg, buy_filled = buy
    sell_avg, sell_filled = sell
    if buy_filled < quantity or sell_filled < quantity:
        return None
    return buy_avg - sell_avg


def slippage_estimate(
    book: OrderBook, action: Action, side: Side, quantity: int
) -> Decimal | None:
    """Expected average fill price minus mid, for a market order of ``quantity``.

    The book is always carried in YES-probability terms, so buying NO (or selling NO)
    walks the *opposite* physical side of the book from buying/selling YES: buying NO at
    price ``p`` is the same fill as selling YES at ``1-p``. This walks the correct side,
    then re-expresses both the fill price and mid in the requested side's own terms before
    differencing, so the sign is always "cost relative to mid *for the side actually held*."
    """
    if quantity <= 0 or not book.bids or not book.asks:
        return None
    mid = book.mid
    if mid is None:
        return None
    walk_asks = (action is Action.BUY and side is Side.YES) or (
        action is Action.SELL and side is Side.NO
    )
    levels = book.asks if walk_asks else book.bids
    result = _walk_levels(levels, quantity)
    if result is None:
        return None
    avg_price, filled = result
    if filled < quantity:
        return None
    if side is Side.YES:
        return avg_price - mid
    return (Decimal(1) - avg_price) - (Decimal(1) - mid)


def available_liquidity(book: OrderBook, max_price: Decimal, side: str) -> int:
    """Total resting size on ``side`` ("bid"/"ask") priced at least as favorably as ``max_price``.

    For asks this means "at or below" ``max_price`` (what you could buy without paying
    more); for bids it means "at or above" ``max_price`` (what you could sell without
    accepting less). Returns ``0`` (never ``None``) for an empty or missing side - "no
    liquidity available" is a well-defined answer, unlike a price that doesn't exist.
    """
    rows = book.bids if side == "bid" else book.asks
    if side == "bid":
        return sum(lvl.size for lvl in rows if lvl.price >= max_price)
    return sum(lvl.size for lvl in rows if lvl.price <= max_price)


@dataclass(frozen=True)
class BookShape:
    """Cheap per-tick snapshot of the microstructure features above, computed together."""

    mid: Decimal | None
    spread: Decimal | None
    relative_spread: float | None
    microprice: Decimal | None
    top_imbalance: float | None
    depth_imbalance_3: float | None
    depth_imbalance_5: float | None
    book_pressure: float | None
    is_crossed: bool
    is_locked: bool


def book_shape(book: OrderBook) -> BookShape:
    return BookShape(
        mid=book.mid,
        spread=spread(book),
        relative_spread=relative_spread(book),
        microprice=book.microprice,
        top_imbalance=top_level_imbalance(book),
        depth_imbalance_3=depth_imbalance(book, 3),
        depth_imbalance_5=depth_imbalance(book, 5),
        book_pressure=book_pressure(book),
        is_crossed=is_book_crossed(book),
        is_locked=is_book_locked(book),
    )


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------


def is_book_crossed(book: OrderBook) -> bool:
    """A crossed book (best bid > best ask) means bad data - strategies must skip it."""
    b, a = book.best_bid, book.best_ask
    if b is None or a is None:
        return False
    return b > a


def is_book_locked(book: OrderBook) -> bool:
    """Best bid equals best ask: zero spread, usually a stale or degenerate quote."""
    b, a = book.best_bid, book.best_ask
    if b is None or a is None:
        return False
    return b == a


# ---------------------------------------------------------------------------
# Trade flow
# ---------------------------------------------------------------------------


class TradeFlowTracker:
    """Accumulates prints over a trailing time window and derives order-flow features.

    ``Trade.aggressor`` is a :class:`Side`: ``YES`` means the taker bought YES (a buy
    print in book terms), ``NO`` means the taker bought NO (a sell print in YES terms).
    Trades with no recorded aggressor contribute to volume/VWAP but not to the signed
    imbalance, since their direction is genuinely unknown rather than assumed neutral.
    """

    def __init__(self, window_seconds: float) -> None:
        self._window: TimeWindow = TimeWindow(window_seconds)

    def add_trade(self, trade: Trade) -> None:
        self._window.add(trade.timestamp, trade)

    def _trades(self, now: datetime) -> list[Trade]:
        return [v for v in self._window.values(now) if isinstance(v, Trade)]

    def aggressor_imbalance(self, now: datetime) -> float | None:
        """(buy volume - sell volume) / total volume, over the window. ``None`` if empty."""
        trades = self._trades(now)
        buy = sum(t.size for t in trades if t.aggressor is Side.YES)
        sell = sum(t.size for t in trades if t.aggressor is Side.NO)
        total = buy + sell
        if total == 0:
            return None
        return (buy - sell) / total

    def trade_intensity(self, now: datetime) -> float | None:
        """Trades per second, using the tracker's configured window as the time base."""
        trades = self._trades(now)
        if not trades:
            return None
        return len(trades) / self._window.seconds

    def signed_volume(self, now: datetime) -> int | None:
        """Net signed size: +size for YES-aggressor prints, -size for NO-aggressor prints."""
        trades = self._trades(now)
        if not trades:
            return None
        total = 0
        for t in trades:
            if t.aggressor is Side.YES:
                total += t.size
            elif t.aggressor is Side.NO:
                total -= t.size
        return total

    def vwap(self, now: datetime) -> Decimal | None:
        trades = self._trades(now)
        if not trades:
            return None
        total_size = sum(t.size for t in trades)
        if total_size == 0:
            return None
        total_cost = sum((t.price * t.size for t in trades), Decimal(0))
        return total_cost / total_size

    def largest_trade(self, now: datetime) -> Trade | None:
        trades = self._trades(now)
        if not trades:
            return None
        return max(trades, key=lambda t: t.size)
