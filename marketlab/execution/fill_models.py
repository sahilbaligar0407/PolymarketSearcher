"""Fill simulation: market orders walk the real book, limit orders rest and get hit.

Everything here operates on a :class:`marketlab.core.instruments.OrderBook`, which is
always expressed in YES-probability terms (a Kalshi NO bid at 30c already appears as a
YES ask at 0.70 by the time it reaches us).  That convention is the source of the one
piece of arithmetic worth over-documenting: **what does it mean to buy or sell NO against
a YES-quoted book?**

Binary contracts satisfy ``P(YES) + P(NO) = 1`` exactly, so every NO order has an
equivalent YES order:

* ``BUY  NO @ p``  ==  ``SELL YES @ (1 - p)``  -> the counterparty is a YES *buyer*, so a
  NO buy consumes the book's **bid** side (highest bid first), at effective NO price
  ``1 - bid.price``. Since bids are sorted descending by price, walking them in their
  existing order visits the *highest* bid (== *lowest* effective NO price == best price
  for a NO buyer) first - no re-sorting needed.
* ``SELL NO @ p``  ==  ``BUY  YES @ (1 - p)``  -> the counterparty is a YES *seller*, so a
  NO sell consumes the book's **ask** side (lowest ask first), at effective NO price
  ``1 - ask.price``. Asks are sorted ascending, so the lowest ask (== *highest* effective
  NO price == best price for a NO seller) is visited first, again with no re-sorting.

So the rule that falls out is simple and side-agnostic: **BUY consumes the side that is
already sorted best-first for a buyer of that side's own probability (asks for YES, bids
for NO); SELL consumes the other side.** No level is ever invented - liquidity exhaustion
produces a partial fill, never a phantom fill at the "last price".

Limit orders never get this treatment when they're marketable at rest time -
:mod:`marketlab.execution.paper_broker` calls :func:`walk_book` for the immediately
crossable portion of a limit order too. What lives in this module besides ``walk_book``
is the family of models for *resting* limit orders: given a book update or a trade print,
how much (if any) of a resting order would plausibly have been filled? All three fill
models share one calling convention: they return ``list[(price, qty)]`` describing new
partial fills; the caller (PaperBroker) is responsible for decrementing
``order_state.remaining_quantity`` by the returned quantity before the next call, and for
stamping every fill produced this way with ``is_maker=True`` - by construction, every fill
that comes out of a resting order was passive.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal

from marketlab.core.instruments import ONE, PROB_QUANTUM, ZERO, OrderBook, Side, Trade
from marketlab.core.orders import Action

# ---------------------------------------------------------------------------
# Market orders: walk the real visible book.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FillResult:
    """Outcome of walking a book for a market order (or the marketable slice of a limit)."""

    #: Levels actually consumed, best price first: [(price, qty), ...].
    fills: tuple[tuple[Decimal, int], ...]
    filled_qty: int
    unfilled_qty: int
    #: Size-weighted average fill price, or None if nothing filled.
    avg_price: Decimal | None
    #: Price of the worst (last) level consumed, or None if nothing filled.
    worst_price: Decimal | None
    #: The book timestamp actually used, for the replay audit trail.
    book_timestamp_used: datetime


def _select_levels(book: OrderBook, action: Action, side: Side) -> tuple:
    """Pick the physical book side a BUY/SELL of YES/NO consumes. See module docstring."""
    if side is Side.YES:
        return book.asks if action is Action.BUY else book.bids
    return book.bids if action is Action.BUY else book.asks


def _price_transform(side: Side):
    """Map a physical book price (always YES-terms) to the order's own side's price."""
    if side is Side.YES:
        return lambda p: p
    return lambda p: (ONE - p)


def walk_book(book: OrderBook, action: Action, side: Side, quantity: int) -> FillResult:
    """Consume real visible liquidity for a market order. Never invents a fill.

    Walks the appropriate side of ``book`` (see module docstring for the NO-side mapping)
    level by level, best price first, until ``quantity`` is exhausted or the book runs
    out. A quantity larger than the visible book produces a partial fill
    (``unfilled_qty > 0``); an empty book produces a zero fill with no exception.
    """
    if quantity <= 0:
        raise ValueError("quantity must be positive")

    levels = _select_levels(book, action, side)
    transform = _price_transform(side)

    fills: list[tuple[Decimal, int]] = []
    remaining = quantity
    for lvl in levels:
        if remaining <= 0:
            break
        if lvl.size <= 0:
            continue
        take = min(remaining, lvl.size)
        fills.append((transform(lvl.price), take))
        remaining -= take

    filled_qty = quantity - remaining
    avg_price: Decimal | None = None
    worst_price: Decimal | None = None
    if fills:
        total_cost = sum((p * Decimal(q) for p, q in fills), ZERO)
        avg_price = (total_cost / Decimal(filled_qty)).quantize(
            PROB_QUANTUM, rounding=ROUND_HALF_EVEN
        )
        worst_price = fills[-1][0]

    return FillResult(
        fills=tuple(fills),
        filled_qty=filled_qty,
        unfilled_qty=remaining,
        avg_price=avg_price,
        worst_price=worst_price,
        book_timestamp_used=book.timestamp,
    )


# ---------------------------------------------------------------------------
# Limit orders: three models of "did my resting order get hit".
# ---------------------------------------------------------------------------


@dataclass
class OrderState:
    """Everything a :class:`LimitFillModel` needs about one resting order.

    This is deliberately a small, model-agnostic shell: ``model_state`` is a scratch dict
    each model uses for its own bookkeeping (e.g. ``QueueFillModel`` stores
    ``"queue_ahead"`` there) so the shared dataclass never has to grow model-specific
    fields. The caller (PaperBroker) owns ``remaining_quantity`` and must decrement it by
    the quantity of every fill a model returns before the next call - the models
    themselves never mutate it, only ``model_state``.
    """

    order_id: str
    canonical_id: str
    side: Side
    action: Action
    #: Price in the order's own side terms (a NO order's limit is a NO price).
    limit_price: Decimal
    remaining_quantity: int
    placed_at: datetime
    model_state: dict = field(default_factory=dict)

    @property
    def book_side_and_price(self) -> tuple[str, Decimal]:
        """Map (side, action, limit_price) onto the physical YES-book side/price this
        order rests on, so every model can compare against book/trade prices (always
        YES-terms) uniformly regardless of whether the order itself is YES or NO.
        """
        if self.side is Side.YES:
            book_price = self.limit_price
            book_side = "bid" if self.action is Action.BUY else "ask"
        else:
            book_price = ONE - self.limit_price
            book_side = "ask" if self.action is Action.BUY else "bid"
        return book_side, book_price


class LimitFillModel(abc.ABC):
    """Common interface for the three resting-limit-order fill models."""

    @abc.abstractmethod
    def on_book_update(self, order_state: OrderState, book: OrderBook) -> list[tuple[Decimal, int]]:
        """React to a new book snapshot. Return new fills, if any."""

    @abc.abstractmethod
    def on_trade(self, order_state: OrderState, trade: Trade) -> list[tuple[Decimal, int]]:
        """React to a public print. Return new fills, if any."""


class TouchFillModel(LimitFillModel):
    """Fills as soon as the market *touches* the limit price.

    Optimistic; research diagnostic only. It assumes the resting order is filled in full
    the instant the opposing best price reaches the limit, regardless of how much size
    actually traded and regardless of queue position. Never use this to size real risk -
    use it only to see the best case a strategy could ever have gotten.
    """

    def _touched(self, order_state: OrderState, price: Decimal) -> bool:
        book_side, book_price = order_state.book_side_and_price
        if book_side == "bid":
            # We're a resting buyer; we're touched once a seller is willing at/below us.
            return price <= book_price
        return price >= book_price

    def on_book_update(self, order_state: OrderState, book: OrderBook) -> list[tuple[Decimal, int]]:
        if order_state.remaining_quantity <= 0:
            return []
        book_side, _ = order_state.book_side_and_price
        opposing = book.best_ask if book_side == "bid" else book.best_bid
        if opposing is None or not self._touched(order_state, opposing):
            return []
        qty = order_state.remaining_quantity
        return [(order_state.limit_price, qty)]

    def on_trade(self, order_state: OrderState, trade: Trade) -> list[tuple[Decimal, int]]:
        if order_state.remaining_quantity <= 0:
            return []
        if not self._touched(order_state, trade.price):
            return []
        qty = order_state.remaining_quantity
        return [(order_state.limit_price, qty)]


class TradeThroughFillModel(LimitFillModel):
    """Fills only when the market actually *trades through* the limit price.

    The conservative default. A print strictly better than the limit (for a resting buy,
    a sale at a price below the limit) proves the market walked past the limit level, so
    the whole remaining quantity is credited. A print *at* the limit is ambiguous - it may
    have gone to other resting orders at the same level - so it only counts once enough
    cumulative volume has printed at that exact price to plausibly cover this order's
    remaining size (``on_book_update`` never fills anything by itself; only real prints
    do). This deliberately ignores true queue position (see ``QueueFillModel`` for that),
    which is why it fills at least as much as ``QueueFillModel`` and, because it requires
    actual traded volume rather than a mere quote touch, no more than ``TouchFillModel``.
    """

    _ACCUM_KEY = "traded_at_limit"

    def on_book_update(self, order_state: OrderState, book: OrderBook) -> list[tuple[Decimal, int]]:
        # Fills here are earned only by real prints, never by a quote merely moving.
        return []

    def on_trade(self, order_state: OrderState, trade: Trade) -> list[tuple[Decimal, int]]:
        qty = order_state.remaining_quantity
        if qty <= 0:
            return []
        _, book_price = order_state.book_side_and_price
        book_side, _ = order_state.book_side_and_price
        if book_side == "bid":
            strictly_through = trade.price < book_price
        else:
            strictly_through = trade.price > book_price
        at_limit = trade.price == book_price

        if strictly_through:
            # A print through the limit proves the market reached past this order - but
            # only for the size that actually printed. Crediting the whole order filled
            # 25 contracts off 3-contract prints (FINDINGS 58).
            order_state.model_state[self._ACCUM_KEY] = 0
            return [(order_state.limit_price, min(qty, trade.size))]

        if at_limit:
            accum = order_state.model_state.get(self._ACCUM_KEY, 0) + trade.size
            if accum >= qty:
                order_state.model_state[self._ACCUM_KEY] = 0
                return [(order_state.limit_price, qty)]
            order_state.model_state[self._ACCUM_KEY] = accum
        return []


class QueueFillModel(LimitFillModel):
    """Estimates true queue position at the resting price level. Preferred advanced model.

    On placement (the first ``on_book_update`` this model sees for an order), the size
    already resting at the order's price level becomes ``queue_ahead`` - everyone
    presumed to be in front of us. Every subsequent print *at that exact price* first
    drains ``queue_ahead``; only volume left over after the queue is cleared can fill this
    order. Cancellations ahead of us are invisible in a book snapshot (a level shrinking
    could be a cancel or a trade we didn't see printed), so we deliberately never decay
    ``queue_ahead`` from a book update alone - only confirmed prints move it. That
    conservatism is exactly why this model fills at most as much as ``TradeThroughFillModel``.
    """

    _QUEUE_KEY = "queue_ahead"

    def _resting_size_at(self, order_state: OrderState, book: OrderBook) -> int:
        book_side, book_price = order_state.book_side_and_price
        levels = book.bids if book_side == "bid" else book.asks
        for lvl in levels:
            if lvl.price == book_price:
                return lvl.size
        return 0

    def on_book_update(self, order_state: OrderState, book: OrderBook) -> list[tuple[Decimal, int]]:
        if self._QUEUE_KEY not in order_state.model_state:
            order_state.model_state[self._QUEUE_KEY] = self._resting_size_at(order_state, book)
        # Never fill from a book update alone: see docstring on non-decay of the queue.
        return []

    def on_trade(self, order_state: OrderState, trade: Trade) -> list[tuple[Decimal, int]]:
        qty = order_state.remaining_quantity
        if qty <= 0:
            return []
        _, book_price = order_state.book_side_and_price
        if trade.price != book_price:
            return []
        queue_ahead = order_state.model_state.get(self._QUEUE_KEY, 0)
        consumed_by_others = min(trade.size, queue_ahead)
        order_state.model_state[self._QUEUE_KEY] = queue_ahead - consumed_by_others
        remainder = trade.size - consumed_by_others
        if remainder <= 0:
            return []
        fill_qty = min(remainder, qty)
        return [(order_state.limit_price, fill_qty)]


_MODEL_REGISTRY: dict[str, type[LimitFillModel]] = {
    "TOUCH": TouchFillModel,
    "TRADE_THROUGH": TradeThroughFillModel,
    "QUEUE": QueueFillModel,
}


def build_limit_fill_model(name: str) -> LimitFillModel:
    """Resolve ``settings.execution.limit_fill_model`` ("TOUCH"|"TRADE_THROUGH"|"QUEUE")."""
    try:
        return _MODEL_REGISTRY[name.upper()]()
    except KeyError as exc:
        raise ValueError(f"unknown limit fill model: {name!r}") from exc
