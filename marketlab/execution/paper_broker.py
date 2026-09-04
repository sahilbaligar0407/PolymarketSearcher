"""The PaperBroker: the heart of trustworthy research.

Every number MarketLab's research reports produce is downstream of this module deciding,
honestly, what would have happened to an order. That means three disciplines are
non-negotiable and enforced structurally, not just by convention:

1. **No look-ahead.** An order decided at time T can only ever see the book as it stood at
   or before ``T + total_latency``. ``_book_as_of`` is the single chokepoint for this - it
   is a hard error class of bug for any other code path in this module to read a book by
   any means other than that method.
2. **Never invent liquidity.** Every fill comes from either walking a real
   :class:`~marketlab.core.instruments.OrderBook` (:func:`marketlab.execution.fill_models.walk_book`)
   or from a documented resting-order fill model. There is no "if last_price <= my_limit,
   fill me" shortcut anywhere in this file.
3. **Settlement is venue-authoritative only.** :meth:`PaperBroker.settle` takes
   ``winning_side`` as a given, verified fact. It is never inferred from a model's
   probability, a headline, or a tweet - see the comment on that method.

``PaperBroker`` never sends anything over a network; it refuses outright in ``LIVE`` and
``DATA_ONLY`` modes (see :meth:`submit`).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from decimal import ROUND_CEILING, ROUND_HALF_EVEN, Decimal
from typing import Protocol, runtime_checkable

from marketlab.clock import Clock
from marketlab.core.broker import Broker, Mode
from marketlab.core.instruments import (
    ONE,
    PROB_QUANTUM,
    ZERO,
    Category,
    MarketStatus,
    NormalizedMarket,
    OrderBook,
    Side,
    Trade,
)
from marketlab.core.orders import (
    Action,
    Fill,
    Order,
    OrderIntent,
    OrderStatus,
    OrderType,
    RejectReason,
    TimeInForce,
)
from marketlab.core.portfolio import Portfolio
from marketlab.execution.fill_models import LimitFillModel, OrderState, walk_book
from marketlab.execution.latency import LatencyModel
from marketlab.execution.risk_gateway import RiskGateway
from marketlab.settings import Settings

# ---------------------------------------------------------------------------
# Fees
# ---------------------------------------------------------------------------


@runtime_checkable
class FeeCalculator(Protocol):
    """Duck-typed fee calculator injected into PaperBroker (and KalshiLiveBroker)."""

    def calculate(
        self, market: NormalizedMarket, price: Decimal, quantity: int, is_maker: bool
    ) -> Decimal: ...


class KalshiFeeCalculator:
    """Reference implementation of :class:`~marketlab.core.instruments.Fees`.

    Kalshi's quadratic fee is ``ceil_to_cent(rate * C * P * (1-P))``, floored at
    ``min_fee_cents`` whenever a nonzero fee is owed. This exists here as the concrete
    default; nothing prevents another team from injecting a different ``FeeCalculator``.
    """

    def calculate(
        self, market: NormalizedMarket, price: Decimal, quantity: int, is_maker: bool
    ) -> Decimal:
        fees = market.fees
        rate = fees.maker_rate if is_maker else fees.taker_rate
        if rate <= ZERO or quantity <= 0:
            return ZERO
        if fees.formula == "kalshi_quadratic":
            raw_dollars = rate * Decimal(quantity) * price * (ONE - price)
        else:
            raw_dollars = rate * Decimal(quantity) * price
        cents = (raw_dollars * 100).to_integral_value(rounding=ROUND_CEILING)
        cents = max(cents, Decimal(fees.min_fee_cents))
        return cents / Decimal(100)


# ---------------------------------------------------------------------------
# Storage duck-type
# ---------------------------------------------------------------------------


@runtime_checkable
class StateStoreLike(Protocol):
    """The subset of another team's ``StateStore`` PaperBroker actually calls.

    Duck-typed so we are never blocked on that module's existence. `store` passed to the
    constructor need only implement whichever of these it wants persisted; PaperBroker
    checks with ``hasattr`` before calling, and ``store`` may be ``None`` entirely.
    """

    def save_order(self, order: Order) -> None: ...
    def save_fill(self, fill: Fill) -> None: ...
    def load_open_orders(self) -> list[Order]: ...


def _avg_price(fills: list[Fill] | tuple[Fill, ...]) -> Decimal | None:
    total_qty = sum(f.quantity for f in fills)
    if total_qty == 0:
        return None
    total_cost = sum((f.price * Decimal(f.quantity) for f in fills), ZERO)
    return (total_cost / Decimal(total_qty)).quantize(PROB_QUANTUM, rounding=ROUND_HALF_EVEN)


class PaperBroker(Broker):
    """Simulates fills against real books with real latency. Never touches a venue."""

    def __init__(
        self,
        *,
        mode: Mode,
        clock: Clock,
        latency_model: LatencyModel,
        limit_fill_model: LimitFillModel,
        fee_calculator: FeeCalculator,
        book_provider: Callable[[str], OrderBook | None],
        market_provider: Callable[[str], NormalizedMarket | None],
        portfolio_provider: Callable[[str], Portfolio],
        risk_gateway: RiskGateway,
        settings: Settings,
        store: StateStoreLike | None = None,
    ) -> None:
        self.mode = mode
        self.clock = clock
        self.latency_model = latency_model
        self.limit_fill_model = limit_fill_model
        self.fee_calculator = fee_calculator
        self.book_provider = book_provider
        self.market_provider = market_provider
        self.portfolio_provider = portfolio_provider
        self.risk_gateway = risk_gateway
        self.settings = settings
        self.store = store

        self._orders: dict[str, Order] = {}
        self._resting: dict[str, OrderState] = {}
        self._resting_by_market: dict[str, set[str]] = {}
        #: Append-only, ascending-by-timestamp per market. The only source of truth
        #: `_book_as_of` reads from; `book_provider` is merged into it, never read directly.
        self._book_history: dict[str, list[OrderBook]] = {}
        self._experiments_by_market: dict[str, set[str]] = {}

        # PRD-required metrics.
        self.stale_data_skips = 0
        self.risk_gate_skips = 0
        self.rejected_orders = 0
        self.cancel_count = 0
        self.maker_fills = 0
        self.taker_fills = 0

    # -- book history / point-in-time lookups -------------------------------------

    def _record_book(self, book: OrderBook) -> None:
        history = self._book_history.setdefault(book.canonical_id, [])
        if history and history[-1].timestamp == book.timestamp and history[-1] == book:
            return  # exact duplicate (e.g. a REST fallback re-fetching the same snapshot)
        idx = len(history)
        while idx > 0 and history[idx - 1].timestamp > book.timestamp:
            idx -= 1
        history.insert(idx, book)

    def _book_as_of(self, canonical_id: str, as_of: datetime) -> OrderBook | None:
        """The most recent book with ``timestamp <= as_of``. Never a later one.

        This is the single chokepoint that prevents look-ahead: even if
        ``book_provider`` hands back a book stamped *after* ``as_of`` (e.g. it always
        returns "the current" snapshot), that book is recorded for future lookups but is
        never itself returned here, and the last book that actually satisfies
        ``timestamp <= as_of`` is used instead. If no such book exists, ``None`` is
        returned and the caller rejects ``NO_LIQUIDITY``.
        """
        fallback = self.book_provider(canonical_id)
        if fallback is not None:
            self._record_book(fallback)
        candidate: OrderBook | None = None
        for b in self._book_history.get(canonical_id, ()):
            if b.timestamp <= as_of:
                candidate = b
            else:
                break
        return candidate

    # -- risk-gateway context (documented single-portfolio approximations) --------

    def _category_exposures(
        self, portfolio: Portfolio, market: NormalizedMarket
    ) -> dict[Category, Decimal]:
        # PaperBroker only ever holds one Portfolio at a time (via portfolio_provider);
        # true cross-strategy category aggregation belongs to a portfolio-level service
        # we don't own. This is a documented single-portfolio approximation.
        return {market.category: portfolio.exposure()}

    def _cluster_exposures(
        self, portfolio: Portfolio, market: NormalizedMarket
    ) -> dict[str, Decimal]:
        # No explicit "cluster id" exists on NormalizedMarket; event_id is the most
        # obviously-correlated grouping available without an external correlation graph.
        return {market.event_id: portfolio.exposure()}

    def _daily_pnl(self, portfolio: Portfolio) -> Decimal:
        # Portfolio carries no start-of-day snapshot, so "daily" is approximated here as
        # cumulative P&L to date for the sleeve. A real day boundary needs a scheduler in
        # the runner/daemon that snapshots equity at UTC midnight and hands the delta in.
        return portfolio.realized_pnl + portfolio.unrealized_pnl()

    def _track_experiment(self, canonical_id: str, experiment_id: str) -> None:
        self._experiments_by_market.setdefault(canonical_id, set()).add(experiment_id)

    # -- persistence ----------------------------------------------------------------

    def _persist_order(self, order: Order) -> None:
        if self.store is not None and hasattr(self.store, "save_order"):
            self.store.save_order(order)

    def _persist_fill(self, fill: Fill) -> None:
        if self.store is not None and hasattr(self.store, "save_fill"):
            self.store.save_fill(fill)

    def load_open_orders(self, store: StateStoreLike | None = None) -> None:
        """Recovery: rebuild resting-order state from a StateStore after a restart.

        Calls ``store.load_open_orders() -> list[Order]``. Exact queue position
        (``QueueFillModel``'s ``queue_ahead``) cannot be recovered - we can't know who was
        really resting ahead of us before the crash - so it is conservatively
        reinitialized as if the order were freshly placed the moment the next book update
        arrives for that market. The bankroll itself is never touched here: it lives in
        whatever ``portfolio_provider`` returns, which is this broker's caller's
        responsibility to have restored from durable storage before wiring it in.
        """
        store = store or self.store
        if store is None:
            return
        for order in store.load_open_orders():
            self._orders[order.order_id] = order
            self._track_experiment(order.canonical_id, order.experiment_id)
            if order.is_terminal:
                continue
            remaining = order.unfilled_quantity
            if remaining <= 0 or order.limit_price is None:
                continue
            state = OrderState(
                order_id=order.order_id,
                canonical_id=order.canonical_id,
                side=order.side,
                action=order.action,
                limit_price=order.limit_price,
                remaining_quantity=remaining,
                placed_at=order.simulated_exchange_arrival_timestamp or order.decision_timestamp,
            )
            self._resting[order.order_id] = state
            self._resting_by_market.setdefault(order.canonical_id, set()).add(order.order_id)

    # -- rejection helper -------------------------------------------------------------

    def _reject_new(
        self,
        intent: OrderIntent,
        reason: RejectReason,
        detail: str,
        decision_time: datetime,
        send_time: datetime | None = None,
        arrival_time: datetime | None = None,
        book_timestamp: datetime | None = None,
    ) -> Order:
        order = Order(
            intent_id=intent.intent_id,
            strategy_id=intent.strategy_id,
            experiment_id=intent.experiment_id,
            canonical_id=intent.canonical_id,
            venue=intent.venue,
            side=intent.side,
            action=intent.action,
            order_type=intent.order_type,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
            time_in_force=intent.time_in_force,
            status=OrderStatus.REJECTED,
            reject_reason=reason,
            reject_detail=detail,
            decision_timestamp=decision_time,
            simulated_network_send_timestamp=send_time,
            simulated_exchange_arrival_timestamp=arrival_time,
            book_timestamp_used=book_timestamp,
        )
        self._orders[order.order_id] = order
        self.rejected_orders += 1
        self._persist_order(order)
        return order

    # -- the main entry point ---------------------------------------------------------

    async def submit(self, intent: OrderIntent) -> Order:
        decision_time = intent.decision_time

        # 1. PaperBroker never touches a venue and never operates with no market data.
        if self.mode in (Mode.LIVE, Mode.DATA_ONLY):
            return self._reject_new(
                intent,
                RejectReason.MODE_FORBIDDEN,
                f"PaperBroker cannot submit orders in {self.mode} mode",
                decision_time,
            )

        if intent.order_type is OrderType.LIMIT and intent.limit_price is None:
            return self._reject_new(
                intent, RejectReason.INVALID_PRICE, "limit order with no limit_price", decision_time
            )

        # 2. Latency timestamps.
        send_time = self.latency_model.send_time(decision_time)
        arrival_time = self.latency_model.arrival_time(decision_time)

        market = self.market_provider(intent.canonical_id)
        if market is None:
            return self._reject_new(
                intent, RejectReason.NO_LIQUIDITY, "unknown market", decision_time, send_time, arrival_time
            )

        # 3. Book as of the simulated arrival time. Never a later one.
        book = self._book_as_of(intent.canonical_id, arrival_time)
        if book is None:
            return self._reject_new(
                intent,
                RejectReason.NO_LIQUIDITY,
                "no book available at or before the simulated exchange arrival time",
                decision_time,
                send_time,
                arrival_time,
            )

        # 4. Staleness, measured as of arrival - not as of "now".
        if book.is_stale(arrival_time, self.settings.execution.max_book_age_seconds):
            self.stale_data_skips += 1
            return self._reject_new(
                intent,
                RejectReason.STALE_DATA,
                f"book at {book.timestamp.isoformat()} exceeds max_book_age_seconds="
                f"{self.settings.execution.max_book_age_seconds}s as of {arrival_time.isoformat()}",
                decision_time,
                send_time,
                arrival_time,
                book.timestamp,
            )

        # 5. Market must be open.
        if market.status is not MarketStatus.OPEN:
            return self._reject_new(
                intent,
                RejectReason.MARKET_CLOSED,
                f"market status is {market.status}",
                decision_time,
                send_time,
                arrival_time,
                book.timestamp,
            )

        portfolio = self.portfolio_provider(intent.experiment_id)

        # 6. Risk gate.
        decision = self.risk_gateway.evaluate(
            intent,
            portfolio,
            market,
            book,
            self._category_exposures(portfolio, market),
            self._cluster_exposures(portfolio, market),
            self._daily_pnl(portfolio),
        )
        if not decision.approved:
            self.risk_gate_skips += 1
            return self._reject_new(
                intent,
                decision.reason or RejectReason.RISK_GATE,
                decision.detail,
                decision_time,
                send_time,
                arrival_time,
                book.timestamp,
            )

        # Reference price for slippage: mid *at decision time*, not at arrival - this is
        # deliberately a different (earlier) point-in-time lookup than the book used to
        # execute against, because slippage measures cost relative to the moment the
        # strategy decided to trade, not the moment the (delayed) order actually landed.
        reference_book = self._book_as_of(intent.canonical_id, decision_time) or book
        reference_price = reference_book.mid

        order = Order(
            intent_id=intent.intent_id,
            strategy_id=intent.strategy_id,
            experiment_id=intent.experiment_id,
            canonical_id=intent.canonical_id,
            venue=intent.venue,
            side=intent.side,
            action=intent.action,
            order_type=intent.order_type,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
            time_in_force=intent.time_in_force,
            status=OrderStatus.PENDING,
            decision_timestamp=decision_time,
            simulated_network_send_timestamp=send_time,
            simulated_exchange_arrival_timestamp=arrival_time,
            book_timestamp_used=book.timestamp,
            reference_price=reference_price,
        )

        # 7-8-9. Execute + charge fees + apply to portfolio.
        if intent.order_type is OrderType.MARKET:
            order = self._execute_market(order, book, market, portfolio)
        else:
            order = self._execute_limit(order, book, market, portfolio)

        self._orders[order.order_id] = order
        self._track_experiment(order.canonical_id, order.experiment_id)
        if order.status is OrderStatus.REJECTED:
            self.rejected_orders += 1
        self._persist_order(order)
        return order

    # -- execution ----------------------------------------------------------------------

    def _build_fills(
        self,
        order: Order,
        level_fills: list[tuple[Decimal, int]] | tuple[tuple[Decimal, int], ...],
        market: NormalizedMarket,
        is_maker: bool,
        timestamp: datetime,
        book_timestamp: datetime,
    ) -> list[Fill]:
        fills = []
        for price, qty in level_fills:
            fee = self.fee_calculator.calculate(market, price, qty, is_maker)
            fills.append(
                Fill(
                    order_id=order.order_id,
                    canonical_id=order.canonical_id,
                    venue=order.venue,
                    side=order.side,
                    action=order.action,
                    price=price,
                    quantity=qty,
                    fee=fee,
                    timestamp=timestamp,
                    is_maker=is_maker,
                    book_timestamp_used=book_timestamp,
                    level_breakdown=((price, qty),),
                )
            )
        return fills

    def _execute_market(
        self, order: Order, book: OrderBook, market: NormalizedMarket, portfolio: Portfolio
    ) -> Order:
        result = walk_book(book, order.action, order.side, order.quantity)
        if result.filled_qty == 0:
            return order.model_copy(
                update={
                    "status": OrderStatus.REJECTED,
                    "reject_reason": RejectReason.NO_LIQUIDITY,
                    "reject_detail": "no visible liquidity on the relevant side of the book",
                }
            )
        fills = self._build_fills(
            order,
            result.fills,
            market,
            is_maker=False,
            timestamp=order.simulated_exchange_arrival_timestamp or order.decision_timestamp,
            book_timestamp=book.timestamp,
        )
        for f in fills:
            portfolio.apply_fill(f)
            self._persist_fill(f)
        self.taker_fills += len(fills)
        status = OrderStatus.FILLED if result.unfilled_qty == 0 else OrderStatus.PARTIALLY_FILLED
        return order.model_copy(
            update={
                "status": status,
                "filled_quantity": result.filled_qty,
                "average_fill_price": result.avg_price,
                "worst_fill_price": result.worst_price,
                "fees_paid": sum((f.fee for f in fills), ZERO),
                "fills": tuple(fills),
            }
        )

    def _execute_limit(
        self, order: Order, book: OrderBook, market: NormalizedMarket, portfolio: Portfolio
    ) -> Order:
        assert order.limit_price is not None
        result = walk_book(book, order.action, order.side, order.quantity)

        # Truncate to the marketable-at-or-better-than-limit prefix. `result.fills` is
        # sorted best price first, so the first level that violates the limit ends the
        # crossable run - everything after it is strictly worse.
        crossable: list[tuple[Decimal, int]] = []
        crossed_qty = 0
        for price, qty in result.fills:
            if order.action is Action.BUY and price > order.limit_price:
                break
            if order.action is Action.SELL and price < order.limit_price:
                break
            crossable.append((price, qty))
            crossed_qty += qty

        if order.time_in_force is TimeInForce.FOK and crossed_qty < order.quantity:
            return order.model_copy(
                update={
                    "status": OrderStatus.REJECTED,
                    "reject_reason": RejectReason.NO_LIQUIDITY,
                    "reject_detail": "FOK could not be filled in full immediately",
                }
            )

        fills: list[Fill] = []
        if crossable:
            fills = self._build_fills(
                order,
                crossable,
                market,
                is_maker=False,
                timestamp=order.simulated_exchange_arrival_timestamp or order.decision_timestamp,
                book_timestamp=book.timestamp,
            )
            for f in fills:
                portfolio.apply_fill(f)
                self._persist_fill(f)
            self.taker_fills += len(fills)

        remaining = order.quantity - crossed_qty
        order = order.model_copy(
            update={
                "filled_quantity": crossed_qty,
                "average_fill_price": _avg_price(fills) if fills else None,
                "worst_fill_price": fills[-1].price if fills else None,
                "fees_paid": sum((f.fee for f in fills), ZERO),
                "fills": tuple(fills),
            }
        )

        if remaining <= 0:
            return order.model_copy(update={"status": OrderStatus.FILLED})

        if order.time_in_force is TimeInForce.IOC:
            status = OrderStatus.PARTIALLY_FILLED if crossed_qty > 0 else OrderStatus.CANCELED
            return order.model_copy(update={"status": status})

        # GTC / GTD: rest the remainder.
        status = OrderStatus.PARTIALLY_FILLED if crossed_qty > 0 else OrderStatus.OPEN
        order = order.model_copy(update={"status": status})
        self._rest_order(order, remaining)
        return order

    def _rest_order(self, order: Order, remaining_quantity: int) -> None:
        assert order.limit_price is not None
        state = OrderState(
            order_id=order.order_id,
            canonical_id=order.canonical_id,
            side=order.side,
            action=order.action,
            limit_price=order.limit_price,
            remaining_quantity=remaining_quantity,
            placed_at=order.simulated_exchange_arrival_timestamp or order.decision_timestamp,
        )
        self._resting[order.order_id] = state
        self._resting_by_market.setdefault(order.canonical_id, set()).add(order.order_id)
        history = self._book_history.get(order.canonical_id)
        if history:
            # Seed queue position immediately against the book we just executed against,
            # rather than waiting for the next externally-driven on_book_update.
            self.limit_fill_model.on_book_update(state, history[-1])

    # -- resting order lifecycle: driven by the runner's event feed --------------------

    def _settle_limit_fills(
        self,
        order_id: str,
        state: OrderState,
        new_fills: list[tuple[Decimal, int]],
        event_timestamp: datetime,
    ) -> list[Fill]:
        if not new_fills:
            return []
        order = self._orders.get(order_id)
        if order is None:
            return []
        market = self.market_provider(order.canonical_id)
        portfolio = self.portfolio_provider(order.experiment_id)

        # Never let a fill model over-fill beyond what's actually left on the order.
        requested_qty = sum(q for _, q in new_fills)
        total_qty = min(requested_qty, state.remaining_quantity)

        fills: list[Fill] = []
        remaining_to_build = total_qty
        for price, qty in new_fills:
            if remaining_to_build <= 0:
                break
            take = min(qty, remaining_to_build)
            fee = self.fee_calculator.calculate(market, price, take, True) if market else ZERO
            fills.append(
                Fill(
                    order_id=order_id,
                    canonical_id=order.canonical_id,
                    venue=order.venue,
                    side=order.side,
                    action=order.action,
                    price=price,
                    quantity=take,
                    fee=fee,
                    timestamp=event_timestamp,
                    is_maker=True,
                    book_timestamp_used=event_timestamp,
                    level_breakdown=((price, take),),
                )
            )
            remaining_to_build -= take

        for f in fills:
            portfolio.apply_fill(f)
            self._persist_fill(f)
        self.maker_fills += len(fills)
        state.remaining_quantity -= total_qty

        all_fills = order.fills + tuple(fills)
        filled_quantity = order.filled_quantity + total_qty
        status = OrderStatus.FILLED if state.remaining_quantity <= 0 else OrderStatus.PARTIALLY_FILLED
        order = order.model_copy(
            update={
                "fills": all_fills,
                "filled_quantity": filled_quantity,
                "average_fill_price": _avg_price(all_fills),
                "worst_fill_price": fills[-1].price if fills else order.worst_fill_price,
                "fees_paid": order.fees_paid + sum((f.fee for f in fills), ZERO),
                "status": status,
            }
        )
        self._orders[order_id] = order
        self._persist_order(order)

        if state.remaining_quantity <= 0:
            self._resting.pop(order_id, None)
            ids = self._resting_by_market.get(order.canonical_id)
            if ids:
                ids.discard(order_id)
        return fills

    async def on_book_update(self, book: OrderBook) -> list[Fill]:
        """Feed a new book snapshot in. Drives resting limit orders and grows the
        point-in-time book history used by `_book_as_of`."""
        self._record_book(book)
        order_ids = list(self._resting_by_market.get(book.canonical_id, ()))
        produced: list[Fill] = []
        for oid in order_ids:
            state = self._resting.get(oid)
            if state is None or state.remaining_quantity <= 0:
                continue
            new_fills = self.limit_fill_model.on_book_update(state, book)
            produced.extend(self._settle_limit_fills(oid, state, new_fills, book.timestamp))
        return produced

    async def on_trade(self, trade: Trade) -> list[Fill]:
        """Feed a public print in. Drives resting limit orders."""
        order_ids = list(self._resting_by_market.get(trade.canonical_id, ()))
        produced: list[Fill] = []
        for oid in order_ids:
            state = self._resting.get(oid)
            if state is None or state.remaining_quantity <= 0:
                continue
            new_fills = self.limit_fill_model.on_trade(state, trade)
            produced.extend(self._settle_limit_fills(oid, state, new_fills, trade.timestamp))
        return produced

    # -- Broker ABC -----------------------------------------------------------------

    async def cancel(self, order_id: str) -> Order | None:
        order = self._orders.get(order_id)
        if order is None:
            return None
        if order.is_terminal:
            return order
        order = order.model_copy(update={"status": OrderStatus.CANCELED})
        self._orders[order_id] = order
        self._resting.pop(order_id, None)
        ids = self._resting_by_market.get(order.canonical_id)
        if ids:
            ids.discard(order_id)
        self.cancel_count += 1
        self._persist_order(order)
        return order

    async def open_orders(self, strategy_id: str | None = None) -> list[Order]:
        return [
            o
            for o in self._orders.values()
            if not o.is_terminal and (strategy_id is None or o.strategy_id == strategy_id)
        ]

    async def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    # -- settlement / expiry ----------------------------------------------------------

    async def settle(self, canonical_id: str, winning_side: Side | None) -> None:
        """Settle open positions from the venue's authoritative resolution only.

        ``winning_side`` MUST come from a verified venue settlement event (Kalshi's
        official resolution feed) - never from a model's probability estimate, a news
        headline, a copy-trader's guess, or a tweet. This method trusts its caller
        completely and does not attempt to verify the outcome itself; that verification
        is the caller's contract to uphold. Any resting orders on the now-settled market
        are canceled, since the venue would never fill against a resolved market.
        """
        for exp_id in list(self._experiments_by_market.get(canonical_id, ())):
            portfolio = self.portfolio_provider(exp_id)
            portfolio.settle(canonical_id, winning_side)

        for oid in list(self._resting_by_market.get(canonical_id, ())):
            order = self._orders.get(oid)
            if order is None or order.is_terminal:
                continue
            order = order.model_copy(update={"status": OrderStatus.CANCELED})
            self._orders[oid] = order
            self.cancel_count += 1
            self._persist_order(order)
            self._resting.pop(oid, None)
        self._resting_by_market.pop(canonical_id, None)

    async def expire_at_close(self, canonical_id: str) -> list[Order]:
        """Cancel (expire) any GTD resting orders on a market that just closed."""
        expired: list[Order] = []
        for oid in list(self._resting_by_market.get(canonical_id, ())):
            order = self._orders.get(oid)
            if order is None or order.time_in_force is not TimeInForce.GTD:
                continue
            order = order.model_copy(update={"status": OrderStatus.EXPIRED})
            self._orders[oid] = order
            self._resting.pop(oid, None)
            self._resting_by_market[canonical_id].discard(oid)
            self._persist_order(order)
            expired.append(order)
        return expired
