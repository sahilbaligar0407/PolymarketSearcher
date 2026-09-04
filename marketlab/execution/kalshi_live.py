"""KalshiLiveBroker: the only code path in MarketLab that can place a real order.

Disabled by default. Every one of the following must hold or the broker refuses before
any network call ever happens:

* ``settings.mode is Mode.LIVE``
* ``settings.secrets.live_armed`` (the env var must be exactly ``YES_I_ACCEPT_REAL_LOSS``,
  see ``settings.LIVE_ARM_TOKEN``)
* the caller passed ``acknowledge_real_money_risk=True`` explicitly to the constructor
* valid Kalshi credentials are configured
* the risk gateway approves the specific intent (necessarily per-order, checked in `submit`)
* the data feed is not stale (necessarily per-order, checked in `submit`)

The first four are static/config-shaped and are re-checked at construction time AND at
the top of every ``submit``/``cancel``/``open_orders`` call, so merely *instantiating*
this class with any of them unmet is impossible, and a long-lived process can never drift
into a live-armed state without deliberately re-arming it - "structurally impossible to
reach by accident" per the PRD. The last two are inherently per-order and can only be
evaluated against a live intent and a live book, so they reject gracefully (return a
REJECTED ``Order``) in ``submit`` rather than raising - an ordinary risk-gate or
stale-data rejection is a normal operating event, not a structural violation of live-
trading safety, and callers need to be able to handle it the same way they handle any
other broker rejection.

``KalshiRestAdapter`` (authenticated HTTP session/signing) is owned by another team and
does not exist as an importable module yet, and deliberately does not implement order
creation itself. We depend on it only through the ``KalshiRestAdapterLike`` Protocol
below so we are never blocked on it - and so the order-specific HTTP calls
(``POST /portfolio/orders``, ``DELETE /portfolio/orders/{id}``, ``GET /portfolio/orders``)
live here, where this team owns them.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Protocol, runtime_checkable

from marketlab.clock import Clock
from marketlab.core.broker import Broker, Mode
from marketlab.core.instruments import NormalizedMarket, OrderBook, Side, probability_to_cents
from marketlab.core.orders import (
    Order,
    OrderIntent,
    OrderStatus,
    OrderType,
    RejectReason,
    TimeInForce,
)
from marketlab.core.portfolio import Portfolio
from marketlab.execution.risk_gateway import RiskGateway
from marketlab.settings import LIVE_ARM_TOKEN, Settings


class LiveTradingBlocked(RuntimeError):
    """Raised when a structural (config/identity) live-trading precondition is unmet.

    Distinct from an ordinary ``RejectReason`` rejection: this means the code path should
    never have been reachable at all, not that this particular order was bad.
    """


@runtime_checkable
class KalshiRestAdapterLike(Protocol):
    """Minimal shape KalshiLiveBroker needs from the (not-yet-written) KalshiRestAdapter.

    That adapter owns authentication/signing/retries/backoff; we only need one
    authenticated request primitive from it. Whichever team builds the real adapter
    should expose at least this coroutine.
    """

    async def request(
        self, method: str, path: str, *, json: dict | None = None, params: dict | None = None
    ) -> dict: ...


def _unmet_static_preconditions(
    settings: Settings,
    acknowledge_real_money_risk: bool,
    rest_adapter: object | None,
) -> list[str]:
    """Every precondition evaluable without a live intent or a live book."""
    unmet: list[str] = []
    if settings.mode is not Mode.LIVE:
        unmet.append(f"settings.mode is {settings.mode}, not LIVE")
    if not settings.secrets.live_armed:
        unmet.append(
            "settings.secrets.live_armed is False "
            f"(LIVE_TRADING_ENABLED must equal exactly {LIVE_ARM_TOKEN!r})"
        )
    if not acknowledge_real_money_risk:
        unmet.append("acknowledge_real_money_risk was not passed as True to the constructor")
    if not settings.secrets.kalshi_api_key_id or not settings.secrets.kalshi_private_key_path:
        unmet.append("Kalshi credentials are not configured (api key id / private key path)")
    if rest_adapter is None:
        unmet.append("no rest_adapter was provided")
    return unmet


class KalshiLiveBroker(Broker):
    """Places real orders on Kalshi. Disabled unless every precondition holds.

    Constructing this object with any static precondition unmet raises
    ``LiveTradingBlocked`` immediately - it never silently becomes a "mostly disabled"
    object that would only fail later, at the worst possible moment, on first submit.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        clock: Clock,
        rest_adapter: KalshiRestAdapterLike,
        risk_gateway: RiskGateway,
        market_provider: Callable[[str], NormalizedMarket | None],
        book_provider: Callable[[str], OrderBook | None],
        portfolio_provider: Callable[[str], Portfolio],
        acknowledge_real_money_risk: bool = False,
        max_book_age_seconds: float | None = None,
        store: object | None = None,
    ) -> None:
        unmet = _unmet_static_preconditions(settings, acknowledge_real_money_risk, rest_adapter)
        if unmet:
            raise LiveTradingBlocked("KalshiLiveBroker refuses to construct: " + "; ".join(unmet))

        self.mode = settings.mode
        self.settings = settings
        self.clock = clock
        self.rest_adapter = rest_adapter
        self.risk_gateway = risk_gateway
        self.market_provider = market_provider
        self.book_provider = book_provider
        self.portfolio_provider = portfolio_provider
        self.acknowledge_real_money_risk = acknowledge_real_money_risk
        self.max_book_age_seconds = (
            max_book_age_seconds
            if max_book_age_seconds is not None
            else settings.execution.max_book_age_seconds
        )
        self.store = store

        self._orders: dict[str, Order] = {}
        self.rejected_orders = 0
        #: Test hook: proves a blocked submit/cancel/query never dials out.
        self.network_calls_made = 0

    @classmethod
    def preflight(
        cls,
        settings: Settings,
        acknowledge_real_money_risk: bool = False,
        rest_adapter: object | None = None,
    ) -> list[str]:
        """Every unmet *static* precondition, for ``marketlab live doctor``.

        Deliberately usable without a fully-constructed (or even constructible) broker -
        ``doctor`` needs to report exactly what's missing when construction would fail.
        Risk-gateway approval and data-feed staleness are per-order, not static, so they
        are not evaluated here; they are only ever checked inside ``submit``.
        """
        return _unmet_static_preconditions(settings, acknowledge_real_money_risk, rest_adapter)

    def _check_static_or_raise(self, action: str) -> None:
        unmet = _unmet_static_preconditions(
            self.settings, self.acknowledge_real_money_risk, self.rest_adapter
        )
        if unmet:
            # Settings is mutable (ConfigDict(frozen=False)); a long-lived process could
            # theoretically have these preconditions change after construction. Never
            # trust that they held at construction time to mean they still hold now.
            raise LiveTradingBlocked(f"KalshiLiveBroker refuses to {action}: " + "; ".join(unmet))

    def _reject(self, intent: OrderIntent, reason: RejectReason, detail: str) -> Order:
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
            decision_timestamp=intent.decision_time,
        )
        self._orders[order.order_id] = order
        self.rejected_orders += 1
        return order

    async def submit(self, intent: OrderIntent) -> Order:
        self._check_static_or_raise("submit")

        # Per-order dynamic preconditions reject gracefully rather than raise.
        book = self.book_provider(intent.canonical_id)
        now = self.clock.now()
        if book is None or book.is_stale(now, self.max_book_age_seconds):
            return self._reject(intent, RejectReason.STALE_DATA, "data feed is stale or missing")

        market = self.market_provider(intent.canonical_id)
        if market is None:
            return self._reject(intent, RejectReason.NO_LIQUIDITY, "unknown market")

        portfolio = self.portfolio_provider(intent.experiment_id)
        decision = self.risk_gateway.evaluate(intent, portfolio, market, book, {}, {}, Decimal(0))
        if not decision.approved:
            return self._reject(intent, decision.reason or RejectReason.RISK_GATE, decision.detail)

        # Every precondition holds. This is the only place in the entire codebase allowed
        # to place a real order.
        return await self._place_order(intent, market)

    async def _place_order(self, intent: OrderIntent, market: NormalizedMarket) -> Order:
        payload: dict = {
            "ticker": market.venue_market_id,
            "action": intent.action.value,
            "side": intent.side.value,
            "count": intent.quantity,
            "type": intent.order_type.value,
        }
        if intent.order_type is OrderType.LIMIT and intent.limit_price is not None:
            cents = probability_to_cents(intent.limit_price)
            payload["yes_price" if intent.side is Side.YES else "no_price"] = cents
        if intent.time_in_force is TimeInForce.IOC:
            payload["time_in_force"] = "immediate_or_cancel"

        self.network_calls_made += 1
        response = await self.rest_adapter.request("POST", "/portfolio/orders", json=payload)
        venue_order_id = str(response.get("order", {}).get("order_id", "")) or None

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
            status=OrderStatus.OPEN,
            decision_timestamp=intent.decision_time,
            venue_order_id=venue_order_id,
        )
        self._orders[order.order_id] = order
        return order

    async def cancel(self, order_id: str) -> Order | None:
        order = self._orders.get(order_id)
        if order is None or order.venue_order_id is None:
            return order
        self._check_static_or_raise("cancel")
        self.network_calls_made += 1
        await self.rest_adapter.request("DELETE", f"/portfolio/orders/{order.venue_order_id}")
        order = order.model_copy(update={"status": OrderStatus.CANCELED})
        self._orders[order_id] = order
        return order

    async def open_orders(self, strategy_id: str | None = None) -> list[Order]:
        self._check_static_or_raise("query open orders")
        self.network_calls_made += 1
        await self.rest_adapter.request("GET", "/portfolio/orders", params={"status": "open"})
        return [
            o
            for o in self._orders.values()
            if not o.is_terminal and (strategy_id is None or o.strategy_id == strategy_id)
        ]

    async def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)
