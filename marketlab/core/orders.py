"""Order lifecycle contracts.

Strategies emit :class:`OrderIntent`.  They never construct an :class:`Order`, never see a
venue order id, and never learn whether the intent became a simulated fill or a real one.
That asymmetry is what keeps backtest, paper and live behaviour identical.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from marketlab.core.instruments import ZERO, Side, Venue


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(StrEnum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    #: Cancel automatically at market close.
    GTD = "gtd"


class Action(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(StrEnum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class RejectReason(StrEnum):
    RISK_GATE = "risk_gate"
    STALE_DATA = "stale_data"
    NO_LIQUIDITY = "no_liquidity"
    MARKET_CLOSED = "market_closed"
    INSUFFICIENT_CASH = "insufficient_cash"
    BELOW_MIN_ORDER = "below_min_order"
    INVALID_PRICE = "invalid_price"
    MODE_FORBIDDEN = "mode_forbidden"
    VENUE_ERROR = "venue_error"
    DUPLICATE = "duplicate"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class OrderIntent(BaseModel):
    """What a strategy wants to happen. The broker decides what actually happens.

    ``rationale`` and ``features`` are mandatory in spirit: every trade decision must be
    reconstructable afterwards, so the risk gateway rejects intents with an empty
    rationale when ``strict_audit`` is on.
    """

    model_config = ConfigDict(frozen=True)

    intent_id: str = Field(default_factory=lambda: _new_id("int"))
    strategy_id: str
    experiment_id: str
    canonical_id: str
    venue: Venue
    side: Side
    action: Action
    quantity: int
    order_type: OrderType = OrderType.LIMIT
    limit_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    #: Strategy's own timestamp for the decision, taken from the injected clock.
    decision_time: datetime
    #: Human-readable explanation. Never "AI said so".
    rationale: str = ""
    #: Feature snapshot that produced the decision, for attribution and replay audit.
    features: dict = Field(default_factory=dict)
    #: Evidence ids (news:, sec:, trader:, market:) backing the decision.
    evidence_ids: tuple[str, ...] = ()
    #: Model probability that motivated the trade, if the strategy emits one.
    model_probability: Decimal | None = None
    #: Expected edge net of fees the strategy believed it had at decision time.
    expected_edge: Decimal | None = None
    #: Set by the strategy when it wants an existing resting order replaced.
    replaces_order_id: str | None = None

    @field_validator("quantity")
    @classmethod
    def _positive_qty(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("quantity must be positive")
        return v

    @field_validator("limit_price")
    @classmethod
    def _price_in_range(cls, v: Decimal | None) -> Decimal | None:
        if v is not None and not (ZERO <= v <= Decimal(1)):
            raise ValueError(f"limit_price must be a probability in [0,1], got {v}")
        return v


class Fill(BaseModel):
    """One execution against one price level."""

    model_config = ConfigDict(frozen=True)

    fill_id: str = Field(default_factory=lambda: _new_id("fil"))
    order_id: str
    canonical_id: str
    venue: Venue
    side: Side
    action: Action
    price: Decimal
    quantity: int
    fee: Decimal = ZERO
    timestamp: datetime
    #: True when the fill was passive (rested and got hit).
    is_maker: bool = False
    #: Book timestamp the simulator consumed, so replay can be audited.
    book_timestamp_used: datetime | None = None
    #: Levels consumed, for slippage attribution: [(price, qty), ...]
    level_breakdown: tuple[tuple[Decimal, int], ...] = ()

    @property
    def notional(self) -> Decimal:
        return self.price * Decimal(self.quantity)


class Order(BaseModel):
    """Broker-side order state.  Constructed only by a broker, never by a strategy."""

    model_config = ConfigDict(frozen=True)

    order_id: str = Field(default_factory=lambda: _new_id("ord"))
    intent_id: str
    strategy_id: str
    experiment_id: str
    canonical_id: str
    venue: Venue
    side: Side
    action: Action
    order_type: OrderType
    quantity: int
    limit_price: Decimal | None
    time_in_force: TimeInForce
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_fill_price: Decimal | None = None
    worst_fill_price: Decimal | None = None
    fees_paid: Decimal = ZERO
    reject_reason: RejectReason | None = None
    reject_detail: str = ""

    # --- latency audit trail (PRD requires all four) ---
    decision_timestamp: datetime
    simulated_network_send_timestamp: datetime | None = None
    simulated_exchange_arrival_timestamp: datetime | None = None
    book_timestamp_used: datetime | None = None

    #: Reference price at decision time, used to compute realized slippage.
    reference_price: Decimal | None = None
    #: Venue order id once a live venue accepts it; always None in PAPER.
    venue_order_id: str | None = None
    fills: tuple[Fill, ...] = ()

    @property
    def unfilled_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }

    @property
    def slippage(self) -> Decimal | None:
        """Signed adverse price difference vs the reference price at decision time."""
        if self.average_fill_price is None or self.reference_price is None:
            return None
        diff = self.average_fill_price - self.reference_price
        return diff if self.action is Action.BUY else -diff
