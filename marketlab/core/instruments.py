"""Instrument and market normalization.

Every venue's event contract is projected onto :class:`NormalizedMarket` so that
strategies never see venue-specific price units.  Kalshi quotes integer cents, Polymarket
quotes decimal dollars; both become a ``Probability`` in ``[0, 1]`` backed by ``Decimal``.

Rule: never mix cents, dollars and probabilities in strategy code.
"""

from __future__ import annotations

from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Probabilities are carried at 1/10000 resolution internally so that a 1-cent Kalshi tick
# and a 0.001 Polymarket tick both round-trip exactly.
PROB_QUANTUM = Decimal("0.0001")
ZERO = Decimal(0)
ONE = Decimal(1)


class Venue(StrEnum):
    """Execution / data venues.

    Only ``KALSHI`` is ever an execution venue for this deployment.  Polymarket global is
    a read-only intelligence source (markets, books, public trader activity) because new
    positions are geoblocked from the United States.
    """

    KALSHI = "kalshi"
    POLY_GLOBAL = "poly-global"
    POLY_US = "poly-us"
    ALPACA = "alpaca"
    SYNTHETIC = "synthetic"


#: Venues that may ever be handed a live create-order call.
EXECUTION_VENUES: frozenset[Venue] = frozenset({Venue.KALSHI, Venue.POLY_US})

#: Venues that are hard-wired read-only regardless of configuration.
READ_ONLY_VENUES: frozenset[Venue] = frozenset({Venue.POLY_GLOBAL, Venue.ALPACA})


class MarketStatus(StrEnum):
    UNOPENED = "unopened"
    OPEN = "open"
    PAUSED = "paused"
    CLOSED = "closed"
    SETTLED = "settled"
    CANCELED = "canceled"
    UNKNOWN = "unknown"


class OutcomeType(StrEnum):
    BINARY = "binary"
    SCALAR = "scalar"
    CATEGORICAL = "categorical"


class Category(StrEnum):
    CRYPTO = "crypto"
    SPORTS = "sports"
    POLITICS = "politics"
    ECONOMICS = "economics"
    FINANCE = "finance"
    WEATHER = "weather"
    TECH = "tech"
    ENTERTAINMENT = "entertainment"
    SCIENCE = "science"
    OTHER = "other"


class Side(StrEnum):
    """Which leg of a binary contract."""

    YES = "yes"
    NO = "no"

    @property
    def opposite(self) -> Side:
        return Side.NO if self is Side.YES else Side.YES


def to_probability(value: Decimal | float | int | str) -> Decimal:
    """Coerce a value into a quantized probability in ``[0, 1]``.

    Raises ``ValueError`` outside the bounds rather than silently clamping, because a
    price outside ``[0, 1]`` means an adapter mixed up its units.
    """
    d = Decimal(str(value)) if not isinstance(value, Decimal) else value
    if d < ZERO or d > ONE:
        raise ValueError(f"probability out of range: {d!r}")
    return d.quantize(PROB_QUANTUM, rounding=ROUND_HALF_EVEN)


def cents_to_probability(cents: int | Decimal) -> Decimal:
    """Kalshi integer cents (0-100) -> probability."""
    return to_probability(Decimal(str(cents)) / Decimal(100))


def probability_to_cents(p: Decimal) -> int:
    """Probability -> Kalshi integer cents, rounding to nearest cent."""
    return int((p * Decimal(100)).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))


class Fees(BaseModel):
    """Venue fee model, fetched from the venue where the API exposes it."""

    model_config = ConfigDict(frozen=True)

    #: Kalshi charges ``ceil(fee_rate * C * P * (1-P))`` per contract; Polymarket charges
    #: a proportional taker fee.  ``formula`` says which is in force.
    formula: str = "kalshi_quadratic"
    taker_rate: Decimal = Decimal("0.07")
    maker_rate: Decimal = Decimal("0.00")
    settlement_rate: Decimal = Decimal("0.00")
    min_fee_cents: int = 1


class NormalizedMarket(BaseModel):
    """Venue-agnostic view of one tradeable event contract."""

    model_config = ConfigDict(frozen=True)

    canonical_id: str
    venue: Venue
    venue_market_id: str
    event_id: str
    title: str
    description: str = ""
    resolution_rules: str = ""
    resolution_source: str = ""
    category: Category = Category.OTHER
    subcategory: str = ""
    outcome_type: OutcomeType = OutcomeType.BINARY
    yes_symbol: str = ""
    no_symbol: str = ""
    open_time: datetime | None = None
    close_time: datetime | None = None
    expected_resolution_time: datetime | None = None
    timezone: str = "UTC"
    tick_size: Decimal = Decimal("0.01")
    min_order: int = 1
    max_payout_per_contract: Decimal = ONE
    status: MarketStatus = MarketStatus.UNKNOWN
    fees: Fees = Field(default_factory=Fees)
    liquidity: Decimal = ZERO
    volume: Decimal = ZERO
    #: Contracts currently outstanding. Kalshi reports this fractionally (``open_interest_fp``);
    #: it is a better liquidity proxy than ``volume`` for a market that has been open a while.
    open_interest: Decimal = ZERO
    #: Free-form venue payload retained for auditability; never read by strategies.
    raw: dict = Field(default_factory=dict, repr=False)

    @field_validator("tick_size")
    @classmethod
    def _tick_positive(cls, v: Decimal) -> Decimal:
        if v <= ZERO:
            raise ValueError("tick_size must be positive")
        return v

    @property
    def is_tradeable(self) -> bool:
        return self.status is MarketStatus.OPEN

    @property
    def is_execution_venue(self) -> bool:
        return self.venue in EXECUTION_VENUES


class BookLevel(BaseModel):
    """One visible price level.  ``size`` is in contracts."""

    model_config = ConfigDict(frozen=True)

    price: Decimal
    size: int


class OrderBook(BaseModel):
    """Top-of-book through depth for one side of one market.

    ``bids`` are sorted descending by price, ``asks`` ascending.  Both are expressed in
    YES-probability terms: a NO bid at 0.30 appears as a YES ask at 0.70 only after the
    adapter folds it, which each adapter documents explicitly.
    """

    model_config = ConfigDict(frozen=True)

    canonical_id: str
    venue: Venue
    timestamp: datetime
    bids: tuple[BookLevel, ...] = ()
    asks: tuple[BookLevel, ...] = ()
    #: When the venue stamped the book, if it differs from ingest time.
    venue_timestamp: datetime | None = None
    sequence: int | None = None

    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return ((b + a) / Decimal(2)).quantize(PROB_QUANTUM, rounding=ROUND_HALF_EVEN)

    @property
    def spread(self) -> Decimal | None:
        b, a = self.best_bid, self.best_ask
        return None if b is None or a is None else a - b

    @property
    def microprice(self) -> Decimal | None:
        """Size-weighted mid: leans toward the side with less resting size."""
        if not self.bids or not self.asks:
            return None
        bs, as_ = Decimal(self.bids[0].size), Decimal(self.asks[0].size)
        total = bs + as_
        if total == ZERO:
            return self.mid
        mp = (self.bids[0].price * as_ + self.asks[0].price * bs) / total
        return mp.quantize(PROB_QUANTUM, rounding=ROUND_HALF_EVEN)

    def depth(self, levels: int, side: str) -> int:
        rows = self.bids if side == "bid" else self.asks
        return sum(lvl.size for lvl in rows[:levels])

    def is_stale(self, now: datetime, max_age_seconds: float) -> bool:
        return (now - self.timestamp).total_seconds() > max_age_seconds


class Trade(BaseModel):
    """A public print observed on a venue."""

    model_config = ConfigDict(frozen=True)

    canonical_id: str
    venue: Venue
    timestamp: datetime
    price: Decimal
    size: int
    aggressor: Side | None = None
    trade_id: str = ""
