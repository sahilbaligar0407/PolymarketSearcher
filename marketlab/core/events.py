"""The event bus vocabulary.

Every input the engine consumes becomes one of these events.  Each carries the four
timestamps point-in-time discipline requires:

``event_time``     when the thing happened in the world
``published_time`` when the source made it available
``first_seen_time`` when MarketLab first observed it   <- the one strategies may use
``ingested_time``  when it was written to storage

A strategy at decision time ``T`` may only see records with ``first_seen_time <= T``.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from marketlab.core.instruments import (
    Category,
    MarketStatus,
    NormalizedMarket,
    OrderBook,
    Side,
    Trade,
    Venue,
)


class EventType(StrEnum):
    MARKET_UPDATE = "market_update"
    BOOK_UPDATE = "book_update"
    TRADE = "trade"
    MARKET_STATUS = "market_status"
    SETTLEMENT = "settlement"
    NEWS = "news"
    SOCIAL = "social"
    FILING = "filing"
    EXTERNAL_PRICE = "external_price"
    TRADER_ACTION = "trader_action"
    WEATHER = "weather"
    ECONOMIC = "economic"
    SPORTS_STATE = "sports_state"
    TIMER = "timer"


class SourceClass(StrEnum):
    """Evidentiary weight of an information source. An SEC filing is not a tweet."""

    OFFICIAL_PRIMARY = "official_primary"
    MAJOR_WIRE = "major_wire"
    MAJOR_NEWS = "major_news"
    SPECIALIST = "specialist"
    SOCIAL_VERIFIED = "social_verified"
    SOCIAL_UNVERIFIED = "social_unverified"
    UNKNOWN = "unknown"


class BaseEvent(BaseModel):
    """Common timestamp discipline for everything on the bus."""

    model_config = ConfigDict(frozen=True)

    event_type: EventType
    #: When the underlying thing happened.
    event_time: datetime
    #: When the source published it (may equal event_time).
    published_time: datetime | None = None
    #: When MarketLab observed it. THIS is the field strategies gate on.
    first_seen_time: datetime
    #: When it hit storage.
    ingested_time: datetime | None = None
    source: str = ""
    source_class: SourceClass = SourceClass.UNKNOWN

    def visible_at(self, decision_time: datetime) -> bool:
        """Point-in-time gate. False means a strategy must not look at this record."""
        return self.first_seen_time <= decision_time


class MarketUpdateEvent(BaseEvent):
    event_type: EventType = EventType.MARKET_UPDATE
    market: NormalizedMarket


class BookUpdateEvent(BaseEvent):
    event_type: EventType = EventType.BOOK_UPDATE
    book: OrderBook

    @property
    def canonical_id(self) -> str:
        return self.book.canonical_id


class TradeEvent(BaseEvent):
    event_type: EventType = EventType.TRADE
    trade: Trade

    @property
    def canonical_id(self) -> str:
        return self.trade.canonical_id


class MarketStatusEvent(BaseEvent):
    event_type: EventType = EventType.MARKET_STATUS
    canonical_id: str
    venue: Venue
    status: MarketStatus


class SettlementEvent(BaseEvent):
    """Authoritative resolution from the venue. Never inferred from a model or a tweet."""

    event_type: EventType = EventType.SETTLEMENT
    canonical_id: str
    venue: Venue
    #: 1 = YES resolved true, 0 = NO, None = canceled/void.
    winning_side: Side | None
    settlement_value: Decimal | None = None
    voided: bool = False


class NewsEvent(BaseEvent):
    event_type: EventType = EventType.NEWS
    news_id: str
    title: str
    url: str = ""
    body: str = ""
    #: Hash of the normalized body, used for cross-source dedup.
    body_hash: str = ""
    canonical_url: str = ""
    language: str = "en"
    entities: tuple[str, ...] = ()
    tickers: tuple[str, ...] = ()
    tone: float | None = None
    #: Set when this story was recognised as a republication of an earlier one.
    duplicate_of: str | None = None


class SocialEvent(BaseEvent):
    event_type: EventType = EventType.SOCIAL
    post_id: str
    platform: str
    author: str
    text: str
    text_hash: str = ""
    url: str = ""
    mentioned_tickers: tuple[str, ...] = ()
    mentioned_entities: tuple[str, ...] = ()
    #: Populated by the public-statement classifier: tariff, sanctions, praise, ...
    action_type: str = ""
    policy_topic: str = ""
    market_hours_state: str = ""
    novelty: float | None = None


class FilingEvent(BaseEvent):
    event_type: EventType = EventType.FILING
    accession: str
    cik: str
    ticker: str = ""
    form_type: str = ""
    company: str = ""
    url: str = ""
    #: Form 4 specifics when applicable.
    insider_name: str = ""
    insider_role: str = ""
    transaction_code: str = ""
    transaction_value: Decimal | None = None
    #: Days between the transaction and public disclosure. Part of the simulation.
    disclosure_lag_days: float | None = None


class ExternalPriceEvent(BaseEvent):
    """A reference price from outside the prediction venues: BTC spot, SPY, a bookmaker."""

    event_type: EventType = EventType.EXTERNAL_PRICE
    symbol: str
    price: Decimal
    #: For sportsbook lines converted to vig-free probabilities.
    implied_probability: Decimal | None = None
    venue: str = ""


class TraderActionEvent(BaseEvent):
    """A public trade by a tracked Polymarket wallet.

    Read-only intelligence. The follower simulation always executes on the Kalshi book
    available *after* ``first_seen_time`` plus the configured follower delay - never at
    the source trader's own price.
    """

    event_type: EventType = EventType.TRADER_ACTION
    wallet: str
    username: str = ""
    canonical_id: str = ""
    poly_market_id: str = ""
    poly_condition_id: str = ""
    title: str = ""
    outcome: str = ""
    side: Side | None = None
    action: str = ""
    price: Decimal | None = None
    size: Decimal | None = None
    usd_size: Decimal | None = None
    category: Category = Category.OTHER
    transaction_hash: str = ""


class WeatherEvent(BaseEvent):
    event_type: EventType = EventType.WEATHER
    station: str
    office: str = ""
    variable: str = ""
    value: Decimal | None = None
    forecast_for: datetime | None = None
    issued_at: datetime | None = None


class EconomicEvent(BaseEvent):
    event_type: EventType = EventType.ECONOMIC
    series_id: str
    value: Decimal | None = None
    observation_date: datetime | None = None
    #: ALFRED vintage date. Using a later vintage in a backtest is look-ahead.
    vintage_date: datetime | None = None
    release_name: str = ""


class SportsStateEvent(BaseEvent):
    event_type: EventType = EventType.SPORTS_STATE
    game_id: str
    league: str = ""
    state: str = ""
    home: str = ""
    away: str = ""
    home_score: int | None = None
    away_score: int | None = None
    period: str = ""
    started: bool = False
    final: bool = False


class TimerEvent(BaseEvent):
    event_type: EventType = EventType.TIMER
    interval_seconds: float = 1.0
    tag: str = ""


Event = (
    MarketUpdateEvent
    | BookUpdateEvent
    | TradeEvent
    | MarketStatusEvent
    | SettlementEvent
    | NewsEvent
    | SocialEvent
    | FilingEvent
    | ExternalPriceEvent
    | TraderActionEvent
    | WeatherEvent
    | EconomicEvent
    | SportsStateEvent
    | TimerEvent
)


class Alert(BaseModel):
    """Operational alert written to storage and the console."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    severity: str
    component: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)
