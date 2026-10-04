"""The strategy plugin contract.

Hard rules, enforced by construction:

* a strategy never receives API credentials
* a strategy never calls a venue SDK
* a strategy never calls ``datetime.now()`` - it uses ``self.clock``
* a strategy cannot tell whether it is in BACKTEST, PAPER or LIVE
* a strategy cannot bypass the risk gateway

A strategy reacts to events, mutates its own private state, and returns intents from
:meth:`generate_intents`.  The runner calls the handlers then drains the intents.
"""

from __future__ import annotations

import abc
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from marketlab.clock import Clock
from marketlab.core.events import (
    BookUpdateEvent,
    EconomicEvent,
    ExternalPriceEvent,
    FilingEvent,
    MarketStatusEvent,
    MarketUpdateEvent,
    NewsEvent,
    SettlementEvent,
    SocialEvent,
    SportsStateEvent,
    TimerEvent,
    TradeEvent,
    TraderActionEvent,
    WeatherEvent,
)
from marketlab.core.instruments import NormalizedMarket, OrderBook
from marketlab.core.orders import Fill, Order, OrderIntent


class ProbabilityForecast(BaseModel):
    """A probability estimate, recorded whether or not it produces a trade.

    Forecasts are scored on Brier score and calibration independently of P&L, because a
    model can be well calibrated and traded badly, or vice versa.
    """

    model_config = ConfigDict(frozen=True)

    strategy_id: str
    experiment_id: str
    canonical_id: str
    as_of: datetime
    p_yes: Decimal
    confidence: Decimal = Decimal("0.5")
    #: Market midpoint at forecast time, so improvement-vs-market is computable later.
    market_probability: Decimal | None = None
    abstain: bool = False
    evidence_ids: tuple[str, ...] = ()
    features: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class StrategyContext:
    """Read-only view of the world handed to a strategy.

    Deliberately narrow: books, markets, the clock, and point-in-time-filtered history.
    No credentials, no broker, no network.
    """

    __slots__ = ("clock", "_books", "_markets", "_marks", "params")

    def __init__(
        self,
        clock: Clock,
        books: dict[str, OrderBook],
        markets: dict[str, NormalizedMarket],
        marks: dict[str, Decimal],
        params: dict[str, Any] | None = None,
    ) -> None:
        self.clock = clock
        self._books = books
        self._markets = markets
        self._marks = marks
        self.params = params or {}

    def now(self) -> datetime:
        return self.clock.now()

    def book(self, canonical_id: str) -> OrderBook | None:
        return self._books.get(canonical_id)

    def market(self, canonical_id: str) -> NormalizedMarket | None:
        return self._markets.get(canonical_id)

    def markets(self) -> list[NormalizedMarket]:
        return list(self._markets.values())

    def mid(self, canonical_id: str) -> Decimal | None:
        book = self._books.get(canonical_id)
        return book.mid if book else None


class UniverseContext(StrategyContext):
    """A sleeve's context: lookups by id stay global, but ``markets()`` is its universe.

    Strategies iterate ``markets()`` on every timer tick. Over the shared context that is
    every market the engine knows - Kalshi and Polymarket, thousands of them - for every
    sleeve, every second; it also let universe-agnostic loops (the random control, for
    one) trade markets outside the sleeve's declared universe. ``members`` is maintained
    by the runner as markets arrive.
    """

    __slots__ = ("_members",)

    def __init__(self, base: StrategyContext, members: dict[str, NormalizedMarket]) -> None:
        super().__init__(base.clock, base._books, base._markets, base._marks, base.params)
        self._members = members

    def markets(self) -> list[NormalizedMarket]:
        return list(self._members.values())


class Strategy(abc.ABC):  # noqa: B024 - see the handler note below
    """Base class for every strategy variant.

    Subclasses override only the handlers they care about.  ``generate_intents`` is called
    after every batch of events; returning an empty list is always valid.
    """

    #: Stable name used in the research registry and experiment ids.
    name: str = "unnamed"
    #: Bump when logic changes. A changed version is a NEW experiment, never a silent edit.
    version: str = "1.0.0"
    #: Evidence class from docs/research_registry.yaml: A-G.
    evidence_class: str = "E"
    #: Which universes this strategy subscribes to.
    universes: tuple[str, ...] = ()

    def __init__(
        self,
        strategy_id: str,
        experiment_id: str,
        ctx: StrategyContext,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.strategy_id = strategy_id
        self.experiment_id = experiment_id
        self.ctx = ctx
        self.params: dict[str, Any] = dict(params or {})
        self._pending: list[OrderIntent] = []
        self._forecasts: list[ProbabilityForecast] = []

    # ---- event handlers (all optional) ----
    # Deliberately concrete no-ops rather than @abstractmethod: a strategy overrides only
    # the handlers it cares about, and a momentum strategy has no business being forced to
    # implement on_weather. ruff's B027 flags exactly this shape, so it is silenced per
    # method below rather than by weakening the base class.
    def on_market_update(self, event: MarketUpdateEvent) -> None: ...  # noqa: B027
    def on_book_update(self, event: BookUpdateEvent) -> None: ...  # noqa: B027
    def on_trade(self, event: TradeEvent) -> None: ...  # noqa: B027
    def on_market_status(self, event: MarketStatusEvent) -> None: ...  # noqa: B027
    def on_settlement(self, event: SettlementEvent) -> None: ...  # noqa: B027
    def on_news(self, event: NewsEvent) -> None: ...  # noqa: B027
    def on_social(self, event: SocialEvent) -> None: ...  # noqa: B027
    def on_filing(self, event: FilingEvent) -> None: ...  # noqa: B027
    def on_external_price(self, event: ExternalPriceEvent) -> None: ...  # noqa: B027
    def on_trader_action(self, event: TraderActionEvent) -> None: ...  # noqa: B027
    def on_weather(self, event: WeatherEvent) -> None: ...  # noqa: B027
    def on_economic(self, event: EconomicEvent) -> None: ...  # noqa: B027
    def on_sports_state(self, event: SportsStateEvent) -> None: ...  # noqa: B027
    def on_timer(self, event: TimerEvent) -> None: ...  # noqa: B027

    # ---- broker feedback ----
    def on_order_update(self, order: Order) -> None: ...  # noqa: B027
    def on_fill(self, fill: Fill) -> None: ...  # noqa: B027

    # ---- output ----
    def generate_intents(self) -> list[OrderIntent]:
        """Drain and return intents accumulated by the handlers."""
        out, self._pending = self._pending, []
        return out

    def drain_forecasts(self) -> list[ProbabilityForecast]:
        out, self._forecasts = self._forecasts, []
        return out

    # ---- helpers for subclasses ----
    def emit(self, intent: OrderIntent) -> None:
        self._pending.append(intent)

    def forecast(self, forecast: ProbabilityForecast) -> None:
        self._forecasts.append(forecast)

    def now(self) -> datetime:
        return self.ctx.clock.now()

    def param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.strategy_id} v{self.version}>"
