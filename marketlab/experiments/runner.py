"""The tournament engine: N strategies, one shared event stream, one $50 sleeve each.

Design notes that matter for correctness:

* Portfolio ownership: the **broker** applies fills to the sleeve's ``Portfolio``,
  because only the broker knows the fee charged on each fill. The runner observes
  fills for bookkeeping and does not re-apply them. See ``broker_owns_portfolio``;
  it defaults to ``True`` so the dangerous case (double-counting every fill) cannot
  happen by omission. A portfolio-free test fake must pass ``False`` explicitly.

* **Marks convention.** ``StrategyContext`` and this runner's own ``self._marks`` are
  keyed by plain ``canonical_id -> YES-probability mid``. ``Portfolio.equity()`` /
  ``.mark()`` / ``.is_dead()`` instead expect ``"canonical_id|side" -> price`` (see
  ``Portfolio.key``). ``_marks_for_portfolio`` bridges the two, folding a NO position's
  mark to ``1 - mid`` per the YES-probability convention documented on ``OrderBook``.

* **Universe routing.** Nothing in ``marketlab.core`` records which configured universe
  (``configs/universes.yaml``) a given ``canonical_id`` belongs to - that classification
  is the market-ingestion team's job, since they are the ones who expand each universe's
  series prefixes into concrete markets. ``market_registry`` is therefore expected to
  expose ``universes_for(canonical_id) -> Iterable[str]`` (see ``MarketRegistryLike``
  below); if it does not, this runner degrades to "unroutable" for that market (logged
  once) rather than broadcasting to every sleeve, which would defeat the entire point of
  the routing requirement.
"""

from __future__ import annotations

import contextlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from marketlab.clock import Clock
from marketlab.core.broker import Broker
from marketlab.core.events import (
    BaseEvent,
    BookUpdateEvent,
    EconomicEvent,
    EventType,
    MarketUpdateEvent,
    SettlementEvent,
    TimerEvent,
    TradeEvent,
    TraderActionEvent,
    WeatherEvent,
)
from marketlab.core.instruments import ONE, NormalizedMarket, OrderBook, Side
from marketlab.core.orders import OrderStatus, RejectReason
from marketlab.core.portfolio import Portfolio, SleeveStatus
from marketlab.core.strategy import Strategy, StrategyContext
from marketlab.experiments import identity as identity_mod
from marketlab.experiments.identity import ExperimentIdentity
from marketlab.experiments.registry import ExperimentRegistry
from marketlab.experiments.sweep import SweepError, VariantSpec, load_strategy_class
from marketlab.logging import get_logger
from marketlab.settings import Settings
from marketlab.storage.state import ExperimentStatus, StateStore

log = get_logger(__name__)

try:
    from marketlab.analytics.metrics import profit_factor as _profit_factor
except ImportError:  # pragma: no cover - defensive; the real module exists as of writing.
    def _profit_factor(pnls: list[Decimal]) -> Decimal | None:  # type: ignore[misc]
        wins = sum((p for p in pnls if p > 0), Decimal(0))
        losses = sum((-p for p in pnls if p < 0), Decimal(0))
        return (wins / losses) if losses else None


#: Handlers dispatched by event type. Every value must be a real Strategy method name.
_EVENT_HANDLER_NAMES: dict[EventType, str] = {
    EventType.MARKET_UPDATE: "on_market_update",
    EventType.BOOK_UPDATE: "on_book_update",
    EventType.TRADE: "on_trade",
    EventType.MARKET_STATUS: "on_market_status",
    EventType.SETTLEMENT: "on_settlement",
    EventType.NEWS: "on_news",
    EventType.SOCIAL: "on_social",
    EventType.FILING: "on_filing",
    EventType.EXTERNAL_PRICE: "on_external_price",
    EventType.TRADER_ACTION: "on_trader_action",
    EventType.WEATHER: "on_weather",
    EventType.ECONOMIC: "on_economic",
    EventType.SPORTS_STATE: "on_sports_state",
    EventType.TIMER: "on_timer",
}

#: Sleeve statuses excluded from the active roster and from event dispatch.
_INACTIVE_STATUSES = frozenset({ExperimentStatus.DEAD, ExperimentStatus.DISABLED})

#: Buffer this many forecasts before committing them in one transaction.
_FORECAST_FLUSH_THRESHOLD = 500


@runtime_checkable
class MarketRegistryLike(Protocol):
    """The subset of a market registry this runner actually calls.

    Duck-typed on purpose: the real registry belongs to the ingestion/matching team.
    ``universes_for`` is this module's own required extension (see module docstring);
    a registry lacking it still works for ``get``-based caching, just without routing.
    """

    def get(self, canonical_id: str) -> NormalizedMarket | None: ...
    def universes_for(self, canonical_id: str) -> Any: ...


@runtime_checkable
class BookRegistryLike(Protocol):
    def get(self, canonical_id: str) -> OrderBook | None: ...


@dataclass
class _Sleeve:
    experiment_id: str
    strategy_id: str
    variant: VariantSpec
    strategy: Strategy
    portfolio: Portfolio
    status: ExperimentStatus
    consecutive_failures: int = 0
    error_count: int = 0
    orders_submitted: int = 0
    orders_rejected: int = 0
    stale_data_skips: int = 0
    risk_gate_skips: int = 0
    pnl_history: list[Decimal] = field(default_factory=list)


@dataclass(frozen=True)
class SleeveResult:
    experiment_id: str
    strategy_name: str
    category: str
    days_alive: float
    initial_capital: Decimal
    equity: Decimal
    net_pnl: Decimal
    roi: Decimal
    max_drawdown: Decimal
    trade_count: int
    profit_factor: Decimal | None
    calibration: Decimal | None
    status: str


def _event_canonical_id(event: BaseEvent) -> str | None:
    cid = getattr(event, "canonical_id", None)
    if cid:
        return str(cid)
    market = getattr(event, "market", None)
    if market is not None:
        return str(market.canonical_id)
    book = getattr(event, "book", None)
    if book is not None:
        return str(book.canonical_id)
    return None


class ExperimentRunner:
    """Drives every registered sleeve against one shared event stream."""

    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        store: StateStore,
        broker: Broker,
        market_registry: MarketRegistryLike,
        book_registry: BookRegistryLike,
        ai_provider: Any | None = None,
        *,
        max_consecutive_failures: int = 10,
        broker_owns_portfolio: bool = True,
    ) -> None:
        self.settings = settings
        self.clock = clock
        self.store = store
        self.broker = broker
        self.market_registry = market_registry
        self.book_registry = book_registry
        self.ai_provider = ai_provider
        self.max_consecutive_failures = max_consecutive_failures
        #: Who applies fills to the sleeve's ``Portfolio``.
        #:
        #: ``PaperBroker`` and ``KalshiLiveBroker`` both mutate the portfolio they are
        #: handed through ``portfolio_provider`` - they have to, because only the broker
        #: knows the fee charged on each fill. So the default is that the broker owns it
        #: and the runner merely observes fills for bookkeeping.
        #:
        #: Set this to ``False`` only for a portfolio-free broker (such as a test fake
        #: implementing the bare ``Broker`` ABC, which has no portfolio concept). Getting
        #: it wrong in that direction loses P&L silently; getting it wrong in the other
        #: direction double-counts every fill, which is why the safe value is the default.
        self.broker_owns_portfolio = broker_owns_portfolio
        #: experiment_id -> why its strategy could not be constructed.
        self._construction_failures: dict[str, str] = {}
        #: Forecasts awaiting a batched write. See _flush_forecasts.
        self._forecast_buffer: list[Any] = []

        self.registry = ExperimentRegistry(store, clock)

        self._sleeves: dict[str, _Sleeve] = {}
        self._books: dict[str, OrderBook] = {}
        self._markets: dict[str, NormalizedMarket] = {}
        #: canonical_id -> YES-probability mid, for StrategyContext and mark-to-market.
        self._marks: dict[str, Decimal] = {}
        self._ctx = StrategyContext(clock=clock, books=self._books, markets=self._markets, marks=self._marks)

        self._bankroll_per_variant = Decimal(
            str((settings.strategies.get("meta", {}) or {}).get("bankroll_per_variant", "50.00"))
        )
        self._last_snapshot: datetime | None = None
        self._warned_missing_universes_for = False

        universes = settings.universes.get("universes", {}) or {}
        self._category_universe: dict[str, set[str]] = {}
        self._station_universe: dict[str, set[str]] = {}
        self._series_universe: dict[str, set[str]] = {}
        for uname, udef in universes.items():
            udef = udef or {}
            cat = udef.get("category")
            if cat:
                self._category_universe.setdefault(str(cat), set()).add(uname)
            for station in (udef.get("stations", {}) or {}).values():
                self._station_universe.setdefault(station, set()).add(uname)
            for series in udef.get("fred_series", []) or []:
                self._series_universe.setdefault(series, set()).add(uname)

    # ------------------------------------------------------------------
    # sleeve lifecycle
    # ------------------------------------------------------------------

    async def load_or_create_sleeves(self, variants: list[VariantSpec]) -> None:
        """Reload existing PAPER sleeves from storage; create fresh $50 sleeves for the rest.

        The rule that matters: a restart never resets a bankroll. Whether a sleeve is
        "existing" is decided purely by whether ``store.load_portfolio`` returns
        something, not by the experiment's recorded status - that keeps this correct
        even if a status transition and a portfolio save ever fall out of step.
        """
        counts: Counter[str] = Counter(v.strategy_name for v in variants)
        for strategy_name, n in counts.items():
            self.registry.record_variant_count(strategy_name, n)

        for variant in variants:
            strategy_cls = self._load_strategy_class_safe(variant)
            if strategy_cls is None:
                continue

            strategy_id = f"{variant.strategy_name}__{variant.universe}__{identity_mod.parameter_hash(variant.params)}"
            strategy_version = str(getattr(strategy_cls, "version", "1.0.0"))
            identity = self._build_identity(variant, strategy_version)

            # Resume an existing live sleeve for this cohort rather than minting a new
            # experiment id. `start_timestamp` is part of the immutable identity hash, so
            # without this every process restart would create a fresh $50 bankroll for
            # all ~400 variants and orphan the previous run - which is exactly the
            # "never create a fresh paper bankroll because the process restarted" rule.
            #
            # A DEAD or DISABLED sleeve is deliberately not resumed: that is a new cohort.
            existing = None
            with contextlib.suppress(Exception):
                existing = self.store.find_live_experiment_by_cohort(identity.cohort_key)
            if existing is not None:
                experiment = existing
                log.debug(
                    "runner.cohort_resumed",
                    experiment_id=experiment.experiment_id,
                    cohort=identity.cohort_key,
                )
            else:
                experiment = self.registry.register(
                    identity, variant.params, variant.universe, cohort=identity.cohort_key
                )

            if experiment.status in _INACTIVE_STATUSES:
                log.info(
                    "runner.sleeve_excluded",
                    experiment_id=experiment.experiment_id,
                    status=str(experiment.status),
                )
                continue

            existing_portfolio = self.store.load_portfolio(experiment.experiment_id)
            if existing_portfolio is not None:
                portfolio = existing_portfolio
                log.info("runner.sleeve_restored", experiment_id=experiment.experiment_id, equity=str(portfolio.equity()))
            else:
                now = self.clock.now()
                portfolio = Portfolio(
                    experiment_id=experiment.experiment_id,
                    strategy_id=strategy_id,
                    initial_capital=self._bankroll_per_variant,
                    cash=self._bankroll_per_variant,
                    high_water_mark=self._bankroll_per_variant,
                    created_at=now,
                )
                self.store.save_portfolio(portfolio, as_of=now)
                if experiment.status == ExperimentStatus.IDEA:
                    self.registry.transition(experiment.experiment_id, ExperimentStatus.BACKTESTING, "sleeve created")
                    self.registry.transition(experiment.experiment_id, ExperimentStatus.PAPER, "starting paper run")

            # A strategy that refuses to construct must cost only its own sleeve. Before
            # this guard, one strategy raising in __init__ aborted load_or_create_sleeves
            # entirely and the tournament came up with 19 of 388 sleeves - the failure
            # looked like a slow startup rather than a crash.
            try:
                strategy = strategy_cls(
                    strategy_id=strategy_id,
                    experiment_id=experiment.experiment_id,
                    ctx=self._ctx,
                    params=variant.params,
                )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "runner.strategy_construct_failed",
                    experiment_id=experiment.experiment_id,
                    strategy=variant.strategy_name,
                    universe=variant.universe,
                    error=str(exc),
                    exc_info=True,
                )
                with contextlib.suppress(Exception):
                    self.registry.transition(
                        experiment.experiment_id,
                        ExperimentStatus.DISABLED,
                        f"construction failed: {type(exc).__name__}: {exc}",
                    )
                self._construction_failures[experiment.experiment_id] = f"{type(exc).__name__}: {exc}"
                continue

            saved_state = self.store.load_strategy_state(experiment.experiment_id)
            restore = getattr(strategy, "restore_state", None)
            if saved_state and callable(restore):
                restore(saved_state)

            refreshed = self.registry.get(experiment.experiment_id)
            current_status = refreshed.status if refreshed is not None else experiment.status
            self._sleeves[experiment.experiment_id] = _Sleeve(
                experiment_id=experiment.experiment_id,
                strategy_id=strategy_id,
                variant=variant,
                strategy=strategy,
                portfolio=portfolio,
                status=current_status,
            )

    def _flush_forecasts(self, force: bool = False) -> int:
        """Write buffered forecasts in one transaction.

        Forecasts are the calibration record - every one is kept, whether or not it led
        to a trade - so this must never silently drop them; a failed flush keeps the
        buffer for the next attempt.
        """
        if not self._forecast_buffer:
            return 0
        if not force and len(self._forecast_buffer) < _FORECAST_FLUSH_THRESHOLD:
            return 0
        batch = self._forecast_buffer
        self._forecast_buffer = []
        try:
            return int(self.store.save_forecasts_bulk(batch))
        except Exception as exc:  # noqa: BLE001
            log.error("runner.forecast_flush_failed", count=len(batch), error=str(exc))
            # Put them back rather than losing the calibration record.
            self._forecast_buffer = batch + self._forecast_buffer
            return 0

    def _load_strategy_class_safe(self, variant: VariantSpec) -> type[Strategy] | None:
        try:
            return load_strategy_class(variant.strategy_class_path)  # type: ignore[return-value]
        except SweepError as exc:
            log.error(
                "runner.strategy_class_unavailable",
                strategy=variant.strategy_name,
                path=variant.strategy_class_path,
                error=str(exc),
            )
            return None

    def _build_identity(self, variant: VariantSpec, strategy_version: str) -> ExperimentIdentity:
        llm_model_id = None
        if variant.requires_ai and self.ai_provider is not None:
            llm_model_id = getattr(self.ai_provider, "model", None)
        return ExperimentIdentity(
            strategy_name=variant.strategy_name,
            strategy_version=strategy_version,
            git_commit=identity_mod.git_commit(),
            parameter_hash=identity_mod.parameter_hash(variant.params),
            market_universe=variant.universe,
            venue="kalshi",
            data_version=identity_mod.DATA_VERSION,
            execution_model_version=identity_mod.EXECUTION_MODEL_VERSION,
            feature_version=identity_mod.FEATURE_VERSION,
            llm_model_id=llm_model_id,
            prompt_hash=None,
            start_timestamp=self.clock.now(),
            starting_bankroll=self._bankroll_per_variant,
        )

    # ------------------------------------------------------------------
    # marks bridge (see module docstring)
    # ------------------------------------------------------------------

    def _marks_for_portfolio(self, portfolio: Portfolio) -> dict[str, Decimal]:
        out: dict[str, Decimal] = {}
        for key, pos in portfolio.positions.items():
            mid = self._marks.get(pos.canonical_id)
            if mid is None:
                continue
            out[key] = mid if pos.side is Side.YES else (ONE - mid)
        return out

    # ------------------------------------------------------------------
    # routing
    # ------------------------------------------------------------------

    def _update_caches(self, event: BaseEvent) -> None:
        if isinstance(event, MarketUpdateEvent):
            self._markets[event.market.canonical_id] = event.market
        elif isinstance(event, BookUpdateEvent):
            self._books[event.book.canonical_id] = event.book
            if event.book.mid is not None:
                self._marks[event.book.canonical_id] = event.book.mid
        elif isinstance(event, TradeEvent):
            self._marks[event.trade.canonical_id] = event.trade.price

    def _target_universes(self, event: BaseEvent) -> set[str] | None:
        """``None`` means broadcast (event carries no market-specific routing signal)."""
        if isinstance(event, TimerEvent):
            return None
        cid = _event_canonical_id(event)
        if cid:
            if hasattr(self.market_registry, "universes_for"):
                try:
                    return set(self.market_registry.universes_for(cid))
                except Exception:
                    log.error("runner.universes_for_failed", canonical_id=cid, exc_info=True)
                    return set()
            if not self._warned_missing_universes_for:
                log.warning(
                    "runner.market_registry_missing_universes_for",
                    detail="market_registry has no universes_for(); per-market events cannot be "
                    "routed and will be dropped rather than broadcast to every sleeve",
                )
                self._warned_missing_universes_for = True
            return set()
        if isinstance(event, WeatherEvent):
            return set(self._station_universe.get(event.station, set()))
        if isinstance(event, EconomicEvent):
            return set(self._series_universe.get(event.series_id, set()))
        if isinstance(event, TraderActionEvent):
            return set(self._category_universe.get(str(event.category), set()))
        # NewsEvent, SocialEvent, FilingEvent, SportsStateEvent, ExternalPriceEvent without
        # a canonical_id: no reliable per-market signal exists upstream of the ingestion/
        # signals teams' own classifiers, so broadcast. These are all far lower-frequency
        # than book/trade ticks, so broadcasting does not threaten the 400-sleeve scaling
        # goal the way per-tick events would.
        return None

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------

    async def dispatch(self, event: BaseEvent) -> None:
        self._update_caches(event)
        targets = self._target_universes(event)
        handler_name = _EVENT_HANDLER_NAMES.get(event.event_type)
        for sleeve in list(self._sleeves.values()):
            if sleeve.status in _INACTIVE_STATUSES:
                continue
            if targets is not None and sleeve.variant.universe not in targets:
                continue
            await self._process_sleeve_event(sleeve, event, handler_name)

    async def _process_sleeve_event(
        self, sleeve: _Sleeve, event: BaseEvent, handler_name: str | None
    ) -> None:
        try:
            if handler_name is not None:
                handler = getattr(sleeve.strategy, handler_name, None)
                if handler is not None:
                    handler(event)

            if isinstance(event, SettlementEvent):
                sleeve.portfolio.settle(event.canonical_id, event.winning_side)

            # Buffered, not written per forecast: a per-row commit here blocked the
            # event loop for minutes once the tournament was emitting tens of thousands
            # of forecasts a minute. Flushed by _flush_forecasts (on tick and snapshot).
            self._forecast_buffer.extend(sleeve.strategy.drain_forecasts())

            for intent in sleeve.strategy.generate_intents():
                await self._submit_intent(sleeve, intent)

            sleeve.consecutive_failures = 0
            self._check_death(sleeve)
        except Exception:
            self._on_sleeve_error(sleeve, event)

    async def _submit_intent(self, sleeve: _Sleeve, intent: Any) -> None:
        order = await self.broker.submit(intent)
        sleeve.orders_submitted += 1
        if order.status is OrderStatus.REJECTED:
            sleeve.orders_rejected += 1
            if order.reject_reason is RejectReason.STALE_DATA:
                sleeve.stale_data_skips += 1
            elif order.reject_reason is RejectReason.RISK_GATE:
                sleeve.risk_gate_skips += 1
        sleeve.strategy.on_order_update(order)
        for fill in order.fills:
            before = sleeve.portfolio.realized_pnl
            if not self.broker_owns_portfolio:
                sleeve.portfolio.apply_fill(fill)
            delta = sleeve.portfolio.realized_pnl - before
            if delta != 0:
                sleeve.pnl_history.append(delta)
            sleeve.strategy.on_fill(fill)

    def _on_sleeve_error(self, sleeve: _Sleeve, event: BaseEvent) -> None:
        sleeve.error_count += 1
        sleeve.consecutive_failures += 1
        log.error(
            "runner.strategy_error",
            experiment_id=sleeve.experiment_id,
            strategy=sleeve.variant.strategy_name,
            event_type=str(event.event_type),
            consecutive_failures=sleeve.consecutive_failures,
            exc_info=True,
        )
        if sleeve.consecutive_failures >= self.max_consecutive_failures:
            self._disable_sleeve(sleeve, f"{sleeve.consecutive_failures} consecutive handler failures")

    def _disable_sleeve(self, sleeve: _Sleeve, reason: str) -> None:
        sleeve.status = ExperimentStatus.DISABLED
        try:
            self.registry.transition(sleeve.experiment_id, ExperimentStatus.DISABLED, reason)
        except Exception:
            log.error("runner.disable_transition_failed", experiment_id=sleeve.experiment_id, exc_info=True)
        log.warning("runner.strategy_disabled", experiment_id=sleeve.experiment_id, reason=reason)

    def _check_death(self, sleeve: _Sleeve) -> None:
        if sleeve.status in _INACTIVE_STATUSES:
            return
        sleeve.portfolio.mark(self._marks_for_portfolio(sleeve.portfolio))
        if sleeve.portfolio.is_dead(self.settings.risk.death_floor):
            self._kill_sleeve(sleeve)

    def _kill_sleeve(self, sleeve: _Sleeve) -> None:
        now = self.clock.now()
        sleeve.portfolio.status = SleeveStatus.DEAD
        sleeve.portfolio.died_at = now
        sleeve.status = ExperimentStatus.DEAD
        self.store.save_portfolio(sleeve.portfolio, as_of=now)
        try:
            self.registry.transition(sleeve.experiment_id, ExperimentStatus.DEAD, "equity below death floor")
        except Exception:
            log.error("runner.death_transition_failed", experiment_id=sleeve.experiment_id, exc_info=True)
        log.warning(
            "runner.sleeve_dead",
            experiment_id=sleeve.experiment_id,
            final_equity=str(sleeve.portfolio.equity()),
        )

    # ------------------------------------------------------------------
    # tick / snapshot
    # ------------------------------------------------------------------

    async def tick(self) -> None:
        now = self.clock.now()
        await self.dispatch(
            TimerEvent(event_time=now, first_seen_time=now, interval_seconds=self.settings.paper.tick_seconds)
        )
        self._flush_forecasts()
        if self._last_snapshot is None or (now - self._last_snapshot).total_seconds() >= self.settings.paper.snapshot_seconds:
            await self.snapshot()

    async def snapshot(self) -> None:
        now = self.clock.now()
        # Force a flush so a snapshot's balances and its forecasts describe the same
        # instant; a partially-written calibration record is worse than a slow one.
        self._flush_forecasts(force=True)
        for sleeve in self._sleeves.values():
            sleeve.portfolio.mark(self._marks_for_portfolio(sleeve.portfolio))
            self.store.save_portfolio(sleeve.portfolio, as_of=now)
        self._last_snapshot = now
        board = self.leaderboard()
        log.info("runner.snapshot", at=now.isoformat(), n_sleeves=len(self._sleeves), n_active=sum(1 for s in self._sleeves.values() if s.status not in _INACTIVE_STATUSES))
        for row in board[:10]:
            log.info("runner.leaderboard_row", experiment_id=row.experiment_id, equity=str(row.equity), roi=str(row.roi), status=row.status)

    # ------------------------------------------------------------------
    # leaderboard
    # ------------------------------------------------------------------

    def leaderboard(self) -> list[SleeveResult]:
        now = self.clock.now()
        universes = self.settings.universes.get("universes", {}) or {}
        results: list[SleeveResult] = []
        for sleeve in self._sleeves.values():
            p = sleeve.portfolio
            marks = self._marks_for_portfolio(p)
            equity = p.equity(marks)
            net_pnl = p.realized_pnl + p.unrealized_pnl(marks)
            roi = (net_pnl / p.initial_capital) if p.initial_capital else Decimal(0)
            created = p.created_at or now
            days_alive = max((now - created).total_seconds() / 86400.0, 0.0)
            pf = _profit_factor(sleeve.pnl_history) if sleeve.pnl_history else None
            category = str((universes.get(sleeve.variant.universe, {}) or {}).get("category", ""))
            results.append(
                SleeveResult(
                    experiment_id=sleeve.experiment_id,
                    strategy_name=sleeve.variant.strategy_name,
                    category=category,
                    days_alive=days_alive,
                    initial_capital=p.initial_capital,
                    equity=equity,
                    net_pnl=net_pnl,
                    roi=roi,
                    max_drawdown=p.max_drawdown,
                    trade_count=p.trade_count,
                    profit_factor=pf,
                    # Full calibration (Brier score vs settlement) belongs to the analytics
                    # layer, which has both forecasts and settlements; this runner only
                    # supplies the raw ingredients (forecasts saved via save_forecast()).
                    calibration=None,
                    status=str(sleeve.status),
                )
            )
        results.sort(key=lambda r: r.experiment_id)
        return results
