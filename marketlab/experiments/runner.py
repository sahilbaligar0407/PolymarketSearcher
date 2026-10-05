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

from marketlab.ai.typesafe import build_jev_client
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
from marketlab.core.strategy import Strategy, StrategyContext, UniverseContext
from marketlab.experiments import identity as identity_mod
from marketlab.experiments.identity import ExperimentIdentity
from marketlab.experiments.registry import ExperimentRegistry
from marketlab.experiments.sweep import SweepError, VariantSpec, load_strategy_class
from marketlab.logging import get_logger
from marketlab.matching.book import MatchBook, MatchView
from marketlab.matching.cross_venue import approved_for_automation
from marketlab.settings import Settings
from marketlab.signals.holdings import HoldingsBook
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

#: Forecast recording is throttled per (experiment, market). Left unthrottled, the
#: baseline strategy alone emits one forecast per subscribed market per 1s tick across
#: ~400 sleeves - measured at 623k rows in nine minutes, projecting ~90M/day and tens of
#: gigabytes of SQLite. Calibration needs a representative sample of forecasts, not every
#: redundant restatement of an unchanged number, so a forecast is kept when either enough
#: time has passed OR the probability actually moved.
_FORECAST_MIN_INTERVAL_SECONDS = 60.0
_FORECAST_MIN_MOVE = Decimal("0.01")
#: After a risk-gate rejection, the same sleeve's same (market, side, action) intent is
#: held back this long instead of re-submitted. The gate's answer does not change tick to
#: tick, and the retries were ~1,300 rejected order rows per strategy per 6 minutes.
_RISK_BACKOFF_SECONDS = 60.0


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
    #: (canonical_id, side, action) -> epoch seconds until which that intent is held back.
    risk_backoff: dict[tuple[str, str, str], float] = field(default_factory=dict)
    risk_backoff_skips: int = 0


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


#: Strategy families that read ``params["matches"]``.
MATCH_CONSUMERS = frozenset({"cross_venue", "copy_trader", "copy_basket"})
#: Strategies driven by the leaderboard holdings snapshot (marketlab.signals.holdings).
HOLDINGS_CONSUMERS = frozenset({"holdings_blind", "holdings_edge", "holdings_confirm"})


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
        ai_stack: Any | None = None,
        portfolio_registry: Any | None = None,
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
        #: The tiered AI runtime (see marketlab.ai.stack). Without one, AI sleeves fall
        #: back to the single ``ai_provider`` with no escalation and no shared caches.
        self.ai_stack = ai_stack
        self._ai_skipped: Counter[str] = Counter()
        #: Approved cross-venue pairs, refreshed from storage; see marketlab.matching.book.
        self.match_book = MatchBook()
        #: wallet -> TraderScore for QUALIFIED wallets; one live dict shared by every
        #: copy sleeve and mutated in place on refresh.
        self.qualified_roster: dict[str, float] = {}
        #: Top-wallet holdings, written by the ingest loop and read by holdings sleeves.
        self.holdings = HoldingsBook()
        #: TypeSafe Jev client for the holdings "jev" confirmations; None without a key.
        self.jev = build_jev_client(settings)
        self._matches_refreshed_at: datetime | None = None
        self.trader_actions_translated = 0
        #: Shared with the broker so BOTH mutate the same Portfolio object.
        #:
        #: The broker looks a sleeve's portfolio up by experiment_id through its
        #: `portfolio_provider`; this runner builds its own from the store. Without
        #: registering ours here those are two different objects for the same sleeve:
        #: the broker applies fills to one while the runner snapshots and settles the
        #: other, so persisted equity and the exposure the risk gateway sees drift apart.
        self.portfolio_registry = portfolio_registry
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
        #: (experiment_id, canonical_id) -> (last recorded time, last recorded p_yes).
        self._forecast_last: dict[tuple[str, str], tuple[datetime, Decimal]] = {}
        self.forecasts_throttled = 0

        self.registry = ExperimentRegistry(store, clock)

        self._sleeves: dict[str, _Sleeve] = {}
        self._books: dict[str, OrderBook] = {}
        self._markets: dict[str, NormalizedMarket] = {}
        #: canonical_id -> YES-probability mid, for StrategyContext and mark-to-market.
        self._marks: dict[str, Decimal] = {}
        self._ctx = StrategyContext(clock=clock, books=self._books, markets=self._markets, marks=self._marks)
        #: universe -> {canonical_id: market}; each sleeve's UniverseContext reads its own.
        self._universe_members: dict[str, dict[str, NormalizedMarket]] = {}

        #: strategy_name -> the version the code ships, filled as sleeves load.
        self._current_versions: dict[str, str] = {}
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

            ai_params = self._ai_params(variant)
            if ai_params is None:
                continue
            # A universe's station map (configs/universes.yaml) routes weather events
            # here, but the weather strategy also needs it to map a market to its NWS
            # station. It was never passed on, so every weather market was skipped.
            stations = ((self.settings.universes.get("universes", {}) or {}).get(variant.universe, {}) or {}).get("stations")
            if stations and "stations" not in variant.params:
                ai_params = {**ai_params, "stations": dict(stations)}
            ai_params = {**ai_params, "bankroll": str(self._bankroll_per_variant)}
            if variant.strategy_name in MATCH_CONSUMERS:
                ai_params = {
                    **ai_params,
                    "matches": MatchView(self.match_book, variant.universe, self._universes_for),
                    "roster": self.qualified_roster,
                }
            if variant.strategy_name in HOLDINGS_CONSUMERS:
                ai_params = {
                    **ai_params,
                    "holdings": self.holdings,
                    # Every approved twin, not just this universe's: the holdings signal is
                    # global, and most twins found by discovery sit in no sleeve universe.
                    "matches": self.match_book,
                    "jev": self.jev,
                    "evidence": self.ai_stack.evidence if self.ai_stack is not None else None,
                }

            strategy_id = f"{variant.strategy_name}__{variant.universe}__{identity_mod.parameter_hash(variant.params)}"
            strategy_version = str(getattr(strategy_cls, "version", "1.0.0"))
            self._current_versions[variant.strategy_name] = strategy_version
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
                    ctx=UniverseContext(self._ctx, self._universe_members.setdefault(variant.universe, {})),
                    # Live collaborators ride alongside the declared params but never
                    # enter variant.params, which is hashed into the experiment identity.
                    params={**variant.params, **ai_params},
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

            if self.portfolio_registry is not None:
                with contextlib.suppress(Exception):
                    self.portfolio_registry.register(portfolio)

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
        self.retire_superseded()

    async def cancel_orphaned_orders(self) -> int:
        """Cancel resting orders that belong to no running sleeve.

        Recovery reloads every open order from storage, including the quotes of sleeves
        that are retired or were superseded (still holding a position, so kept until it
        settles). Their strategies no longer run, so nothing would ever manage those
        orders - but they could still fill and keep changing a dead sleeve's book.
        """
        running = {eid for eid, sl in self._sleeves.items() if sl.status not in _INACTIVE_STATUSES}
        cancelled = 0
        try:
            orders = await self.broker.open_orders()
        except Exception:  # noqa: BLE001
            return 0
        for order in orders:
            if order.experiment_id in running or order.is_terminal:
                continue
            with contextlib.suppress(Exception):
                if await self.broker.cancel(order.order_id) is not None:
                    cancelled += 1
        if cancelled:
            log.info("runner.orphaned_orders_cancelled", count=cancelled)
        return cancelled

    def retire_superseded(self) -> int:
        """DISABLE live sleeves that the current config has explicitly replaced.

        Only explicit signals count: a different starting bankroll, a strategy that is
        now ``enabled: false``, or an older strategy version than the code ships. Any
        other cohort that simply was not re-created this boot (an AI arm whose tier is
        down, a universe marked unavailable) is left alone, since that can be transient.
        A sleeve still holding a position is also left alone until it settles, so its
        P&L completes. Without this, every bankroll or version change left the old
        cohort on the dashboard as hundreds of "active" sleeves that never trade.
        """
        strategies_cfg = (self.settings.strategies.get("strategies", {}) or {})
        running = set(self._sleeves)
        retired = 0
        try:
            experiments = self.store.list_experiments()
        except Exception:  # noqa: BLE001
            return 0
        for exp in experiments:
            if exp.experiment_id in running or str(getattr(exp.status, "value", exp.status)) != "PAPER":
                continue
            sdef = strategies_cfg.get(exp.strategy_name) or {}
            reason = None
            if exp.starting_bankroll != self._bankroll_per_variant:
                reason = f"bankroll {exp.starting_bankroll} replaced by {self._bankroll_per_variant}"
            elif sdef and not sdef.get("enabled", True):
                reason = "strategy disabled in configs/strategies.yaml"
            elif (
                exp.strategy_name in self._current_versions
                and exp.strategy_version != self._current_versions[exp.strategy_name]
            ):
                reason = f"version {exp.strategy_version} replaced by {self._current_versions[exp.strategy_name]}"
            if reason is None:
                continue
            try:
                portfolio = self.store.load_portfolio(exp.experiment_id)
            except Exception:  # noqa: BLE001
                continue
            if portfolio is not None and any(pos.quantity > 0 for pos in portfolio.positions.values()):
                continue
            try:
                self.registry.transition(exp.experiment_id, ExperimentStatus.DISABLED, f"superseded: {reason}")
                retired += 1
            except Exception:  # noqa: BLE001 - an illegal transition is simply skipped
                log.warning("runner.retire_failed", experiment_id=exp.experiment_id, exc_info=True)
        if retired:
            log.info("runner.superseded_retired", count=retired)
        return retired

    def _should_record_forecast(self, forecast: Any) -> bool:
        """Throttle per (experiment, market): keep it if time passed or the number moved.

        Dropping an unchanged restatement loses nothing - the previous row already says
        the same thing at a slightly earlier instant - while keeping every genuine move,
        which is what calibration and Brier scoring actually need.
        """
        key = (forecast.experiment_id, forecast.canonical_id)
        previous = self._forecast_last.get(key)
        now = forecast.as_of
        if previous is not None:
            last_time, last_p = previous
            moved = abs(forecast.p_yes - last_p) >= _FORECAST_MIN_MOVE
            elapsed = (now - last_time).total_seconds()
            if not moved and elapsed < _FORECAST_MIN_INTERVAL_SECONDS:
                self.forecasts_throttled += 1
                return False
        self._forecast_last[key] = (now, forecast.p_yes)
        return True

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

    def _ai_params(self, variant: VariantSpec) -> dict[str, Any] | None:
        """Collaborators for an AI sleeve; ``{}`` for a non-AI one; None = cannot run.

        An arm whose tiers are not configured (no Jev URL, no OpenAI budget) is skipped
        before registration, so it never appears on the leaderboard as a sleeve that
        merely "never traded".
        """
        if not variant.requires_ai:
            return {}
        if self.ai_stack is None:
            return {"llm_provider": self.ai_provider} if self.ai_provider is not None else {}
        stack_name = str(variant.params.get("ai_stack", "local"))
        resolved = self.ai_stack.resolve(stack_name)
        if resolved is None:
            if self._ai_skipped[stack_name] == 0:
                log.info("runner.ai_arm_unavailable", ai_stack=stack_name,
                         detail="a required AI tier is not configured; arm not created")
            self._ai_skipped[stack_name] += 1
            return None
        primary, escalation = resolved
        return {
            "llm_provider": primary,
            "escalation_provider": escalation,
            "assessment_cache": self.ai_stack.assessments,
            "retrieval_store": self.ai_stack.evidence,
        }

    def _llm_model_id(self, variant: VariantSpec) -> str | None:
        if not variant.requires_ai:
            return None
        if self.ai_stack is not None:
            resolved = self.ai_stack.resolve(str(variant.params.get("ai_stack", "local")))
            if resolved is not None:
                primary, escalation = resolved
                ids = [f"{primary.name}:{primary.model}"]
                if escalation is not None:
                    ids.append(f"{escalation.name}:{escalation.model}")
                return "+".join(ids)
        return getattr(self.ai_provider, "model", None) if self.ai_provider is not None else None

    def _build_identity(self, variant: VariantSpec, strategy_version: str) -> ExperimentIdentity:
        llm_model_id = self._llm_model_id(variant)
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
            try:
                for universe in self._universes_for(event.market.canonical_id) or ():
                    self._universe_members.setdefault(universe, {})[event.market.canonical_id] = event.market
            except Exception:  # noqa: BLE001 - routing failure must not lose the market update
                log.error("runner.universe_membership_failed", canonical_id=event.market.canonical_id, exc_info=True)
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

    def _universes_for(self, canonical_id: str) -> Any:
        return self.market_registry.universes_for(canonical_id) if hasattr(self.market_registry, "universes_for") else ()

    def refresh_matches(self, force: bool = False) -> int:
        """Reload the approved cross-venue pairs from storage (at most every 5 minutes)."""
        now = self.clock.now()
        if not force and self._matches_refreshed_at is not None and (now - self._matches_refreshed_at).total_seconds() < 300:
            return len(self.match_book)
        self._matches_refreshed_at = now
        try:
            rows = self.store.approved_matches()
        except Exception:  # noqa: BLE001 - keep the previous book on a read failure
            log.error("runner.match_refresh_failed", exc_info=True)
            return len(self.match_book)
        self.match_book.replace([m for m in rows if approved_for_automation(m)])
        try:
            roster = {r["wallet"]: float(r["score"]) for r in self.store.trader_scores(status="QUALIFIED")}
        except Exception:  # noqa: BLE001 - a store without the table simply has no roster
            roster = dict(self.qualified_roster)
        self.qualified_roster.clear()
        self.qualified_roster.update(roster)
        log.info("runner.matches_refreshed", approved=len(self.match_book), qualified_traders=len(roster))
        return len(self.match_book)

    def _translate_trader_action(self, event: TraderActionEvent) -> TraderActionEvent:
        """Re-point a Polymarket trade at its proven Kalshi twin.

        Wallet activity arrives keyed by the Polymarket market, with ``side`` set only for
        literal Yes/No outcomes - so a buy of "Padres" reached copy_trader with a ``poly:``
        id and no side, and was refused every time. With an approved YES==YES pair, buying
        Polymarket outcome[0] is a Kalshi YES and buying outcome[1] is a Kalshi NO. Sells
        are left untranslated: exiting a position is not a signal to open one.
        """
        match = self.match_book.for_poly(event.canonical_id)
        if match is None or str(event.action).lower() != "buy":
            return event
        poly_market = self._markets.get(event.canonical_id)
        if poly_market is None:
            return event
        outcome = event.outcome.strip().lower()
        if outcome and outcome == poly_market.yes_symbol.strip().lower():
            side = Side.YES
        elif outcome and outcome == poly_market.no_symbol.strip().lower():
            side = Side.NO
        else:
            return event
        self.trader_actions_translated += 1
        return event.model_copy(update={"canonical_id": match.canonical_id_a, "side": side})

    async def dispatch(self, event: BaseEvent) -> None:
        if isinstance(event, TraderActionEvent) and event.canonical_id.startswith("poly:"):
            event = self._translate_trader_action(event)
        self._update_caches(event)
        if self.ai_stack is not None:
            with contextlib.suppress(Exception):
                self.ai_stack.evidence.record(event)
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
            for forecast in sleeve.strategy.drain_forecasts():
                if self._should_record_forecast(forecast):
                    self._forecast_buffer.append(forecast)

            # Cancels before new intents: a requoting strategy pulls its old quotes and
            # posts new ones in the same handler call, and the old ones must be gone
            # before the risk gate sizes the new ones. market_maker exposes this hook;
            # nothing drained it, so its quotes were never pulled.
            cancel_requests = getattr(sleeve.strategy, "cancel_requests", None)
            if callable(cancel_requests):
                for order_id in cancel_requests():
                    await self._cancel_order(sleeve, order_id)

            for intent in sleeve.strategy.generate_intents():
                await self._submit_intent(sleeve, intent)

            sleeve.consecutive_failures = 0
            self._check_death(sleeve)
        except Exception:
            self._on_sleeve_error(sleeve, event)

    async def _cancel_order(self, sleeve: _Sleeve, order_id: str) -> None:
        """Cancel one of this sleeve's own resting orders and report the result back.

        The broker is shared by every sleeve, so an id from another experiment (or an
        unknown or already-finished order) is ignored rather than cancelled.
        """
        order = await self.broker.get_order(order_id)
        if order is None or order.experiment_id != sleeve.experiment_id or order.is_terminal:
            return
        cancelled = await self.broker.cancel(order_id)
        if cancelled is not None:
            sleeve.strategy.on_order_update(cancelled)

    async def _submit_intent(self, sleeve: _Sleeve, intent: Any) -> None:
        key = (str(intent.canonical_id), str(intent.side), str(intent.action))
        now_ts = self.clock.now().timestamp()
        if sleeve.risk_backoff.get(key, 0.0) > now_ts:
            sleeve.risk_backoff_skips += 1
            return
        order = await self.broker.submit(intent)
        sleeve.orders_submitted += 1
        if order.status is OrderStatus.REJECTED:
            sleeve.orders_rejected += 1
            if order.reject_reason is RejectReason.STALE_DATA:
                sleeve.stale_data_skips += 1
            elif order.reject_reason is RejectReason.RISK_GATE:
                sleeve.risk_gate_skips += 1
                sleeve.risk_backoff[key] = now_ts + _RISK_BACKOFF_SECONDS
        sleeve.strategy.on_order_update(order)
        if order.filled_quantity > 0:
            try:
                self.store.save_decision(intent, order, sleeve.experiment_id)
            except Exception:  # noqa: BLE001 - the audit record must never break trading
                log.error("runner.decision_persist_failed", order_id=order.order_id, exc_info=True)
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
        self._start_ai_inference()
        self.refresh_matches()
        self._flush_forecasts()
        if self._last_snapshot is None or (now - self._last_snapshot).total_seconds() >= self.settings.paper.snapshot_seconds:
            await self.snapshot()

    def _start_ai_inference(self) -> None:
        """Let AI sleeves drain their inference queues in the background.

        Inference is slow (seconds to minutes) so it is never awaited here; the intents
        it produces are submitted on the sleeve's next dispatch.
        """
        for sleeve in self._sleeves.values():
            if sleeve.status in _INACTIVE_STATUSES:
                continue
            start = getattr(sleeve.strategy, "start_background_inference", None)
            if start is None:
                continue
            try:
                start()
            except Exception:  # noqa: BLE001 - one sleeve's AI must never stall the tick
                log.error("runner.ai_inference_start_failed", experiment_id=sleeve.experiment_id, exc_info=True)

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
