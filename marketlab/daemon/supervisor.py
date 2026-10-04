"""The supervisor: the process that runs forever.

``Supervisor.run()`` performs the full boot sequence (settings -> logging -> store ->
AI provider -> adapters -> geoblock -> registries -> risk gateway + broker -> strategy
variants -> experiment runner -> recovery -> ingest -> main loop) and then drains events
until asked to stop. It is the one place that assembles every other team's frozen module
into a running system.

Two integration points deserve a note because they touch modules other teams are still
writing:

* ``marketlab.experiments`` (``generate_variants``, ``ExperimentRunner``,
  ``PromotionEngine``) is imported lazily, inside the methods that use it, and every
  call is wrapped so a missing or partially-built module degrades to "ingestion runs,
  the tournament doesn't" rather than a crash.
* The health gate (``docs/live_safety.md``'s "the system stops trading rather than
  trading blind") is enforced structurally by wrapping whatever broker is built in
  :class:`_HealthGatedBroker`, which intercepts every ``submit()`` regardless of which
  strategy or runner code path produced the intent -- this does not depend on the
  experiment runner remembering to check health itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from marketlab.ai.provider import LLMProvider, detect_provider
from marketlab.ai.stack import AIStack, build_ai_stack
from marketlab.clock import Clock, LiveClock
from marketlab.core.broker import Broker, Mode
from marketlab.core.events import (
    BookUpdateEvent,
    Event,
    MarketStatusEvent,
    MarketUpdateEvent,
    SettlementEvent,
    TimerEvent,
    TradeEvent,
)
from marketlab.core.orders import Order, OrderIntent, OrderStatus, RejectReason
from marketlab.daemon.health import HealthMonitor
from marketlab.daemon.ingest import IngestService, load_ingest_config
from marketlab.daemon.recovery import RecoveryService
from marketlab.daemon.registry import BookRegistry, MarketRegistry, PortfolioRegistry
from marketlab.daemon.watchdog import HangWatchdog
from marketlab.execution.fill_models import build_limit_fill_model
from marketlab.execution.kalshi_live import KalshiLiveBroker, LiveTradingBlocked
from marketlab.execution.latency import LatencyModel
from marketlab.execution.paper_broker import KalshiFeeCalculator, PaperBroker
from marketlab.execution.risk_gateway import RiskGateway
from marketlab.logging import configure_logging, get_logger
from marketlab.settings import Settings
from marketlab.storage.parquet import AsyncParquetWriter, ParquetWriter
from marketlab.storage.state import AsyncStateStore, StateStore

log = get_logger(__name__)

#: How often the promotion engine is re-evaluated. Deliberately much slower than the
#: tick/snapshot cadences -- promotion is a judgment about weeks of history, not seconds.
PROMOTION_INTERVAL_SECONDS = 6 * 3600.0
HEARTBEAT_INTERVAL_SECONDS = 30.0
#: No progress for this long kills the process so START_TRADING.bat restarts it.
BOOT_HANG_LIMIT_SECONDS = 1200.0
RUN_HANG_LIMIT_SECONDS = 600.0
MAIN_LOOP_QUEUE_TIMEOUT = 1.0
#: Yield to the event loop this often while draining a hot queue, so ingest
#: tasks and the heartbeat are not starved by a CPU-bound dispatch burst.
_YIELD_EVERY_N_EVENTS = 25


class _HealthGatedBroker(Broker):
    """Wraps any :class:`Broker` so every ``submit()`` is gated on feed health first.

    This is the structural form of "the system stops trading rather than trading
    blind": regardless of what code path produced an :class:`OrderIntent` (a strategy,
    a copy-trading follower simulation, a manual test), it reaches the real broker only
    if :meth:`HealthMonitor.trading_allowed` currently says yes.
    """

    def __init__(self, inner: Broker, health: HealthMonitor, clock: Clock) -> None:
        self.mode = inner.mode
        self._inner = inner
        self._health = health
        self._clock = clock
        self.submit_count = 0
        self.blocked_count = 0

    @property
    def inner(self) -> Broker:
        return self._inner

    async def submit(self, intent: OrderIntent) -> Order:
        self.submit_count += 1
        allowed, detail = self._health.trading_allowed(self._clock.now())
        if not allowed:
            self.blocked_count += 1
            log.warning("trading_blocked_intent_rejected", intent_id=intent.intent_id, detail=detail)
            return Order(
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
                reject_reason=RejectReason.STALE_DATA,
                reject_detail=f"trading halted: {detail}",
                decision_timestamp=intent.decision_time,
            )
        return await self._inner.submit(intent)

    async def cancel(self, order_id: str) -> Order | None:
        return await self._inner.cancel(order_id)

    async def open_orders(self, strategy_id: str | None = None) -> list[Order]:
        return await self._inner.open_orders(strategy_id)

    async def get_order(self, order_id: str) -> Order | None:
        return await self._inner.get_order(order_id)


@dataclass
class SupervisorStatus:
    pid: int
    mode: str
    started_at: str | None
    uptime_seconds: float
    events_processed: int
    markets_tracked: int
    sleeves_alive: int
    sleeves_dead: int
    orders_submitted: int
    orders_blocked: int
    trading_allowed: bool
    trading_detail: str
    last_heartbeat: str
    feed_health: dict[str, Any]
    ai: dict[str, Any] = field(default_factory=dict)
    risk_rejects: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


async def _ensure_ollama_running(base_url: str) -> None:
    """Start a local Ollama server if one is installed but not answering.

    The watchdog restarts the daemon after a reboot or crash; the local AI tier should
    come back with it rather than staying dark until someone starts Ollama by hand.
    """
    import os
    import shutil
    import subprocess

    import httpx

    async def _up() -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                return (await client.get(f"{base_url.rstrip('/')}/api/tags")).status_code == 200
        except httpx.HTTPError:
            return False

    if await _up():
        return
    exe = shutil.which("ollama")
    if exe is None and os.name == "nt":
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        exe = str(candidate) if candidate.exists() else None
    if exe is None:
        log.warning("ollama_not_installed", detail="local AI tier will abstain")
        return
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0)
    try:
        subprocess.Popen(  # noqa: S603, ASYNC220 - fire-and-forget detached server
            [exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags
        )
    except OSError as exc:
        log.warning("ollama_start_failed", error=str(exc))
        return
    for _ in range(20):
        await asyncio.sleep(1.5)
        if await _up():
            log.info("ollama_started", exe=exe)
            return
    log.warning("ollama_start_timeout", detail="continuing; local tier abstains until it answers")


class Supervisor:
    """Boots and runs the whole daemon. One instance per process."""

    def __init__(self, settings: Settings, *, clock: Clock | None = None) -> None:
        self.settings = settings
        self.clock: Clock = clock or LiveClock()

        self.store: StateStore | None = None
        self.async_store: AsyncStateStore | None = None
        self.parquet: AsyncParquetWriter | None = None
        self.ai_provider: LLMProvider | None = None

        self.health: HealthMonitor | None = None
        self.ingest: IngestService | None = None
        self.market_registry: MarketRegistry | None = None
        self.book_registry: BookRegistry | None = None
        self.portfolio_registry: PortfolioRegistry | None = None
        self.risk_gateway: RiskGateway | None = None
        self._raw_broker: Broker | None = None
        self.broker: _HealthGatedBroker | None = None
        self.runner: Any | None = None
        self.ai_stack: AIStack | None = None
        self._watchdog: HangWatchdog | None = None
        self.variants: list[Any] = []

        self.started_at: datetime | None = None
        self.events_processed = 0
        self._stop_event = asyncio.Event()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._last_heartbeat: datetime | None = None
        self._last_report_date: date | None = None

    # ------------------------------------------------------------------
    # boot sequence
    # ------------------------------------------------------------------

    async def run(self, *, acknowledge_real_money_risk: bool = False) -> None:
        self.settings.ensure_dirs()
        configure_logging(self.settings.log_dir, level="INFO")
        log.info("supervisor_boot_begin", mode=self.settings.mode.value)
        # Boot touches the network (Ollama, geoblock, Kalshi recovery); give it 20 min.
        self._watchdog = HangWatchdog(BOOT_HANG_LIMIT_SECONDS)
        self._watchdog.start()

        self.store = StateStore.open(self.settings.db_path)
        self.async_store = AsyncStateStore(self.store)
        self.parquet = AsyncParquetWriter(ParquetWriter(self.settings.parquet_dir))
        await _ensure_ollama_running(self.settings.sources.ollama_base)
        self.ai_provider = await detect_provider(self.settings)
        log.info("ai_provider_ready", provider=self.ai_provider.name, model=self.ai_provider.model)

        self.health = HealthMonitor(self.clock, store=self.store)
        ingest_cfg = load_ingest_config()
        self.ingest = IngestService(
            self.settings, self.clock, store=self.store, parquet=self.parquet, health=self.health,
            ingest_config=ingest_cfg,
        )
        self.market_registry = self.ingest.markets
        self.book_registry = self.ingest.books
        self.portfolio_registry = PortfolioRegistry(default_bankroll=self.settings.paper.bankroll_per_strategy)

        geo = await self.ingest.geoblock_check()
        log.info("geoblock_status", blocked=geo.blocked, country=geo.country)

        self.risk_gateway = RiskGateway(self.settings.risk)
        self._raw_broker = await self._build_broker(acknowledge_real_money_risk)
        self.broker = _HealthGatedBroker(self._raw_broker, self.health, self.clock)

        try:
            self.ai_stack = await build_ai_stack(
                self.settings, self.ai_provider, self.clock, self.market_registry.get
            )
        except Exception as exc:  # noqa: BLE001 - the AI layer is an enrichment, never a dependency
            log.error("ai_stack_build_failed", error=str(exc), exc_info=True)
            self.ai_stack = None

        self.variants = self._generate_variants()
        self.runner = await self._build_runner()

        recovery = RecoveryService(
            self.clock,
            market_registry=self.market_registry,
            book_registry=self.book_registry,
            portfolio_registry=self.portfolio_registry,
            kalshi_rest=self.ingest.kalshi_rest,
        )
        self._watchdog.beat("recovery")
        recovery_summary = await recovery.restore(self.store, self._raw_broker, self.runner)
        log.info("recovery_summary", detail=recovery_summary.render())

        await self.ingest.start()
        self.started_at = self.clock.now()
        self._install_signal_handlers()
        self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())

        self._watchdog.set_limit(RUN_HANG_LIMIT_SECONDS, "running")
        log.info("supervisor_boot_complete", mode=self.settings.mode.value)
        try:
            await self._main_loop()
        finally:
            await self._shutdown()

    async def _build_broker(self, acknowledge_real_money_risk: bool) -> Broker:
        assert self.ingest is not None and self.market_registry is not None
        assert self.book_registry is not None and self.portfolio_registry is not None and self.risk_gateway is not None

        if self.settings.mode is Mode.LIVE:
            rest_adapter = _KalshiLiveRestBridge(self.ingest.kalshi_rest)
            unmet = KalshiLiveBroker.preflight(
                self.settings, acknowledge_real_money_risk=acknowledge_real_money_risk, rest_adapter=rest_adapter
            )
            if unmet:
                log.error("live_trading_blocked", unmet=unmet)
                raise LiveTradingBlocked("refusing to start LIVE: " + "; ".join(unmet))
            log.warning("live_broker_armed", detail="KalshiLiveBroker constructed -- real orders can be submitted")
            return KalshiLiveBroker(
                settings=self.settings,
                clock=self.clock,
                rest_adapter=rest_adapter,
                risk_gateway=self.risk_gateway,
                market_provider=self._market_lookup,
                book_provider=self.book_registry.get,
                portfolio_provider=self.portfolio_registry.get_or_create,
                acknowledge_real_money_risk=acknowledge_real_money_risk,
                store=self.store,
            )

        latency_model = LatencyModel(
            signal_to_order_ms=self.settings.execution.signal_to_order_ms,
            network_latency_ms=self.settings.execution.network_latency_ms,
            processing_latency_ms=self.settings.execution.processing_latency_ms,
        )
        fill_model = build_limit_fill_model(self.settings.execution.limit_fill_model)
        broker_mode = self.settings.mode if self.settings.mode.allows_orders else Mode.PAPER
        store_bridge = _PaperBrokerStoreBridge(self.store) if self.store is not None else None
        return PaperBroker(
            mode=broker_mode,
            clock=self.clock,
            latency_model=latency_model,
            limit_fill_model=fill_model,
            fee_calculator=KalshiFeeCalculator(),
            book_provider=self.book_registry.get,
            market_provider=self._market_lookup,
            portfolio_provider=self.portfolio_registry.get_or_create,
            risk_gateway=self.risk_gateway,
            settings=self.settings,
            store=store_bridge,
        )

    def _market_lookup(self, canonical_id: str):
        """Registry first, then the persisted catalogue.

        The in-memory registry is a bounded LRU (``ingest.max_tracked_markets``) whose
        contents churn as each refresh cycle re-selects the tracked set. The runner's own
        market map has a different lifetime, so a strategy can legitimately decide on a
        market that has since been evicted - the broker then saw "unknown market" and
        refused the order. Nearly every rejection in the first live run was this, not a
        real liquidity or risk condition.

        The StateStore holds every market ingested, so falling back to it makes the
        broker's view a superset of every strategy's view.
        """
        market = self.market_registry.get(canonical_id) if self.market_registry else None
        if market is not None:
            return market
        if self.store is None:
            return None
        try:
            return self.store.get_market(canonical_id)
        except Exception:  # noqa: BLE001 - a lookup miss must never break order handling
            return None

    def _generate_variants(self) -> list[Any]:
        try:
            from marketlab.experiments.sweep import generate_variants
        except Exception as exc:  # noqa: BLE001 - experiments module may not exist yet
            log.warning(
                "experiments_module_unavailable",
                error=str(exc),
                detail="marketlab.experiments.sweep not importable; no strategy variants generated. "
                "Ingestion and paper accounting still run.",
            )
            return []
        try:
            return list(generate_variants(self.settings.strategies, self.settings.universes))
        except Exception as exc:  # noqa: BLE001
            log.error("generate_variants_failed", error=str(exc), exc_info=True)
            return []

    async def _build_runner(self) -> Any | None:
        try:
            from marketlab.experiments.runner import ExperimentRunner
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "experiments_runner_unavailable",
                error=str(exc),
                detail="marketlab.experiments.runner not importable; the strategy tournament will not "
                "run this session, but ingestion, the broker, and paper accounting still do.",
            )
            return None
        if self.store is None or self.broker is None or self.market_registry is None or self.book_registry is None:
            log.warning("experiment_runner_missing_dependency", detail="store/broker/registries not ready")
            return None
        try:
            runner = ExperimentRunner(
                self.settings,
                self.clock,
                self.store,
                self.broker,
                self.market_registry,
                self.book_registry,
                ai_provider=self.ai_provider,
                ai_stack=self.ai_stack,
                portfolio_registry=self.portfolio_registry,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("experiment_runner_construction_failed", error=str(exc), exc_info=True)
            return None
        if hasattr(runner, "load_or_create_sleeves"):
            try:
                await runner.load_or_create_sleeves(self.variants)
            except Exception as exc:  # noqa: BLE001
                log.error("load_or_create_sleeves_failed", error=str(exc), exc_info=True)
        return runner

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    async def _handle_event(self, event: Event) -> None:
        if isinstance(event, MarketUpdateEvent) and self.market_registry is not None:
            self.market_registry.upsert(event.market)
        elif isinstance(event, BookUpdateEvent) and self.book_registry is not None:
            self.book_registry.upsert(event.book)
            if self._raw_broker is not None and hasattr(self._raw_broker, "on_book_update"):
                with contextlib.suppress(Exception):
                    await self._raw_broker.on_book_update(event.book)
        elif isinstance(event, TradeEvent):
            if self._raw_broker is not None and hasattr(self._raw_broker, "on_trade"):
                with contextlib.suppress(Exception):
                    await self._raw_broker.on_trade(event.trade)
        elif isinstance(event, MarketStatusEvent) and self.market_registry is not None:
            existing = self.market_registry.get(event.canonical_id)
            if existing is not None:
                self.market_registry.upsert(existing.model_copy(update={"status": event.status}))
        elif isinstance(event, SettlementEvent) and self.portfolio_registry is not None:
            for portfolio in self.portfolio_registry.all().values():
                if any(k.startswith(event.canonical_id) for k in portfolio.positions):
                    portfolio.settle(event.canonical_id, event.winning_side)
            if self._raw_broker is not None and hasattr(self._raw_broker, "settle"):
                with contextlib.suppress(Exception):
                    await self._raw_broker.settle(event.canonical_id, event.winning_side)

        if self.runner is not None and hasattr(self.runner, "dispatch"):
            try:
                await self.runner.dispatch(event)
            except Exception as exc:  # noqa: BLE001 - a bad event must never kill a multi-day run
                log.error("runner_dispatch_failed", event_type=event.event_type, error=str(exc), exc_info=True)

    async def _on_tick(self, now: datetime) -> None:
        if self.runner is not None and hasattr(self.runner, "tick"):
            with contextlib.suppress(Exception):
                await self.runner.tick()
        timer = TimerEvent(
            event_time=now, first_seen_time=now, source="supervisor",
            interval_seconds=self.settings.paper.tick_seconds, tag="tick",
        )
        if self.runner is not None and hasattr(self.runner, "dispatch"):
            with contextlib.suppress(Exception):
                await self.runner.dispatch(timer)

    async def _on_snapshot(self, now: datetime) -> None:
        if self.runner is not None and hasattr(self.runner, "snapshot"):
            with contextlib.suppress(Exception):
                await self.runner.snapshot()
        if self.store is not None and self.portfolio_registry is not None and self.book_registry is not None:
            marks = self.book_registry.mark_prices()
            for portfolio in self.portfolio_registry.all().values():
                portfolio.mark(marks)
                with contextlib.suppress(Exception):
                    self.store.save_portfolio(portfolio, as_of=now)

    async def _on_promotion(self, now: datetime) -> None:
        """Construct the promotion engine and prove the wiring is live.

        ``PromotionEngine.evaluate(experiment, metrics, forecasts=())`` needs a real
        ``PromotionMetrics`` (frequency class, resolved-trade clustering, forward days,
        fill model, ...) per experiment -- a computation the experiments team owns and
        that is not part of this daemon's data model. Rather than fabricate that input
        (a wrong promotion decision is worse than a deferred one -- see
        ``docs/live_safety.md``'s "strategy holds status QUALIFIED or better" live
        gate), the supervisor only confirms the engine constructs cleanly against the
        live strategies config on this cadence; per-experiment evaluation is a job the
        experiments team runs with its own metrics pipeline. See the integration report
        for the full rationale.
        """
        if self.runner is None:
            return
        try:
            from marketlab.experiments.promotion import PromotionEngine
        except Exception as exc:  # noqa: BLE001
            log.warning("promotion_engine_unavailable", error=str(exc))
            return
        try:
            PromotionEngine(self.settings.strategies)
            log.info("promotion_engine_ready", detail="deferring per-experiment evaluate() to the experiments team's metrics pipeline")
        except Exception as exc:  # noqa: BLE001
            log.error("promotion_engine_construction_failed", error=str(exc), exc_info=True)

    async def _write_daily_report(self, now: datetime) -> None:
        if self.health is None or self.portfolio_registry is None:
            return
        try:
            from marketlab.analytics.reports import daily_json_report
        except Exception as exc:  # noqa: BLE001
            log.warning("reports_module_unavailable", error=str(exc))
            return

        alive, dead = self.portfolio_registry.alive_count(self.settings.risk.death_floor)
        data_health = self.health.summary(now)
        venue_health = {k: v for k, v in data_health.items() if k.startswith(("kalshi", "poly"))}

        best_net_pnl: dict[str, Any] = {}
        best_risk_adjusted: dict[str, Any] = {}
        largest_drawdown: dict[str, Any] = {}
        if self.runner is not None and hasattr(self.runner, "leaderboard"):
            try:
                board = list(self.runner.leaderboard())
                if board:
                    top = max(board, key=lambda r: getattr(r, "net_pnl", Decimal(0)) or Decimal(0))
                    best_net_pnl = {
                        "strategy_id": getattr(top, "strategy_id", ""),
                        "net_pnl": str(getattr(top, "net_pnl", "")),
                    }
                    risk_top = max(
                        board, key=lambda r: getattr(r, "return_per_max_exposure", 0.0) or 0.0
                    )
                    best_risk_adjusted = {
                        "strategy_id": getattr(risk_top, "strategy_id", ""),
                        "return_per_max_exposure": getattr(risk_top, "return_per_max_exposure", None),
                    }
                    dd_top = max(board, key=lambda r: getattr(r, "max_drawdown", Decimal(0)) or Decimal(0))
                    largest_drawdown = {
                        "strategy_id": getattr(dd_top, "strategy_id", ""),
                        "max_drawdown": str(getattr(dd_top, "max_drawdown", "")),
                    }
            except Exception as exc:  # noqa: BLE001
                log.warning("daily_report_leaderboard_failed", error=str(exc))

        try:
            path = daily_json_report(
                report_date=now.date(),
                strategies_running=len(self.portfolio_registry),
                strategies_alive=alive,
                strategies_dead=dead,
                best_net_pnl=best_net_pnl,
                best_risk_adjusted=best_risk_adjusted,
                largest_drawdown=largest_drawdown,
                data_health=data_health,
                venue_health=venue_health,
                output_dir=self.settings.reports_dir,
            )
            log.info("daily_report_written", path=str(path))
        except Exception as exc:  # noqa: BLE001
            log.error("daily_report_failed", error=str(exc), exc_info=True)

    async def _main_loop(self) -> None:
        assert self.ingest is not None
        tick_interval = self.settings.paper.tick_seconds
        snapshot_interval = self.settings.paper.snapshot_seconds
        last_tick = self.clock.now()
        last_snapshot = self.clock.now()
        last_promotion = self.clock.now()

        while not self._stop_event.is_set():
            try:
                try:
                    event = await asyncio.wait_for(self.ingest.queue.get(), timeout=MAIN_LOOP_QUEUE_TIMEOUT)
                except TimeoutError:
                    event = None
                if event is not None:
                    await self._handle_event(event)
                    self.events_processed += 1
                    # Hand control back to the scheduler periodically. When the queue is
                    # never empty, `queue.get()` completes synchronously and dispatching
                    # to ~400 sleeves is pure CPU, so this loop can monopolise the event
                    # loop indefinitely: ingest tasks stop being serviced and the
                    # heartbeat never fires, which makes a saturated daemon look like a
                    # dead one. An explicit yield is the only thing that breaks that.
                    if self.events_processed % _YIELD_EVERY_N_EVENTS == 0:
                        await asyncio.sleep(0)

                now = self.clock.now()
                if (now - last_tick).total_seconds() >= tick_interval:
                    await self._on_tick(now)
                    last_tick = now
                if (now - last_snapshot).total_seconds() >= snapshot_interval:
                    await self._on_snapshot(now)
                    last_snapshot = now
                if (now - last_promotion).total_seconds() >= PROMOTION_INTERVAL_SECONDS:
                    await self._on_promotion(now)
                    last_promotion = now
                report_date = now.date()
                if self._last_report_date != report_date:
                    await self._write_daily_report(now)
                    self._last_report_date = report_date
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a single bad event must never kill the run
                log.error("main_loop_error", error=str(exc), exc_info=True)

    # ------------------------------------------------------------------
    # heartbeat / status
    # ------------------------------------------------------------------

    def build_status(self) -> SupervisorStatus:
        import os

        now = self.clock.now()
        uptime = (now - self.started_at).total_seconds() if self.started_at else 0.0
        alive, dead = (0, 0)
        if self.portfolio_registry is not None:
            alive, dead = self.portfolio_registry.alive_count(self.settings.risk.death_floor)
        trading_allowed, trading_detail = (True, "")
        feed_health: dict[str, Any] = {}
        if self.health is not None:
            trading_allowed, trading_detail = self.health.trading_allowed(now)
            feed_health = self.health.summary(now)
        return SupervisorStatus(
            pid=os.getpid(),
            mode=self.settings.mode.value,
            started_at=self.started_at.isoformat() if self.started_at else None,
            uptime_seconds=round(uptime, 1),
            events_processed=self.events_processed,
            markets_tracked=len(self.market_registry) if self.market_registry is not None else 0,
            sleeves_alive=alive,
            sleeves_dead=dead,
            orders_submitted=self.broker.submit_count if self.broker is not None else 0,
            orders_blocked=self.broker.blocked_count if self.broker is not None else 0,
            trading_allowed=trading_allowed,
            trading_detail=trading_detail,
            last_heartbeat=now.isoformat(),
            feed_health=feed_health,
            ai=self._ai_status(),
            risk_rejects=dict(self.risk_gateway.reject_counts) if self.risk_gateway is not None else {},
        )

    def _ai_status(self) -> dict[str, Any]:
        if self.ai_stack is None:
            return {"tiers": {}, "arms": []}
        out: dict[str, Any] = {
            "tiers": {k: f"{v.name}:{v.model}" for k, v in self.ai_stack.providers.items()},
            "arms": self.ai_stack.available_stacks(),
            "evidence": self.ai_stack.evidence.sizes(),
        }
        openai = self.ai_stack.providers.get("openai")
        ledger = getattr(openai, "ledger", None)
        if ledger is not None:
            out["openai_spent_today_usd"] = str(ledger.spent_today())
            out["openai_daily_cap_usd"] = str(ledger.daily_cap)
        return out

    def _status_path(self) -> Path:
        return self.settings.data_dir / "daemon_status.json"

    async def _heartbeat_loop(self) -> None:
        # The heartbeat is how an operator knows the run is alive. It must never die
        # quietly: a bare `ensure_future` task that raises is never awaited, so the
        # exception is swallowed and the daemon looks healthy while reporting nothing.
        # Every iteration is therefore individually guarded.
        log.info("heartbeat_loop_started", interval_seconds=HEARTBEAT_INTERVAL_SECONDS)
        while not self._stop_event.is_set():
            await self.clock.sleep(HEARTBEAT_INTERVAL_SECONDS)
            if self._stop_event.is_set():
                break
            if self._watchdog is not None:
                self._watchdog.beat()
            try:
                status = self.build_status()
                self._last_heartbeat = self.clock.now()
                log.info(
                    "heartbeat",
                    uptime_seconds=status.uptime_seconds,
                    events_processed=status.events_processed,
                    markets_tracked=status.markets_tracked,
                    sleeves_alive=status.sleeves_alive,
                    sleeves_dead=status.sleeves_dead,
                    orders_submitted=status.orders_submitted,
                    trading_allowed=status.trading_allowed,
                )
                self._status_path().write_text(status.to_json(), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                log.error("heartbeat_failed", error=str(exc), exc_info=True)

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        def _handler(signum: int, _frame: Any = None) -> None:
            log.warning("shutdown_signal_received", signal=signum)
            loop.call_soon_threadsafe(self._stop_event.set)

        for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            with contextlib.suppress(ValueError, OSError, RuntimeError):
                signal.signal(sig, _handler)

    def request_shutdown(self) -> None:
        self._stop_event.set()

    async def _shutdown(self) -> None:
        log.info("supervisor_shutdown_begin")
        if self._watchdog is not None:
            self._watchdog.stop()
        if self.ingest is not None:
            with contextlib.suppress(Exception):
                await self.ingest.stop()
        if self.parquet is not None:
            with contextlib.suppress(Exception):
                await self.parquet.flush_all()
        if self.store is not None and self.portfolio_registry is not None:
            now = self.clock.now()
            for portfolio in self.portfolio_registry.all().values():
                with contextlib.suppress(Exception):
                    self.store.save_portfolio(portfolio, as_of=now)
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._heartbeat_task
        if self.ai_provider is not None:
            with contextlib.suppress(Exception):
                await self.ai_provider.close()
        if self.ai_stack is not None:
            for provider in self.ai_stack.providers.values():
                if provider is not self.ai_provider:
                    with contextlib.suppress(Exception):
                        await provider.close()
        if self.store is not None:
            with contextlib.suppress(Exception):
                self.store.close()
        with contextlib.suppress(Exception):
            self._status_path().unlink(missing_ok=True)
        log.info("supervisor_shutdown_complete", events_processed=self.events_processed)


class _PaperBrokerStoreBridge:
    """Adapts :class:`StateStore` to ``PaperBroker``'s ``StateStoreLike`` protocol.

    A genuine contract mismatch between the execution and storage teams' frozen
    modules: ``PaperBroker.StateStoreLike`` expects ``save_order(order)``,
    ``save_fill(fill)`` and ``load_open_orders() -> list[Order]``, but the real
    ``StateStore`` requires an ``experiment_id`` on both save methods and has no
    ``load_open_orders`` at all (its equivalent is ``open_orders(experiment_id=None)``).
    Calling ``PaperBroker(store=a_real_StateStore)`` directly would raise ``TypeError``
    on the very first order. This bridge fixes it on our side without touching either
    frozen file -- see the integration report for the precise mismatch.
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self._experiment_by_order: dict[str, str] = {}
        #: order_id -> fills seen before that order row existed. See save_fill.
        self._pending_fills: dict[str, list[Any]] = {}

    def save_order(self, order: Any) -> None:
        self._experiment_by_order[order.order_id] = order.experiment_id
        try:
            self._store.save_order(order, order.experiment_id)
        except Exception as exc:  # noqa: BLE001
            log.error("persist_order_failed", order_id=order.order_id, error=str(exc))
            return
        # The order row now exists, so any fills that arrived early can be written.
        for fill in self._pending_fills.pop(order.order_id, []):
            self._write_fill(fill, order.experiment_id)

    def save_fill(self, fill: Any) -> None:
        """Buffer fills that arrive before their order row exists.

        ``PaperBroker._execute_market`` persists each fill inside the execution helper but
        only persists the *order* after that helper returns, so on the filling path the
        fill reaches storage first. ``fills.order_id`` has a foreign key to
        ``orders(order_id)``, so the insert raised ``IntegrityError`` - which propagated
        out of the broker, into the strategy's dispatch, and got the strategy counted as
        failing. Every filled order in the first live run was lost this way while the
        rejected ones (which never fill) persisted fine, so the tables looked like
        "nothing ever fills" rather than "the writes are failing".
        """
        experiment_id = self._experiment_by_order.get(fill.order_id)
        if experiment_id is None:
            self._pending_fills.setdefault(fill.order_id, []).append(fill)
            return
        self._write_fill(fill, experiment_id)

    def _write_fill(self, fill: Any, experiment_id: str) -> None:
        # A persistence failure must never propagate into strategy code.
        try:
            self._store.save_fill(fill, experiment_id)
        except Exception as exc:  # noqa: BLE001
            log.error("persist_fill_failed", order_id=fill.order_id, error=str(exc))

    def load_open_orders(self) -> list[Any]:
        return list(self._store.open_orders())


class _KalshiLiveRestBridge:
    """Adapts :class:`KalshiRestAdapter` to the ``KalshiRestAdapterLike`` protocol
    :class:`KalshiLiveBroker` expects (a single authenticated ``request()`` coroutine).

    ``KalshiRestAdapter`` (owned by the Kalshi-adapter team) deliberately implements no
    order-placement methods and exposes no public generic ``request()`` -- see its
    module docstring. This bridge is the minimal glue needed to reuse its signing/auth
    machinery for ``POST /portfolio/orders`` and friends without duplicating RSA-PSS
    signing here. See the final integration report for why this reaches into a couple
    of that adapter's private members rather than a public API.
    """

    def __init__(self, rest_adapter: Any) -> None:
        self._rest = rest_adapter

    async def request(
        self, method: str, path: str, *, json: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        rest = self._rest
        headers = rest._auth_headers(method, path)  # noqa: SLF001 - see class docstring
        http = rest._http  # noqa: SLF001
        if method.upper() == "GET":
            return await http.get_json(path, params=params, headers=headers)
        if method.upper() == "POST":
            return await http.post_json(path, json_body=json, headers=headers)
        # DELETE and anything else: use the underlying httpx client directly, since
        # HttpAdapter only exposes get_json/post_json wrappers.
        return await rest._get(path, params=params, auth=True) if method.upper() == "GET" else await self._raw(method, path, json, params, headers)

    async def _raw(
        self, method: str, path: str, json_body: dict[str, Any] | None, params: dict[str, Any] | None, headers: dict[str, str] | None
    ) -> dict[str, Any]:
        rest = self._rest
        resp = await rest._http._client.request(method, path, params=params, headers=headers, json=json_body)  # noqa: SLF001
        resp.raise_for_status()
        return resp.json() if resp.content else {}
