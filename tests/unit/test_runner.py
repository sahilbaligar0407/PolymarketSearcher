"""Tests for marketlab.experiments.runner (and a few registry behaviours it leans on)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from marketlab.clock import SimulatedClock
from marketlab.core.broker import Broker, Mode
from marketlab.core.events import BookUpdateEvent, SettlementEvent
from marketlab.core.instruments import BookLevel, OrderBook, Side, Venue
from marketlab.core.orders import (
    Action,
    Fill,
    Order,
    OrderIntent,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from marketlab.core.strategy import Strategy
from marketlab.experiments.identity import ExperimentIdentity
from marketlab.experiments.registry import (
    ExperimentRegistry,
    IllegalTransitionError,
    LiveNotAllowedError,
)
from marketlab.experiments.runner import ExperimentRunner
from marketlab.experiments.sweep import VariantSpec
from marketlab.settings import Settings
from marketlab.storage.state import ExperimentStatus, StateStore

T0 = datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class RecordingStrategy(Strategy):
    """Counts every handler call it receives; never trades."""

    name = "recording"
    version = "1.0.0"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.book_update_calls = 0

    def on_book_update(self, event: BookUpdateEvent) -> None:
        self.book_update_calls += 1


class ExplodingStrategy(Strategy):
    """Raises on every book update - used to test failure isolation."""

    name = "exploding"
    version = "1.0.0"

    def on_book_update(self, event: BookUpdateEvent) -> None:
        raise RuntimeError("this strategy is broken on purpose")


class GreedyLossStrategy(Strategy):
    """Buys almost the whole sleeve once, then loses it all on settlement."""

    name = "greedy_loss"
    version = "1.0.0"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._bought = False

    def on_book_update(self, event: BookUpdateEvent) -> None:
        if self._bought:
            return
        self._bought = True
        self.emit(
            OrderIntent(
                strategy_id=self.strategy_id,
                experiment_id=self.experiment_id,
                canonical_id=event.book.canonical_id,
                venue=Venue.KALSHI,
                side=Side.YES,
                action=Action.BUY,
                quantity=50,
                order_type=OrderType.LIMIT,
                limit_price=Decimal("0.99"),
                time_in_force=TimeInForce.GTC,
                decision_time=self.now(),
                rationale="test: buy nearly everything",
            )
        )


class FakeBroker(Broker):
    """Fills every intent in full, immediately, at its limit price. No portfolio side
    effects - matching the real Broker ABC's contract that the runner owns the
    portfolio (see runner.py's module docstring)."""

    def __init__(self) -> None:
        self.mode = Mode.PAPER
        self._orders: dict[str, Order] = {}

    async def submit(self, intent: OrderIntent) -> Order:
        price = intent.limit_price if intent.limit_price is not None else Decimal("0.50")
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
            status=OrderStatus.FILLED,
            filled_quantity=intent.quantity,
            average_fill_price=price,
            decision_timestamp=intent.decision_time,
        )
        fill = Fill(
            order_id=order.order_id,
            canonical_id=intent.canonical_id,
            venue=intent.venue,
            side=intent.side,
            action=intent.action,
            price=price,
            quantity=intent.quantity,
            fee=Decimal(0),
            timestamp=intent.decision_time,
            is_maker=False,
        )
        order = order.model_copy(update={"fills": (fill,)})
        self._orders[order.order_id] = order
        return order

    async def cancel(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    async def open_orders(self, strategy_id: str | None = None) -> list[Order]:
        return []

    async def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)


class FakeMarketRegistry:
    """Minimal MarketRegistryLike: a hand-fed canonical_id -> universes map."""

    def __init__(self, mapping: dict[str, set[str]]) -> None:
        self._mapping = mapping

    def get(self, canonical_id: str):
        return None

    def universes_for(self, canonical_id: str):
        return self._mapping.get(canonical_id, set())


class FakeBookRegistry:
    def get(self, canonical_id: str):
        return None


def _book_event(canonical_id: str, at: datetime, bid: str = "0.49", ask: str = "0.51") -> BookUpdateEvent:
    book = OrderBook(
        canonical_id=canonical_id,
        venue=Venue.KALSHI,
        timestamp=at,
        bids=(BookLevel(price=Decimal(bid), size=100),),
        asks=(BookLevel(price=Decimal(ask), size=100),),
    )
    return BookUpdateEvent(event_time=at, first_seen_time=at, book=book)


def _settings(bankroll: str = "50.00") -> Settings:
    return Settings(
        strategies={"meta": {"bankroll_per_variant": bankroll}},
        universes={
            "universes": {
                "uni_1": {"category": "crypto", "available": True},
                "uni_2": {"category": "crypto", "available": True},
            }
        },
    )


def _variant(strategy_name: str, class_path: str, universe: str = "uni_1") -> VariantSpec:
    return VariantSpec(
        strategy_name=strategy_name,
        strategy_class_path=class_path,
        universe=universe,
        params={},
        evidence_class="E",
        trades=True,
        requires_ai=False,
    )


_RECORDING_PATH = "tests.unit.test_runner:RecordingStrategy"
_EXPLODING_PATH = "tests.unit.test_runner:ExplodingStrategy"
_GREEDY_LOSS_PATH = "tests.unit.test_runner:GreedyLossStrategy"


def _make_runner(db_path, market_map: dict[str, set[str]], clock: SimulatedClock, **kwargs) -> ExperimentRunner:
    store = StateStore(db_path)
    return ExperimentRunner(
        settings=_settings(),
        clock=clock,
        store=store,
        broker=FakeBroker(),
        market_registry=FakeMarketRegistry(market_map),
        book_registry=FakeBookRegistry(),
        # FakeBroker implements the bare Broker ABC, which has no portfolio concept, so
        # it never mutates a Portfolio. The real PaperBroker does (it has to - only the
        # broker knows each fill's fee), which is why the runner defaults to
        # broker_owns_portfolio=True. A portfolio-free fake must opt out explicitly or
        # every fill would silently go unrecorded here.
        broker_owns_portfolio=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Crash recovery - the headline test
# ---------------------------------------------------------------------------


async def test_restart_recovery_never_resets_bankroll(tmp_path) -> None:
    db_path = tmp_path / "marketlab.db"
    market_map = {"KXBTC-TEST": {"uni_1"}}
    clock = SimulatedClock(T0)

    runner1 = _make_runner(db_path, market_map, clock)
    variant = _variant("greedy_loss", _GREEDY_LOSS_PATH)
    await runner1.load_or_create_sleeves([variant])
    assert len(runner1._sleeves) == 1
    experiment_id = next(iter(runner1._sleeves))

    # Trigger a real fill: cash should now be well below the $50 starting bankroll.
    await runner1.dispatch(_book_event("KXBTC-TEST", clock.now()))
    sleeve1 = runner1._sleeves[experiment_id]
    assert sleeve1.portfolio.cash == Decimal("0.50")  # 50.00 - 50*0.99
    await runner1.snapshot()
    runner1.store.close()

    # "Restart": brand new runner, brand new StateStore pointed at the same file.
    clock2 = SimulatedClock(clock.now())
    runner2 = _make_runner(db_path, market_map, clock2)
    await runner2.load_or_create_sleeves([variant])
    assert experiment_id in runner2._sleeves
    sleeve2 = runner2._sleeves[experiment_id]

    assert sleeve2.portfolio.cash == sleeve1.portfolio.cash
    assert sleeve2.portfolio.cash != sleeve2.portfolio.initial_capital
    assert sleeve2.portfolio.initial_capital == Decimal("50.00")
    assert sleeve2.portfolio.positions.keys() == sleeve1.portfolio.positions.keys()
    for key, pos in sleeve1.portfolio.positions.items():
        assert sleeve2.portfolio.positions[key].quantity == pos.quantity
        assert sleeve2.portfolio.positions[key].average_price == pos.average_price
    runner2.store.close()


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


async def test_strategy_error_isolation_disables_after_n_failures(tmp_path) -> None:
    market_map = {"KXBTC-TEST": {"uni_1"}}
    clock = SimulatedClock(T0)
    runner = _make_runner(tmp_path / "db.sqlite", market_map, clock, max_consecutive_failures=3)

    variants = [
        _variant("exploding", _EXPLODING_PATH),
        _variant("recording", _RECORDING_PATH),
    ]
    await runner.load_or_create_sleeves(variants)
    assert len(runner._sleeves) == 2

    exploding_id = next(eid for eid, s in runner._sleeves.items() if s.variant.strategy_name == "exploding")
    recording_id = next(eid for eid, s in runner._sleeves.items() if s.variant.strategy_name == "recording")

    for _ in range(5):
        await runner.dispatch(_book_event("KXBTC-TEST", clock.now()))
        clock.advance(1)

    exploding_sleeve = runner._sleeves[exploding_id]
    recording_sleeve = runner._sleeves[recording_id]

    assert exploding_sleeve.status is ExperimentStatus.DISABLED
    assert exploding_sleeve.consecutive_failures >= 3
    stored = runner.store.get_experiment(exploding_id)
    assert stored is not None and stored.status is ExperimentStatus.DISABLED

    # The healthy sleeve must be completely unaffected: it saw every single event.
    assert recording_sleeve.status is ExperimentStatus.PAPER
    assert recording_sleeve.strategy.book_update_calls == 5
    runner.store.close()


# ---------------------------------------------------------------------------
# Death rule
# ---------------------------------------------------------------------------


async def test_dead_sleeve_stops_receiving_events_and_is_readable_from_storage(tmp_path) -> None:
    db_path = tmp_path / "db.sqlite"
    market_map = {"KXBTC-TEST": {"uni_1"}}
    clock = SimulatedClock(T0)
    runner = _make_runner(db_path, market_map, clock)

    variant = _variant("greedy_loss", _GREEDY_LOSS_PATH)
    await runner.load_or_create_sleeves([variant])
    experiment_id = next(iter(runner._sleeves))

    # 1. Buy almost everything.
    await runner.dispatch(_book_event("KXBTC-TEST", clock.now()))
    sleeve = runner._sleeves[experiment_id]
    assert sleeve.status is not ExperimentStatus.DEAD  # not dead yet, just fully invested

    # 2. Lose it all on settlement: the sleeve must now die.
    clock.advance(60)
    settlement = SettlementEvent(
        event_time=clock.now(), first_seen_time=clock.now(),
        canonical_id="KXBTC-TEST", venue=Venue.KALSHI, winning_side=Side.NO,
    )
    await runner.dispatch(settlement)
    assert sleeve.status is ExperimentStatus.DEAD
    final_equity = sleeve.portfolio.equity()
    assert final_equity < Decimal("1.00")

    # 3. Further book updates must not reach the now-dead strategy.
    clock.advance(60)
    await runner.dispatch(_book_event("KXBTC-TEST", clock.now()))
    # GreedyLossStrategy has no counter, so assert indirectly: no exception, and the
    # experiment status/portfolio are untouched by the extra dispatch.
    assert sleeve.status is ExperimentStatus.DEAD
    assert sleeve.portfolio.equity() == final_equity

    # 4. Still readable directly from storage, with its final equity intact.
    stored_experiment = runner.store.get_experiment(experiment_id)
    assert stored_experiment is not None
    assert stored_experiment.status is ExperimentStatus.DEAD
    stored_portfolio = runner.store.load_portfolio(experiment_id)
    assert stored_portfolio is not None
    assert stored_portfolio.equity() == final_equity
    runner.store.close()


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


async def test_events_route_only_to_subscribed_strategies(tmp_path) -> None:
    market_map = {"KXBTC-UNI1": {"uni_1"}}  # note: no mapping at all for a uni_2 market
    clock = SimulatedClock(T0)
    runner = _make_runner(tmp_path / "db.sqlite", market_map, clock)

    variants = [
        _variant("recording_uni1", _RECORDING_PATH, universe="uni_1"),
        _variant("recording_uni2", _RECORDING_PATH, universe="uni_2"),
    ]
    await runner.load_or_create_sleeves(variants)
    uni1_id = next(eid for eid, s in runner._sleeves.items() if s.variant.universe == "uni_1")
    uni2_id = next(eid for eid, s in runner._sleeves.items() if s.variant.universe == "uni_2")

    await runner.dispatch(_book_event("KXBTC-UNI1", clock.now()))

    assert runner._sleeves[uni1_id].strategy.book_update_calls == 1
    assert runner._sleeves[uni2_id].strategy.book_update_calls == 0
    runner.store.close()


# ---------------------------------------------------------------------------
# Registry: legal-transition table, idempotence, and cohort immutability
# ---------------------------------------------------------------------------


def _identity(universe: str = "btc_1h", start: datetime = T0) -> ExperimentIdentity:
    return ExperimentIdentity(
        strategy_name="momentum",
        strategy_version="1.0.0",
        git_commit="abc123",
        parameter_hash="deadbeef0000",
        market_universe=universe,
        venue="kalshi",
        data_version="data.v1",
        execution_model_version="exec.v1",
        feature_version="feat.v1",
        llm_model_id=None,
        prompt_hash=None,
        start_timestamp=start,
        starting_bankroll=Decimal("50.00"),
    )


def test_register_is_idempotent(tmp_path) -> None:
    store = StateStore(tmp_path / "db.sqlite")
    clock = SimulatedClock(T0)
    registry = ExperimentRegistry(store, clock)
    identity = _identity()

    first = registry.register(identity, {"lookback_seconds": 300}, "btc_1h")
    second = registry.register(identity, {"lookback_seconds": 300}, "btc_1h")
    assert first.experiment_id == second.experiment_id
    assert len(store.list_experiments()) == 1
    store.close()


def test_illegal_transition_raises(tmp_path) -> None:
    store = StateStore(tmp_path / "db.sqlite")
    clock = SimulatedClock(T0)
    registry = ExperimentRegistry(store, clock)
    identity = _identity()
    experiment = registry.register(identity, {}, "btc_1h")

    with pytest.raises(IllegalTransitionError):
        registry.transition(experiment.experiment_id, ExperimentStatus.CHAMPION, "skip the queue")
    store.close()


def test_dead_is_terminal(tmp_path) -> None:
    store = StateStore(tmp_path / "db.sqlite")
    clock = SimulatedClock(T0)
    registry = ExperimentRegistry(store, clock)
    experiment = registry.register(_identity(), {}, "btc_1h")
    registry.transition(experiment.experiment_id, ExperimentStatus.BACKTESTING, "go")
    registry.transition(experiment.experiment_id, ExperimentStatus.PAPER, "go")
    registry.transition(experiment.experiment_id, ExperimentStatus.DEAD, "equity below floor")

    with pytest.raises(IllegalTransitionError):
        registry.transition(experiment.experiment_id, ExperimentStatus.BACKTESTING, "resurrect?")
    store.close()


def test_live_transition_requires_allow_live(tmp_path) -> None:
    store = StateStore(tmp_path / "db.sqlite")
    clock = SimulatedClock(T0)
    registry = ExperimentRegistry(store, clock)
    experiment = registry.register(_identity(), {}, "btc_1h")
    for status in (ExperimentStatus.BACKTESTING, ExperimentStatus.PAPER, ExperimentStatus.QUALIFIED, ExperimentStatus.CHAMPION):
        registry.transition(experiment.experiment_id, status, "advance")

    with pytest.raises(LiveNotAllowedError):
        registry.transition(experiment.experiment_id, ExperimentStatus.LIVE_SMALL, "go live")

    # With explicit human sign-off, the same transition succeeds.
    updated = registry.transition(experiment.experiment_id, ExperimentStatus.LIVE_SMALL, "go live", allow_live=True)
    assert updated.status is ExperimentStatus.LIVE_SMALL
    store.close()


def test_new_cohort_preserves_dead_record_and_gets_a_new_id(tmp_path) -> None:
    store = StateStore(tmp_path / "db.sqlite")
    clock = SimulatedClock(T0)
    registry = ExperimentRegistry(store, clock)
    experiment = registry.register(_identity(), {"lookback_seconds": 300}, "btc_1h")
    registry.transition(experiment.experiment_id, ExperimentStatus.BACKTESTING, "go")
    registry.transition(experiment.experiment_id, ExperimentStatus.PAPER, "go")
    registry.transition(experiment.experiment_id, ExperimentStatus.DEAD, "equity below floor")

    before = store.get_experiment(experiment.experiment_id)
    assert before is not None

    clock.advance(3600)
    new_identity = registry.new_cohort(experiment.experiment_id)
    assert new_identity.experiment_id != experiment.experiment_id
    assert new_identity.strategy_name == "momentum"
    assert new_identity.parameter_hash == "deadbeef0000"
    assert new_identity.start_timestamp == clock.now()

    # The dead record itself must be completely untouched by computing a new cohort.
    after = store.get_experiment(experiment.experiment_id)
    assert after is not None
    assert after == before
    assert after.status is ExperimentStatus.DEAD

    # Registering the new cohort creates a second, independent row - not an edit.
    new_experiment = registry.register(new_identity, {"lookback_seconds": 300}, "btc_1h")
    assert new_experiment.experiment_id != experiment.experiment_id
    assert len(store.list_experiments(strategy="momentum")) == 2
    store.close()


def test_new_cohort_requires_dead_status(tmp_path) -> None:
    store = StateStore(tmp_path / "db.sqlite")
    clock = SimulatedClock(T0)
    registry = ExperimentRegistry(store, clock)
    experiment = registry.register(_identity(), {}, "btc_1h")

    with pytest.raises(ValueError):
        registry.new_cohort(experiment.experiment_id)
    store.close()


# ---------------------------------------------------------------------------
# AI stack wiring and decision audit
# ---------------------------------------------------------------------------


class ParamCapturingStrategy(Strategy):
    """Does nothing; the test inspects the params it was constructed with."""

    name = "param_capture"
    version = "1.0.0"


def _ai_variant(stack: str) -> VariantSpec:
    return VariantSpec(
        strategy_name="param_capture",
        strategy_class_path="tests.unit.test_runner:ParamCapturingStrategy",
        universe="uni_1",
        params={"ai_stack": stack},
        evidence_class="F",
        trades=True,
        requires_ai=True,
    )


async def test_ai_arms_get_their_tiers_and_unavailable_arms_are_never_registered(tmp_path) -> None:
    from marketlab.ai.provider import DisabledProvider
    from marketlab.ai.stack import AIStack, AssessmentCache, EvidenceCache

    clock = SimulatedClock(T0)
    local, openai = DisabledProvider(), DisabledProvider()
    stack = AIStack({"local": local, "openai": openai}, EvidenceCache(lambda c: None), AssessmentCache(clock))
    runner = _make_runner(tmp_path / "m.db", {}, clock, ai_stack=stack)
    await runner.load_or_create_sleeves([_ai_variant("local"), _ai_variant("hybrid"), _ai_variant("jev")])

    assert len(runner._sleeves) == 2
    by_stack = {sl.strategy.params["ai_stack"]: sl.strategy.params for sl in runner._sleeves.values()}
    assert set(by_stack) == {"local", "hybrid"}
    assert by_stack["local"]["llm_provider"] is local and by_stack["local"]["escalation_provider"] is None
    assert by_stack["hybrid"]["escalation_provider"] is openai
    assert by_stack["hybrid"]["retrieval_store"] is stack.evidence
    # Collaborators never leak into the hashed, persisted parameters.
    stored = [runner.store.get_experiment(eid) for eid in runner._sleeves]
    assert all("llm_provider" not in (e.parameters or {}) for e in stored)
    assert {e.llm_model_id for e in stored} == {"disabled:none", "disabled:none+disabled:none"}


async def test_filled_orders_persist_their_rationale(tmp_path) -> None:
    clock = SimulatedClock(T0)
    runner = _make_runner(tmp_path / "m.db", {"KXBTC-TEST": {"uni_1"}}, clock)
    await runner.load_or_create_sleeves([_variant("greedy_loss", _GREEDY_LOSS_PATH)])
    await runner.dispatch(_book_event("KXBTC-TEST", T0))
    rows = runner.store._conn.execute("SELECT rationale, filled_quantity FROM decisions").fetchall()
    assert [tuple(r) for r in rows] == [("test: buy nearly everything", 50)]


async def test_sleeve_context_lists_only_its_universe(tmp_path) -> None:
    from marketlab.core.events import MarketUpdateEvent
    from marketlab.core.instruments import MarketStatus, NormalizedMarket

    clock = SimulatedClock(T0)
    runner = _make_runner(tmp_path / "m.db", {"kalshi:in": {"uni_1"}, "kalshi:out": {"uni_2"}}, clock)
    await runner.load_or_create_sleeves([_variant("recording", _RECORDING_PATH)])
    for cid in ("kalshi:in", "kalshi:out", "poly:x"):
        market = NormalizedMarket(canonical_id=cid, venue=Venue.KALSHI, venue_market_id=cid, event_id="e",
                                  title=cid, status=MarketStatus.OPEN)
        await runner.dispatch(MarketUpdateEvent(event_time=T0, first_seen_time=T0, market=market))
    sleeve = next(iter(runner._sleeves.values()))
    assert [m.canonical_id for m in sleeve.strategy.ctx.markets()] == ["kalshi:in"]
    assert sleeve.strategy.ctx.market("kalshi:out") is not None  # lookups stay global
