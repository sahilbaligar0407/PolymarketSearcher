"""Acceptance tests for marketlab.execution.paper_broker.PaperBroker.

Also hosts the required precondition tests for marketlab.execution.kalshi_live -
KalshiLiveBroker has no PRD-mandated dedicated test file, and this team's file list does
not include one, so its "structurally impossible to reach by accident" guarantees are
verified in the class at the bottom of this file instead.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from marketlab.clock import SimulatedClock
from marketlab.core.broker import Mode
from marketlab.core.instruments import (
    BookLevel,
    Category,
    MarketStatus,
    NormalizedMarket,
    OrderBook,
    Side,
    Trade,
    Venue,
)
from marketlab.core.orders import (
    Action,
    OrderIntent,
    OrderStatus,
    OrderType,
    RejectReason,
)
from marketlab.core.portfolio import Portfolio
from marketlab.execution.fill_models import TradeThroughFillModel
from marketlab.execution.kalshi_live import KalshiLiveBroker, LiveTradingBlocked
from marketlab.execution.latency import LatencyModel
from marketlab.execution.paper_broker import KalshiFeeCalculator, PaperBroker
from marketlab.execution.risk_gateway import RiskGateway
from marketlab.settings import ExecutionConfig, RiskConfig, Secrets, Settings

TS = datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


class ZeroFeeCalculator:
    def calculate(self, market, price, quantity, is_maker):
        return Decimal("0")


class FlatFeeCalculator:
    def __init__(self, fee_per_contract: Decimal) -> None:
        self.fee_per_contract = fee_per_contract

    def calculate(self, market, price, quantity, is_maker):
        return self.fee_per_contract * Decimal(quantity)


class FakeStore:
    """Minimal StateStoreLike double for the recovery test."""

    def __init__(self) -> None:
        self.orders: dict = {}
        self.fills: list = []

    def save_order(self, order) -> None:
        self.orders[order.order_id] = order

    def save_fill(self, fill) -> None:
        self.fills.append(fill)

    def load_open_orders(self) -> list:
        return [o for o in self.orders.values() if not o.is_terminal]


def _market(**kwargs) -> NormalizedMarket:
    defaults = dict(
        canonical_id="mkt-1",
        venue=Venue.KALSHI,
        venue_market_id="MKT-1",
        event_id="evt-1",
        title="Will X happen?",
        category=Category.POLITICS,
        status=MarketStatus.OPEN,
        min_order=1,
    )
    defaults.update(kwargs)
    return NormalizedMarket(**defaults)


def _book(ts=TS, bids=(), asks=(), canonical_id="mkt-1") -> OrderBook:
    return OrderBook(canonical_id=canonical_id, venue=Venue.KALSHI, timestamp=ts, bids=bids, asks=asks)


def _portfolio(**kwargs) -> Portfolio:
    defaults = dict(experiment_id="exp-1", strategy_id="strat-1")
    defaults.update(kwargs)
    return Portfolio(**defaults)


def _intent(**kwargs) -> OrderIntent:
    defaults = dict(
        strategy_id="strat-1",
        experiment_id="exp-1",
        canonical_id="mkt-1",
        venue=Venue.KALSHI,
        side=Side.YES,
        action=Action.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
        decision_time=TS,
        rationale="unit test rationale",
    )
    defaults.update(kwargs)
    return OrderIntent(**defaults)


def _permissive_risk_gateway() -> RiskGateway:
    return RiskGateway(
        RiskConfig(
            max_single_event_loss_pct=Decimal("1"),
            max_strategy_exposure_pct=Decimal("1"),
            max_category_exposure_pct=Decimal("1"),
            max_correlated_cluster_pct=Decimal("1"),
            daily_loss_pause_pct=Decimal("1"),
            total_drawdown_pause_pct=Decimal("1"),
            strict_audit=False,
        )
    )


def _make_broker(
    *,
    clock,
    market_provider,
    book_provider,
    portfolio_provider,
    risk_gateway=None,
    latency_model=None,
    limit_fill_model=None,
    fee_calculator=None,
    settings=None,
    store=None,
    mode=Mode.PAPER,
) -> PaperBroker:
    return PaperBroker(
        mode=mode,
        clock=clock,
        latency_model=latency_model or LatencyModel(0, 0, 0),
        limit_fill_model=limit_fill_model or TradeThroughFillModel(),
        fee_calculator=fee_calculator or ZeroFeeCalculator(),
        book_provider=book_provider,
        market_provider=market_provider,
        portfolio_provider=portfolio_provider,
        risk_gateway=risk_gateway or _permissive_risk_gateway(),
        settings=settings or Settings(mode=mode),
        store=store,
    )


# ---------------------------------------------------------------------------
# Mode / MARKET_CLOSED / STALE_DATA / cash
# ---------------------------------------------------------------------------


async def test_live_and_data_only_modes_refused():
    for mode in (Mode.LIVE, Mode.DATA_ONLY):
        broker = _make_broker(
            clock=SimulatedClock(TS),
            market_provider=lambda cid: _market(),
            book_provider=lambda cid: _book(),
            portfolio_provider=lambda eid: _portfolio(),
            mode=mode,
        )
        order = await broker.submit(_intent())
        assert order.status is OrderStatus.REJECTED
        assert order.reject_reason is RejectReason.MODE_FORBIDDEN


async def test_stale_book_rejected_and_counted():
    market = _market()
    stale_book = _book(ts=TS - timedelta(seconds=100))
    settings = Settings(mode=Mode.PAPER, execution=ExecutionConfig(max_book_age_seconds=5.0))
    broker = _make_broker(
        clock=SimulatedClock(TS),
        market_provider=lambda cid: market,
        book_provider=lambda cid: stale_book,
        portfolio_provider=lambda eid: _portfolio(),
        settings=settings,
    )
    order = await broker.submit(_intent())
    assert order.status is OrderStatus.REJECTED
    assert order.reject_reason is RejectReason.STALE_DATA
    assert broker.stale_data_skips == 1


async def test_closed_market_rejected():
    market = _market(status=MarketStatus.CLOSED)
    broker = _make_broker(
        clock=SimulatedClock(TS),
        market_provider=lambda cid: market,
        book_provider=lambda cid: _book(),
        portfolio_provider=lambda eid: _portfolio(),
    )
    order = await broker.submit(_intent())
    assert order.status is OrderStatus.REJECTED
    assert order.reject_reason is RejectReason.MARKET_CLOSED


async def test_insufficient_cash_rejected():
    market = _market()
    book = _book(asks=(BookLevel(price=Decimal("0.50"), size=100),))
    portfolio = _portfolio(cash=Decimal("1.00"))
    broker = _make_broker(
        clock=SimulatedClock(TS),
        market_provider=lambda cid: market,
        book_provider=lambda cid: book,
        portfolio_provider=lambda eid: portfolio,
    )
    intent = _intent(order_type=OrderType.LIMIT, limit_price=Decimal("0.50"), quantity=10)
    order = await broker.submit(intent)
    assert order.status is OrderStatus.REJECTED
    assert order.reject_reason is RejectReason.INSUFFICIENT_CASH
    assert broker.risk_gate_skips == 1


# ---------------------------------------------------------------------------
# Latency / no look-ahead
# ---------------------------------------------------------------------------


async def test_no_lookahead_uses_book_at_or_before_simulated_arrival():
    """A book sequence that would give a better price only under look-ahead."""
    latency_model = LatencyModel(signal_to_order_ms=500, network_latency_ms=300, processing_latency_ms=200)
    assert latency_model.total_latency_ms == 1000  # arrival = decision + 1s

    market = _market()
    book_old = _book(ts=TS, asks=(BookLevel(price=Decimal("0.90"), size=100),))
    book_new = _book(ts=TS + timedelta(seconds=2), asks=(BookLevel(price=Decimal("0.10"), size=100),))

    # book_provider always hands back "the current" snapshot, exactly as a real
    # REST/websocket feed would - the broker must still refuse to use it if it's from
    # after the simulated arrival time.
    def book_provider(cid):
        return book_new

    portfolio = _portfolio()
    broker = _make_broker(
        clock=SimulatedClock(TS),
        market_provider=lambda cid: market,
        book_provider=book_provider,
        portfolio_provider=lambda eid: portfolio,
        latency_model=latency_model,
    )
    await broker.on_book_update(book_old)
    await broker.on_book_update(book_new)

    order = await broker.submit(_intent(order_type=OrderType.MARKET, quantity=10))

    assert order.status is OrderStatus.FILLED
    assert order.average_fill_price == Decimal("0.90"), "must use the worse, pre-arrival price"
    assert order.book_timestamp_used == TS
    assert order.simulated_exchange_arrival_timestamp == TS + timedelta(seconds=1)


# ---------------------------------------------------------------------------
# Fees
# ---------------------------------------------------------------------------


async def test_fees_charged_and_reduce_cash():
    market = _market()
    book = _book(asks=(BookLevel(price=Decimal("0.50"), size=100),))
    portfolio = _portfolio(cash=Decimal("50.00"))
    broker = _make_broker(
        clock=SimulatedClock(TS),
        market_provider=lambda cid: market,
        book_provider=lambda cid: book,
        portfolio_provider=lambda eid: portfolio,
        fee_calculator=FlatFeeCalculator(Decimal("0.02")),
    )
    order = await broker.submit(_intent(order_type=OrderType.MARKET, quantity=10))
    assert order.status is OrderStatus.FILLED
    expected_fee = Decimal("0.02") * 10
    assert order.fees_paid == expected_fee
    assert portfolio.cash == Decimal("50.00") - Decimal("0.50") * 10 - expected_fee


def test_kalshi_fee_calculator_quadratic_formula():
    market = _market()  # default Fees(): taker_rate 0.07, formula kalshi_quadratic
    calc = KalshiFeeCalculator()
    fee = calc.calculate(market, Decimal("0.50"), 10, is_maker=False)
    # ceil_to_cent(0.07 * 10 * 0.5 * 0.5) = ceil_to_cent(0.175) = 0.18
    assert fee == Decimal("0.18")
    assert calc.calculate(market, Decimal("0.50"), 10, is_maker=True) == Decimal("0")  # maker_rate 0.00


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------


async def test_settlement_yes_win_cash_and_pnl():
    market = _market()
    book = _book(asks=(BookLevel(price=Decimal("0.40"), size=100),))
    portfolio = _portfolio(cash=Decimal("50.00"))
    broker = _make_broker(
        clock=SimulatedClock(TS),
        market_provider=lambda cid: market,
        book_provider=lambda cid: book,
        portfolio_provider=lambda eid: portfolio,
    )
    order = await broker.submit(_intent(order_type=OrderType.MARKET, quantity=10, side=Side.YES))
    assert order.status is OrderStatus.FILLED
    cash_after_buy = portfolio.cash
    assert cash_after_buy == Decimal("50.00") - Decimal("4.00")

    await broker.settle("mkt-1", Side.YES)
    assert portfolio.cash == cash_after_buy + Decimal("10.00")
    assert portfolio.realized_pnl == Decimal("6.00")


async def test_settlement_no_win_position_worth_zero():
    market = _market()
    book = _book(asks=(BookLevel(price=Decimal("0.40"), size=100),))
    portfolio = _portfolio(cash=Decimal("50.00"))
    broker = _make_broker(
        clock=SimulatedClock(TS),
        market_provider=lambda cid: market,
        book_provider=lambda cid: book,
        portfolio_provider=lambda eid: portfolio,
    )
    await broker.submit(_intent(order_type=OrderType.MARKET, quantity=10, side=Side.YES))
    cash_after_buy = portfolio.cash

    await broker.settle("mkt-1", Side.NO)  # YES position loses
    assert portfolio.cash == cash_after_buy  # no payout
    pos = portfolio.positions[Portfolio.key("mkt-1", Side.YES)]
    assert pos.quantity == 0


@pytest.mark.parametrize("winner", [Side.YES, Side.NO])
async def test_yes_and_no_parity_settles_to_exactly_one_dollar_per_contract(winner):
    market = _market()
    book = _book(
        asks=(BookLevel(price=Decimal("0.40"), size=100),),
        bids=(BookLevel(price=Decimal("0.55"), size=100),),
    )
    portfolio = _portfolio(cash=Decimal("50.00"))
    broker = _make_broker(
        clock=SimulatedClock(TS),
        market_provider=lambda cid: market,
        book_provider=lambda cid: book,
        portfolio_provider=lambda eid: portfolio,
    )
    await broker.submit(_intent(order_type=OrderType.MARKET, quantity=10, side=Side.YES, action=Action.BUY))
    await broker.submit(_intent(order_type=OrderType.MARKET, quantity=10, side=Side.NO, action=Action.BUY))
    cash_before_settle = portfolio.cash

    await broker.settle("mkt-1", winner)

    assert portfolio.cash - cash_before_settle == Decimal("10.00")


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


async def test_cancel_removes_resting_order():
    market = _market()
    book = _book(asks=(BookLevel(price=Decimal("0.60"), size=5),))  # too expensive to cross a 0.50 buy limit
    portfolio = _portfolio()
    broker = _make_broker(
        clock=SimulatedClock(TS),
        market_provider=lambda cid: market,
        book_provider=lambda cid: book,
        portfolio_provider=lambda eid: portfolio,
    )
    order = await broker.submit(
        _intent(order_type=OrderType.LIMIT, limit_price=Decimal("0.50"), quantity=10)
    )
    assert order.status is OrderStatus.OPEN
    assert len(await broker.open_orders()) == 1

    canceled = await broker.cancel(order.order_id)
    assert canceled.status is OrderStatus.CANCELED
    assert broker.cancel_count == 1
    assert await broker.open_orders() == []


# ---------------------------------------------------------------------------
# Restart / recovery
# ---------------------------------------------------------------------------


async def test_restart_recovers_resting_orders_and_preserves_bankroll():
    market = _market()
    book = _book(asks=(BookLevel(price=Decimal("0.60"), size=5),))
    portfolio = _portfolio(cash=Decimal("37.00"))
    store = FakeStore()

    broker1 = _make_broker(
        clock=SimulatedClock(TS),
        market_provider=lambda cid: market,
        book_provider=lambda cid: book,
        portfolio_provider=lambda eid: portfolio,
        store=store,
    )
    order = await broker1.submit(
        _intent(order_type=OrderType.LIMIT, limit_price=Decimal("0.50"), quantity=10)
    )
    assert order.status is OrderStatus.OPEN

    # A brand-new broker instance, standing in for a process restart. Same store, and the
    # same (externally-persisted) Portfolio - PaperBroker itself never resets a bankroll.
    broker2 = _make_broker(
        clock=SimulatedClock(TS),
        market_provider=lambda cid: market,
        book_provider=lambda cid: book,
        portfolio_provider=lambda eid: portfolio,
        store=store,
    )
    broker2.load_open_orders(store)

    resumed = await broker2.open_orders()
    assert len(resumed) == 1
    assert resumed[0].order_id == order.order_id
    assert resumed[0].limit_price == Decimal("0.50")
    assert portfolio.cash == Decimal("37.00")

    await broker2.on_trade(
        Trade(canonical_id="mkt-1", venue=Venue.KALSHI, timestamp=TS, price=Decimal("0.48"), size=10)
    )
    finalized = await broker2.get_order(order.order_id)
    assert finalized.status is OrderStatus.FILLED


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


async def _run_scenario(seed: int) -> list:
    market = _market()
    book = _book(
        asks=(
            BookLevel(price=Decimal("0.55"), size=50),
            BookLevel(price=Decimal("0.60"), size=50),
        )
    )
    portfolio = _portfolio()
    latency_model = LatencyModel(
        signal_to_order_ms=100, network_latency_ms=50, processing_latency_ms=25, jitter_ms=10, seed=seed
    )
    broker = _make_broker(
        clock=SimulatedClock(TS),
        market_provider=lambda cid: market,
        book_provider=lambda cid: book,
        portfolio_provider=lambda eid: portfolio,
        latency_model=latency_model,
    )
    orders = []
    for _ in range(5):
        orders.append(await broker.submit(_intent(order_type=OrderType.MARKET, quantity=10)))
    return orders


def _normalize(order) -> tuple:
    dumped = order.model_dump(exclude={"order_id", "intent_id", "fills"})
    fills = tuple(f.model_dump(exclude={"fill_id", "order_id"}) for f in order.fills)
    return (dumped, fills)


async def test_determinism_same_seed_same_event_sequence_byte_identical():
    orders_a = await _run_scenario(seed=42)
    orders_b = await _run_scenario(seed=42)
    assert [_normalize(o) for o in orders_a] == [_normalize(o) for o in orders_b]


async def test_determinism_different_seed_can_diverge():
    orders_a = await _run_scenario(seed=1)
    orders_b = await _run_scenario(seed=2)
    # Not a hard guarantee for every possible seed pair, but with jitter_ms=10 over 5
    # draws the arrival timestamps overwhelmingly diverge - this is a sanity check that
    # the seed is actually wired into the jitter rather than being ignored.
    assert [_normalize(o) for o in orders_a] != [_normalize(o) for o in orders_b]


# ---------------------------------------------------------------------------
# KalshiLiveBroker: structurally-impossible-to-reach-by-accident preconditions.
# ---------------------------------------------------------------------------


class FakeKalshiRestAdapter:
    def __init__(self) -> None:
        self.calls: list = []

    async def request(self, method, path, *, json=None, params=None):
        self.calls.append((method, path, json, params))
        if method == "POST":
            return {"order": {"order_id": "kalshi-order-1"}}
        return {}


def _armed_settings() -> Settings:
    return Settings(
        mode=Mode.LIVE,
        secrets=Secrets(
            live_trading_enabled="YES_I_ACCEPT_REAL_LOSS",
            kalshi_api_key_id="key-123",
            kalshi_private_key_path="/tmp/key.pem",
        ),
    )


def _unarmed_settings() -> Settings:
    return Settings(
        mode=Mode.PAPER,
        secrets=Secrets(live_trading_enabled="NO", kalshi_api_key_id="", kalshi_private_key_path=""),
    )


def _kalshi_kwargs(**overrides):
    market = _market()
    book = _book(asks=(BookLevel(price=Decimal("0.50"), size=10),))
    kwargs = dict(
        clock=SimulatedClock(TS),
        rest_adapter=FakeKalshiRestAdapter(),
        risk_gateway=_permissive_risk_gateway(),
        market_provider=lambda cid: market,
        book_provider=lambda cid: book,
        portfolio_provider=lambda eid: _portfolio(),
        acknowledge_real_money_risk=True,
    )
    kwargs.update(overrides)
    return kwargs


class TestKalshiLiveBrokerPreconditions:
    def test_every_static_precondition_individually_blocks_construction(self):
        armed = _armed_settings()

        # Wrong mode.
        with pytest.raises(LiveTradingBlocked):
            KalshiLiveBroker(settings=_unarmed_settings(), **_kalshi_kwargs())

        # Not armed via env token.
        not_armed = Settings(
            mode=Mode.LIVE,
            secrets=Secrets(
                live_trading_enabled="NO", kalshi_api_key_id="key-123", kalshi_private_key_path="/tmp/key.pem"
            ),
        )
        with pytest.raises(LiveTradingBlocked):
            KalshiLiveBroker(settings=not_armed, **_kalshi_kwargs())

        # acknowledge_real_money_risk not passed (defaults False).
        with pytest.raises(LiveTradingBlocked):
            KalshiLiveBroker(settings=armed, **_kalshi_kwargs(acknowledge_real_money_risk=False))

        # No credentials.
        no_creds = Settings(
            mode=Mode.LIVE,
            secrets=Secrets(live_trading_enabled="YES_I_ACCEPT_REAL_LOSS", kalshi_api_key_id="", kalshi_private_key_path=""),
        )
        with pytest.raises(LiveTradingBlocked):
            KalshiLiveBroker(settings=no_creds, **_kalshi_kwargs())

        # No rest_adapter.
        with pytest.raises(LiveTradingBlocked):
            KalshiLiveBroker(settings=armed, **_kalshi_kwargs(rest_adapter=None))

    def test_preflight_lists_every_unmet_condition_without_constructing(self):
        unmet = KalshiLiveBroker.preflight(_unarmed_settings(), acknowledge_real_money_risk=False, rest_adapter=None)
        assert len(unmet) >= 4  # mode, arming, acknowledge flag, credentials (adapter too)
        assert KalshiLiveBroker.preflight(_armed_settings(), acknowledge_real_money_risk=True, rest_adapter=object()) == []

    async def test_fully_armed_construction_succeeds_and_places_an_order(self):
        adapter = FakeKalshiRestAdapter()
        broker = KalshiLiveBroker(settings=_armed_settings(), **_kalshi_kwargs(rest_adapter=adapter))
        assert KalshiLiveBroker.preflight(_armed_settings(), True, adapter) == []

        order = await broker.submit(_intent())
        assert order.status is OrderStatus.OPEN
        assert order.venue_order_id == "kalshi-order-1"
        assert adapter.calls  # the network call actually happened on the green path

    async def test_stale_data_rejects_without_any_network_call(self):
        adapter = FakeKalshiRestAdapter()
        stale_book = _book(ts=TS - timedelta(hours=1))
        broker = KalshiLiveBroker(
            **_kalshi_kwargs(rest_adapter=adapter, book_provider=lambda cid: stale_book),
            settings=_armed_settings(),
        )
        order = await broker.submit(_intent())
        assert order.status is OrderStatus.REJECTED
        assert order.reject_reason is RejectReason.STALE_DATA
        assert adapter.calls == []
        assert broker.network_calls_made == 0

    async def test_settings_disarmed_after_construction_blocks_submit(self):
        """Defense in depth: Settings is mutable, so re-check on every call."""
        settings = _armed_settings()
        adapter = FakeKalshiRestAdapter()
        broker = KalshiLiveBroker(settings=settings, **_kalshi_kwargs(rest_adapter=adapter))
        settings.mode = Mode.PAPER  # simulate a config drift after construction
        with pytest.raises(LiveTradingBlocked):
            await broker.submit(_intent())
        assert adapter.calls == []


async def test_daily_loss_pause_resets_on_the_next_day(tmp_path) -> None:
    """The daily-loss pause must be daily, not permanent.

    `_daily_pnl` previously returned CUMULATIVE P&L, so any sleeve that ever fell 10%
    below its starting bankroll was paused forever. Measured effect in production: 178,000
    of 178,318 orders in one 18-hour window refused with "daily P&L breaches
    daily_loss_pause_pct", and fill volume fell roughly eighteen-fold.

    The pause exists so the system can be diagnosed rather than spend the rest of the
    bankroll in a known-broken state - which requires that it reset.
    """
    from datetime import timedelta

    clock = SimulatedClock(TS)
    portfolio = _portfolio(experiment_id="EXP_DAILY")
    broker = _make_broker(
        clock=clock,
        market_provider=lambda cid: _market(),
        book_provider=lambda cid: _book(),
        portfolio_provider=lambda eid: portfolio,
    )

    # Put the sleeve well past the 10% daily-loss threshold on day one.
    portfolio.realized_pnl = Decimal("-8.00")
    portfolio.cash = Decimal("42.00")

    day_one = broker._daily_pnl(portfolio)
    assert day_one == Decimal(0), "first observation of a day is the day's own baseline"

    # Lose more within the same day -> the pause must engage.
    portfolio.realized_pnl = Decimal("-14.00")
    assert broker._daily_pnl(portfolio) == Decimal("-6.00")

    # Roll to the next UTC day: the day's loss resets even though cumulative P&L is
    # still deeply negative.
    clock.set(clock.now() + timedelta(days=1))
    assert broker._daily_pnl(portfolio) == Decimal(0), (
        "a new day must start from zero, otherwise the pause is permanent"
    )
    # ...and cumulative P&L is untouched: the sleeve's history is not rewritten.
    assert portfolio.realized_pnl == Decimal("-14.00")


async def test_book_history_is_trimmed_but_point_in_time_lookup_still_works() -> None:
    from datetime import timedelta

    from marketlab.execution import paper_broker as pb

    broker = _make_broker(
        clock=SimulatedClock(TS),
        market_provider=lambda cid: _market(),
        book_provider=lambda cid: None,
        portfolio_provider=lambda eid: _portfolio(),
    )
    for i in range(2000):  # one book a second for ~33 minutes
        await broker.on_book_update(_book(ts=TS + timedelta(seconds=i)))
    history = broker._book_history["mkt-1"]
    assert len(history) <= pb.BOOK_HISTORY_SECONDS + 2
    newest = TS + timedelta(seconds=1999)
    found = broker._book_as_of("mkt-1", newest - timedelta(milliseconds=500))
    assert found is not None and found.timestamp == newest - timedelta(seconds=1)
