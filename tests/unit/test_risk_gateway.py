"""Acceptance tests for marketlab.execution.risk_gateway.RiskGateway."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from marketlab.core.instruments import (
    BookLevel,
    Category,
    MarketStatus,
    NormalizedMarket,
    OrderBook,
    Side,
    Venue,
)
from marketlab.core.orders import Action, OrderIntent, OrderType, RejectReason
from marketlab.core.portfolio import Portfolio, Position
from marketlab.execution.risk_gateway import RiskGateway
from marketlab.settings import RiskConfig

TS = datetime(2026, 1, 1, tzinfo=UTC)


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


def _book(mid_bid=Decimal("0.49"), mid_ask=Decimal("0.51")) -> OrderBook:
    return OrderBook(
        canonical_id="mkt-1",
        venue=Venue.KALSHI,
        timestamp=TS,
        bids=(BookLevel(price=mid_bid, size=1000),),
        asks=(BookLevel(price=mid_ask, size=1000),),
    )


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
        order_type=OrderType.LIMIT,
        limit_price=Decimal("0.50"),
        decision_time=TS,
        rationale="unit test rationale",
    )
    defaults.update(kwargs)
    return OrderIntent(**defaults)


def _gateway(**overrides) -> RiskGateway:
    return RiskGateway(RiskConfig(**overrides))


# ---------------------------------------------------------------------------
# Hard block
# ---------------------------------------------------------------------------


def test_polymarket_global_hard_blocked_even_with_permissive_config():
    gw = _gateway(
        leverage=True,
        borrowing=True,
        martingale=True,
        strict_audit=False,
        max_single_event_loss_pct=Decimal("1"),
        max_strategy_exposure_pct=Decimal("1"),
        max_category_exposure_pct=Decimal("1"),
        max_correlated_cluster_pct=Decimal("1"),
        daily_loss_pause_pct=Decimal("1"),
        total_drawdown_pause_pct=Decimal("1"),
    )
    intent = _intent(venue=Venue.POLY_GLOBAL, rationale="")
    decision = gw.evaluate(intent, _portfolio(), _market(), _book(), {}, {}, Decimal(0))
    assert not decision.approved
    assert decision.reason is RejectReason.MODE_FORBIDDEN


# ---------------------------------------------------------------------------
# Cash / leverage
# ---------------------------------------------------------------------------


def test_insufficient_cash_rejected():
    gw = _gateway()
    portfolio = _portfolio(cash=Decimal("1.00"))
    intent = _intent(quantity=10, limit_price=Decimal("0.50"))  # cost 5.00 > cash 1.00
    decision = gw.evaluate(intent, portfolio, _market(), _book(), {}, {}, Decimal(0))
    assert not decision.approved
    assert decision.reason is RejectReason.INSUFFICIENT_CASH


# ---------------------------------------------------------------------------
# Venue minimum order size: reject, never resize
# ---------------------------------------------------------------------------


def test_below_min_order_rejected_and_never_resized():
    gw = _gateway()
    market = _market(min_order=50)
    intent = _intent(quantity=5)
    decision = gw.evaluate(intent, _portfolio(), market, _book(), {}, {}, Decimal(0))
    assert not decision.approved
    assert decision.reason is RejectReason.BELOW_MIN_ORDER
    assert intent.quantity == 5


def test_venue_min_above_risk_cap_is_skipped_not_resized():
    # min_order=1000 would cost $500 - resizing up to meet it would blow through every
    # exposure cap on a $50 sleeve. The gateway must reject outright either way.
    gw = _gateway(max_strategy_exposure_pct=Decimal("0.20"))
    market = _market(min_order=1000)
    intent = _intent(quantity=1)
    decision = gw.evaluate(intent, _portfolio(), market, _book(), {}, {}, Decimal(0))
    assert not decision.approved
    assert decision.reason is RejectReason.BELOW_MIN_ORDER
    assert intent.quantity == 1  # never bumped up to the venue minimum


# ---------------------------------------------------------------------------
# Exposure caps
# ---------------------------------------------------------------------------


def test_single_event_loss_cap_rejected():
    gw = _gateway(max_single_event_loss_pct=Decimal("0.04"))  # 4% of $50 = $2.00
    intent = _intent(quantity=10, limit_price=Decimal("0.50"))  # cost $5.00
    decision = gw.evaluate(intent, _portfolio(), _market(), _book(), {}, {}, Decimal(0))
    assert not decision.approved
    assert decision.reason is RejectReason.RISK_GATE
    assert gw.reject_counts.get("single_event_loss") == 1


def test_strategy_exposure_cap_rejected():
    gw = _gateway(max_strategy_exposure_pct=Decimal("0.10"), max_single_event_loss_pct=Decimal("1"))
    portfolio = _portfolio()
    portfolio.positions[Portfolio.key("mkt-1", Side.YES)] = Position(
        canonical_id="mkt-1", side=Side.YES, quantity=8, average_price=Decimal("0.50")
    )
    intent = _intent(quantity=10, limit_price=Decimal("0.50"))
    decision = gw.evaluate(intent, portfolio, _market(), _book(), {}, {}, Decimal(0))
    assert not decision.approved
    assert decision.reason is RejectReason.RISK_GATE
    assert gw.reject_counts.get("strategy_exposure") == 1


def test_category_exposure_cap_rejected():
    gw = _gateway(
        max_category_exposure_pct=Decimal("0.10"),
        max_single_event_loss_pct=Decimal("1"),
        max_strategy_exposure_pct=Decimal("1"),
    )
    intent = _intent(quantity=10, limit_price=Decimal("0.50"))  # cost 5.00
    category_exposures = {Category.POLITICS: Decimal("4.00")}  # 4 + 5 = 9 > cap(5)
    decision = gw.evaluate(
        intent, _portfolio(), _market(category=Category.POLITICS), _book(), category_exposures, {}, Decimal(0)
    )
    assert not decision.approved
    assert decision.reason is RejectReason.RISK_GATE
    assert gw.reject_counts.get("category_exposure") == 1


def test_cluster_exposure_cap_rejected():
    gw = _gateway(
        max_correlated_cluster_pct=Decimal("0.10"),
        max_single_event_loss_pct=Decimal("1"),
        max_strategy_exposure_pct=Decimal("1"),
        max_category_exposure_pct=Decimal("1"),
    )
    intent = _intent(quantity=10, limit_price=Decimal("0.50"))  # cost 5.00
    cluster_exposures = {"evt-1": Decimal("4.00")}  # 4 + 5 = 9 > cap(5)
    decision = gw.evaluate(intent, _portfolio(), _market(event_id="evt-1"), _book(), {}, cluster_exposures, Decimal(0))
    assert not decision.approved
    assert decision.reason is RejectReason.RISK_GATE
    assert gw.reject_counts.get("cluster_exposure") == 1


# ---------------------------------------------------------------------------
# Pause gates: reducing orders always get through
# ---------------------------------------------------------------------------


def test_daily_loss_pause_blocks_increasing_but_allows_reducing():
    gw = _gateway(daily_loss_pause_pct=Decimal("0.10"))  # pauses at daily_pnl <= -5.00
    portfolio = _portfolio()

    increasing = _intent(action=Action.BUY, quantity=1, limit_price=Decimal("0.50"))
    decision = gw.evaluate(increasing, portfolio, _market(), _book(), {}, {}, Decimal("-6.00"))
    assert not decision.approved
    assert decision.reason is RejectReason.RISK_GATE
    assert gw.reject_counts.get("daily_loss_pause") == 1

    portfolio.positions[Portfolio.key("mkt-1", Side.YES)] = Position(
        canonical_id="mkt-1", side=Side.YES, quantity=5, average_price=Decimal("0.40")
    )
    reducing = _intent(action=Action.SELL, quantity=5, limit_price=Decimal("0.40"))
    decision2 = gw.evaluate(reducing, portfolio, _market(), _book(), {}, {}, Decimal("-6.00"))
    assert decision2.approved


def test_drawdown_pause_blocks_increasing_but_allows_reducing():
    gw = _gateway(total_drawdown_pause_pct=Decimal("0.20"))
    portfolio = _portfolio()
    portfolio.max_drawdown = Decimal("0.25")

    increasing = _intent(action=Action.BUY, quantity=1, limit_price=Decimal("0.50"))
    decision = gw.evaluate(increasing, portfolio, _market(), _book(), {}, {}, Decimal(0))
    assert not decision.approved
    assert gw.reject_counts.get("drawdown_pause") == 1

    portfolio.positions[Portfolio.key("mkt-1", Side.YES)] = Position(
        canonical_id="mkt-1", side=Side.YES, quantity=5, average_price=Decimal("0.40")
    )
    reducing = _intent(action=Action.SELL, quantity=5, limit_price=Decimal("0.40"))
    decision2 = gw.evaluate(reducing, portfolio, _market(), _book(), {}, {}, Decimal(0))
    assert decision2.approved


# ---------------------------------------------------------------------------
# Martingale detection
# ---------------------------------------------------------------------------


def _permissive_exposure_gateway(**overrides) -> RiskGateway:
    defaults = dict(
        max_single_event_loss_pct=Decimal("1"),
        max_strategy_exposure_pct=Decimal("1"),
        max_category_exposure_pct=Decimal("1"),
        max_correlated_cluster_pct=Decimal("1"),
    )
    defaults.update(overrides)
    return _gateway(**defaults)


def test_martingale_blocks_large_add_to_underwater_position():
    gw = _permissive_exposure_gateway(martingale=False)
    portfolio = _portfolio()
    # avg 0.70, mark (book mid) ~0.50 -> underwater. Cost basis 7.00.
    portfolio.positions[Portfolio.key("mkt-1", Side.YES)] = Position(
        canonical_id="mkt-1", side=Side.YES, quantity=10, average_price=Decimal("0.70")
    )
    book = _book(mid_bid=Decimal("0.49"), mid_ask=Decimal("0.51"))
    # Adding 10 @ 0.50 = $5.00, a 71% increase to the $7.00 cost basis -> over threshold.
    intent = _intent(action=Action.BUY, quantity=10, limit_price=Decimal("0.50"))
    decision = gw.evaluate(intent, portfolio, _market(), book, {}, {}, Decimal(0))
    assert not decision.approved
    assert decision.reason is RejectReason.RISK_GATE
    assert gw.reject_counts.get("martingale") == 1


def test_martingale_allows_small_add_within_threshold():
    gw = _permissive_exposure_gateway(martingale=False)
    portfolio = _portfolio()
    portfolio.positions[Portfolio.key("mkt-1", Side.YES)] = Position(
        canonical_id="mkt-1", side=Side.YES, quantity=10, average_price=Decimal("0.70")
    )
    book = _book(mid_bid=Decimal("0.49"), mid_ask=Decimal("0.51"))
    # Adding 1 @ 0.50 = $0.50, a ~7% increase to the $7.00 cost basis -> under threshold.
    intent = _intent(action=Action.BUY, quantity=1, limit_price=Decimal("0.50"))
    decision = gw.evaluate(intent, portfolio, _market(), book, {}, {}, Decimal(0))
    assert decision.approved


def test_martingale_allowed_when_config_enables_it():
    gw = _permissive_exposure_gateway(martingale=True)
    portfolio = _portfolio()
    portfolio.positions[Portfolio.key("mkt-1", Side.YES)] = Position(
        canonical_id="mkt-1", side=Side.YES, quantity=10, average_price=Decimal("0.70")
    )
    book = _book(mid_bid=Decimal("0.49"), mid_ask=Decimal("0.51"))
    intent = _intent(action=Action.BUY, quantity=10, limit_price=Decimal("0.50"))
    decision = gw.evaluate(intent, portfolio, _market(), book, {}, {}, Decimal(0))
    assert decision.approved


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def test_strict_audit_rejects_empty_rationale():
    gw = _gateway(strict_audit=True)
    intent = _intent(rationale="")
    decision = gw.evaluate(intent, _portfolio(), _market(), _book(), {}, {}, Decimal(0))
    assert not decision.approved
    assert decision.reason is RejectReason.RISK_GATE
    assert gw.reject_counts.get("strict_audit") == 1


def test_strict_audit_off_allows_empty_rationale():
    gw = _gateway(strict_audit=False)
    intent = _intent(rationale="", quantity=2)  # cost $1.00, under every default cap
    decision = gw.evaluate(intent, _portfolio(), _market(), _book(), {}, {}, Decimal(0))
    assert decision.approved


def test_approved_intent_passes_clean():
    gw = _gateway()
    intent = _intent(quantity=2)  # cost $1.00, under every default cap
    decision = gw.evaluate(intent, _portfolio(), _market(), _book(), {}, {}, Decimal(0))
    assert decision.approved
    assert decision.reason is None
    assert gw.approved_count == 1
