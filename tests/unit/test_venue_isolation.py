"""Polymarket must never be traded. This is the project's one absolute rule.

The operator is in the United States, where global Polymarket is close-only for new
positions. Polymarket is a read-only intelligence source: prices for cross-venue
comparison and public trader activity for the copy family. **Kalshi is the only execution
venue.**

These are regression tests for a real failure. Strategies stamp a constant
``VENUE = Venue.KALSHI`` on every intent they emit, so an intent aimed at a Polymarket
market arrived at the risk gateway *declaring itself a Kalshi order*. The gateway checked
only ``intent.venue`` and let it through. 166 fills landed on ``poly:`` contracts before
anyone noticed - contracts this deployment can never actually trade, quietly corrupting
the P&L of every sleeve that touched one.

Two independent guards now exist, and both are asserted here:

1. ``RiskGateway`` checks the **market's own venue**, not just the intent's claim.
2. ``BaseStrategy.should_skip`` refuses a non-execution-venue market outright.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from marketlab.core.instruments import (
    EXECUTION_VENUES,
    READ_ONLY_VENUES,
    BookLevel,
    Category,
    MarketStatus,
    NormalizedMarket,
    OrderBook,
    Side,
    Venue,
)
from marketlab.core.orders import Action, OrderIntent, OrderType, RejectReason
from marketlab.core.portfolio import Portfolio
from marketlab.execution.risk_gateway import RiskGateway
from marketlab.settings import RiskConfig

TS = datetime(2026, 9, 4, 17, 0, tzinfo=UTC)


def _market(canonical_id: str, venue: Venue) -> NormalizedMarket:
    return NormalizedMarket(
        canonical_id=canonical_id,
        venue=venue,
        venue_market_id=canonical_id.split(":", 1)[-1],
        event_id="EVT",
        title="Some binary claim",
        category=Category.POLITICS,
        status=MarketStatus.OPEN,
        volume=Decimal("10000"),
        open_interest=Decimal("5000"),
    )


def _book(canonical_id: str, venue: Venue) -> OrderBook:
    return OrderBook(
        canonical_id=canonical_id,
        venue=venue,
        timestamp=TS,
        bids=(BookLevel(price=Decimal("0.49"), size=500),),
        asks=(BookLevel(price=Decimal("0.51"), size=500),),
    )


def _intent(canonical_id: str, declared_venue: Venue) -> OrderIntent:
    return OrderIntent(
        strategy_id="s1",
        experiment_id="e1",
        canonical_id=canonical_id,
        venue=declared_venue,
        side=Side.YES,
        action=Action.BUY,
        quantity=1,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("0.51"),
        decision_time=TS,
        rationale="venue isolation regression test",
        features={"mid": "0.50"},
    )


def _portfolio() -> Portfolio:
    return Portfolio(experiment_id="e1", strategy_id="s1", created_at=TS)


def test_polymarket_market_is_rejected_even_when_the_intent_claims_kalshi() -> None:
    """The exact production bug: a Kalshi-stamped intent against a Polymarket market."""
    cid = "poly:0xac02cbb049e46d6a3627c0fdf52"
    decision = RiskGateway(RiskConfig()).evaluate(
        intent=_intent(cid, Venue.KALSHI),  # strategies always stamp KALSHI
        portfolio=_portfolio(),
        market=_market(cid, Venue.POLY_GLOBAL),  # ...but the market is Polymarket
        book=_book(cid, Venue.POLY_GLOBAL),
        category_exposures={},
        cluster_exposures={},
        daily_pnl=Decimal(0),
    )
    assert not decision.approved
    assert decision.reason is RejectReason.MODE_FORBIDDEN
    assert "read-only" in decision.detail


@pytest.mark.parametrize("venue", sorted(READ_ONLY_VENUES, key=str))
def test_no_read_only_venue_can_ever_be_traded(venue: Venue) -> None:
    cid = f"{venue.value}:some-market"
    decision = RiskGateway(RiskConfig()).evaluate(
        intent=_intent(cid, Venue.KALSHI),
        portfolio=_portfolio(),
        market=_market(cid, venue),
        book=_book(cid, venue),
        category_exposures={},
        cluster_exposures={},
        daily_pnl=Decimal(0),
    )
    assert not decision.approved, f"{venue} must never be tradeable"
    assert decision.reason is RejectReason.MODE_FORBIDDEN


def test_kalshi_market_is_still_allowed_through_the_venue_guard() -> None:
    """The guard must not be so blunt that it blocks the one venue we do trade."""
    cid = "kalshi:kxnflgame-26sep13dalnyg-dal"
    decision = RiskGateway(RiskConfig()).evaluate(
        intent=_intent(cid, Venue.KALSHI),
        portfolio=_portfolio(),
        market=_market(cid, Venue.KALSHI),
        book=_book(cid, Venue.KALSHI),
        category_exposures={},
        cluster_exposures={},
        daily_pnl=Decimal(0),
    )
    assert decision.approved, decision.detail


def test_polymarket_is_not_in_the_execution_venue_set() -> None:
    """A configuration-level assertion: the constant itself must stay correct."""
    assert Venue.POLY_GLOBAL not in EXECUTION_VENUES
    assert Venue.POLY_GLOBAL in READ_ONLY_VENUES
    assert Venue.KALSHI in EXECUTION_VENUES


def test_base_strategy_skips_non_execution_venue_markets() -> None:
    """The second, independent guard: a strategy refuses the market before emitting."""
    from marketlab.clock import SimulatedClock
    from marketlab.core.strategy import StrategyContext
    from marketlab.strategies.base import BaseStrategy

    cid = "poly:0xdeadbeef"
    clock = SimulatedClock(TS)
    ctx = StrategyContext(
        clock=clock,
        books={cid: _book(cid, Venue.POLY_GLOBAL)},
        markets={cid: _market(cid, Venue.POLY_GLOBAL)},
        marks={},
    )
    strategy = BaseStrategy("s1", "e1", ctx, params={})
    assert strategy.should_skip(cid) == "not_an_execution_venue"

    kalshi_id = "kalshi:kxnflgame-26sep13dalnyg-dal"
    ctx_ok = StrategyContext(
        clock=clock,
        books={kalshi_id: _book(kalshi_id, Venue.KALSHI)},
        markets={kalshi_id: _market(kalshi_id, Venue.KALSHI)},
        marks={},
    )
    assert BaseStrategy("s1", "e1", ctx_ok, params={}).should_skip(kalshi_id) is None
