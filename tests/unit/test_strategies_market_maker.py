"""Tests for AvellanedaStoikovBinaryStrategy."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from marketlab.core.instruments import MarketStatus, Side, Venue
from marketlab.core.orders import Action, Fill, Order, OrderStatus, OrderType, TimeInForce
from marketlab.strategies.market_maker import AvellanedaStoikovBinaryStrategy
from tests.fixtures.strategy_harness import StrategyHarness, make_book, make_market


def _open_order(order_id: str, canonical_id: str, now) -> Order:
    return Order(
        order_id=order_id,
        intent_id=f"int-{order_id}",
        strategy_id="s",
        experiment_id="e",
        canonical_id=canonical_id,
        venue=Venue.KALSHI,
        side=Side.YES,
        action=Action.BUY,
        order_type=OrderType.LIMIT,
        quantity=5,
        limit_price=Decimal("0.45"),
        time_in_force=TimeInForce.GTC,
        status=OrderStatus.OPEN,
        decision_timestamp=now,
    )


def test_quotes_straddle_the_reservation_price() -> None:
    h = StrategyHarness(AvellanedaStoikovBinaryStrategy, params={"base_spread": "0.05", "mode": "two_sided"})
    h.set_market(make_market("M1"))
    h.feed_book(make_book("M1", [("0.49", 100)], [("0.51", 100)], h.now()))

    assert len(h.intents) == 2
    bid = next(i for i in h.intents if i.action is Action.BUY)
    ask = next(i for i in h.intents if i.action is Action.SELL)
    reservation = Decimal(str(bid.features["reservation_price"]))
    assert bid.limit_price < reservation < ask.limit_price
    assert bid.expected_edge is not None and bid.expected_edge > 0
    assert ask.expected_edge is not None and ask.expected_edge > 0


def test_inventory_skews_quotes_in_the_correct_direction() -> None:
    flat = StrategyHarness(AvellanedaStoikovBinaryStrategy, params={"base_spread": "0.05", "mode": "two_sided"})
    flat.set_market(make_market("M1"))
    flat.feed_book(make_book("M1", [("0.49", 100)], [("0.51", 100)], flat.now()))
    flat_bid = next(i for i in flat.intents if i.action is Action.BUY).limit_price
    flat_ask = next(i for i in flat.intents if i.action is Action.SELL).limit_price

    long_yes = StrategyHarness(AvellanedaStoikovBinaryStrategy, params={"base_spread": "0.05", "mode": "two_sided"})
    long_yes.set_market(make_market("M1"))
    fill = Fill(
        order_id="o1",
        canonical_id="M1",
        venue=Venue.KALSHI,
        side=Side.YES,
        action=Action.BUY,
        price=Decimal("0.50"),
        quantity=10,
        timestamp=long_yes.now(),
    )
    long_yes.feed_fill(fill)
    long_yes.feed_book(make_book("M1", [("0.49", 100)], [("0.51", 100)], long_yes.now()))
    skewed_bid = next(i for i in long_yes.intents if i.action is Action.BUY).limit_price
    skewed_ask = next(i for i in long_yes.intents if i.action is Action.SELL).limit_price

    # Long YES inventory should pull both quotes DOWN: less eager to buy more, more eager
    # to sell down the existing position.
    assert skewed_bid < flat_bid
    assert skewed_ask < flat_ask


def test_no_quote_when_expected_captured_spread_is_below_fees() -> None:
    # At p=0.50, 1 contract: fee = ceil(0.07*0.25*100) = 1.75c -> round-trip unwind cost
    # = 2*$0.0175 = $0.035. A base_spread of $0.02 cannot clear that, so nothing quotes.
    h = StrategyHarness(AvellanedaStoikovBinaryStrategy, params={"base_spread": "0.02", "mode": "two_sided"})
    h.set_market(make_market("M1"))
    h.feed_book(make_book("M1", [("0.49", 100)], [("0.51", 100)], h.now()))
    assert h.intents == []

    # The same $0.05 base_spread used elsewhere clears that $0.035 bar comfortably.
    h2 = StrategyHarness(AvellanedaStoikovBinaryStrategy, params={"base_spread": "0.05", "mode": "two_sided"})
    h2.set_market(make_market("M1"))
    h2.feed_book(make_book("M1", [("0.49", 100)], [("0.51", 100)], h2.now()))
    assert len(h2.intents) == 2


def test_quotes_cancelled_on_market_status_change() -> None:
    h = StrategyHarness(AvellanedaStoikovBinaryStrategy, params={"base_spread": "0.05"})
    h.set_market(make_market("M1"))
    h.feed_book(make_book("M1", [("0.49", 100)], [("0.51", 100)], h.now()))
    order = _open_order("ord-1", "M1", h.now())
    h.feed_order_update(order)

    h.feed_market_status("M1", MarketStatus.CLOSED)
    assert "ord-1" in h.cancels


def test_quotes_cancelled_on_high_impact_news_pause() -> None:
    h = StrategyHarness(AvellanedaStoikovBinaryStrategy, params={"base_spread": "0.05", "news_pause": True})
    h.set_market(make_market("M1"))
    h.feed_book(make_book("M1", [("0.49", 100)], [("0.51", 100)], h.now()))
    order = _open_order("ord-2", "M1", h.now())
    h.feed_order_update(order)

    from marketlab.core.events import SourceClass

    h.feed_news(source_class=SourceClass.OFFICIAL_PRIMARY, title="Breaking")
    assert "ord-2" in h.cancels

    # And no fresh quotes should be posted while the pause is active.
    h.intents.clear()
    h.feed_book(make_book("M1", [("0.49", 100)], [("0.51", 100)], h.now()))
    assert h.intents == []


def test_quotes_stay_within_tick_bounds_near_expiry() -> None:
    h = StrategyHarness(
        AvellanedaStoikovBinaryStrategy,
        params={"base_spread": "0.05", "stop_quoting_seconds": 30, "close_buffer_seconds": 300},
    )
    close_time = h.now() + timedelta(seconds=100)
    h.set_market(make_market("M1", close_time=close_time, tick_size=Decimal("0.01")))
    h.feed_book(make_book("M1", [("0.96", 100)], [("0.98", 100)], h.now()))

    assert len(h.intents) >= 1
    tick = Decimal("0.01")
    for intent in h.intents:
        assert tick <= intent.limit_price <= (Decimal(1) - tick)


def test_no_quoting_once_inside_stop_quoting_window() -> None:
    h = StrategyHarness(AvellanedaStoikovBinaryStrategy, params={"base_spread": "0.05", "stop_quoting_seconds": 30})
    close_time = h.now() + timedelta(seconds=10)
    h.set_market(make_market("M1", close_time=close_time))
    h.feed_book(make_book("M1", [("0.49", 100)], [("0.51", 100)], h.now()))
    assert h.intents == []


def test_every_intent_has_rationale_and_features() -> None:
    h = StrategyHarness(AvellanedaStoikovBinaryStrategy, params={"base_spread": "0.05"})
    h.set_market(make_market("M1"))
    h.feed_book(make_book("M1", [("0.49", 100)], [("0.51", 100)], h.now()))
    assert len(h.intents) >= 1
    for intent in h.intents:
        assert intent.rationale.strip()
        assert intent.features
