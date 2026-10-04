"""Unit tests for marketlab/strategies/btc_event.py."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from marketlab.clock import SimulatedClock
from marketlab.core.events import BookUpdateEvent, ExternalPriceEvent
from marketlab.core.instruments import (
    BookLevel,
    Category,
    MarketStatus,
    NormalizedMarket,
    OrderBook,
    Venue,
)
from marketlab.core.probability import gbm_barrier_touch_probability, gbm_touch_probability
from marketlab.core.strategy import StrategyContext
from marketlab.strategies.btc_event import (
    BtcEventStrategy,
    classify_measurement,
    is_below_threshold_contract,
    parse_btc_strike,
)

TS = datetime(2026, 1, 1, tzinfo=UTC)


def test_strike_parsed_correctly_from_a_real_ticker() -> None:
    assert parse_btc_strike("KXBTCD-26SEP0413-T86799.99") == Decimal("86799.99")


def test_strike_falls_back_to_title_dollar_amount() -> None:
    assert parse_btc_strike("SOME-OTHER-FORMAT", "Will BTC be above $50,000.00?") == Decimal("50000.00")
    assert parse_btc_strike("SOME-OTHER-FORMAT", "no dollar amount here") is None


def test_terminal_vs_barrier_give_different_answers_on_the_same_inputs() -> None:
    spot, strike, sigma, seconds = 95000.0, 100000.0, 0.6, 3600.0
    terminal = gbm_touch_probability(spot, strike, sigma, seconds)
    barrier = gbm_barrier_touch_probability(spot, strike, sigma, seconds)
    assert terminal != barrier
    # A barrier ("touches before expiry") is always at least as likely as the terminal
    # ("is above at expiry") event, since touching is a necessary condition for ending
    # above an out-of-the-money strike under a continuous driftless process.
    assert barrier >= terminal


def test_classify_measurement_terminal_vs_barrier_vs_ambiguous() -> None:
    assert classify_measurement("Will BTC be above $100k at 5pm ET?") == "terminal"
    assert classify_measurement("Will BTC touch $100k before 5pm ET?") == "barrier_touch"
    assert classify_measurement("Will BTC do something with $100k?") is None


def test_is_below_threshold_contract() -> None:
    assert is_below_threshold_contract("Will BTC be below $50,000 at close?") is True
    assert is_below_threshold_contract("Will BTC be above $50,000 at close?") is False


def _market(title: str, close_time: datetime) -> NormalizedMarket:
    return NormalizedMarket(
        canonical_id="kalshi:btc-test",
        venue=Venue.KALSHI,
        venue_market_id="KXBTCD-26SEP0413-T90000.00",
        event_id="btc-evt-1",
        title=title,
        category=Category.CRYPTO,
        status=MarketStatus.OPEN,
        close_time=close_time,
        min_order=1,
    )


def _book(bid: str = "0.40", ask: str = "0.42") -> OrderBook:
    return OrderBook(
        canonical_id="kalshi:btc-test",
        venue=Venue.KALSHI,
        timestamp=TS,
        bids=(BookLevel(price=Decimal(bid), size=100),),
        asks=(BookLevel(price=Decimal(ask), size=100),),
    )


def _ctx(market: NormalizedMarket, book: OrderBook, now: datetime = TS) -> StrategyContext:
    clock = SimulatedClock(now)
    return StrategyContext(clock=clock, books={market.canonical_id: book}, markets={market.canonical_id: market}, marks={})


def _feed_spot_history(strat: BtcEventStrategy, spot: float, n: int = 20, step_seconds: float = 60.0) -> None:
    # Feed a flat-but-noisy series so realized_volatility has something to compute from.
    base_ts = TS - timedelta(seconds=step_seconds * n)
    for i in range(n):
        wiggle = 1.0 if i % 2 == 0 else -1.0
        strat.on_external_price(
            ExternalPriceEvent(
                symbol="BTC-USD",
                price=Decimal(str(spot + wiggle * 50)),
                event_time=base_ts + timedelta(seconds=step_seconds * i),
                first_seen_time=base_ts + timedelta(seconds=step_seconds * i),
            )
        )


def test_ambiguous_wording_abstains() -> None:
    close_time = TS + timedelta(hours=1)
    market = _market("Will BTC do something with $90,000?", close_time)
    book = _book()
    ctx = _ctx(market, book)
    strat = BtcEventStrategy("s1", "e1", ctx, params={"model": "gbm_terminal"})
    _feed_spot_history(strat, 90000.0)

    strat.on_book_update(BookUpdateEvent(book=book, event_time=TS, first_seen_time=TS))

    forecasts = strat.drain_forecasts()
    assert len(forecasts) == 1
    assert forecasts[0].abstain is True
    assert strat.generate_intents() == []


def test_hand_computed_gbm_probability_matches_forecast() -> None:
    close_time = TS + timedelta(hours=1)
    market = _market("Will BTC be above $90,000 at close?", close_time)
    book = _book()
    ctx = _ctx(market, book)
    strat = BtcEventStrategy("s1", "e1", ctx, params={"model": "gbm_terminal", "vol_window_minutes": 15})
    _feed_spot_history(strat, 90000.0)

    strat.on_book_update(BookUpdateEvent(book=book, event_time=TS, first_seen_time=TS))
    forecasts = strat.drain_forecasts()
    assert len(forecasts) == 1
    forecast = forecasts[0]
    assert forecast.abstain is False

    # Recompute the same probability by hand from the features the strategy recorded.
    spot = forecast.features["spot"]
    strike = forecast.features["strike"]
    sigma = forecast.features["sigma_annual"]
    seconds_remaining = forecast.features["seconds_remaining"]
    expected = gbm_touch_probability(spot, strike, sigma, seconds_remaining)
    assert float(forecast.p_yes) == round(expected, 6)


def test_terminal_variant_does_not_trade_barrier_worded_market() -> None:
    close_time = TS + timedelta(hours=1)
    market = _market("Will BTC touch $90,000 before close?", close_time)
    book = _book()
    ctx = _ctx(market, book)
    strat = BtcEventStrategy("s1", "e1", ctx, params={"model": "gbm_terminal"})
    _feed_spot_history(strat, 90000.0)

    strat.on_book_update(BookUpdateEvent(book=book, event_time=TS, first_seen_time=TS))
    # Out of scope for this variant (barrier wording, terminal model) - not ambiguous,
    # so no abstain forecast either, just silence.
    assert strat.drain_forecasts() == []
    assert strat.generate_intents() == []



def test_kalshi_daily_threshold_series_is_terminal_by_structure() -> None:
    from marketlab.strategies.btc_event import classify_measurement

    # The event title is uninformative and the rules say "before 5 PM", which a keyword
    # pass would misread as a barrier. The series structure settles it.
    assert classify_measurement("Bitcoin price on Oct 9, 2026?", "$87,000 or above",
                                "KXBTCD-26OCT0917-T86999.99") == "terminal"
    assert classify_measurement("Bitcoin price range on Sep 4, 2026?", "$79,700 to 79,799.99",
                                "KXBTC-26SEP0419-B79750") is None
    assert classify_measurement("Bitcoin yearly high", "", "KXBTCMAXY-26-T150000") == "barrier_touch"
