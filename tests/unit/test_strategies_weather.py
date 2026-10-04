"""Weather strategy: range-bucket pricing and station-map wiring."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

TS = datetime(2026, 10, 4, tzinfo=UTC)
def test_range_buckets_are_priced_as_buckets_and_sum_to_one() -> None:
    from marketlab.strategies.weather import forecast_to_bucket_probability, parse_temp_bucket

    assert parse_temp_bucket("KXHIGHNY-26OCT03-B68.5") == Decimal("68.5")
    assert parse_temp_bucket("KXHIGHNY-26OCT03-T72") is None
    total = sum(forecast_to_bucket_probability(70.0, m + 0.5, 2.5) for m in range(40, 100, 2))
    assert abs(total - 1.0) < 1e-9
    # A forecast far from the bucket makes it unlikely, not "likely above".
    assert forecast_to_bucket_probability(75.0, 68.5, 2.5) < 0.02


async def test_runner_hands_weather_sleeves_their_station_map(tmp_path) -> None:
    from marketlab.clock import SimulatedClock
    from marketlab.experiments.runner import ExperimentRunner
    from marketlab.experiments.sweep import VariantSpec
    from marketlab.settings import Settings
    from marketlab.storage.state import StateStore
    from tests.unit.test_runner import FakeBookRegistry, FakeBroker, FakeMarketRegistry

    settings = Settings(
        strategies={"meta": {"bankroll_per_variant": "50.00"}},
        universes={"universes": {"wx": {"category": "weather", "stations": {"KXHIGHNY": "KNYC"}}}},
    )
    runner = ExperimentRunner(settings, SimulatedClock(TS), StateStore(tmp_path / "m.db"), FakeBroker(),
                              FakeMarketRegistry({}), FakeBookRegistry(), broker_owns_portfolio=False)
    await runner.load_or_create_sleeves([VariantSpec(
        strategy_name="weather_forecast", strategy_class_path="marketlab.strategies.weather:WeatherForecastStrategy",
        universe="wx", params={}, evidence_class="B", trades=True, requires_ai=False,
    )])
    sleeve = next(iter(runner._sleeves.values()))
    assert sleeve.strategy.param("stations") == {"KXHIGHNY": "KNYC"}


def test_night_lows_never_price_a_daily_high_and_dates_must_match() -> None:
    from datetime import timedelta

    from marketlab.clock import SimulatedClock
    from marketlab.core.events import WeatherEvent
    from marketlab.core.instruments import (
        BookLevel,
        Category,
        MarketStatus,
        NormalizedMarket,
        OrderBook,
        Venue,
    )
    from marketlab.core.strategy import StrategyContext
    from marketlab.strategies.weather import WeatherForecastStrategy, parse_market_date

    assert parse_market_date("KXHIGHLAX-26OCT04-T98").isoformat() == "2026-10-04"
    cid = "kalshi:kxhighlax-26oct04-t98"
    market = NormalizedMarket(canonical_id=cid, venue=Venue.KALSHI, venue_market_id="KXHIGHLAX-26OCT04-T98",
                              event_id="KXHIGHLAX-26OCT04", title="Will the maximum temperature be >98° on Oct 4, 2026?",
                              category=Category.WEATHER, status=MarketStatus.OPEN, close_time=TS + timedelta(hours=20))
    book = OrderBook(canonical_id=cid, venue=Venue.KALSHI, timestamp=TS,
                     bids=(BookLevel(price=Decimal("0.40"), size=50),), asks=(BookLevel(price=Decimal("0.45"), size=50),))
    ctx = StrategyContext(clock=SimulatedClock(TS), books={cid: book}, markets={cid: market}, marks={})
    strat = WeatherForecastStrategy("s", "e", ctx, params={"stations": {"KXHIGHLAX": "KLAX"}})

    def wx(variable: str, value: str, day: int) -> WeatherEvent:
        return WeatherEvent(event_time=TS, first_seen_time=TS, station="KLAX", variable=variable,
                            value=Decimal(value), forecast_for=datetime(2026, 10, day, 12, tzinfo=UTC))

    strat.on_weather(wx("temperature_forecast_night", "66", 4))  # a low: ignored
    strat.on_weather(wx("temperature_forecast_day", "70", 5))  # tomorrow: wrong date
    strat._evaluate_market(cid, market, "KLAX", trigger="test", forecast_delta=None)
    assert strat.drain_forecasts() == []
    strat.on_weather(wx("temperature_forecast_day", "99", 4))
    strat._evaluate_market(cid, market, "KLAX", trigger="test", forecast_delta=None)
    (forecast,) = strat.drain_forecasts()
    assert forecast.p_yes > Decimal("0.5")  # a 99 F high forecast makes ">98" likely
