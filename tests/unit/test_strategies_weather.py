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
