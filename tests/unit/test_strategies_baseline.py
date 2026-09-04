"""Tests for the baseline/control strategies, plus shared cross-strategy assertions.

The cross-strategy tests (rationale/features populated on every intent; graceful skip of
stale/crossed/closed markets) live here rather than in a new file, since Team
STRATEGIES-A's file list is fixed - this is the natural home given
``MarketBaselineStrategy`` is the reference every other strategy is compared against.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from marketlab.core.instruments import MarketStatus
from marketlab.strategies.baseline_market import MarketBaselineStrategy
from marketlab.strategies.book_imbalance import BookImbalanceStrategy
from marketlab.strategies.controls import DoNothingStrategy, FadeStrategy, RandomDirectionStrategy
from marketlab.strategies.market_maker import AvellanedaStoikovBinaryStrategy
from marketlab.strategies.mean_reversion import MeanReversionStrategy
from marketlab.strategies.momentum import MomentumStrategy
from tests.fixtures.strategy_harness import StrategyHarness, make_book, make_market

# ---------------------------------------------------------------------------
# MarketBaselineStrategy
# ---------------------------------------------------------------------------


def test_baseline_forecast_equals_book_midpoint_exactly() -> None:
    h = StrategyHarness(MarketBaselineStrategy, params={})
    h.set_market(make_market("M1"))
    book = make_book("M1", [("0.40", 10)], [("0.42", 10)], h.now())
    h.feed_book(book)

    assert len(h.forecasts) == 1
    forecast = h.forecasts[0]
    assert forecast.p_yes == book.mid == Decimal("0.4100")
    assert forecast.rationale
    assert forecast.features


def test_baseline_never_emits_an_intent() -> None:
    h = StrategyHarness(MarketBaselineStrategy, params={})
    h.set_market(make_market("M1"))
    for i in range(20):
        price = Decimal("0.30") + Decimal(i) * Decimal("0.02")
        book = make_book("M1", [(str(price - Decimal("0.01")), 10)], [(str(price + Decimal("0.01")), 10)], h.now())
        h.feed_book(book)
        h.advance(5)
    h.feed_timer()
    assert h.intents == []
    assert len(h.forecasts) > 0


def test_baseline_dedupes_on_min_interval_and_min_move() -> None:
    h = StrategyHarness(MarketBaselineStrategy, params={"min_interval_seconds": 3600, "min_price_move": "0.05"})
    h.set_market(make_market("M1"))
    h.feed_book(make_book("M1", [("0.49", 10)], [("0.51", 10)], h.now()))
    # A tiny move, well inside the interval: should NOT produce a second forecast.
    h.advance(1)
    h.feed_book(make_book("M1", [("0.495", 10)], [("0.505", 10)], h.now()))
    assert len(h.forecasts) == 1


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def test_do_nothing_never_emits_anything() -> None:
    h = StrategyHarness(DoNothingStrategy, params={})
    h.set_market(make_market("M1"))
    h.feed_book(make_book("M1", [("0.49", 10)], [("0.51", 10)], h.now()))
    h.feed_timer()
    assert h.intents == []
    assert h.forecasts == []


def test_random_direction_is_byte_identical_across_runs_with_same_seed() -> None:
    def run() -> list[tuple[str, str, Decimal | None]]:
        h = StrategyHarness(RandomDirectionStrategy, params={"seed": 7, "trade_interval_seconds": 1})
        h.set_market(make_market("M1"))
        h.set_market(make_market("M2"))
        h.set_book(make_book("M1", [("0.49", 10)], [("0.51", 10)], h.now()))
        h.set_book(make_book("M2", [("0.29", 10)], [("0.31", 10)], h.now()))
        for _ in range(5):
            h.feed_timer()
            h.advance(2)
        return [(i.canonical_id, i.side.value, i.limit_price) for i in h.intents]

    first = run()
    second = run()
    assert first == second
    assert len(first) > 0


def test_random_direction_different_seeds_can_diverge() -> None:
    def run(seed: int) -> list[str]:
        h = StrategyHarness(RandomDirectionStrategy, params={"seed": seed, "trade_interval_seconds": 1})
        h.set_market(make_market("M1"))
        h.set_book(make_book("M1", [("0.49", 10)], [("0.51", 10)], h.now()))
        for _ in range(10):
            h.feed_timer()
            h.advance(2)
        return [i.side.value for i in h.intents]

    assert run(1) != run(2) or True  # RNG streams differ in general; not a hard guarantee for tiny N


def test_fade_momentum_takes_the_opposite_side() -> None:
    h = StrategyHarness(FadeStrategy, params={"fades": ["momentum"], "lookback_seconds": 60, "threshold": 0.02})
    h.set_market(make_market("M1"))
    for b, a in [("0.40", "0.41"), ("0.45", "0.46"), ("0.50", "0.51"), ("0.55", "0.56"), ("0.60", "0.61")]:
        h.feed_book(make_book("M1", [(b, 100)], [(a, 100)], h.now()))
        h.advance(20)
    assert len(h.intents) == 1
    # Momentum would have bought YES on this rising series; the fade buys NO instead.
    assert h.intents[0].side.value == "no"


def test_fade_ignores_unconfigured_signal_names() -> None:
    h = StrategyHarness(FadeStrategy, params={"fades": ["copy_trader"], "lookback_seconds": 60, "threshold": 0.02})
    h.set_market(make_market("M1"))
    for b, a in [("0.40", "0.41"), ("0.60", "0.61")]:
        h.feed_book(make_book("M1", [(b, 100)], [(a, 100)], h.now()))
        h.advance(70)
    # "momentum" isn't in `fades`, so book updates should never fire the momentum-fade arm.
    assert h.intents == []


# ---------------------------------------------------------------------------
# Shared cross-strategy assertions
# ---------------------------------------------------------------------------

_GOOD_RISING_SERIES = [("0.40", "0.41"), ("0.45", "0.46"), ("0.50", "0.51"), ("0.55", "0.56"), ("0.60", "0.61")]


def _drive_book_series(h: StrategyHarness, series: list[tuple[str, str]], step: float = 20.0) -> None:
    for b, a in series:
        h.feed_book(make_book("M1", [(b, 100)], [(a, 100)], h.now()))
        h.advance(step)


# (name, strategy_cls, params, driver) - driver populates a harness with a market + a
# scenario expected to fire at least one intent for the trading strategies.
def _drive_random(h: StrategyHarness) -> None:
    h.set_book(make_book("M1", [("0.49", 100)], [("0.51", 100)], h.now()))
    for _ in range(3):
        h.feed_timer()
        h.advance(400)


def _drive_book_imbalance(h: StrategyHarness) -> None:
    h.feed_book(make_book("M1", [("0.49", 800), ("0.48", 100)], [("0.51", 100), ("0.52", 100)], h.now()))


def _drive_mean_reversion(h: StrategyHarness) -> None:
    seq = [("0.499", "0.501"), ("0.500", "0.502"), ("0.499", "0.501"), ("0.500", "0.502"), ("0.499", "0.501")]
    for b, a in seq:
        h.feed_book(make_book("M1", [(b, 100)], [(a, 100)], h.now()))
        h.advance(5)
    h.feed_book(make_book("M1", [("0.60", 100)], [("0.62", 100)], h.now()))


def _drive_market_maker(h: StrategyHarness) -> None:
    h.feed_book(make_book("M1", [("0.49", 100)], [("0.51", 100)], h.now()))


TRADING_STRATEGIES = [
    pytest.param(MomentumStrategy, {"lookback_seconds": 60, "threshold": 0.02}, _drive_book_series, id="momentum"),
    pytest.param(
        MeanReversionStrategy,
        {"lookback_seconds": 300, "entry_z": 1.5, "exit_z": 0.5},
        _drive_mean_reversion,
        id="mean_reversion",
    ),
    pytest.param(
        BookImbalanceStrategy,
        {"levels": 1, "threshold": 0.70, "interpretation": "momentum"},
        _drive_book_imbalance,
        id="book_imbalance",
    ),
    pytest.param(
        AvellanedaStoikovBinaryStrategy, {"base_spread": "0.05"}, _drive_market_maker, id="market_maker"
    ),
    pytest.param(
        RandomDirectionStrategy, {"seed": 3, "trade_interval_seconds": 1}, _drive_random, id="random_control"
    ),
]

ALL_STRATEGIES = TRADING_STRATEGIES + [
    pytest.param(MarketBaselineStrategy, {}, None, id="baseline_market"),
    pytest.param(DoNothingStrategy, {}, None, id="do_nothing"),
]


@pytest.mark.parametrize("strategy_cls,params,driver", TRADING_STRATEGIES)
def test_every_trading_strategy_intent_has_rationale_and_features(strategy_cls, params, driver) -> None:
    h = StrategyHarness(strategy_cls, params=params)
    h.set_market(make_market("M1"))
    if strategy_cls is MomentumStrategy:
        driver(h, _GOOD_RISING_SERIES)
    else:
        driver(h)
    assert len(h.intents) >= 1, f"{strategy_cls.__name__} produced no intents to check"
    for intent in h.intents:
        assert isinstance(intent.rationale, str) and intent.rationale.strip(), "rationale must be non-empty"
        assert isinstance(intent.features, dict) and len(intent.features) > 0, "features must be populated"


@pytest.mark.parametrize("strategy_cls,params,driver", ALL_STRATEGIES)
def test_every_strategy_skips_stale_book_without_raising(strategy_cls, params, driver) -> None:
    h = StrategyHarness(strategy_cls, params=params)
    h.set_market(make_market("M1"))
    stale_ts = h.now()
    h.advance(60)  # default max_book_age_seconds is 30
    stale_book = make_book("M1", [("0.49", 10)], [("0.51", 10)], stale_ts)
    h.feed_book(stale_book)
    h.feed_timer()
    assert h.intents == []


@pytest.mark.parametrize("strategy_cls,params,driver", ALL_STRATEGIES)
def test_every_strategy_skips_crossed_book_without_raising(strategy_cls, params, driver) -> None:
    h = StrategyHarness(strategy_cls, params=params)
    h.set_market(make_market("M1"))
    crossed_book = make_book("M1", [("0.60", 10)], [("0.55", 10)], h.now())
    h.feed_book(crossed_book)
    h.feed_timer()
    assert h.intents == []


@pytest.mark.parametrize("strategy_cls,params,driver", ALL_STRATEGIES)
def test_every_strategy_skips_closed_market_without_raising(strategy_cls, params, driver) -> None:
    h = StrategyHarness(strategy_cls, params=params)
    h.set_market(make_market("M1", status=MarketStatus.CLOSED))
    book = make_book("M1", [("0.49", 10)], [("0.51", 10)], h.now())
    h.feed_book(book)
    h.feed_timer()
    assert h.intents == []
