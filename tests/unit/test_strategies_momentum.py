"""Tests for MomentumStrategy."""

from __future__ import annotations

from decimal import Decimal

from marketlab.core.events import SourceClass
from marketlab.core.instruments import Side
from marketlab.strategies.momentum import MomentumStrategy
from tests.fixtures.strategy_harness import StrategyHarness, make_book, make_market

RISING = [("0.40", "0.41"), ("0.45", "0.46"), ("0.50", "0.51"), ("0.55", "0.56"), ("0.60", "0.61")]
FLAT = [("0.49", "0.51")] * 6


def _feed(h: StrategyHarness, series: list[tuple[str, str]], step: float = 20.0) -> None:
    for b, a in series:
        h.feed_book(make_book("M1", [(b, 100)], [(a, 100)], h.now()))
        h.advance(step)


def test_rising_series_past_threshold_buys_yes() -> None:
    h = StrategyHarness(MomentumStrategy, params={"lookback_seconds": 60, "threshold": 0.02})
    h.set_market(make_market("M1"))
    _feed(h, RISING)
    assert len(h.intents) == 1
    intent = h.intents[0]
    assert intent.side is Side.YES
    assert intent.action.value == "buy"
    assert intent.expected_edge is not None and intent.expected_edge > 0


def test_flat_series_produces_nothing() -> None:
    h = StrategyHarness(MomentumStrategy, params={"lookback_seconds": 60, "threshold": 0.02})
    h.set_market(make_market("M1"))
    _feed(h, FLAT)
    assert h.intents == []


def test_volatility_filter_suppresses_entry_in_high_vol_regime() -> None:
    # Net move from 0.30 -> 0.50 clears any reasonable threshold, but the path is very
    # choppy (large step-to-step swings), so a volatility filter should veto it.
    choppy_but_net_up = [("0.30", "0.32"), ("0.55", "0.57"), ("0.20", "0.22"), ("0.65", "0.67"), ("0.50", "0.52")]

    filtered = StrategyHarness(
        MomentumStrategy,
        params={"lookback_seconds": 60, "threshold": 0.02, "volatility_filter": True, "volatility_threshold": 0.05},
    )
    filtered.set_market(make_market("M1"))
    _feed(filtered, choppy_but_net_up)
    assert filtered.intents == []

    unfiltered = StrategyHarness(
        MomentumStrategy, params={"lookback_seconds": 60, "threshold": 0.02, "volatility_filter": False}
    )
    unfiltered.set_market(make_market("M1"))
    _feed(unfiltered, choppy_but_net_up)
    assert len(unfiltered.intents) == 1


def test_logit_boundary_move_does_not_trigger_while_midrange_move_does() -> None:
    """Encodes the exact reason bounded_momentum exists (see module docstring).

    logit(0.02)-logit(0.01) ~= 0.70 is numerically *larger* than logit(0.60)-logit(0.50)
    ~= 0.41, so comparing the logit delta itself to `threshold` cannot distinguish these
    cases - which is exactly why the entry gate below compares the *raw* probability
    delta instead. A raw threshold of 0.02 does the job: 0.01->0.02 is a 1-point move
    (below threshold), 0.50->0.60 is a 10-point move (above it).
    """
    boundary = StrategyHarness(MomentumStrategy, params={"lookback_seconds": 60, "threshold": 0.02})
    boundary.set_market(make_market("M1", tick_size=Decimal("0.0001")))
    boundary.feed_book(make_book("M1", [("0.0099", 1000)], [("0.0101", 1000)], boundary.now()))
    boundary.advance(70)
    boundary.feed_book(make_book("M1", [("0.0199", 1000)], [("0.0201", 1000)], boundary.now()))
    assert boundary.intents == []

    midrange = StrategyHarness(MomentumStrategy, params={"lookback_seconds": 60, "threshold": 0.02})
    midrange.set_market(make_market("M1"))
    midrange.feed_book(make_book("M1", [("0.4900", 1000)], [("0.5100", 1000)], midrange.now()))
    midrange.advance(70)
    midrange.feed_book(make_book("M1", [("0.5900", 1000)], [("0.6100", 1000)], midrange.now()))
    assert len(midrange.intents) == 1
    assert midrange.intents[0].side is Side.YES


def test_book_confirmation_requires_agreement() -> None:
    params = {"lookback_seconds": 60, "threshold": 0.02, "require_book_confirmation": True}

    # Price rose, but the book is ask-heavy (imbalance < 0.5) - contradicts continuation.
    contradicting = StrategyHarness(MomentumStrategy, params=params)
    contradicting.set_market(make_market("M1"))
    contradicting.feed_book(make_book("M1", [("0.40", 10)], [("0.41", 500)], contradicting.now()))
    contradicting.advance(70)
    contradicting.feed_book(make_book("M1", [("0.60", 10)], [("0.61", 500)], contradicting.now()))
    assert contradicting.intents == []

    # Price rose and the book is bid-heavy (imbalance > 0.5) - confirms continuation.
    confirming = StrategyHarness(MomentumStrategy, params=params)
    confirming.set_market(make_market("M1"))
    confirming.feed_book(make_book("M1", [("0.40", 500)], [("0.41", 10)], confirming.now()))
    confirming.advance(70)
    confirming.feed_book(make_book("M1", [("0.60", 500)], [("0.61", 10)], confirming.now()))
    assert len(confirming.intents) == 1


def test_news_suppression_blocks_entries_after_high_impact_news() -> None:
    params = {"lookback_seconds": 60, "threshold": 0.02, "news_suppress_seconds": 120}
    h = StrategyHarness(MomentumStrategy, params=params)
    h.set_market(make_market("M1"))
    h.feed_book(make_book("M1", [("0.40", 100)], [("0.41", 100)], h.now()))
    h.advance(10)
    h.feed_news(source_class=SourceClass.OFFICIAL_PRIMARY, title="Fed announcement")
    h.advance(60)
    h.feed_book(make_book("M1", [("0.60", 100)], [("0.61", 100)], h.now()))
    assert h.intents == []  # still within the 120s suppression window

    # Once the suppression window has elapsed, a fresh move (anchored off the now-105
    # price) fires normally.
    h.advance(120)
    h.feed_book(make_book("M1", [("0.70", 100)], [("0.71", 100)], h.now()))
    assert len(h.intents) == 1


def test_append_sample_keeps_one_entry_per_spacing() -> None:
    from datetime import UTC, datetime, timedelta
    from decimal import Decimal

    from marketlab.strategies.base import BaseStrategy

    t0 = datetime(2026, 10, 4, tzinfo=UTC)
    history: list = []
    for i in range(100):  # 100 updates over 10s, 0.1s apart -> two 5s buckets
        BaseStrategy.append_sample(history, t0 + timedelta(seconds=i / 10), Decimal(i), 5.0)
    BaseStrategy.append_sample(history, t0 + timedelta(seconds=12), Decimal("500"), 5.0)
    assert history == [
        (t0, Decimal(49)),
        (t0 + timedelta(seconds=5), Decimal(99)),
        (t0 + timedelta(seconds=12), Decimal("500")),
    ]
