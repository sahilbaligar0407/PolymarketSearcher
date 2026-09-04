"""Tests for BookImbalanceStrategy."""

from __future__ import annotations

from marketlab.core.instruments import Side
from marketlab.strategies.book_imbalance import BookImbalanceStrategy
from tests.fixtures.strategy_harness import StrategyHarness, make_book, make_market

BID_HEAVY_BIDS = [("0.49", 800), ("0.48", 100)]
BID_HEAVY_ASKS = [("0.51", 100), ("0.52", 100)]


def test_momentum_and_reversal_emit_opposite_sides_on_identical_book() -> None:
    momentum = StrategyHarness(
        BookImbalanceStrategy, params={"levels": 1, "threshold": 0.70, "interpretation": "momentum"}
    )
    momentum.set_market(make_market("M1"))
    momentum.feed_book(make_book("M1", BID_HEAVY_BIDS, BID_HEAVY_ASKS, momentum.now()))

    reversal = StrategyHarness(
        BookImbalanceStrategy, params={"levels": 1, "threshold": 0.70, "interpretation": "reversal"}
    )
    reversal.set_market(make_market("M1"))
    reversal.feed_book(make_book("M1", BID_HEAVY_BIDS, BID_HEAVY_ASKS, reversal.now()))

    assert len(momentum.intents) == 1 and len(reversal.intents) == 1
    assert momentum.intents[0].side is Side.YES  # bid-heavy -> continuation -> buy YES
    assert reversal.intents[0].side is Side.NO  # bid-heavy -> fade -> buy NO
    assert momentum.intents[0].side is not reversal.intents[0].side


def test_ask_heavy_book_flips_both_interpretations() -> None:
    ask_heavy_bids = [("0.49", 100), ("0.48", 100)]
    ask_heavy_asks = [("0.51", 800), ("0.52", 100)]

    momentum = StrategyHarness(
        BookImbalanceStrategy, params={"levels": 1, "threshold": 0.70, "interpretation": "momentum"}
    )
    momentum.set_market(make_market("M1"))
    momentum.feed_book(make_book("M1", ask_heavy_bids, ask_heavy_asks, momentum.now()))

    reversal = StrategyHarness(
        BookImbalanceStrategy, params={"levels": 1, "threshold": 0.70, "interpretation": "reversal"}
    )
    reversal.set_market(make_market("M1"))
    reversal.feed_book(make_book("M1", ask_heavy_bids, ask_heavy_asks, reversal.now()))

    assert momentum.intents[0].side is Side.NO  # ask-heavy -> continuation down -> buy NO
    assert reversal.intents[0].side is Side.YES  # ask-heavy -> fade -> buy YES


def test_imbalance_below_threshold_produces_no_intent() -> None:
    balanced_bids = [("0.49", 55)]
    balanced_asks = [("0.51", 45)]
    h = StrategyHarness(BookImbalanceStrategy, params={"levels": 1, "threshold": 0.70, "interpretation": "momentum"})
    h.set_market(make_market("M1"))
    h.feed_book(make_book("M1", balanced_bids, balanced_asks, h.now()))
    assert h.intents == []


def test_depth_imbalance_uses_configured_levels() -> None:
    # Top level is balanced, but levels 2-3 are heavily bid-skewed - only a levels=3 read
    # should catch this.
    bids = [("0.49", 50), ("0.48", 400), ("0.47", 400)]
    asks = [("0.51", 50), ("0.52", 20), ("0.53", 20)]

    top_only = StrategyHarness(BookImbalanceStrategy, params={"levels": 1, "threshold": 0.70, "interpretation": "momentum"})
    top_only.set_market(make_market("M1"))
    top_only.feed_book(make_book("M1", bids, asks, top_only.now()))
    assert top_only.intents == []

    deep = StrategyHarness(BookImbalanceStrategy, params={"levels": 3, "threshold": 0.70, "interpretation": "momentum"})
    deep.set_market(make_market("M1"))
    deep.feed_book(make_book("M1", bids, asks, deep.now()))
    assert len(deep.intents) == 1
    assert deep.intents[0].side is Side.YES


def test_intent_carries_rationale_and_features() -> None:
    h = StrategyHarness(BookImbalanceStrategy, params={"levels": 1, "threshold": 0.70, "interpretation": "momentum"})
    h.set_market(make_market("M1"))
    h.feed_book(make_book("M1", BID_HEAVY_BIDS, BID_HEAVY_ASKS, h.now()))
    assert len(h.intents) == 1
    intent = h.intents[0]
    assert intent.rationale.strip()
    assert intent.features["depth_imbalance"] is not None
    assert intent.features["interpretation"] == "momentum"
