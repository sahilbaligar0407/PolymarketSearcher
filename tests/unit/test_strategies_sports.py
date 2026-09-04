"""Unit tests for marketlab/strategies/sports_consensus.py."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from marketlab.clock import SimulatedClock
from marketlab.core.events import BookUpdateEvent, ExternalPriceEvent, SportsStateEvent
from marketlab.core.instruments import (
    BookLevel,
    Category,
    MarketStatus,
    NormalizedMarket,
    OrderBook,
    Side,
    Venue,
)
from marketlab.core.orders import Action, Order, OrderStatus, OrderType, TimeInForce
from marketlab.core.probability import american_to_probability, remove_vig
from marketlab.core.strategy import StrategyContext
from marketlab.strategies.sports_consensus import SportsConsensusStrategy

TS = datetime(2026, 1, 1, tzinfo=UTC)
CANONICAL_ID = "kalshi:chiefs-win-w1"
GAME_ID = "game-1"


def _market() -> NormalizedMarket:
    return NormalizedMarket(
        canonical_id=CANONICAL_ID,
        venue=Venue.KALSHI,
        venue_market_id="KXNFLGAME-TEST",
        event_id=GAME_ID,
        title="Will the Chiefs win?",
        category=Category.SPORTS,
        status=MarketStatus.OPEN,
        min_order=1,
    )


def _book(bid: str = "0.40", ask: str = "0.42") -> OrderBook:
    return OrderBook(
        canonical_id=CANONICAL_ID,
        venue=Venue.KALSHI,
        timestamp=TS,
        bids=(BookLevel(price=Decimal(bid), size=100),),
        asks=(BookLevel(price=Decimal(ask), size=100),),
    )


def _ctx(book: OrderBook) -> StrategyContext:
    clock = SimulatedClock(TS)
    market = _market()
    return StrategyContext(clock=clock, books={CANONICAL_ID: book}, markets={CANONICAL_ID: market}, marks={})


def _feed_book_odds(strat: SportsConsensusStrategy, home_odds: int, away_odds: int, bookmaker: str = "book1") -> None:
    strat.on_external_price(
        ExternalPriceEvent(
            symbol=f"{GAME_ID}:{bookmaker}:home",
            price=Decimal(home_odds),
            event_time=TS,
            first_seen_time=TS,
        )
    )
    strat.on_external_price(
        ExternalPriceEvent(
            symbol=f"{GAME_ID}:{bookmaker}:away",
            price=Decimal(away_odds),
            event_time=TS,
            first_seen_time=TS,
        )
    )


# ---------------------------------------------------------------------------
# American odds are never compared raw - -110/-110 de-vigs to 0.50
# ---------------------------------------------------------------------------


def test_minus_110_both_sides_devigs_to_half() -> None:
    home_raw = american_to_probability(-110)
    away_raw = american_to_probability(-110)
    home_fair, away_fair = remove_vig([home_raw, away_raw])
    assert home_fair == Decimal("0.5000")
    assert away_fair == Decimal("0.5000")


def test_strategy_consensus_uses_devigged_probability_not_raw_odds() -> None:
    book = _book(bid="0.55", ask="0.57")
    ctx = _ctx(book)
    strat = SportsConsensusStrategy("s1", "e1", ctx, params={"source": "vig_free_consensus", "min_edge": "0.01"})
    _feed_book_odds(strat, home_odds=-110, away_odds=-110)

    strat.on_book_update(BookUpdateEvent(book=book, event_time=TS, first_seen_time=TS))

    forecasts = strat.drain_forecasts()
    assert len(forecasts) == 1
    # -110/-110 raw is 0.5238 each (sums to > 1); the consensus fed to the model must be
    # the de-vigged 0.50, never the raw vig-inclusive number.
    assert forecasts[0].p_yes == Decimal("0.5000")
    assert forecasts[0].p_yes != american_to_probability(-110)


# ---------------------------------------------------------------------------
# Headline test: pregame signal stops the instant the game starts
# ---------------------------------------------------------------------------


def test_pregame_signal_stops_the_instant_game_starts() -> None:
    book = _book(bid="0.30", ask="0.32")  # far from the 0.50 consensus - a big pregame edge
    ctx = _ctx(book)
    strat = SportsConsensusStrategy("s1", "e1", ctx, params={"source": "vig_free_consensus", "min_edge": "0.01"})
    _feed_book_odds(strat, home_odds=-110, away_odds=-110)

    strat.on_sports_state(
        SportsStateEvent(game_id=GAME_ID, started=False, final=False, event_time=TS, first_seen_time=TS)
    )
    strat.on_book_update(BookUpdateEvent(book=book, event_time=TS, first_seen_time=TS))
    pregame_intents = strat.generate_intents()
    assert len(pregame_intents) == 1  # a real edge exists pregame

    # Now the game starts.
    strat.on_sports_state(
        SportsStateEvent(game_id=GAME_ID, started=True, final=False, event_time=TS, first_seen_time=TS)
    )
    strat.drain_forecasts()
    strat.on_book_update(BookUpdateEvent(book=book, event_time=TS, first_seen_time=TS))
    live_intents = strat.generate_intents()
    assert live_intents == []


def test_pregame_signal_stops_even_with_other_pregame_looking_fields() -> None:
    book = _book(bid="0.30", ask="0.32")
    ctx = _ctx(book)
    strat = SportsConsensusStrategy("s1", "e1", ctx, params={"source": "vig_free_consensus", "min_edge": "0.01"})
    _feed_book_odds(strat, home_odds=-110, away_odds=-110)

    # started=True must dominate regardless of score/period fields.
    strat.on_sports_state(
        SportsStateEvent(
            game_id=GAME_ID,
            started=True,
            final=False,
            home_score=0,
            away_score=0,
            event_time=TS,
            first_seen_time=TS,
        )
    )
    strat.on_book_update(BookUpdateEvent(book=book, event_time=TS, first_seen_time=TS))
    assert strat.generate_intents() == []


def test_game_start_retires_a_tracked_resting_order() -> None:
    book = _book(bid="0.30", ask="0.32")
    ctx = _ctx(book)
    strat = SportsConsensusStrategy("s1", "e1", ctx, params={"source": "vig_free_consensus", "min_edge": "0.01"})

    order = Order(
        intent_id="int_1",
        strategy_id="s1",
        experiment_id="e1",
        canonical_id=CANONICAL_ID,
        venue=Venue.KALSHI,
        side=Side.YES,
        action=Action.BUY,
        order_type=OrderType.LIMIT,
        quantity=5,
        limit_price=Decimal("0.30"),
        time_in_force=TimeInForce.GTC,
        status=OrderStatus.OPEN,
        decision_timestamp=TS,
    )
    strat.on_order_update(order)

    strat.on_sports_state(
        SportsStateEvent(game_id=GAME_ID, started=True, final=False, event_time=TS, first_seen_time=TS)
    )

    intents = strat.generate_intents()
    assert len(intents) == 1
    cancel_intent = intents[0]
    assert cancel_intent.replaces_order_id == order.order_id
    assert cancel_intent.time_in_force is TimeInForce.IOC
