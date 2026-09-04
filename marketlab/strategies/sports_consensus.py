"""Sportsbook-consensus relative value, gated hard by the pregame/live boundary.

**Never compare American odds directly to a contract price.** Every conversion goes
through :func:`marketlab.core.probability.american_to_probability` and
:func:`marketlab.signals.sports.consensus_probability` (which removes each bookmaker's
own vig before averaging - see that function's docstring for why averaging first and
normalizing last would be wrong) rather than being re-derived here.

**The headline rule this module exists to enforce**: a sports strategy must stop using
pregame state the instant the game goes live. Every pregame signal (sportsbook consensus,
Elo baseline, line movement) is gated behind
:func:`marketlab.signals.sports.should_use_pregame_signal`, which returns ``False`` the
moment :class:`~marketlab.core.events.SportsStateEvent.started` is ``True`` - regardless
of score, clock, or anything else. See
``tests/unit/test_strategies_sports.py::test_pregame_signal_stops_the_instant_game_starts``.

``source`` interprets the params grid as follows (a documented judgment call - the PRD
names the three arms without spelling out their exact semantics):

* ``vig_free_consensus`` - model probability is the multi-book, per-book-de-vigged
  consensus (the standard arm).
* ``market_only`` - model probability comes from the Elo baseline alone, with **no**
  dependency on a paid sportsbook-odds feed at all. This tests whether a free, purely
  structural rating model has anything to add on its own.
* ``disagreement`` - same consensus model as ``vig_free_consensus``, but additionally
  requires the Elo baseline to agree in *direction* with the consensus before trading -
  a stricter, rarer, higher-conviction signal that two independent sources agree the
  market is mispriced, rather than one source alone.

**Order behaviour at official game start.** A strategy has no broker access and cannot
call ``cancel()`` directly (see ``marketlab.core.broker.Broker`` - only the runner holds
a broker handle). The two things this layer *can* guarantee are: (1) it immediately stops
emitting new pregame-signal intents the instant the game goes live (enforced above), and
(2) for any resting order it believes it still has open on this contract (tracked via
``on_order_update``), it emits a superseding ``replaces_order_id`` intent with
``time_in_force=IOC`` the instant the game starts, so a fill-or-nothing attempt is made
and the order does not linger unattended through live play. This uses only mechanisms
already in the frozen :class:`~marketlab.core.orders.OrderIntent` contract; it is exempt
from the positive-edge gate because it is risk housekeeping, not a profit-seeking trade.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from marketlab.core.events import BookUpdateEvent, ExternalPriceEvent, SportsStateEvent
from marketlab.core.instruments import ONE, Side
from marketlab.core.orders import Action, Order, OrderStatus, OrderType, TimeInForce
from marketlab.core.probability import american_to_probability
from marketlab.core.strategy import ProbabilityForecast
from marketlab.signals.sports import (
    GameState,
    consensus_probability,
    elo_win_probability,
    should_use_pregame_signal,
)
from marketlab.strategies.base import BaseStrategy, clamp_probability

DEFAULT_HOME_ADVANTAGE = 65.0
DEFAULT_ELO_RATING = 1500.0


class SportsConsensusStrategy(BaseStrategy):
    """Trades a Kalshi sports contract against a vig-free sportsbook consensus (or Elo),
    strictly retired the instant the underlying game goes live.
    """

    name = "sports_consensus"
    version = "1.0.0"
    evidence_class = "B"

    def __init__(self, strategy_id: str, experiment_id: str, ctx: Any, params: dict | None = None) -> None:
        super().__init__(strategy_id, experiment_id, ctx, params)
        #: game_id -> GameState, updated by on_sports_state. The join key between a
        #: SportsStateEvent's game_id and a Kalshi market is NormalizedMarket.event_id -
        #: a documented judgment call: no field literally named "game_id" exists on
        #: NormalizedMarket, and event_id is the natural, already-used correlation key
        #: (the risk gateway uses it the same way for its correlated-cluster cap).
        self._game_states: dict[str, GameState] = {}
        #: game_id -> {bookmaker: {"home": american_odds, "away": american_odds}}
        self._book_odds: dict[str, dict[str, dict[str, Decimal]]] = {}
        #: game_id -> Elo ratings dict shared across the pregame baseline for this game.
        self._elo_ratings: dict[str, float] = {}
        #: canonical_id -> last known open order_id, for the game-start cancel gesture.
        self._resting_orders: dict[str, str] = {}

    # ------------------------------------------------------------------ inputs

    def on_sports_state(self, event: SportsStateEvent) -> None:
        previous = self._game_states.get(event.game_id)
        was_pregame = previous is None or should_use_pregame_signal(previous)
        state = GameState(
            game_id=event.game_id,
            league=event.league,
            started=event.started,
            final=event.final,
            home_score=event.home_score,
            away_score=event.away_score,
        )
        self._game_states[event.game_id] = state
        if was_pregame and not should_use_pregame_signal(state):
            self._handle_game_start(event.game_id)

    def on_external_price(self, event: ExternalPriceEvent) -> None:
        # Convention (documented judgment call): symbol = "<game_id>:<bookmaker>:<home|away>",
        # price = raw American odds for that outcome. This keeps de-vig math entirely in
        # marketlab.core.probability / marketlab.signals.sports - nothing here re-derives it.
        parts = event.symbol.split(":")
        if len(parts) != 3 or parts[2] not in ("home", "away"):
            return
        game_id, bookmaker, outcome = parts
        book = self._book_odds.setdefault(game_id, {}).setdefault(bookmaker, {})
        book[outcome] = event.price

    def on_order_update(self, order: Order) -> None:
        if order.status is OrderStatus.OPEN:
            self._resting_orders[order.canonical_id] = order.order_id
        elif order.is_terminal:
            self._resting_orders.pop(order.canonical_id, None)

    # ------------------------------------------------------------------ per-market evaluation

    def on_book_update(self, event: BookUpdateEvent) -> None:
        self._evaluate(event.canonical_id)

    def _game_state_for(self, market_event_id: str) -> GameState | None:
        return self._game_states.get(market_event_id)

    def _consensus_for(self, game_id: str) -> Decimal | None:
        raw = self._book_odds.get(game_id, {})
        book_probs: dict[str, list[Decimal]] = {}
        for bookmaker, sides in raw.items():
            if "home" not in sides or "away" not in sides:
                continue
            book_probs[bookmaker] = [
                american_to_probability(int(sides["home"])),
                american_to_probability(int(sides["away"])),
            ]
        if not book_probs:
            return None
        return consensus_probability(book_probs)

    def _elo_probability(self, game_id: str) -> Decimal | None:
        # Elo needs home/away identity, which we don't have a dedicated field for here;
        # a neutral, symmetric baseline (both teams start at the default rating) is used
        # in the absence of a tracked rating history for this game - honestly weak, but
        # exactly what "market_only: no external odds dependency" promises.
        ratings = self._elo_ratings
        p_home = elo_win_probability(
            ratings.get(f"{game_id}:home", DEFAULT_ELO_RATING),
            ratings.get(f"{game_id}:away", DEFAULT_ELO_RATING),
            DEFAULT_HOME_ADVANTAGE,
        )
        return clamp_probability(Decimal(str(round(p_home, 6))))

    def _evaluate(self, canonical_id: str) -> None:
        if self.should_skip(canonical_id) is not None:
            return
        market = self.ctx.market(canonical_id)
        book = self.ctx.book(canonical_id)
        assert market is not None and book is not None

        game_state = self._game_state_for(market.event_id)
        if game_state is not None and not should_use_pregame_signal(game_state):
            # Live or final: the pregame signal is retired unconditionally, regardless of
            # `pregame_only` - that param only controls whether this strategy trades AT
            # ALL once live, never whether the pregame signal itself may keep firing.
            return
        if bool(self.param("pregame_only", True)) is False and game_state is not None and game_state.started:
            # Explicitly out of scope for v1: an in-play model is a different strategy.
            return

        source = str(self.param("source", "vig_free_consensus"))
        consensus = self._consensus_for(market.event_id)
        elo_p = self._elo_probability(market.event_id)

        if source == "market_only":
            model_probability = elo_p
        else:
            model_probability = consensus
            if source == "disagreement" and consensus is not None and elo_p is not None:
                consensus_favors_home = consensus >= Decimal("0.5")
                elo_favors_home = elo_p >= Decimal("0.5")
                if consensus_favors_home != elo_favors_home:
                    return

        if model_probability is None:
            return

        price_yes = self.executable_price(book, Side.YES, Action.BUY)
        price_no = self.executable_price(book, Side.NO, Action.BUY)
        candidates: list[tuple[Side, Decimal, Decimal]] = []
        if price_yes is not None:
            candidates.append((Side.YES, price_yes, self.edge_after_costs(model_probability, price_yes, market, Side.YES)))
        if price_no is not None:
            no_p = clamp_probability(ONE - model_probability)
            candidates.append((Side.NO, price_no, self.edge_after_costs(no_p, price_no, market, Side.NO)))
        if not candidates:
            return
        side, price, edge = max(candidates, key=lambda c: c[2])

        min_edge = Decimal(str(self.param("min_edge", "0.03")))
        if edge < min_edge:
            return
        cooldown = float(self.param("cooldown_seconds", 60.0))
        if self.on_cooldown(canonical_id, cooldown):
            return

        features: dict[str, Any] = {
            "source": source,
            "consensus_probability": float(consensus) if consensus is not None else None,
            "elo_probability": float(elo_p) if elo_p is not None else None,
            "market_probability": float(book.mid) if book.mid is not None else None,
            "model_probability": float(model_probability),
            "side": side.value,
            "game_started": game_state.started if game_state is not None else None,
        }
        self.forecast(
            ProbabilityForecast(
                strategy_id=self.strategy_id,
                experiment_id=self.experiment_id,
                canonical_id=canonical_id,
                as_of=self.now(),
                p_yes=model_probability,
                market_probability=book.mid,
                features=features,
                rationale=f"source={source} model_p={model_probability} vs market.",
            )
        )
        intent = self.make_intent(
            canonical_id=canonical_id,
            side=side,
            action=Action.BUY,
            quantity=self.sensible_quantity(price),
            order_type=OrderType.LIMIT,
            limit_price=price,
            rationale=(
                f"Sports consensus ({source}): model_p={model_probability} vs "
                f"executable {side.value} price {price}; edge {edge} clears "
                f"min_edge {min_edge}."
            ),
            features={**features, "executable_price": float(price)},
            model_probability=model_probability,
            expected_edge=edge,
        )
        if self.emit_if_profitable(intent):
            self.mark_fired(canonical_id)

    # ------------------------------------------------------------------ game-start housekeeping

    def _handle_game_start(self, game_id: str) -> None:
        for market in self.ctx.markets():
            if market.event_id != game_id:
                continue
            order_id = self._resting_orders.get(market.canonical_id)
            if order_id is None:
                continue
            book = self.ctx.book(market.canonical_id)
            if book is None or book.best_bid is None:
                continue
            # Best-effort IOC re-quote at the current bid, tagged as replacing the stale
            # resting order - see module docstring for why this is the strongest
            # guarantee a strategy (no broker access) can give.
            intent = self.make_intent(
                canonical_id=market.canonical_id,
                side=Side.YES,
                action=Action.SELL,
                quantity=1,
                order_type=OrderType.LIMIT,
                limit_price=book.best_bid,
                time_in_force=TimeInForce.IOC,
                rationale=(
                    f"Game {game_id} started: retiring pregame resting order {order_id} "
                    "with an IOC re-quote rather than leaving it unattended through live play."
                ),
                features={"game_id": game_id, "retired_order_id": order_id},
                replaces_order_id=order_id,
            )
            self.emit(intent)  # housekeeping, exempt from the positive-edge gate
            self._resting_orders.pop(market.canonical_id, None)


__all__ = ["SportsConsensusStrategy"]
