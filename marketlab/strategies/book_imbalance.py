"""Order-book imbalance (evidence class E).

Params (see ``configs/strategies.yaml``):
    levels          in {1, 3, 5}
    threshold       in {0.60, 0.70, 0.80}
    interpretation  in {momentum, reversal}

We do not assume which way a lopsided book points. ``momentum`` treats heavy resting size
on one side as directional pressure that will keep pushing price that way (continuation).
``reversal`` treats the same lopsided book as exhausted/absorbed pressure about to snap
back (fade). Both are wired up as separate, testable arms that trade *opposite* sides of
the identical book - that disagreement is the experiment.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from marketlab.core.events import BookUpdateEvent, TradeEvent
from marketlab.core.instruments import Side
from marketlab.core.orders import Action, OrderType
from marketlab.signals.orderbook import (
    TradeFlowTracker,
    depth_imbalance,
    microprice,
    spread,
    top_level_imbalance,
    weighted_depth_imbalance,
)
from marketlab.strategies.base import BaseStrategy, clamp_probability

#: Decay used for the weighted-depth imbalance feature - see
#: ``signals/orderbook.py::weighted_depth_imbalance`` for units (per probability-point).
WEIGHTED_DEPTH_DECAY = 20.0
DEFAULT_TRADE_FLOW_WINDOW_SECONDS = 60.0
DEFAULT_SIGNAL_GAIN = 0.3
DEFAULT_MAX_ADJUSTMENT = Decimal("0.08")


class BookImbalanceStrategy(BaseStrategy):
    """Trades resting-size imbalance, in either the momentum or reversal interpretation."""

    name = "book_imbalance"
    version = "1.0.0"
    evidence_class = "E"

    def on_trade(self, event: TradeEvent) -> None:
        st = self.state(event.canonical_id)
        window = float(self.param("trade_flow_window_seconds", DEFAULT_TRADE_FLOW_WINDOW_SECONDS))
        tracker: TradeFlowTracker = st.setdefault("trade_flow", TradeFlowTracker(window))
        tracker.add_trade(event.trade)

    def on_book_update(self, event: BookUpdateEvent) -> None:
        self._evaluate(event.canonical_id)

    # ------------------------------------------------------------------

    def _evaluate(self, canonical_id: str) -> None:
        if self.should_skip(canonical_id) is not None:
            return
        book = self.ctx.book(canonical_id)
        market = self.ctx.market(canonical_id)
        assert book is not None and market is not None
        mid = book.mid
        if mid is None:
            return

        levels = int(self.param("levels", 1))
        threshold = float(self.param("threshold", 0.70))
        interpretation = str(self.param("interpretation", "momentum"))
        cooldown = float(self.param("cooldown_seconds", 30.0))
        now = self.now()

        top_imbalance = top_level_imbalance(book)
        depth_imb = depth_imbalance(book, levels)
        weighted_imb = weighted_depth_imbalance(book, levels, WEIGHTED_DEPTH_DECAY)
        mp = microprice(book)
        sp = spread(book)

        st = self.state(canonical_id)
        tracker: TradeFlowTracker | None = st.get("trade_flow")
        aggressor_imbalance = tracker.aggressor_imbalance(now) if tracker is not None else None
        trade_intensity = tracker.trade_intensity(now) if tracker is not None else None

        features: dict[str, Any] = {
            "levels": levels,
            "threshold": threshold,
            "interpretation": interpretation,
            "top_level_imbalance": top_imbalance,
            "depth_imbalance": depth_imb,
            "weighted_depth_imbalance": weighted_imb,
            "microprice": float(mp) if mp is not None else None,
            "spread": float(sp) if sp is not None else None,
            "aggressor_imbalance": aggressor_imbalance,
            "trade_intensity": trade_intensity,
        }

        if depth_imb is None:
            return

        if depth_imb >= threshold:
            book_state = "bid_heavy"
        elif depth_imb <= 1.0 - threshold:
            book_state = "ask_heavy"
        else:
            return  # imbalance not extreme enough on either side

        # momentum: bid-heavy -> expect continued upward pressure -> buy YES.
        # reversal: bid-heavy -> expect exhausted buying, snap back -> buy NO.
        if interpretation == "momentum":
            side = Side.YES if book_state == "bid_heavy" else Side.NO
        elif interpretation == "reversal":
            side = Side.NO if book_state == "bid_heavy" else Side.YES
        else:
            raise ValueError(f"unknown interpretation: {interpretation!r}")

        if self.on_cooldown(canonical_id, cooldown):
            return

        # Conviction scales with how far past the threshold the imbalance sits, signed to
        # match the side actually chosen so the strategy never claims edge against itself.
        strength = (depth_imb - 0.5) if side is Side.YES else (0.5 - depth_imb)
        strength = abs(strength)
        gain = float(self.param("signal_gain", DEFAULT_SIGNAL_GAIN))
        max_adj = Decimal(str(self.param("max_adjustment", DEFAULT_MAX_ADJUSTMENT)))
        adjustment = min(max_adj, Decimal(str(strength * gain)))
        model_p = clamp_probability(mid + adjustment if side is Side.YES else mid - adjustment)

        price = self.executable_price(book, side, Action.BUY)
        if price is None:
            return
        edge = self.edge_after_costs(model_p, price, market, side)
        quantity = self.sensible_quantity(price)

        features.update(
            {
                "book_state": book_state,
                "model_probability": float(model_p),
                "executable_price": float(price),
                "expected_edge": float(edge),
                "side": side.value,
            }
        )
        intent = self.make_intent(
            canonical_id=canonical_id,
            side=side,
            action=Action.BUY,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=price,
            rationale=(
                f"Book imbalance ({interpretation}): depth_imbalance[{levels}]={depth_imb:.3f} "
                f"is {book_state} (threshold={threshold:.2f}) -> BUY {side.value}; "
                f"model_p={model_p}, executable_price={price}, edge={edge}."
            ),
            features=features,
            model_probability=model_p,
            expected_edge=edge,
        )
        if self.emit_if_profitable(intent):
            self.mark_fired(canonical_id)
