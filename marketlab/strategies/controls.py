"""The control group: three strategies that exist to be compared against, not to win.

``RandomDirectionStrategy`` and ``FadeStrategy`` are cheap ways to find out whether a
"real" strategy's apparent edge survives the simplest null hypotheses: "is this just
directional trading at this frequency" and "is the thing I'm fading actually anti-correlated
with outcomes". ``DoNothingStrategy`` is the zero line every leaderboard needs.
"""

from __future__ import annotations

import random
from decimal import Decimal
from typing import Any

from marketlab.core.events import (
    BookUpdateEvent,
    TimerEvent,
    TraderActionEvent,
)
from marketlab.core.instruments import ONE, Side
from marketlab.core.orders import Action, OrderType
from marketlab.signals.technical import bounded_momentum
from marketlab.strategies.base import BaseStrategy, clamp_probability


class RandomDirectionStrategy(BaseStrategy):
    """Trades a random direction at a configurable frequency. Deterministic given a seed.

    This is the frequency-matched null: if a real strategy's Sharpe doesn't clear what
    picking a coin-flip direction at the same cadence achieves, the real strategy has
    nothing to show for its signal.
    """

    name = "random_control"
    version = "1.0.0"
    evidence_class = "B"

    def __init__(self, strategy_id: str, experiment_id: str, ctx: Any, params: dict | None = None) -> None:
        super().__init__(strategy_id, experiment_id, ctx, params)
        # A private RNG seeded once at construction. Given the same seed and the same
        # sequence of timer ticks / eligible markets, every draw is reproduced exactly -
        # that is the whole point of a control.
        self._rng = random.Random(int(self.param("seed", 0)))

    def on_timer(self, event: TimerEvent) -> None:
        del event
        interval = float(self.param("trade_interval_seconds", 300.0))
        quantity_risk_fraction = Decimal(str(self.param("risk_fraction", "0.05")))
        for market in self.ctx.markets():
            canonical_id = market.canonical_id
            if self.should_skip(canonical_id) is not None:
                continue
            if self.on_cooldown(canonical_id, interval):
                continue
            # Always draw, even markets that end up skipped below, so the RNG stream is a
            # deterministic function of (seed, tick, market list) alone.
            side = Side.YES if self._rng.random() < 0.5 else Side.NO
            book = self.ctx.book(canonical_id)
            assert book is not None
            price = self.executable_price(book, side, Action.BUY)
            if price is None:
                continue
            quantity = self.sensible_quantity(price, risk_fraction=quantity_risk_fraction)
            intent = self.make_intent(
                canonical_id=canonical_id,
                side=side,
                action=Action.BUY,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                limit_price=price,
                rationale=(
                    f"Random control: seed={self.param('seed', 0)} drew side={side.value} "
                    f"at price={price} (no signal, frequency-matched null)."
                ),
                features={
                    "seed": self.param("seed", 0),
                    "drawn_side": side.value,
                    "price": float(price),
                },
                model_probability=None,
                expected_edge=None,
            )
            # The control is not claiming edge, so it is exempt from the positive-edge
            # gate other strategies must pass - that gate would make a null control
            # untestable by construction. Risk sizing/limits still apply downstream.
            self.emit(intent)
            self.mark_fired(canonical_id)


class FadeStrategy(BaseStrategy):
    """Takes the opposite side of a named signal.

    Two ways to implement "fade another strategy" were available: (a) wire a live
    in-process signal bus so this strategy subscribes to another *running instance's*
    emitted intents, or (b) have this strategy recompute the target signal itself, using
    only the shared event stream, and invert it. **(b) is what's implemented here** - it
    needs no cross-instance wiring or shared mutable registry, works identically in a
    backtest or a live run, and does not create an import/lifecycle dependency between
    unrelated strategy instances (or, for ``copy_trader``, on Team STRATEGIES-B's module).
    Config selects which named signals to fade via ``params["fades"]``, e.g.
    ``["momentum", "copy_trader"]``.
    """

    name = "fade_control"
    version = "1.0.0"
    evidence_class = "B"

    def on_book_update(self, event: BookUpdateEvent) -> None:
        if "momentum" in self._fades():
            self._fade_momentum(event.canonical_id)

    def on_trader_action(self, event: TraderActionEvent) -> None:
        if "copy_trader" in self._fades():
            self._fade_copy_trader(event)

    # ------------------------------------------------------------------

    def _fades(self) -> tuple[str, ...]:
        raw = self.param("fades", ())
        return tuple(raw) if not isinstance(raw, str) else (raw,)

    def _fade_momentum(self, canonical_id: str) -> None:
        """Recompute a minimal continuation-momentum signal and take the other side."""
        if self.should_skip(canonical_id) is not None:
            return
        book = self.ctx.book(canonical_id)
        market = self.ctx.market(canonical_id)
        assert book is not None and market is not None
        mid = book.mid
        if mid is None:
            return

        lookback = float(self.param("lookback_seconds", 300.0))
        threshold = float(self.param("threshold", 0.02))
        cooldown = float(self.param("cooldown_seconds", lookback))

        st = self.state(canonical_id)
        history: list[tuple[Any, Decimal]] = st.setdefault("price_history", [])
        now = self.now()
        self.append_sample(history, now, mid)
        self.prune_series(history, now, lookback * 4)

        anchor = self.find_anchor(history, lookback, now)
        if anchor is None or self.on_cooldown(canonical_id, cooldown):
            return
        logit_diff = bounded_momentum(mid, anchor)
        if abs(logit_diff) < threshold:
            return

        # The target signal (momentum continuation) would BUY YES on a rise, BUY NO on a
        # fall. Fading it means taking exactly the opposite side.
        momentum_side = Side.YES if logit_diff > 0 else Side.NO
        fade_side = momentum_side.opposite
        price = self.executable_price(book, fade_side, Action.BUY)
        if price is None:
            return
        model_p = clamp_probability(ONE - mid if fade_side is Side.NO else mid)
        edge = self.edge_after_costs(model_p, price, market, fade_side)
        quantity = self.sensible_quantity(price)
        intent = self.make_intent(
            canonical_id=canonical_id,
            side=fade_side,
            action=Action.BUY,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=price,
            rationale=(
                f"Fading momentum: logit move {logit_diff:.4f} over {lookback:.0f}s "
                f"(p {anchor}->{mid}) would signal BUY {momentum_side.value}; "
                f"fade takes BUY {fade_side.value} instead."
            ),
            features={
                "logit_diff": logit_diff,
                "lookback_seconds": lookback,
                "threshold": threshold,
                "momentum_side": momentum_side.value,
                "fade_side": fade_side.value,
                "price": float(price),
            },
            model_probability=model_p,
            expected_edge=edge,
        )
        if self.emit_if_profitable(intent):
            self.mark_fired(canonical_id)

    def _fade_copy_trader(self, event: TraderActionEvent) -> None:
        """Recompute the naive "copy this wallet's side" signal and take the other side."""
        canonical_id = event.canonical_id
        if not canonical_id or event.side is None:
            return
        if self.should_skip(canonical_id) is not None:
            return
        book = self.ctx.book(canonical_id)
        market = self.ctx.market(canonical_id)
        assert book is not None and market is not None
        delay = float(self.param("follower_delay_seconds", 5.0))
        if (self.now() - event.first_seen_time).total_seconds() < delay:
            return
        cooldown = float(self.param("cooldown_seconds", 30.0))
        if self.on_cooldown(canonical_id, cooldown):
            return

        naive_copy_side = event.side  # the raw "buy what they bought" signal
        fade_side = naive_copy_side.opposite
        price = self.executable_price(book, fade_side, Action.BUY)
        if price is None:
            return
        mid = book.mid
        if mid is None:
            return
        model_p = clamp_probability(ONE - mid if fade_side is Side.NO else mid)
        edge = self.edge_after_costs(model_p, price, market, fade_side)
        quantity = self.sensible_quantity(price)
        intent = self.make_intent(
            canonical_id=canonical_id,
            side=fade_side,
            action=Action.BUY,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=price,
            rationale=(
                f"Fading copy_trader: wallet={event.wallet} took {naive_copy_side.value} "
                f"on {canonical_id}; fade takes BUY {fade_side.value} instead, "
                f"{delay:.0f}s after first_seen."
            ),
            features={
                "wallet": event.wallet,
                "naive_copy_side": naive_copy_side.value,
                "fade_side": fade_side.value,
                "follower_delay_seconds": delay,
                "price": float(price),
            },
            model_probability=model_p,
            expected_edge=edge,
        )
        if self.emit_if_profitable(intent):
            self.mark_fired(canonical_id)


class DoNothingStrategy(BaseStrategy):
    """The null benchmark. Never emits an intent or a forecast."""

    name = "do_nothing"
    version = "1.0.0"
    evidence_class = "B"
