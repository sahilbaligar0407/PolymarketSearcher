"""Time-series momentum on contract prices (evidence class A).

Params (see ``configs/strategies.yaml``):
    lookback_seconds   in {60, 300, 900, 3600}
    threshold          in {0.01, 0.02, 0.04} - a *raw probability-point* entry threshold
    volatility_filter  in {false, true}

Plus PRD-required variant knobs not swept by the default grid (off by default, so the base
grid is pure continuation): ``require_book_confirmation``, ``news_suppress_seconds``,
``reference_symbol`` / ``require_reference_confirmation``.

**Why the entry gate uses a raw delta, and where logit-space actually earns its keep.**
It is tempting to compare :func:`bounded_momentum`'s logit difference directly against
``threshold``, but that does not work: logit's derivative ``1/(p(1-p))`` is *steepest* near
0/1, so the *same* small raw move (0.01 -> 0.02) produces a *larger* logit difference
(~0.70) than a ten-times-bigger raw move through the middle (0.50 -> 0.60, ~0.41) - the
opposite of what a naive "big logit move => trade" rule wants. Comparing the raw delta to
``threshold`` is what actually reproduces the desired behaviour (a 1-point move at the
boundary does not clear a 2-point threshold; a 10-point move through the middle does), and
that comparison is unaffected by *where* in ``[0, 1]`` the market sits. Logit-space is used
instead for what it is actually good at: turning an observed move into a *forward*
probability estimate without producing nonsense near the boundaries. We extrapolate a
fraction of the observed logit move and map back through the sigmoid, so a strong move
starting near 0.98 gets compressed toward 1.0 instead of a raw-probability extrapolation
blowing straight past it.

Reference-asset (BTC spot) series use ordinary time-based :func:`momentum`, per its own
docstring, since they are not bounded to ``[0, 1]``.
"""

from __future__ import annotations

import math
from datetime import datetime
from decimal import Decimal
from typing import Any

from marketlab.core.events import BookUpdateEvent, ExternalPriceEvent, NewsEvent
from marketlab.core.instruments import Side
from marketlab.core.orders import Action, OrderType
from marketlab.signals.orderbook import top_level_imbalance
from marketlab.signals.rolling import RollingWindow
from marketlab.signals.technical import bounded_momentum, momentum
from marketlab.strategies.base import BaseStrategy, clamp_probability

#: Fraction of the observed logit move extrapolated forward as continuation. A momentum
#: strategy is claiming the market hasn't fully caught up yet, not that it knows the
#: terminal probability, so this stays well under 1.0.
DEFAULT_SIGNAL_GAIN = 0.5
#: Cap on the extrapolated logit step, regardless of how large the observed move was - a
#: sanity bound so one big move can't imply an absurdly overconfident forward price.
DEFAULT_MAX_LOGIT_STEP = 1.0
DEFAULT_NEWS_TONE_THRESHOLD = 0.7


class MomentumStrategy(BaseStrategy):
    """Momentum continuation on contract price, with optional confirmation filters."""

    name = "momentum"
    version = "1.0.0"
    evidence_class = "A"

    def __init__(self, strategy_id: str, experiment_id: str, ctx: Any, params: dict | None = None) -> None:
        super().__init__(strategy_id, experiment_id, ctx, params)
        self._reference_series: dict[str, dict[str, Any]] = {}
        self._last_high_impact_news_at: datetime | None = None

    def on_book_update(self, event: BookUpdateEvent) -> None:
        self._evaluate(event.canonical_id)

    def on_external_price(self, event: ExternalPriceEvent) -> None:
        st = self._reference_state(event.symbol)
        history: list[tuple[datetime, Decimal]] = st["history"]
        history.append((event.first_seen_time, event.price))
        lookback = float(self.param("lookback_seconds", 300.0))
        self.prune_series(history, event.first_seen_time, lookback * 4)

    def on_news(self, event: NewsEvent) -> None:
        tone_threshold = float(self.param("news_impact_tone_threshold", DEFAULT_NEWS_TONE_THRESHOLD))
        if self.is_high_impact_news(event, tone_threshold):
            self._last_high_impact_news_at = event.first_seen_time

    # ------------------------------------------------------------------

    def _reference_state(self, symbol: str) -> dict[str, Any]:
        # Reference-asset series are keyed by symbol, not canonical_id, since one symbol
        # (e.g. BTC-USD) informs many contracts. Kept in a dedicated dict rather than
        # per-market state so it survives eviction of any single market's state.
        return self._reference_series.setdefault(symbol, {"history": []})

    def _news_suppressed(self) -> bool:
        suppress_seconds = float(self.param("news_suppress_seconds", 0.0))
        if suppress_seconds <= 0:
            return False
        last = self._last_high_impact_news_at
        if last is None:
            return False
        return (self.now() - last).total_seconds() < suppress_seconds

    def _evaluate(self, canonical_id: str) -> None:
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
        now = self.now()

        st = self.state(canonical_id)
        history: list[tuple[datetime, Decimal]] = st.setdefault("price_history", [])
        history.append((now, mid))
        self.prune_series(history, now, lookback * 4)

        returns: RollingWindow = st.setdefault("returns", RollingWindow(30))
        prev = st.get("prev_mid")
        if prev is not None:
            returns.push(float(mid - prev))
        st["prev_mid"] = mid

        anchor = self.find_anchor(history, lookback, now)
        if anchor is None:
            return
        raw_delta = mid - anchor
        logit_diff = bounded_momentum(mid, anchor)
        features: dict[str, Any] = {
            "p_now": float(mid),
            "p_anchor": float(anchor),
            "raw_delta": float(raw_delta),
            "logit_diff": logit_diff,
            "lookback_seconds": lookback,
            "threshold": threshold,
        }
        # Entry gate is a raw probability-point threshold (see module docstring for why
        # logit-diff magnitude cannot be compared directly to `threshold`).
        if abs(raw_delta) < Decimal(str(threshold)):
            return
        if self.on_cooldown(canonical_id, cooldown):
            return

        if self._news_suppressed():
            features["news_suppressed"] = True
            return

        volatility_filter = bool(self.param("volatility_filter", False))
        vol = returns.std()
        features["volatility"] = vol
        features["volatility_filter"] = volatility_filter
        if volatility_filter:
            max_vol = float(self.param("volatility_threshold", 0.05))
            if vol is not None and vol > max_vol:
                return

        side = Side.YES if raw_delta > 0 else Side.NO

        if bool(self.param("require_book_confirmation", False)):
            imbalance = top_level_imbalance(book)
            features["book_imbalance"] = imbalance
            if imbalance is None:
                return
            # Continuation up needs bid-heavy confirmation; continuation down needs
            # ask-heavy confirmation. Disagreement means the book doesn't back the move.
            if side is Side.YES and imbalance <= 0.5:
                return
            if side is Side.NO and imbalance >= 0.5:
                return

        reference_symbol = self.param("reference_symbol")
        if reference_symbol and bool(self.param("require_reference_confirmation", False)):
            ref_history = self._reference_state(str(reference_symbol))["history"]
            ref_mom = momentum(ref_history, lookback, now)
            features["reference_momentum"] = ref_mom
            if ref_mom is None:
                return
            if side is Side.YES and ref_mom <= 0:
                return
            if side is Side.NO and ref_mom >= 0:
                return

        # Extrapolate a fraction of the observed logit move forward, then map back through
        # the sigmoid - this is where bounded_momentum earns its keep: the same `gain`
        # fraction of a move starting near an extreme lands close to the extreme instead of
        # overshooting past 0/1 the way extrapolating the raw probability would.
        gain = float(self.param("signal_gain", DEFAULT_SIGNAL_GAIN))
        max_step = float(self.param("max_logit_step", DEFAULT_MAX_LOGIT_STEP))
        step = max(-max_step, min(max_step, logit_diff * gain))
        predicted_logit = _logit(float(mid)) + step
        model_p = clamp_probability(Decimal(str(_sigmoid(predicted_logit))))

        price = self.executable_price(book, side, Action.BUY)
        if price is None:
            return
        edge = self.edge_after_costs(model_p, price, market, side)
        quantity = self.sensible_quantity(price)

        features.update(
            {
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
                f"Momentum continuation: logit move {logit_diff:.4f} over {lookback:.0f}s "
                f"(p {anchor}->{mid}) exceeds threshold {threshold:.3f}; "
                f"model_p={model_p}, executable_price={price}, edge={edge}."
            ),
            features=features,
            model_probability=model_p,
            expected_edge=edge,
        )
        if self.emit_if_profitable(intent):
            self.mark_fired(canonical_id)


_LOGIT_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, _LOGIT_EPS), 1.0 - _LOGIT_EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


__all__ = ["MomentumStrategy"]
