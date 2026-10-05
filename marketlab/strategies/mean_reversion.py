"""Mean reversion on contract prices (evidence class A).

Params (see ``configs/strategies.yaml``):
    lookback_seconds  in {300, 900, 3600}
    entry_z           in {1.5, 2.0, 2.5}
    exit_z            in {0.5}

Run entirely separately from :mod:`marketlab.strategies.momentum`. The whole point of
running both is finding out *which regime* (trend-following or mean-reverting) actually
describes a given market/horizon; blending the two signals into one score would throw that
question away before it could be answered.

Four reversion signals are computed, per the PRD:

1. **z-score of short-term returns** - the primary, always-available entry trigger.
2. **deviation from a trade-print VWAP** - via :class:`TradeFlowTracker`, recorded as a
   feature; requires trade prints, which not every market has enough of.
3. **deviation from a slow EWMA "consensus" fair value** - a longer-halflife EWMA of the
   midpoint stands in for "what the market has consistently thought this is worth",
   distinct from the fast, noisy last tick.
4. **cross-venue divergence against an injected consensus price** - fed via
   :class:`~marketlab.core.events.ExternalPriceEvent` (``implied_probability``, e.g. a
   Polymarket-derived read on the same event). If no such event has arrived for this
   market, this arm is simply absent from ``features`` rather than guessed at.

Only signal (1) gates the entry decision below, to keep the test surface (and the
promotion evidence) unambiguous: one clear, always-on trigger, with the other three
recorded for analysis. See the strategy's class docstring for how they could be split into
separate variants later.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from marketlab.core.events import BookUpdateEvent, ExternalPriceEvent, TradeEvent
from marketlab.core.instruments import Side
from marketlab.core.orders import Action, OrderType
from marketlab.signals.orderbook import TradeFlowTracker
from marketlab.signals.rolling import EWMA, RollingWindow
from marketlab.signals.technical import deviation_from_vwap, zscore
from marketlab.strategies.base import BaseStrategy, clamp_probability

#: How much of the observed deviation-from-mean we expect to close as "reversion" - a
#: partial, not full, forecasted reversion (the strategy is not claiming to know the exact
#: fair value, only that the current print is stretched relative to its recent range).
DEFAULT_REVERSION_GAIN = 0.5
DEFAULT_MAX_ADJUSTMENT = 0.10
#: Halflife of the slow "consensus" EWMA, as a multiple of lookback_seconds.
CONSENSUS_HALFLIFE_MULTIPLIER = 3.0
#: How stale an injected cross-venue consensus price may be before it's ignored.
MAX_CONSENSUS_AGE_SECONDS = 120.0


class MeanReversionStrategy(BaseStrategy):
    """Fades stretched short-term moves back toward the market's own recent center."""

    name = "mean_reversion"
    version = "1.1.0"  # 1.1.0: FINDINGS 57
    evidence_class = "A"

    def on_trade(self, event: TradeEvent) -> None:
        st = self.state(event.canonical_id)
        lookback = float(self.param("lookback_seconds", 900.0))
        tracker: TradeFlowTracker = st.setdefault("trade_flow", TradeFlowTracker(lookback))
        tracker.add_trade(event.trade)

    def on_external_price(self, event: ExternalPriceEvent) -> None:
        # `symbol` doubles as the canonical_id when a signal is injected per-market (as
        # opposed to momentum's reference-asset series, which is keyed by a spot symbol
        # like BTC-USD). If nothing matches this market's canonical_id, the arm is simply
        # never populated for it - "skip that arm" per the PRD.
        if event.implied_probability is None:
            return
        st = self.state(event.symbol)
        st["consensus_price"] = event.implied_probability
        st["consensus_time"] = event.first_seen_time

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

        lookback = float(self.param("lookback_seconds", 900.0))
        entry_z = float(self.param("entry_z", 2.0))
        exit_z = float(self.param("exit_z", 0.5))
        cooldown = float(self.param("cooldown_seconds", lookback / 4))
        now = self.now()

        st = self.state(canonical_id)
        returns: RollingWindow = st.setdefault("returns", RollingWindow(60))
        consensus_ewma: EWMA = st.setdefault(
            "consensus_ewma", EWMA(lookback * CONSENSUS_HALFLIFE_MULTIPLIER)
        )
        prev = st.get("prev_mid")
        if prev is not None:
            returns.push(float(mid - prev))
        st["prev_mid"] = mid
        consensus_ewma.update(float(mid), now)

        features: dict[str, Any] = {"lookback_seconds": lookback, "entry_z": entry_z, "exit_z": exit_z}

        mean = returns.mean()
        std = returns.std()
        latest_return = returns.last()
        z = zscore(latest_return, mean, std) if latest_return is not None and mean is not None else None
        features["return_zscore"] = z

        # Arm 2: deviation from trade-print VWAP (diagnostic only).
        tracker: TradeFlowTracker | None = st.get("trade_flow")
        if tracker is not None:
            vwap_value = tracker.vwap(now)
            features["vwap"] = float(vwap_value) if vwap_value is not None else None
            features["vwap_deviation"] = deviation_from_vwap(mid, vwap_value)

        # Arm 3: deviation from the slow EWMA "consensus" fair value (diagnostic only).
        consensus_fair = consensus_ewma.value
        features["ewma_consensus"] = consensus_fair
        features["ewma_deviation"] = (float(mid) - consensus_fair) if consensus_fair is not None else None

        # Arm 4: injected cross-venue consensus, if one has arrived and is fresh.
        consensus_price: Decimal | None = st.get("consensus_price")
        consensus_time: datetime | None = st.get("consensus_time")
        if consensus_price is not None and consensus_time is not None:
            age = (now - consensus_time).total_seconds()
            if age <= MAX_CONSENSUS_AGE_SECONDS:
                features["cross_venue_consensus"] = float(consensus_price)
                features["cross_venue_deviation"] = float(mid - consensus_price)
            # else: stale - omitted rather than reported on a guess.

        if z is None:
            return

        # Hysteresis: once armed, an extreme trips an entry; the arm doesn't reset until
        # the z-score has actually come back inside the exit band, so a single stretched
        # episode can't fire repeatedly on every tick while it stays extreme.
        armed = st.get("armed", True)
        if abs(z) <= exit_z:
            st["armed"] = True
        if abs(z) < entry_z:
            return
        if not armed:
            return
        if self.on_cooldown(canonical_id, cooldown):
            return

        # Overbought (z > 0, price ran up relative to its recent mean) -> bet on reversion
        # down -> buy NO. Oversold (z < 0) -> buy YES.
        side = Side.NO if z > 0 else Side.YES

        # The stretched quantity is the latest *move* (a return), so revert a fraction of
        # that move. Until 1.1.0 this was `mid - mean_return`: a price minus a ~0 return,
        # i.e. ~mid, so the adjustment sat at max_adj and model_p was always mid - 0.10 -
        # it could only ever buy NO, on every up-tick (FINDINGS 57).
        deviation = float(latest_return) - (mean or 0.0)
        gain = float(self.param("reversion_gain", DEFAULT_REVERSION_GAIN))
        max_adj = float(self.param("max_adjustment", DEFAULT_MAX_ADJUSTMENT))
        adjustment = max(-max_adj, min(max_adj, deviation * gain))
        model_p = clamp_probability(mid - Decimal(str(adjustment)))

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
                "mean_return": mean,
                "return_std": std,
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
                f"Mean reversion: return z-score={z:.2f} (|z|>={entry_z:.2f}) over "
                f"{lookback:.0f}s, mid={mid} vs rolling mean_return={mean}; "
                f"model_p={model_p}, executable_price={price}, edge={edge}."
            ),
            features=features,
            model_probability=model_p,
            expected_edge=edge,
        )
        if self.emit_if_profitable(intent):
            self.mark_fired(canonical_id)
            st["armed"] = False
