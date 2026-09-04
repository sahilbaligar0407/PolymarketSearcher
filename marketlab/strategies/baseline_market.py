"""The market baseline: ``p_yes = current midpoint``.

Every sophisticated model in this repository must beat this. It never emits an order
(``trades: false`` in ``configs/strategies.yaml``) - its only job is to put a forecast on
the scoreboard so Brier score / calibration comparisons have a trivial reference point.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from marketlab.core.events import BookUpdateEvent, TimerEvent, TradeEvent
from marketlab.core.strategy import ProbabilityForecast
from marketlab.strategies.base import BaseStrategy

#: Emit a fresh forecast at least this often even if the price hasn't moved, so a quiet
#: market still gets a periodic reading rather than going dark.
DEFAULT_MIN_INTERVAL_SECONDS = 60.0
#: Emit immediately (subject to the interval-independent dedup below) when the midpoint
#: moves by at least this much - "meaningful book change" per the PRD.
DEFAULT_MIN_PRICE_MOVE = Decimal("0.01")


class MarketBaselineStrategy(BaseStrategy):
    """Forecasts the book midpoint. Never trades."""

    name = "market_baseline"
    version = "1.0.0"
    evidence_class = "B"

    def on_trade(self, event: TradeEvent) -> None:
        # Recorded purely for the "last trade" feature on the next forecast; the baseline
        # itself never reacts to trades.
        st = self.state(event.canonical_id)
        st["last_trade_price"] = event.trade.price
        st["last_trade_time"] = event.trade.timestamp

    def on_timer(self, event: TimerEvent) -> None:
        del event
        for market in self.ctx.markets():
            self._maybe_forecast(market.canonical_id, force=True)

    def on_book_update(self, event: BookUpdateEvent) -> None:
        self._maybe_forecast(event.canonical_id, force=False)

    # ------------------------------------------------------------------

    def _maybe_forecast(self, canonical_id: str, force: bool) -> None:
        if self.should_skip(canonical_id) is not None:
            return
        book = self.ctx.book(canonical_id)
        assert book is not None  # should_skip already guaranteed this
        mid = book.mid
        if mid is None:
            return

        st = self.state(canonical_id)
        last_mid: Decimal | None = st.get("last_forecast_mid")
        last_time = st.get("last_forecast_time")
        min_interval = float(self.param("min_interval_seconds", DEFAULT_MIN_INTERVAL_SECONDS))
        min_move = Decimal(str(self.param("min_price_move", DEFAULT_MIN_PRICE_MOVE)))

        due_on_interval = last_time is None or (self.now() - last_time).total_seconds() >= min_interval
        moved_enough = last_mid is None or abs(mid - last_mid) >= min_move
        if not (force or due_on_interval or moved_enough):
            return

        features: dict[str, Any] = {
            "best_bid": float(book.best_bid) if book.best_bid is not None else None,
            "best_ask": float(book.best_ask) if book.best_ask is not None else None,
            "midpoint": float(mid),
            "spread": float(book.spread) if book.spread is not None else None,
            "weighted_mid": float(book.microprice) if book.microprice is not None else None,
            "last_trade_price": float(st["last_trade_price"]) if "last_trade_price" in st else None,
        }
        self.forecast(
            ProbabilityForecast(
                strategy_id=self.strategy_id,
                experiment_id=self.experiment_id,
                canonical_id=canonical_id,
                as_of=self.now(),
                p_yes=mid,
                market_probability=mid,
                confidence=Decimal("0.5"),
                features=features,
                rationale=(
                    f"Baseline p_yes=midpoint={mid}: best_bid={book.best_bid}, "
                    f"best_ask={book.best_ask}, spread={book.spread}, "
                    f"weighted_mid={book.microprice}."
                ),
            )
        )
        st["last_forecast_mid"] = mid
        st["last_forecast_time"] = self.now()
