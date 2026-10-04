"""BTC threshold contracts, priced with driftless GBM.

Universes: ``btc_15m``, ``btc_1h``, ``btc_daily`` (see ``configs/universes.yaml``).
**Kalshi lists no 1-minute or 5-minute BTC contract** as of the 2026-09-04 series
catalogue probe recorded in ``docs/CONTRACTS.md`` - ``btc_1m``/``btc_5m`` are declared
``available: false`` there, so those horizons are simply untestable on this venue. The
``btc_15m`` arm exists partly *because* it is the shortest horizon that does exist: it is
as much a test of whether a home/local setup is latency-competitive at all as it is a test
of the GBM model. **A losing result on ``btc_15m`` is itself a finding** (evidence that
retail infrastructure cannot compete at that horizon), not a bug to chase away.

**Terminal vs. barrier is a real modelling error to get wrong, not a style choice.**
"Will BTC be above $X at 5pm ET" is a *terminal* question - :func:`gbm_touch_probability`
answers exactly that. "Will BTC reach $X before 5pm ET" is a *barrier* question -
:func:`gbm_barrier_touch_probability` answers that instead, and the two give materially
different numbers for the same spot/strike/vol/time. This module resolves which one
applies from the market's own wording (:func:`classify_measurement`, mirroring the barrier
keyword list :mod:`marketlab.matching.extract` uses so the two independently-invoked
classifiers can't quietly diverge), and **abstains rather than guesses** when the wording
is ambiguous - recorded as a :class:`~marketlab.core.strategy.ProbabilityForecast` with
``abstain=True`` so the abstention itself is auditable, never silently dropped.

A configured ``model`` param (``gbm_terminal`` | ``gbm_barrier``) is not merely a formula
choice - it also scopes *which markets this experiment arm is even eligible to trade*: a
``gbm_terminal`` variant only acts on markets whose own wording says "terminal", and a
``gbm_barrier`` variant only acts on "barrier_touch" wording. A variant never applies its
configured formula to a market whose wording says the other type; that would be exactly
the modelling error the PRD calls out.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from marketlab.core.events import BookUpdateEvent, ExternalPriceEvent, TimerEvent
from marketlab.core.instruments import Category, Side
from marketlab.core.orders import Action, OrderType
from marketlab.core.probability import gbm_barrier_touch_probability, gbm_touch_probability
from marketlab.core.strategy import ProbabilityForecast
from marketlab.signals.technical import momentum, realized_volatility
from marketlab.strategies.base import BaseStrategy, clamp_probability

# ---------------------------------------------------------------------------
# Ticker / title parsing
# ---------------------------------------------------------------------------

#: Kalshi BTC ticker suffix, e.g. "KXBTCD-26SEP0413-T86799.99" -> strike 86799.99.
_TICKER_STRIKE_RE = re.compile(r"-T(?P<strike>[\d,]+(?:\.\d+)?)\b")
_TITLE_STRIKE_RE = re.compile(r"\$\s*(?P<num>[\d,]+(?:\.\d+)?)")

#: Mirrors marketlab.matching.extract's keyword lists exactly, so this strategy's
#: terminal/barrier judgment never quietly diverges from the matcher's.
_BARRIER_KEYWORDS = ("before", "any point", "any time", "touches", "touch", "ever reaches", "at any moment")
_TERMINAL_KEYWORDS = ("at close", "closing price", "final price", " at ", "as of")
_BELOW_KEYWORDS = ("below", "under", "or lower", "at or below")
_ABOVE_KEYWORDS = ("above", "over", "or higher", "at or above", "exceed")


def parse_btc_strike(ticker: str, title: str = "") -> Decimal | None:
    """Strike from the ticker's ``-T<value>`` suffix, falling back to a ``$`` amount in the title."""
    m = _TICKER_STRIKE_RE.search(ticker or "")
    if m:
        try:
            return Decimal(m.group("strike").replace(",", ""))
        except InvalidOperation:
            pass
    m2 = _TITLE_STRIKE_RE.search(title or "")
    if m2:
        try:
            return Decimal(m2.group("num").replace(",", ""))
        except InvalidOperation:
            pass
    return None


#: Kalshi series whose structure fixes the measurement, regardless of wording. Every
#: KXBTCD-...-T<strike> contract settles on "the 60-second BRTI average before <time> is
#: above <strike> at <time>" - a terminal question. The event title ("Bitcoin price on
#: Oct 9, 2026?") says nothing either way, and the rules text contains "before", so a
#: keyword pass over it would wrongly read barrier. Measured: 1,381 KXBTCD contracts
#: abstained for a week on exactly that.
_TERMINAL_TICKER_RE = re.compile(r"^KXBTCD-[0-9A-Z]+-T[\d.]+$")
_BARRIER_SERIES = ("KXBTCMAX", "KXBTCMIN")


def classify_measurement(title: str, description: str = "", ticker: str = "") -> str | None:
    """``'terminal'`` | ``'barrier_touch'`` | ``None`` (ambiguous -> caller must abstain)."""
    lowered = f"{title}\n{description}".lower()
    worded: str | None = None
    if any(k in lowered for k in _BARRIER_KEYWORDS):
        worded = "barrier_touch"
    elif any(k in lowered for k in _TERMINAL_KEYWORDS):
        worded = "terminal"
    upper = (ticker or "").upper()
    structural: str | None = None
    if _TERMINAL_TICKER_RE.match(upper):
        structural = "terminal"
    elif upper.startswith(_BARRIER_SERIES):
        structural = "barrier_touch"
    if structural is not None and worded is not None and structural != worded:
        return None  # the contract's own words contradict its series: never guess
    return structural or worded


def is_below_threshold_contract(title: str) -> bool:
    """True if YES means "below strike" rather than the (more common) "above strike"."""
    lowered = title.lower()
    if any(k in lowered for k in _ABOVE_KEYWORDS):
        return False
    return any(k in lowered for k in _BELOW_KEYWORDS)


class BtcEventStrategy(BaseStrategy):
    """Prices BTC threshold contracts against a driftless-GBM model of spot."""

    name = "btc_event"
    version = "1.1.0"
    evidence_class = "B"

    def __init__(self, strategy_id: str, experiment_id: str, ctx: Any, params: dict | None = None) -> None:
        super().__init__(strategy_id, experiment_id, ctx, params)
        #: (timestamp, spot) pairs, shared across every BTC market this instance tracks -
        #: one spot feed informs many contracts, so it is not per-market state.
        self._spot_history: list[tuple[datetime, Decimal]] = []

    # ------------------------------------------------------------------ inputs

    def on_external_price(self, event: ExternalPriceEvent) -> None:
        reference_symbols = self.param("reference_symbols", ("BTC-USD",))
        if event.symbol not in reference_symbols:
            return
        self.append_sample(self._spot_history, event.first_seen_time, event.price)
        vol_window_seconds = float(self.param("vol_window_minutes", 60.0)) * 60.0
        self.prune_series(self._spot_history, event.first_seen_time, vol_window_seconds * 8)

    def on_book_update(self, event: BookUpdateEvent) -> None:
        self._evaluate(event.canonical_id)

    def on_timer(self, event: TimerEvent) -> None:
        del event
        for market in self.ctx.markets():
            if market.category is Category.CRYPTO and "BTC" in market.venue_market_id.upper():
                self._evaluate(market.canonical_id)

    # ------------------------------------------------------------------ evaluation

    def _current_spot(self, now: datetime) -> Decimal | None:
        eligible = [(ts, p) for ts, p in self._spot_history if ts <= now]
        return eligible[-1][1] if eligible else None

    def _evaluate(self, canonical_id: str) -> None:
        if self.should_skip(canonical_id) is not None:
            return
        market = self.ctx.market(canonical_id)
        book = self.ctx.book(canonical_id)
        assert market is not None and book is not None
        if market.category is not Category.CRYPTO or "BTC" not in market.venue_market_id.upper():
            return

        now = self.now()
        measurement = classify_measurement(market.title, market.description, market.venue_market_id)
        if measurement is None:
            self.forecast(
                ProbabilityForecast(
                    strategy_id=self.strategy_id,
                    experiment_id=self.experiment_id,
                    canonical_id=canonical_id,
                    as_of=now,
                    p_yes=Decimal("0.5"),
                    abstain=True,
                    market_probability=book.mid,
                    rationale=(
                        f"Abstaining: '{market.title}' wording is ambiguous between a "
                        "terminal ('above at close') and a barrier ('touches before "
                        "close') question - choosing wrong is a modelling error, not a "
                        "style choice, so no model probability is claimed."
                    ),
                    features={"measurement": None, "ticker": market.venue_market_id},
                )
            )
            return

        configured_model = str(self.param("model", "gbm_terminal"))
        expected_measurement = "terminal" if configured_model == "gbm_terminal" else "barrier_touch"
        if measurement != expected_measurement:
            # Out of scope for this variant - not ambiguous, just the other question type.
            return

        strike = parse_btc_strike(market.venue_market_id, market.title)
        if strike is None:
            return
        if market.close_time is None:
            return
        seconds_remaining = (market.close_time - now).total_seconds()
        if seconds_remaining <= 0:
            return

        spot = self._current_spot(now)
        if spot is None:
            return

        vol_window_minutes = float(self.param("vol_window_minutes", 60.0))
        prices = [p for _, p in self._spot_history]
        timestamps = [ts for ts, _ in self._spot_history]
        sigma = realized_volatility(prices, timestamps, vol_window_minutes * 60.0)
        if sigma is None or sigma <= 0:
            return

        momentum_lookback = min(300.0, vol_window_minutes * 60.0 / 2.0)
        mom = momentum(list(zip(timestamps, prices, strict=True)), momentum_lookback, now)

        if configured_model == "gbm_terminal":
            p_above = gbm_touch_probability(float(spot), float(strike), sigma, seconds_remaining)
        else:
            p_above = gbm_barrier_touch_probability(float(spot), float(strike), sigma, seconds_remaining)

        below = is_below_threshold_contract(f"{market.title} {market.description}")
        p_yes = (1.0 - p_above) if below else p_above
        model_probability = clamp_probability(Decimal(str(round(p_yes, 6))))

        features: dict[str, Any] = {
            "spot": float(spot),
            "strike": float(strike),
            "sigma_annual": sigma,
            "seconds_remaining": seconds_remaining,
            "momentum": mom,
            "measurement": measurement,
            "model": configured_model,
            "vol_window_minutes": vol_window_minutes,
            "below_threshold_contract": below,
            "model_probability": float(model_probability),
            "market_probability": float(book.mid) if book.mid is not None else None,
        }
        self.forecast(
            ProbabilityForecast(
                strategy_id=self.strategy_id,
                experiment_id=self.experiment_id,
                canonical_id=canonical_id,
                as_of=now,
                p_yes=model_probability,
                market_probability=book.mid,
                features=features,
                rationale=(
                    f"{configured_model}: spot={spot} strike={strike} sigma={sigma:.4f} "
                    f"t={seconds_remaining:.0f}s -> p_yes={model_probability}."
                ),
            )
        )

        price_yes = self.executable_price(book, Side.YES, Action.BUY)
        price_no = self.executable_price(book, Side.NO, Action.BUY)
        candidates: list[tuple[Side, Decimal, Decimal]] = []
        if price_yes is not None:
            candidates.append((Side.YES, price_yes, self.edge_after_costs(model_probability, price_yes, market, Side.YES)))
        if price_no is not None:
            no_model_p = clamp_probability(Decimal(1) - model_probability)
            candidates.append((Side.NO, price_no, self.edge_after_costs(no_model_p, price_no, market, Side.NO)))
        if not candidates:
            return
        side, price, edge = max(candidates, key=lambda c: c[2])

        entry_edge = Decimal(str(self.param("entry_edge", "0.02")))
        if edge < entry_edge:
            return
        cooldown = float(self.param("cooldown_seconds", 30.0))
        if self.on_cooldown(canonical_id, cooldown):
            return

        intent = self.make_intent(
            canonical_id=canonical_id,
            side=side,
            action=Action.BUY,
            quantity=self.sensible_quantity(price),
            order_type=OrderType.LIMIT,
            limit_price=price,
            rationale=(
                f"BTC {configured_model} model_p={model_probability} vs executable "
                f"{side.value} price {price}: edge {edge} clears entry_edge {entry_edge}."
            ),
            features={**features, "executable_price": float(price), "side": side.value},
            model_probability=model_probability,
            expected_edge=edge,
        )
        if self.emit_if_profitable(intent):
            self.mark_fired(canonical_id)


__all__ = ["BtcEventStrategy", "classify_measurement", "is_below_threshold_contract", "parse_btc_strike"]
