"""Cross-venue relative value: Polymarket price as a SIGNAL, traded only on Kalshi.

**This is not arbitrage.** There is no second leg on Polymarket - new Polymarket
positions are geoblocked for this deployment (see ``docs/CONTRACTS.md``). What this
strategy does is: when an *approved* match says a Kalshi contract and a Polymarket
contract describe the identical bet, and the two venues (plus, optionally, a vig-free
sportsbook/quant consensus) disagree about the probability, that disagreement is a
*signal* that the Kalshi price may be wrong - and the resulting trade is a single-venue
Kalshi order sized by the edge, with the Polymarket price recorded as evidence. Calling
this "arbitrage" would misstate what is actually being risked: only one side ever
executes, so there is no locked-in payoff, only a probability-weighted bet that Kalshi's
price converges toward the reference.

**Only an approved match may drive a trade.** ``marketlab.matching.approved_for_automation``
is reused directly rather than re-derived, for the same reason
``marketlab.signals.sports`` reuses ``remove_vig`` instead of reimplementing it: two
independently-maintained copies of "is this match safe to trade on" would eventually
disagree and nobody would notice until real capital was on the line. Every refusal
(unapproved match, stale book, thin liquidity, contradictory signals) is counted in
:attr:`CrossVenueRelativeValueStrategy.refusal_counts` - how often a big-looking gap gets
refused is itself a research finding, not a footnote.

**Match delivery.** :class:`~marketlab.core.strategy.StrategyContext` is deliberately
narrow (clock, books, markets, marks, params) and there is no ``MarketMatchEvent`` in
:mod:`marketlab.core.events`. Match records are therefore supplied out of band, through
``params["matches"]``: an iterable of objects shaped like
``marketlab.matching.MarketMatch`` (or any duck-typed equivalent exposing
``canonical_id_a``, ``canonical_id_b``, ``match_confidence``, ``same_outcome_boolean``,
``human_review_required``). This is a documented judgment call - the alternative (having
this strategy run ``CrossVenueMatcher`` itself) would mean an event-driven strategy
running the async, cross-venue matching pipeline inline, which is both slow and outside
what a strategy is allowed to do (no network, no storage). The runner is expected to
refresh ``params["matches"]`` from the matcher's persisted output on its own schedule.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from marketlab.core.events import ExternalPriceEvent, TimerEvent
from marketlab.core.instruments import Side
from marketlab.core.orders import Action, OrderType
from marketlab.matching import approved_for_automation
from marketlab.strategies.base import BaseStrategy

DEFAULT_MIN_OPEN_INTEREST = Decimal("1")
DEFAULT_MAX_BOOK_STALENESS_SECONDS = 10.0
DEFAULT_MIN_LIQUIDITY_CONTRACTS = 5


def _match_attr(match: Any, name: str) -> Any:
    """Read an attribute off a match record, tolerating a plain dict too (duck-typing)."""
    if isinstance(match, dict):
        return match.get(name)
    return getattr(match, name, None)


class CrossVenueRelativeValueStrategy(BaseStrategy):
    """Trades the Kalshi leg of an approved cross-venue match toward the Polymarket/
    consensus signal. Never touches Polymarket execution.
    """

    name = "cross_venue"
    version = "1.1.0"  # 1.1.0: FINDINGS 57
    evidence_class = "B"

    def __init__(self, strategy_id: str, experiment_id: str, ctx: Any, params: dict | None = None) -> None:
        super().__init__(strategy_id, experiment_id, ctx, params)
        #: reason -> count. Deliberately public: "how often did we refuse" is a result.
        self.refusal_counts: dict[str, int] = {}
        #: canonical_id (Kalshi) -> latest vig-free reference probability, set by
        #: on_external_price. Optional - a match can still trade on Polymarket alone.
        self._reference_probability: dict[str, Decimal] = {}

    # ------------------------------------------------------------------ triggers

    def on_timer(self, event: TimerEvent) -> None:
        del event
        # A full sweep of every approved pair, at most every few seconds: the 1 s tick
        # times ~30 sleeves times hundreds of discovered pairs is CPU the loop cannot spare.
        if self.on_cooldown("__sweep__", float(self.param("sweep_seconds", 5.0))):
            return
        self.mark_fired("__sweep__")
        for match in self.param("matches", ()) or ():
            self._evaluate_match(match)

    def on_external_price(self, event: ExternalPriceEvent) -> None:
        # Judgment call: a reference (vig-free sportsbook / quant) feed is expected to
        # publish its ExternalPriceEvent.symbol as the Kalshi canonical_id it prices, so
        # this strategy can look it up without re-deriving vig removal itself.
        if event.implied_probability is not None:
            self._reference_probability[event.symbol] = event.implied_probability

    # ------------------------------------------------------------------ evaluation

    def _refuse(self, reason: str) -> None:
        self.refusal_counts[reason] = self.refusal_counts.get(reason, 0) + 1

    def _evaluate_match(self, match: Any) -> None:
        canonical_id_a = _match_attr(match, "canonical_id_a")  # Kalshi
        canonical_id_b = _match_attr(match, "canonical_id_b")  # Polymarket (read-only)
        if not canonical_id_a or not canonical_id_b:
            self._refuse("malformed_match")
            return

        min_confidence = Decimal(str(self.param("require_match_confidence", "0.90")))
        if not approved_for_automation(match, min_confidence=min_confidence):
            self._refuse("unapproved_match")
            return

        # --- Kalshi leg must be genuinely tradeable ----------------------------------
        skip_reason = self.should_skip(canonical_id_a)
        if skip_reason is not None:
            self._refuse(f"kalshi_{skip_reason}")
            return
        market_a = self.ctx.market(canonical_id_a)
        book_a = self.ctx.book(canonical_id_a)
        assert market_a is not None and book_a is not None

        min_open_interest = Decimal(str(self.param("min_open_interest", DEFAULT_MIN_OPEN_INTEREST)))
        if market_a.open_interest < min_open_interest:
            self._refuse("thin_liquidity_kalshi")
            return

        # --- Polymarket leg: read-only signal, but must be fresh and two-sided -------
        book_b = self.ctx.book(canonical_id_b)
        if book_b is None:
            self._refuse("no_poly_book")
            return
        max_staleness = float(self.param("max_book_staleness_seconds", DEFAULT_MAX_BOOK_STALENESS_SECONDS))
        now = self.now()
        if book_b.is_stale(now, max_staleness):
            self._refuse("stale_poly_book")
            return
        if not book_b.bids or not book_b.asks:
            self._refuse("thin_liquidity_poly")
            return
        min_liquidity = int(self.param("min_liquidity_contracts", DEFAULT_MIN_LIQUIDITY_CONTRACTS))
        if book_b.depth(1, "bid") < min_liquidity or book_b.depth(1, "ask") < min_liquidity:
            self._refuse("thin_liquidity_poly")
            return

        poly_global_mid = book_b.mid
        kalshi_mid = book_a.mid
        if poly_global_mid is None or kalshi_mid is None:
            self._refuse("no_mid")
            return

        time_delta_seconds = abs((book_a.timestamp - book_b.timestamp).total_seconds())

        reference_probability = self._reference_probability.get(canonical_id_a)
        kalshi_minus_poly = kalshi_mid - poly_global_mid
        kalshi_minus_consensus = (
            kalshi_mid - reference_probability if reference_probability is not None else None
        )

        # Extra robustness: when both an independent reference and Polymarket disagree
        # in *direction* about which side of 0.5 is favored, the signal is contradictory
        # rather than merely noisy, and this strategy stands down rather than picking one.
        if reference_probability is not None:
            poly_favors_yes = poly_global_mid >= Decimal("0.5")
            ref_favors_yes = reference_probability >= Decimal("0.5")
            if poly_favors_yes != ref_favors_yes:
                self._refuse("contradictory_signal")
                return

        model_probability = poly_global_mid

        price_yes = self.executable_price(book_a, Side.YES, Action.BUY)
        price_no = self.executable_price(book_a, Side.NO, Action.BUY)
        candidates: list[tuple[Side, Decimal, Decimal]] = []
        if price_yes is not None:
            edge_yes = self.edge_after_costs(model_probability, price_yes, market_a, Side.YES)
            candidates.append((Side.YES, price_yes, edge_yes))
        if price_no is not None:
            # P(YES) for both sides: edge_after_costs -> expected_edge converts it to P(NO) itself.
            # Passing 1 - P here flipped it twice and bought NO when the model said YES
            # (FINDINGS 57).
            edge_no = self.edge_after_costs(model_probability, price_no, market_a, Side.NO)
            candidates.append((Side.NO, price_no, edge_no))
        if not candidates:
            self._refuse("no_executable_price")
            return

        side, price, edge = max(candidates, key=lambda c: c[2])
        min_edge = Decimal(str(self.param("min_edge", "0.03")))
        if edge < min_edge:
            self._refuse("edge_below_min")
            return

        cooldown = float(self.param("cooldown_seconds", 60.0))
        if self.on_cooldown(canonical_id_a, cooldown):
            return

        match_confidence = _match_attr(match, "match_confidence")
        features: dict[str, Any] = {
            "kalshi_mid": float(kalshi_mid),
            "poly_global_mid": float(poly_global_mid),
            "reference_probability": float(reference_probability) if reference_probability is not None else None,
            "kalshi_minus_poly": float(kalshi_minus_poly),
            "kalshi_minus_consensus": float(kalshi_minus_consensus) if kalshi_minus_consensus is not None else None,
            "match_confidence": float(match_confidence) if match_confidence is not None else None,
            "book_timestamp_delta_seconds": time_delta_seconds,
            "poly_canonical_id": canonical_id_b,
            "side": side.value,
        }
        rationale = (
            f"Cross-venue relative value: kalshi_mid={kalshi_mid} vs poly_global_mid="
            f"{poly_global_mid} (delta {kalshi_minus_poly}), reference_probability="
            f"{reference_probability}; approved match confidence={match_confidence}; "
            f"books {time_delta_seconds:.1f}s apart; buying {side.value} on Kalshi at "
            f"{price}, edge={edge} clears min_edge={min_edge}. Polymarket is signal-only - "
            f"no order is ever placed there."
        )
        intent = self.make_intent(
            canonical_id=canonical_id_a,
            side=side,
            action=Action.BUY,
            quantity=self.sensible_quantity(price),
            order_type=OrderType.LIMIT,
            limit_price=price,
            rationale=rationale,
            features=features,
            model_probability=model_probability,
            expected_edge=edge,
            evidence_ids=(f"market:{canonical_id_b}",),
        )
        if self.emit_if_profitable(intent):
            self.mark_fired(canonical_id_a)


__all__ = ["CrossVenueRelativeValueStrategy"]
