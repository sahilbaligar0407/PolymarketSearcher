"""Binary-parity relative value between two logically complementary Kalshi contracts.

Some real-world questions are listed on Kalshi as **two separate markets** whose outcomes
are the exact logical negation of one another (e.g. "Will the Chiefs win their game?" and
"Will the Chiefs lose their game?", or "BTC above $X at 5pm" and "BTC at-or-below $X at
5pm"). Exactly one of the pair resolves YES. Buying YES on both legs therefore guarantees a
payout of exactly $1 per matched contract, regardless of the outcome - the classic
"buy both sides of a complementary pair for less than $1" parity trade.

This is a single-venue relative-value trade: both legs are Kalshi, both legs are
:class:`~marketlab.core.orders.OrderIntent` with ``venue=Venue.KALSHI``. There is no
Polymarket leg here at all (that is :mod:`marketlab.strategies.cross_venue`'s job).

**"Looks profitable at the touch" is not the same as "is profitable."** Every one of the
PRD's five conditions is encoded as its own explicit check with its own rejection reason,
tracked in :attr:`BinaryParityStrategy.rejection_counts` so a report can say *why* a
seemingly-fat parity gap was refused, not just that it was:

1. both legs simultaneously executable (``should_skip`` on each leg)
2. the quoted size actually exists (the book is walked to the requested quantity, not
   just the top-of-book touch price)
3. fees included on both legs (:meth:`BaseStrategy.estimate_fee`)
4. slippage included (the same ``slippage_buffer`` param the rest of the codebase uses)
5. resolution semantics identical - enforced by requiring
   :func:`marketlab.matching.detect_complement` to agree the two claims are a true logical
   complement (same entities, same measurement, same resolution authority when known, same
   settlement window), never merely "different questions that happen to look similar"
6. capital can actually be locked for the duration - both legs must have a known,
   agreeing settlement time; an unknown settlement time is refused rather than assumed safe

Both legs are emitted together, sharing a ``parity_pair_id`` in ``features`` so the risk
gateway and analytics can see they are a linked pair, and both carry the identical
``expected_edge`` so :meth:`~marketlab.strategies.base.BaseStrategy.emit_if_profitable`
either accepts or rejects them atomically - there is no code path that emits one leg
without the other.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from marketlab.core.events import BookUpdateEvent, TimerEvent
from marketlab.core.instruments import ONE, BookLevel, NormalizedMarket, Side
from marketlab.core.orders import Action, OrderType
from marketlab.matching import detect_complement, extract_claim
from marketlab.strategies.base import BaseStrategy

#: Contracts requested per leg when probing a parity pair. Kept small and fixed rather
#: than sleeve-scaled: the whole point of walking the book is to prove *this exact size*
#: is available at *this exact* average price, so the size must be picked before pricing.
DEFAULT_PROBE_QUANTITY = 5


def _walk_avg_price(levels: tuple[BookLevel, ...], quantity: int) -> tuple[Decimal, int] | None:
    """Size-weighted average price and fill count walking ``levels`` for ``quantity``.

    A local re-implementation of ``signals.orderbook``'s private ``_walk_levels`` - that
    function is intentionally not exported, so this keeps this module from depending on
    another team's internal helper while doing exactly the same walk. Returns ``None`` for
    a non-positive quantity or an empty side, mirroring that function's contract.
    """
    if quantity <= 0 or not levels:
        return None
    remaining = quantity
    cost = Decimal(0)
    filled = 0
    for lvl in levels:
        take = min(remaining, lvl.size)
        if take <= 0:
            continue
        cost += lvl.price * take
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    if filled == 0:
        return None
    return cost / filled, filled


class BinaryParityStrategy(BaseStrategy):
    """Buys both legs of a complementary Kalshi pair when the combined cost is executably
    cheap enough to guarantee a profit after every real cost.
    """

    name = "binary_parity"
    version = "1.0.0"
    evidence_class = "B"

    def __init__(self, strategy_id: str, experiment_id: str, ctx: Any, params: dict | None = None) -> None:
        super().__init__(strategy_id, experiment_id, ctx, params)
        self._pairs: list[tuple[NormalizedMarket, NormalizedMarket]] = []
        self._pairs_fingerprint: int = -1
        #: reason -> count, for reporting *why* a fat-looking pair never traded.
        self.rejection_counts: dict[str, int] = {}

    # ------------------------------------------------------------------ triggers

    def on_timer(self, event: TimerEvent) -> None:
        del event
        self._refresh_pairs()
        for a, b in self._pairs:
            self._evaluate_pair(a, b)

    def on_book_update(self, event: BookUpdateEvent) -> None:
        self._refresh_pairs()
        cid = event.canonical_id
        for a, b in self._pairs:
            if a.canonical_id == cid or b.canonical_id == cid:
                self._evaluate_pair(a, b)

    # ------------------------------------------------------------------ pair discovery

    def _refresh_pairs(self) -> None:
        markets = self.ctx.markets()
        # Cheap fingerprint: the market list only ever grows/shrinks at universe-refresh
        # cadence, so a length check is enough to avoid rebuilding claims every tick while
        # still catching new markets appearing in-run.
        fingerprint = len(markets)
        if fingerprint == self._pairs_fingerprint:
            return
        self._pairs_fingerprint = fingerprint
        claims = {m.canonical_id: extract_claim(m) for m in markets}
        pairs: list[tuple[NormalizedMarket, NormalizedMarket]] = []
        for i, a in enumerate(markets):
            for b in markets[i + 1 :]:
                if a.category != b.category:
                    continue
                if detect_complement(claims[a.canonical_id], claims[b.canonical_id]):
                    pairs.append((a, b))
        self._pairs = pairs

    # ------------------------------------------------------------------ evaluation

    def _reject(self, pair_key: tuple[str, str], reason: str) -> None:
        self.rejection_counts[reason] = self.rejection_counts.get(reason, 0) + 1

    def _evaluate_pair(self, a: NormalizedMarket, b: NormalizedMarket) -> None:
        pair_key = (a.canonical_id, b.canonical_id)
        cooldown_key = f"parity:{pair_key[0]}:{pair_key[1]}"
        cooldown = float(self.param("cooldown_seconds", 30.0))
        if self.on_cooldown(cooldown_key, cooldown):
            return

        # --- check 1: both legs simultaneously executable ---------------------------
        skip_a = self.should_skip(a.canonical_id)
        skip_b = self.should_skip(b.canonical_id)
        if skip_a is not None:
            self._reject(pair_key, f"leg_a_{skip_a}")
            return
        if skip_b is not None:
            self._reject(pair_key, f"leg_b_{skip_b}")
            return

        # --- check 6: capital can actually be locked for a known duration -----------
        if a.close_time is None or b.close_time is None:
            self._reject(pair_key, "unknown_settlement_time")
            return

        book_a = self.ctx.book(a.canonical_id)
        book_b = self.ctx.book(b.canonical_id)
        assert book_a is not None and book_b is not None  # should_skip guaranteed this

        # --- check 2: the quoted size actually exists (walk, don't trust the touch) --
        quantity = int(self.param("quantity", DEFAULT_PROBE_QUANTITY))
        fill_a = _walk_avg_price(book_a.asks, quantity)
        fill_b = _walk_avg_price(book_b.asks, quantity)
        if fill_a is None or fill_a[1] < quantity:
            self._reject(pair_key, "leg_a_insufficient_depth")
            return
        if fill_b is None or fill_b[1] < quantity:
            self._reject(pair_key, "leg_b_insufficient_depth")
            return
        avg_price_a, _ = fill_a
        avg_price_b, _ = fill_b

        # --- check 3: fees on both legs -----------------------------------------------
        fee_a = self.estimate_fee(a, avg_price_a, quantity) / Decimal(quantity)
        fee_b = self.estimate_fee(b, avg_price_b, quantity) / Decimal(quantity)

        # --- check 4: slippage buffer --------------------------------------------------
        slippage_buffer = Decimal(str(self.param("slippage_buffer", "0.005")))

        # check 5 (resolution semantics identical) was already enforced by
        # detect_complement() at pair-discovery time - a pair only ever reaches this
        # method if the deterministic claim comparison agreed it is a true complement.

        guaranteed_payout = ONE
        total_cost = avg_price_a + avg_price_b
        total_fees = fee_a + fee_b
        edge = guaranteed_payout - total_cost - total_fees - slippage_buffer

        min_edge = Decimal(str(self.param("min_edge", "0.01")))
        if edge < min_edge:
            self._reject(pair_key, "edge_below_min")
            return

        parity_pair_id = f"parity:{pair_key[0]}|{pair_key[1]}"
        rationale = (
            f"Binary parity: buying YES on both {a.canonical_id} (avg {avg_price_a}) and "
            f"{b.canonical_id} (avg {avg_price_b}) costs {total_cost} for a guaranteed "
            f"$1 payout; edge {edge} after fees {total_fees} and slippage buffer "
            f"{slippage_buffer} clears min_edge {min_edge}."
        )
        shared_features: dict[str, Any] = {
            "parity_pair_id": parity_pair_id,
            "leg_a_canonical_id": a.canonical_id,
            "leg_b_canonical_id": b.canonical_id,
            "avg_price_a": float(avg_price_a),
            "avg_price_b": float(avg_price_b),
            "total_cost": float(total_cost),
            "total_fees": float(total_fees),
            "slippage_buffer": float(slippage_buffer),
            "quantity": quantity,
        }
        intent_a = self.make_intent(
            canonical_id=a.canonical_id,
            side=Side.YES,
            action=Action.BUY,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=avg_price_a,
            rationale=rationale,
            features={**shared_features, "leg": "a"},
            model_probability=None,
            expected_edge=edge,
        )
        intent_b = self.make_intent(
            canonical_id=b.canonical_id,
            side=Side.YES,
            action=Action.BUY,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=avg_price_b,
            rationale=rationale,
            features={**shared_features, "leg": "b"},
            model_probability=None,
            expected_edge=edge,
        )
        # Both intents carry the identical `edge`, so emit_if_profitable's positive-edge
        # gate accepts or rejects them together - there is no partial-pair emission path.
        emitted_a = self.emit_if_profitable(intent_a)
        emitted_b = self.emit_if_profitable(intent_b)
        if emitted_a and emitted_b:
            self.mark_fired(cooldown_key)


__all__ = ["BinaryParityStrategy"]
