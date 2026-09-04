"""Shared machinery for every MarketLab strategy.

Every strategy in this repository - both the baseline/microstructure strategies owned by
Team STRATEGIES-A and the event/copy-trading strategies owned by Team STRATEGIES-B -
subclasses :class:`BaseStrategy` instead of :class:`marketlab.core.strategy.Strategy`
directly. It centralises the bookkeeping every strategy would otherwise duplicate:
bounded per-market state, intent construction, fee-aware edge math, book-quality gating,
and cooldown throttling.

Design note on ``in_universe``: :class:`StrategyContext` is deliberately narrow (no
knowledge of ``configs/universes.yaml``). Universe scoping happens upstream, in the runner
that decides which markets get fed into a given experiment's context. ``in_universe`` here
is a light guard confirming the runner actually populated the market/book this strategy was
asked about, not a re-implementation of the universe membership rules.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from decimal import Decimal
from typing import Any

from marketlab.adapters.kalshi.fees import trading_fee
from marketlab.core.events import NewsEvent, SourceClass
from marketlab.core.instruments import (
    EXECUTION_VENUES,
    ONE,
    ZERO,
    MarketStatus,
    NormalizedMarket,
    OrderBook,
    Side,
    Venue,
)
from marketlab.core.orders import Action, OrderIntent, OrderType, TimeInForce
from marketlab.core.probability import expected_edge as _expected_edge
from marketlab.core.strategy import Strategy, StrategyContext
from marketlab.signals.orderbook import is_book_crossed, is_book_locked

#: News source classes treated as "high impact" regardless of tone score - primary/wire
#: sources move markets even when their language is neutral.
HIGH_IMPACT_NEWS_SOURCES = frozenset({SourceClass.OFFICIAL_PRIMARY, SourceClass.MAJOR_WIRE})
DEFAULT_NEWS_TONE_THRESHOLD = 0.7

Series = list[tuple[datetime, Decimal]]

#: Probability epsilon kept away from the exact [0, 1] boundary - logit-space signals and
#: fee formulas are undefined or degenerate exactly at 0 or 1.
_PROB_EPS = Decimal("0.0001")

#: Default per-strategy cap on concurrently tracked markets. Strategies run for days
#: across hundreds of markets; an unbounded state dict is a slow memory leak.
DEFAULT_MAX_TRACKED_MARKETS = 500


def clamp_probability(value: Decimal, lo: Decimal = _PROB_EPS, hi: Decimal = ONE - _PROB_EPS) -> Decimal:
    """Clamp ``value`` into ``[lo, hi]`` without raising, unlike ``to_probability``.

    Strategy-derived probabilities (mid +/- a heuristic adjustment) can legitimately land
    outside ``[0, 1]`` before clamping; ``to_probability`` raises on that, which is correct
    for adapter data but wrong for a model's own arithmetic.
    """
    return max(lo, min(hi, value))


class BaseStrategy(Strategy):
    """Common helpers for the baseline/control/microstructure strategy family.

    Subclasses still implement whichever ``on_*`` handlers they need; this class adds
    concrete helper methods on top of the abstract :class:`Strategy` contract. Nothing
    here is abstract - a subclass that overrides no handlers is simply a strategy that
    never does anything (see :class:`marketlab.strategies.controls.DoNothingStrategy`).
    """

    #: Every intent this strategy family emits goes to Kalshi - the only execution venue
    #: for this deployment. Polymarket is signal-only: it may inform a forecast or a
    #: cross-venue signal, but never appears as ``OrderIntent.venue``.
    VENUE: Venue = Venue.KALSHI

    def __init__(
        self,
        strategy_id: str,
        experiment_id: str,
        ctx: StrategyContext,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(strategy_id, experiment_id, ctx, params)
        self._max_tracked_markets = int(
            self.param("max_tracked_markets", DEFAULT_MAX_TRACKED_MARKETS)
        )
        # OrderedDict so we can evict the least-recently-touched market in O(1) once the
        # bound is hit, instead of growing without limit over a multi-day run.
        self._state: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cooldowns: dict[str, datetime] = {}

    # ------------------------------------------------------------------ per-market state

    def state(self, canonical_id: str) -> dict[str, Any]:
        """Lazily-created, bounded per-market scratch dict.

        Subclasses stash whatever they need here: ``RollingWindow``/``EWMA``/``TimeWindow``
        instances, last-seen prices, cooldown markers. Touching a market moves it to the
        most-recently-used end; once ``max_tracked_markets`` is exceeded the
        least-recently-touched market's state is dropped.
        """
        existing = self._state.get(canonical_id)
        if existing is not None:
            self._state.move_to_end(canonical_id)
            return existing
        new_state: dict[str, Any] = {}
        self._state[canonical_id] = new_state
        if len(self._state) > self._max_tracked_markets:
            self._state.popitem(last=False)
        return new_state

    def in_universe(self, canonical_id: str) -> bool:
        """True when the runner has actually populated this market for this experiment."""
        return self.ctx.market(canonical_id) is not None

    # ------------------------------------------------------------------ cooldown throttle

    def on_cooldown(self, canonical_id: str, seconds: float) -> bool:
        """True if this market fired within the last ``seconds`` of sim time.

        Does not itself record a firing - call :meth:`mark_fired` after actually emitting,
        so a strategy that decides not to trade this tick doesn't consume the cooldown.
        """
        last = self._cooldowns.get(canonical_id)
        if last is None:
            return False
        return (self.now() - last).total_seconds() < seconds

    def mark_fired(self, canonical_id: str) -> None:
        self._cooldowns[canonical_id] = self.now()

    # ------------------------------------------------------------------ book quality gate

    def should_skip(self, canonical_id: str) -> str | None:
        """Centralised stale/crossed/locked/closed checks. Returns a reason or ``None``.

        Every strategy in this package calls this before acting on a market - see the
        hard rule in the PRD: never trade a book you cannot trust.
        """
        market = self.ctx.market(canonical_id)
        if market is None:
            return "no_market"
        # Polymarket markets are visible to strategies deliberately - the cross-venue and
        # copy-trader families need them as SIGNALS. They are never tradeable here. Every
        # strategy stamps VENUE = Venue.KALSHI on its intents, so without this check a
        # strategy iterating ctx.markets() happily emits orders against `poly:` contracts
        # that look like Kalshi orders. Caught only after 166 such fills landed.
        if market.venue not in EXECUTION_VENUES:
            return "not_an_execution_venue"
        if market.status is not MarketStatus.OPEN:
            return "market_not_open"
        book = self.ctx.book(canonical_id)
        if book is None:
            return "no_book"
        max_age = float(self.param("max_book_age_seconds", 30.0))
        if book.is_stale(self.now(), max_age):
            return "stale_book"
        if not book.bids or not book.asks:
            return "one_sided_book"
        if is_book_crossed(book):
            return "crossed_book"
        if is_book_locked(book):
            return "locked_book"
        return None

    # ------------------------------------------------------------------ pricing / costs

    def executable_price(self, book: OrderBook, side: Side, action: Action) -> Decimal | None:
        """The price actually paid/received at the touch for ``action side``.

        The book is always carried in YES-probability terms (see
        ``marketlab/core/instruments.py``), so a NO order has to be translated: buying NO
        means lifting the YES bid, at effective price ``1 - best_bid`` (buying NO at that
        price has an identical payoff to selling YES at ``best_bid``); selling NO means
        hitting the YES ask, at effective price ``1 - best_ask``.
        """
        if side is Side.YES:
            return book.best_ask if action is Action.BUY else book.best_bid
        # Side.NO
        if action is Action.BUY:
            return None if book.best_bid is None else (ONE - book.best_bid)
        return None if book.best_ask is None else (ONE - book.best_ask)

    def estimate_fee(
        self,
        market: NormalizedMarket,
        price: Decimal,
        quantity: int = 1,
        is_maker: bool = False,
    ) -> Decimal:
        """Per-fill fee in dollars, using the market's own fee schedule.

        Deliberately does not use ``trading_fee``'s own ``is_maker`` short-circuit (which
        always zeroes the fee): a market's ``Fees.maker_rate`` might one day be non-zero,
        and selecting the rate first keeps this correct in that case while still returning
        the historically-correct $0.00 today.
        """
        rate = market.fees.maker_rate if is_maker else market.fees.taker_rate
        return trading_fee(price, quantity, rate=rate)

    def edge_after_costs(
        self,
        model_probability: Decimal,
        executable_price: Decimal,
        market: NormalizedMarket,
        side: Side,
        is_maker: bool = False,
    ) -> Decimal:
        """Edge net of fees, slippage buffer and uncertainty buffer - see PRD.

        Fee is computed per-contract (the quadratic formula is linear in contract count,
        so quantity does not change the per-contract fee). ``slippage_buffer`` and
        ``uncertainty_buffer`` come from params so the runner can wire in
        ``settings.execution``'s values; they default to that config's own defaults so a
        strategy tested in isolation still behaves conservatively.
        """
        fee = self.estimate_fee(market, executable_price, quantity=1, is_maker=is_maker)
        slippage_buffer = Decimal(str(self.param("slippage_buffer", Decimal("0.005"))))
        uncertainty_buffer = Decimal(str(self.param("uncertainty_buffer", Decimal("0.01"))))
        return _expected_edge(
            model_probability,
            executable_price,
            fee,
            slippage_buffer,
            uncertainty_buffer,
            side=side,
        )

    # ------------------------------------------------------------------ sizing

    def sensible_quantity(
        self,
        price: Decimal,
        risk_fraction: Decimal = Decimal("0.03"),
        sleeve: Decimal = Decimal("50.00"),
        min_qty: int = 1,
        max_qty: int = 25,
    ) -> int:
        """A small, sleeve-scaled contract count - never "as many as possible".

        Position limits are the risk gateway's job, but the default fraction must sit
        comfortably BELOW ``risk.max_single_event_loss_pct`` (0.04 of a $50 sleeve = $2.00)
        or every intent is rejected before it reaches a book. At 5% a single 61c contract
        order came to $2.44 and was refused; 3% leaves headroom for price and rounding.
        Sizing that reliably trips the risk gate is not conservative, it is broken.
        """
        price = clamp_probability(price)
        budget = sleeve * risk_fraction
        qty = int(budget / price) if price > ZERO else min_qty
        return max(min_qty, min(max_qty, qty))

    # ------------------------------------------------------------------ intent construction

    def make_intent(
        self,
        canonical_id: str,
        side: Side,
        action: Action,
        quantity: int,
        order_type: OrderType,
        limit_price: Decimal | None,
        rationale: str,
        features: dict[str, Any],
        model_probability: Decimal | None = None,
        expected_edge: Decimal | None = None,
        evidence_ids: tuple[str, ...] = (),
        time_in_force: TimeInForce = TimeInForce.GTC,
        replaces_order_id: str | None = None,
    ) -> OrderIntent:
        """Build an :class:`OrderIntent` with the bookkeeping fields filled in.

        ``rationale`` and ``features`` are required arguments (no defaults) on purpose:
        the risk gateway rejects an empty rationale under ``strict_audit``, and an empty
        one is never acceptable here regardless.
        """
        if not rationale:
            raise ValueError("rationale must be non-empty - the audit trail requires it")
        return OrderIntent(
            strategy_id=self.strategy_id,
            experiment_id=self.experiment_id,
            canonical_id=canonical_id,
            venue=self.VENUE,
            side=side,
            action=action,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
            decision_time=self.now(),
            rationale=rationale,
            features=dict(features),
            evidence_ids=tuple(evidence_ids),
            model_probability=model_probability,
            expected_edge=expected_edge,
            replaces_order_id=replaces_order_id,
        )

    @staticmethod
    def prune_series(history: Series, now: datetime, max_age_seconds: float) -> None:
        """Drop entries older than ``max_age_seconds`` from a chronological ``(ts, value)`` list.

        Mutates ``history`` in place. Bounds memory for the simple list-based price
        histories that :mod:`~marketlab.signals.technical`'s ``momentum``/``bounded_momentum``
        need raw ``(timestamp, price)`` pairs for (as opposed to the scalar accumulators in
        ``signals/rolling.py``, which those functions don't consume).
        """
        cutoff = now.timestamp() - max_age_seconds
        while history and history[0][0].timestamp() < cutoff:
            history.pop(0)

    @staticmethod
    def find_anchor(history: Series, lookback_seconds: float, now: datetime) -> Decimal | None:
        """Most recent value at or before ``now - lookback_seconds``; ``None`` if none is old enough."""
        cutoff = now.timestamp() - lookback_seconds
        anchor: Decimal | None = None
        for ts, value in history:
            if ts.timestamp() <= cutoff:
                anchor = value
            else:
                break
        return anchor

    def is_high_impact_news(
        self, event: NewsEvent, tone_threshold: float = DEFAULT_NEWS_TONE_THRESHOLD
    ) -> bool:
        """Shared "does this news event matter enough to pause on" heuristic.

        Used by :class:`~marketlab.strategies.momentum.MomentumStrategy` (suppress entries)
        and :class:`~marketlab.strategies.market_maker.AvellanedaStoikovBinaryStrategy`
        (pull resting quotes). A primary/wire source counts regardless of tone; anything
        else counts only if its tone score is extreme.
        """
        if event.source_class in HIGH_IMPACT_NEWS_SOURCES:
            return True
        return event.tone is not None and abs(event.tone) >= tone_threshold

    def emit_if_profitable(self, intent: OrderIntent) -> bool:
        """Emit ``intent`` only if it carries a strictly positive ``expected_edge``.

        Centralises the PRD's hard rule ("never emit an intent whose expected edge is not
        positive after costs") in one place so every strategy enforces it identically.
        """
        if intent.expected_edge is None or intent.expected_edge <= ZERO:
            return False
        self.emit(intent)
        return True
