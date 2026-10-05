"""Copy trading: follow tracked Polymarket wallets, execute only on Kalshi. Evidence D.

**The mechanism.** A :class:`~marketlab.core.events.TraderActionEvent` arrives from a
tracked Polymarket wallet -> if the matcher has resolved it to a Kalshi contract
(``event.canonical_id``) -> after ``follower_delay_seconds`` of *simulated* time has
actually elapsed -> the strategy trades that Kalshi contract using the Kalshi book
available **then**, not the book (or price) available when the signal arrived.

**Non-negotiables, each with its own dedicated test in ``test_strategies_copy.py``:**

* **Never execute at the source trader's price.** ``event.price`` is recorded in
  ``features["source_price"]`` for audit only; the executed price always comes from
  :meth:`~marketlab.strategies.base.BaseStrategy.executable_price` against the book
  fetched from ``self.ctx.book(canonical_id)`` *after* the delay has elapsed.
* ``follower_delay_seconds=0`` is not a valid strategy arm - it is a diagnostic computed
  in analytics, never traded here. The constructor raises ``ValueError`` rather than
  silently treating 0 as "trade immediately."
* An unmatched Polymarket market (``event.canonical_id`` empty, or a match record present
  but not approved per ``min_match_confidence`` when one is injected) means no trade,
  counted in :attr:`refusal_counts`.
* A wallet whose social-challenge verification status is not in ``allow_trading_status``
  (default: only ``ONCHAIN_OR_API_CONFIRMED``) is never followed - checked against
  ``params["wallet_status"]`` when that table is populated; a wallet absent from that
  table is an ordinary (non-social-challenge) tracked wallet and is not gated by it.
* ``max_stake_per_signal_pct`` caps position size, same knob ``configs/copy_traders.yaml``
  documents.

**Modes** (``raw``, ``specialist``, ``high_conviction``, ``consensus``, ``early``,
``scale_in``, ``fade``) filter which signals become pending copies, or (for ``scale_in``)
adjust sizing, or (for ``fade``) flip the traded side. ``fade`` is a deliberate control
group: it takes the opposite side of what ``raw`` would trade on the identical input, to
measure whether the apparent copy edge is informative at all, or whether the tracked
wallets are simply noise (in which case fading them should be no better than raw, and
possibly worse, and either result is itself the finding).
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Any

from marketlab.core.events import BookUpdateEvent, TimerEvent, TraderActionEvent
from marketlab.core.instruments import ONE, Category, Side, Venue
from marketlab.core.orders import Action, OrderType
from marketlab.logging import get_logger
from marketlab.signals.copy_trader import classify_specialization
from marketlab.signals.rolling import RollingZScore
from marketlab.strategies.base import BaseStrategy, clamp_probability

VALID_MODES = ("raw", "specialist", "high_conviction", "consensus", "early", "scale_in", "fade", "qualified", "qualified_consensus")
DEFAULT_MIN_OPEN_INTEREST = Decimal("1")
DEFAULT_SIZE_Z_MIN = 2.0
DEFAULT_MIN_AGREEING = 3
DEFAULT_CONSENSUS_WINDOW_SECONDS = 300.0
QUALIFIED_MIN_AGREEING = 2
QUALIFIED_CONSENSUS_WINDOW_SECONDS = 6 * 3600.0
#: Documented assumption: the probability-points a copy signal is believed worth, before
#: any forward-validated, mode/wallet-specific number replaces it. Recorded in every
#: intent's features so it is measured, never silently trusted.
DEFAULT_COPY_EDGE_ASSUMPTION = Decimal("0.06")

#: Basket size bounds. The lower bound is a DORMANCY threshold, not a constructor
#: precondition: a fresh deployment has zero validated wallets and must still be able to
#: create the sleeve, so the basket simply does not trade until the roster fills.
_MIN_BASKET_TRADERS = 5
_MAX_BASKET_TRADERS = 20

log = get_logger(__name__)


def _action_to_side(event: TraderActionEvent) -> Side | None:
    if event.side is not None:
        return event.side
    return None


def _assumed_model_probability(price: Decimal, side: Side, edge_assumption: Decimal) -> Decimal:
    """The YES-probability :func:`~marketlab.core.probability.expected_edge` expects.

    ``expected_edge`` always takes ``model_probability`` as ``P(YES)`` and flips it
    internally when ``side`` is NO - so the nudge has to be applied in YES-probability
    space too, not to the raw executable price of whichever side was chosen. Copying a
    YES signal means believing YES is underpriced (P_YES a bit above the YES price);
    copying a NO signal means believing YES is *overpriced* (P_YES a bit *below* the
    YES-equivalent of the NO price).
    """
    implied_yes_price = price if side is Side.YES else (ONE - price)
    nudged = implied_yes_price + edge_assumption if side is Side.YES else implied_yes_price - edge_assumption
    return clamp_probability(nudged)


class CopyTraderStrategy(BaseStrategy):
    """Follows one wallet-agnostic stream of tracked Polymarket trades onto Kalshi."""

    name = "copy_trader"
    version = "1.1.0"
    evidence_class = "D"

    def __init__(self, strategy_id: str, experiment_id: str, ctx: Any, params: dict | None = None) -> None:
        super().__init__(strategy_id, experiment_id, ctx, params)
        self.mode = str(self.param("mode", "raw"))
        if self.mode not in VALID_MODES:
            raise ValueError(f"unknown copy_trader mode: {self.mode!r}")
        self.follower_delay_seconds = float(self.param("follower_delay_seconds", 5.0))
        if self.follower_delay_seconds <= 0:
            raise ValueError(
                "follower_delay_seconds must be > 0; a 0-delay copy is an analytics "
                "diagnostic, never a tradeable strategy arm."
            )
        self._pending_copies: list[dict[str, Any]] = []
        self.refusal_counts: dict[str, int] = {}
        #: wallet -> bounded recent action history, for `specialist`/`early`.
        self._wallet_actions: dict[str, list[TraderActionEvent]] = {}
        #: wallet -> position-change z-scorer on usd_size, for `high_conviction`.
        self._wallet_size_z: dict[str, RollingZScore] = {}
        #: (canonical_id, side) -> [(wallet, timestamp), ...] within the consensus window.
        self._recent_signals: dict[tuple[str, Side], list[tuple[str, Any]]] = {}

    # ------------------------------------------------------------------ inputs

    def on_trader_action(self, event: TraderActionEvent) -> None:
        self._track_wallet_history(event)

        wallet_status = (self.param("wallet_status", {}) or {}).get(event.wallet)
        allow_status = tuple(self.param("allow_trading_status", ("ONCHAIN_OR_API_CONFIRMED",)))
        if wallet_status is not None and wallet_status not in allow_status:
            self._refuse("wallet_status_not_allowed")
            return

        canonical_id, refuse_reason = self._resolve_match(event)
        if canonical_id is None:
            self._refuse(refuse_reason)
            return

        side = _action_to_side(event)
        if side is None:
            self._refuse("no_side")
            return

        if not self._passes_mode_filter(event, canonical_id, side):
            return

        due_at = event.first_seen_time + timedelta(seconds=self.follower_delay_seconds)
        self._pending_copies.append(
            {
                "wallet": event.wallet,
                "canonical_id": canonical_id,
                "side": side,
                "due_at": due_at,
                "first_seen_time": event.first_seen_time,
                "source_price": event.price,
                "usd_size": event.usd_size,
                "size_scale": self._size_scale(event),
                "transaction_hash": event.transaction_hash,
            }
        )

    def on_book_update(self, event: BookUpdateEvent) -> None:
        self._drain_pending(only_canonical_id=event.canonical_id)

    def on_timer(self, event: TimerEvent) -> None:
        del event
        self._drain_pending(only_canonical_id=None)

    # ------------------------------------------------------------------ matching / verification

    def _refuse(self, reason: str) -> None:
        self.refusal_counts[reason] = self.refusal_counts.get(reason, 0) + 1

    def _resolve_match(self, event: TraderActionEvent) -> tuple[str | None, str]:
        if not event.canonical_id or not event.canonical_id.startswith(f"{Venue.KALSHI.value}:"):
            return None, "no_market_match"
        if not bool(self.param("require_market_match", True)):
            return event.canonical_id, ""
        matches: dict[str, Any] = self.param("matches", {}) or {}
        if not matches:
            # No confidence table injected: trust the upstream pipeline's resolution
            # (event.canonical_id already being non-empty is the match itself).
            return event.canonical_id, ""
        record = matches.get(event.poly_condition_id) or matches.get(event.canonical_id)
        if record is None:
            return None, "no_market_match"
        confidence = record.get("match_confidence") if isinstance(record, dict) else getattr(record, "match_confidence", None)
        approved = record.get("approved") if isinstance(record, dict) else getattr(record, "approved", None)
        min_confidence = Decimal(str(self.param("min_match_confidence", "0.90")))
        if approved is False:
            return None, "match_not_approved"
        if confidence is not None and Decimal(str(confidence)) < min_confidence:
            return None, "match_confidence_below_min"
        return event.canonical_id, ""

    # ------------------------------------------------------------------ mode filters

    def _track_wallet_history(self, event: TraderActionEvent) -> None:
        history = self._wallet_actions.setdefault(event.wallet, [])
        history.append(event)
        if len(history) > 200:
            del history[:-200]

    def _size_scale(self, event: TraderActionEvent) -> float:
        """Only meaningful for `scale_in`: this action's size relative to the wallet's
        own median observed size, clamped to a sane range so one outlier can't blow up
        position sizing."""
        history = self._wallet_actions.get(event.wallet, [])
        sizes = sorted(float(a.usd_size) for a in history if a.usd_size is not None)
        if not sizes or event.usd_size is None:
            return 1.0
        median = sizes[len(sizes) // 2]
        if median <= 0:
            return 1.0
        return max(0.25, min(2.0, float(event.usd_size) / median))

    def _passes_mode_filter(self, event: TraderActionEvent, canonical_id: str, side: Side) -> bool:
        if self.mode in ("raw", "fade", "scale_in"):
            return True

        if self.mode in ("qualified", "qualified_consensus"):
            # Only wallets whose realized track record earned QUALIFIED (TraderScore).
            # `roster` is a live mapping the runner refreshes; it is never re-derived here.
            roster = self.param("roster", {}) or {}
            if event.wallet not in roster:
                self._refuse("wallet_not_qualified")
                return False
            if self.mode == "qualified":
                return True
            # qualified_consensus: >= min_agreeing distinct QUALIFIED wallets on the
            # same Kalshi contract and side within the window - the PRD's core signal.

        if self.mode == "specialist":
            history = self._wallet_actions.get(event.wallet, [])
            specialization = classify_specialization(history)
            if event.category is not specialization or specialization is Category.OTHER:
                self._refuse("not_specialist_category")
                return False
            return True

        if self.mode == "high_conviction":
            if event.usd_size is None:
                self._refuse("no_size_for_conviction_check")
                return False
            scorer = self._wallet_size_z.setdefault(event.wallet, RollingZScore(20))
            z = scorer.update(float(event.usd_size))
            size_z_min = float(self.param("size_z_min", DEFAULT_SIZE_Z_MIN))
            if z is None or z < size_z_min:
                self._refuse("below_conviction_threshold")
                return False
            return True

        if self.mode == "early":
            history = self._wallet_actions.get(event.wallet, [])
            prior_on_market = sum(1 for a in history[:-1] if a.canonical_id == event.canonical_id)
            if prior_on_market > 0:
                self._refuse("not_initial_entry")
                return False
            return True

        if self.mode in ("consensus", "qualified_consensus"):
            qualified = self.mode == "qualified_consensus"
            # The QUALIFIED set is small and its members act independently over hours
            # before an event, so its agreement window is longer and two suffice.
            window = float(self.param(
                "consensus_window_seconds",
                QUALIFIED_CONSENSUS_WINDOW_SECONDS if qualified else DEFAULT_CONSENSUS_WINDOW_SECONDS,
            ))
            key = (canonical_id, side)
            signals = self._recent_signals.setdefault(key, [])
            signals.append((event.wallet, event.first_seen_time))
            cutoff = event.first_seen_time.timestamp() - window
            signals[:] = [(w, ts) for w, ts in signals if ts.timestamp() >= cutoff]
            distinct_wallets = {w for w, _ in signals}
            min_agreeing = int(self.param("min_agreeing", QUALIFIED_MIN_AGREEING if qualified else DEFAULT_MIN_AGREEING))
            if len(distinct_wallets) < min_agreeing:
                self._refuse("insufficient_consensus")
                return False
            return True

        return True

    # ------------------------------------------------------------------ delayed execution

    def _drain_pending(self, only_canonical_id: str | None) -> None:
        now = self.now()
        remaining: list[dict[str, Any]] = []
        for entry in self._pending_copies:
            if only_canonical_id is not None and entry["canonical_id"] != only_canonical_id:
                remaining.append(entry)
                continue
            if entry["due_at"] > now:
                remaining.append(entry)
                continue
            self._execute_copy(entry)
        self._pending_copies = remaining

    def _execute_copy(self, entry: dict[str, Any]) -> None:
        canonical_id = entry["canonical_id"]
        if self.should_skip(canonical_id) is not None:
            self._refuse("book_unusable_at_execution")
            return
        market = self.ctx.market(canonical_id)
        book = self.ctx.book(canonical_id)
        assert market is not None and book is not None

        min_open_interest = Decimal(str(self.param("min_open_interest", DEFAULT_MIN_OPEN_INTEREST)))
        if market.open_interest < min_open_interest:
            self._refuse("thin_liquidity")
            return

        side: Side = entry["side"]
        if self.mode == "fade":
            side = side.opposite

        price = self.executable_price(book, side, Action.BUY)
        if price is None:
            self._refuse("no_liquidity_at_execution")
            return

        # We never execute at the source trader's price - the executed price is always
        # `price`, derived from the book fetched *now*, after the delay has elapsed.
        edge_assumption = Decimal(str(self.param("copy_edge_assumption", DEFAULT_COPY_EDGE_ASSUMPTION)))
        model_probability = _assumed_model_probability(price, side, edge_assumption)
        edge = self.edge_after_costs(model_probability, price, market, side)

        max_stake_pct = Decimal(str(self.param("max_stake_per_signal_pct", "0.05")))
        risk_fraction = max_stake_pct
        if self.mode == "scale_in":
            risk_fraction = clamp_probability(max_stake_pct * Decimal(str(entry["size_scale"])), lo=Decimal(0), hi=ONE)
        quantity = self.sensible_quantity(price, risk_fraction=risk_fraction)

        features: dict[str, Any] = {
            "wallet": entry["wallet"],
            "mode": self.mode,
            "follower_delay_seconds": self.follower_delay_seconds,
            "source_price": float(entry["source_price"]) if entry["source_price"] is not None else None,
            "executed_price": float(price),
            "due_at": entry["due_at"].isoformat(),
            "first_seen_time": entry["first_seen_time"].isoformat(),
            "signal_side": entry["side"].value,
            "traded_side": side.value,
            "copy_edge_assumption": float(edge_assumption),
        }
        intent = self.make_intent(
            canonical_id=canonical_id,
            side=side,
            action=Action.BUY,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=price,
            rationale=(
                f"Copy ({self.mode}) wallet={entry['wallet']}: signal side="
                f"{entry['side'].value} first_seen={entry['first_seen_time'].isoformat()}; "
                f"executed {self.follower_delay_seconds:.0f}s later at {price} "
                f"(source_price={entry['source_price']} recorded for audit only, never "
                f"used for execution). edge={edge}."
            ),
            features=features,
            model_probability=model_probability,
            expected_edge=edge,
            evidence_ids=(f"trader:{entry['wallet']}:{entry['transaction_hash'] or entry['due_at'].isoformat()}",),
        )
        if self.emit_if_profitable(intent):
            self.mark_fired(canonical_id)


class CopyBasketStrategy(CopyTraderStrategy):
    """A weighted basket of 5-20 validated wallets, each copied via the same delayed-book
    mechanism as :class:`CopyTraderStrategy`, with basket-level risk controls layered on
    top: per-trader weighting, a correlated-event exposure cap, and a liquidity floor.
    """

    name = "copy_basket"
    version = "1.1.0"
    evidence_class = "D"

    def __init__(self, strategy_id: str, experiment_id: str, ctx: Any, params: dict | None = None) -> None:
        super().__init__(strategy_id, experiment_id, ctx, params)
        traders: dict[str, Decimal] = {
            w: Decimal(str(v)) for w, v in (self.param("traders", {}) or {}).items()
        }
        # An empty roster is the NORMAL day-one state, not an error: traders are recorded
        # as DISCOVERED and then forward-tracked for `tracking.min_forward_days` (21) and
        # `min_forward_trades` (25) before any of them is validated. Raising here killed
        # sleeve creation for the whole tournament on a fresh database.
        #
        # So the strategy constructs, stays dormant, and starts trading only once the
        # roster fills. Too MANY traders is a genuine misconfiguration and still raises,
        # because silently trading 400 wallets is not what the operator asked for.
        if len(traders) > _MAX_BASKET_TRADERS:
            raise ValueError(
                f"CopyBasketStrategy accepts at most {_MAX_BASKET_TRADERS} traders, "
                f"got {len(traders)}"
            )
        self._min_basket_traders = int(self.param("min_basket_traders", _MIN_BASKET_TRADERS))
        if len(traders) < self._min_basket_traders:
            log.info(
                "copy_basket.dormant",
                strategy_id=strategy_id,
                validated_traders=len(traders),
                required=self._min_basket_traders,
                detail=(
                    "no validated traders yet; the basket stays dormant until forward "
                    "tracking qualifies enough wallets"
                ),
            )
        self._traders = traders
        self._weights = self._normalize_weights(traders, str(self.param("weighting", "equal")))
        self._max_correlated_event_pct = Decimal(str(self.param("max_correlated_event_pct", "0.30")))
        #: event_id -> cumulative committed exposure (price * quantity), reset never
        #: within a run; a real deployment would decay this against realized settlement,
        #: which is outside this strategy's read-only view of the world.
        self._event_exposure: dict[str, Decimal] = {}

    @staticmethod
    def _normalize_weights(traders: dict[str, Decimal], weighting: str) -> dict[str, Decimal]:
        if not traders:
            return {}
        if weighting == "equal":
            w = Decimal(1) / Decimal(len(traders))
            return dict.fromkeys(traders, w)
        # forward_validated: normalize the injected reliability scores (never re-derived
        # from a wallet's pre-discovery history - see marketlab.signals.copy_trader's
        # own module docstring rule 2).
        total = sum(traders.values(), Decimal(0))
        if total <= 0:
            return CopyBasketStrategy._normalize_weights(traders, "equal")
        return {w: v / total for w, v in traders.items()}

    def _sync_roster(self) -> None:
        """Follow the runner's live QUALIFIED roster: the top ``basket_size`` by score.

        The roster changes only when wallets are rescored, so the basket re-forms at most
        every ``roster_refresh_seconds``; a wallet leaving the roster stops being copied.
        """
        roster = self.param("roster", None)
        if roster is None or self.on_cooldown("__roster__", float(self.param("roster_refresh_seconds", 600))):
            return
        self.mark_fired("__roster__")
        size = min(int(self.param("basket_size", 10)), _MAX_BASKET_TRADERS)
        top = sorted(roster.items(), key=lambda kv: kv[1], reverse=True)[:size]
        traders = {w: Decimal(str(score)) for w, score in top}
        if set(traders) != set(self._traders):
            self._traders = traders
            self._weights = self._normalize_weights(traders, str(self.param("weighting", "equal")))
            log.info("copy_basket.roster", strategy_id=self.strategy_id, traders=len(traders))

    def on_timer(self, event: TimerEvent) -> None:
        self._sync_roster()
        super().on_timer(event)

    def on_trader_action(self, event: TraderActionEvent) -> None:
        self._sync_roster()
        if len(self._traders) < self._min_basket_traders:
            # Dormant: an under-filled basket is not a basket. Trading 1-4 wallets would
            # be a concentrated single-trader bet wearing a diversified label.
            return
        if event.wallet not in self._traders:
            self._refuse("not_a_basket_member")
            return
        super().on_trader_action(event)

    def _execute_copy(self, entry: dict[str, Any]) -> None:
        canonical_id = entry["canonical_id"]
        market = self.ctx.market(canonical_id)
        if market is None:
            self._refuse("no_market")
            return
        book = self.ctx.book(canonical_id)
        if book is None:
            self._refuse("no_book")
            return
        side: Side = entry["side"]
        price = self.executable_price(book, side, Action.BUY)
        if price is None:
            self._refuse("no_liquidity_at_execution")
            return

        weight = self._weights.get(entry["wallet"], Decimal(0))
        max_stake_pct = Decimal(str(self.param("max_stake_per_signal_pct", "0.05")))
        quantity = self.sensible_quantity(price, risk_fraction=max_stake_pct * weight)
        if quantity <= 0:
            self._refuse("zero_weighted_quantity")
            return

        prospective_notional = price * Decimal(quantity)
        cap = self._max_correlated_event_pct * self.bankroll()
        existing = self._event_exposure.get(market.event_id, Decimal(0))
        if existing + prospective_notional > cap:
            self._refuse("correlated_event_cap")
            return

        min_open_interest = Decimal(str(self.param("min_open_interest", DEFAULT_MIN_OPEN_INTEREST)))
        if market.open_interest < min_open_interest:
            self._refuse("thin_liquidity")
            return

        edge_assumption = Decimal(str(self.param("copy_edge_assumption", DEFAULT_COPY_EDGE_ASSUMPTION)))
        model_probability = _assumed_model_probability(price, side, edge_assumption)
        edge = self.edge_after_costs(model_probability, price, market, side)

        intent = self.make_intent(
            canonical_id=canonical_id,
            side=side,
            action=Action.BUY,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=price,
            rationale=(
                f"Copy basket wallet={entry['wallet']} weight={weight}: executed "
                f"{self.follower_delay_seconds:.0f}s after signal at {price} "
                f"(source_price={entry['source_price']} for audit only); edge={edge}."
            ),
            features={
                "wallet": entry["wallet"],
                "weight": float(weight),
                "source_price": float(entry["source_price"]) if entry["source_price"] is not None else None,
                "executed_price": float(price),
                "event_id": market.event_id,
            },
            model_probability=model_probability,
            expected_edge=edge,
            evidence_ids=(f"trader:{entry['wallet']}:{entry['transaction_hash'] or entry['due_at'].isoformat()}",),
        )
        if self.emit_if_profitable(intent):
            self._event_exposure[market.event_id] = existing + prospective_notional
            self.mark_fired(canonical_id)


__all__ = ["CopyTraderStrategy", "CopyBasketStrategy"]
