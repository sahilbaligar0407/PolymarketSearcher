"""The risk gate every :class:`OrderIntent` must pass before it can become an order.

This is the one place that enforces "no leverage, no borrowing, no martingale, no
Polymarket-global orders" regardless of what a strategy or a config file asks for. Two
design notes worth stating up front because the reference test suite exercises them
directly:

1. ``RejectReason`` (frozen, in ``marketlab.core.orders``) only has a few venue-shaped
   reasons (``MODE_FORBIDDEN``, ``INSUFFICIENT_CASH``, ``BELOW_MIN_ORDER``, ...). Checks
   that map cleanly onto one of those use it; every position/exposure/pause/martingale/
   audit check that has no dedicated enum member reports ``RejectReason.RISK_GATE`` with a
   ``detail`` string that names exactly which limit fired. ``PaperBroker`` is expected to
   copy ``RiskDecision.reason`` verbatim onto the rejected ``Order`` rather than
   collapsing everything to ``RISK_GATE`` itself - that's what lets a MODE_FORBIDDEN or
   INSUFFICIENT_CASH rejection actually show up as that specific reason end to end.
2. The venue-minimum-order check never resizes an intent. If ``intent.quantity`` is below
   ``market.min_order`` the intent is rejected outright (``BELOW_MIN_ORDER``) - there is no
   code path anywhere in this module that mutates ``intent.quantity`` upward to satisfy a
   venue minimum, even when doing so would still fit comfortably inside every risk limit.
   Deciding how big a bet should be is the strategy's job, never the gateway's.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from marketlab.core.instruments import (
    EXECUTION_VENUES,
    ONE,
    ZERO,
    Category,
    NormalizedMarket,
    OrderBook,
    Side,
)
from marketlab.core.orders import Action, OrderIntent, RejectReason
from marketlab.core.portfolio import Portfolio
from marketlab.settings import RiskConfig

#: How much a BUY may add to the cost basis of an already-underwater position (as a
#: fraction of that position's current cost basis) before it counts as "martingale"
#: doubling-down, when ``RiskConfig.martingale`` is False. This exact threshold is not
#: specified anywhere in the PRD; 25% is a documented judgment call - large enough that
#: ordinary dollar-cost-averaging-sized adds don't trip it, small enough to catch a
#: strategy trying to meaningfully double down on a loser.
MARTINGALE_INCREASE_THRESHOLD_PCT = Decimal("0.25")


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: RejectReason | None = None
    detail: str = ""


def _is_reducing(intent: OrderIntent, portfolio: Portfolio) -> bool:
    """A SELL that shrinks an existing position is "reducing"; everything else "increases"
    exposure. There is no shorting in this system (no leverage/borrowing), so a SELL can
    only ever be reducing an existing long - it can never open new risk.
    """
    if intent.action is not Action.SELL:
        return False
    pos = portfolio.positions.get(Portfolio.key(intent.canonical_id, intent.side))
    return pos is not None and pos.quantity > 0


def _estimate_price(intent: OrderIntent, book: OrderBook | None) -> Decimal:
    """Best pre-trade estimate of the price this intent would transact at.

    This is a *gate*, not the executor - it does not walk multiple book levels the way
    ``fill_models.walk_book`` does. Using the limit price (if given) or the best touch
    price is a conservative-enough approximation for cash/exposure checks.
    """
    if intent.limit_price is not None:
        return intent.limit_price
    if book is not None:
        if intent.side is Side.YES:
            p = book.best_ask if intent.action is Action.BUY else book.best_bid
        else:
            opposite = book.best_bid if intent.action is Action.BUY else book.best_ask
            p = (ONE - opposite) if opposite is not None else None
        if p is not None:
            return p
    # No price information at all: fall back to the worst-case midpoint of the probability
    # range rather than pretending we know the price.
    return Decimal("0.5")


class RiskGateway:
    """Evaluates one intent against a portfolio and the configured :class:`RiskConfig`."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        #: Per-check rejection counters, keyed by a short check name (not RejectReason,
        #: since several checks share RISK_GATE - this is what actually distinguishes them
        #: for metrics/dashboards).
        self.reject_counts: dict[str, int] = {}
        self.approved_count = 0

    def _reject(self, check: str, reason: RejectReason, detail: str) -> RiskDecision:
        self.reject_counts[check] = self.reject_counts.get(check, 0) + 1
        return RiskDecision(approved=False, reason=reason, detail=detail)

    def evaluate(
        self,
        intent: OrderIntent,
        portfolio: Portfolio,
        market: NormalizedMarket,
        book: OrderBook | None,
        category_exposures: dict[Category, Decimal],
        cluster_exposures: dict[str, Decimal],
        daily_pnl: Decimal,
    ) -> RiskDecision:
        cfg = self.config

        # --- Hard block: never reachable by configuration, ever. -----------------
        if intent.venue not in EXECUTION_VENUES:
            return self._reject(
                "mode_forbidden",
                RejectReason.MODE_FORBIDDEN,
                f"venue {intent.venue!r} is not an execution venue "
                f"(Polymarket-global is read-only; only {sorted(EXECUTION_VENUES)} may trade)",
            )

        # The intent's declared venue is NOT sufficient on its own. Strategies stamp a
        # constant `VENUE = Venue.KALSHI` on everything they emit, so an intent aimed at a
        # Polymarket market arrives declaring itself a Kalshi order and sails past the
        # check above. That is exactly what happened on the first full run: 166 fills
        # landed on `poly:` markets - contracts this deployment can never actually trade,
        # silently poisoning the P&L of every sleeve that touched one.
        #
        # The market's OWN venue is the authoritative fact, so it is checked here too.
        if market is not None and market.venue not in EXECUTION_VENUES:
            return self._reject(
                "mode_forbidden",
                RejectReason.MODE_FORBIDDEN,
                f"market {market.canonical_id!r} is on {market.venue!r}, which is a "
                f"read-only intelligence source; only {sorted(EXECUTION_VENUES)} may trade",
            )

        if cfg.strict_audit and not intent.rationale.strip():
            return self._reject(
                "strict_audit",
                RejectReason.RISK_GATE,
                "strict_audit: intent has no rationale",
            )

        est_price = _estimate_price(intent, book)
        cost = est_price * Decimal(intent.quantity)
        reducing = _is_reducing(intent, portfolio)

        # --- Venue minimum order size: reject, never resize. ----------------------
        if intent.quantity < market.min_order:
            return self._reject(
                "below_min_order",
                RejectReason.BELOW_MIN_ORDER,
                f"quantity {intent.quantity} is below venue minimum {market.min_order}; "
                "not resizing to meet it",
            )

        # --- Pause gates: existing positions may still be closed. -----------------
        daily_loss_floor = -(cfg.daily_loss_pause_pct * cfg.initial_capital)
        if not reducing and daily_pnl <= daily_loss_floor:
            return self._reject(
                "daily_loss_pause",
                RejectReason.RISK_GATE,
                f"daily P&L {daily_pnl} breaches daily_loss_pause_pct "
                f"({cfg.daily_loss_pause_pct}); only reducing orders are allowed",
            )
        if not reducing and portfolio.max_drawdown >= cfg.total_drawdown_pause_pct:
            return self._reject(
                "drawdown_pause",
                RejectReason.RISK_GATE,
                f"drawdown {portfolio.max_drawdown} breaches total_drawdown_pause_pct "
                f"({cfg.total_drawdown_pause_pct}); only reducing orders are allowed",
            )

        # A reducing order skips every exposure/cash/martingale check below: it can only
        # shrink risk, never add it, and pausing must never trap a sleeve in a losing
        # position it wants to exit.
        if reducing:
            self.approved_count += 1
            return RiskDecision(approved=True)

        # --- No leverage, no borrowing. ---------------------------------------------
        if not cfg.leverage and not cfg.borrowing and intent.action is Action.BUY and cost > portfolio.cash:
            return self._reject(
                "insufficient_cash",
                RejectReason.INSUFFICIENT_CASH,
                f"cost {cost} exceeds available cash {portfolio.cash}",
            )

        # --- Per-event loss cap (approximated at the single-market level: Portfolio has
        # no event_id grouping across canonical_ids, and evaluate() is not handed a
        # market graph, so "event" here means this one canonical_id's own two-sided
        # exposure, not a cross-market aggregation). ---------------------------------
        existing_market_exposure = sum(
            (p.cost_basis for k, p in portfolio.positions.items() if p.canonical_id == market.canonical_id and p.quantity > 0),
            ZERO,
        )
        max_event_loss = cfg.max_single_event_loss_pct * cfg.initial_capital
        if existing_market_exposure + cost > max_event_loss:
            return self._reject(
                "single_event_loss",
                RejectReason.RISK_GATE,
                f"prospective exposure {existing_market_exposure + cost} on "
                f"{market.canonical_id} exceeds max_single_event_loss_pct cap {max_event_loss}",
            )

        # --- Strategy-level exposure cap. --------------------------------------------
        max_strategy_exposure = cfg.max_strategy_exposure_pct * cfg.initial_capital
        if portfolio.exposure() + cost > max_strategy_exposure:
            return self._reject(
                "strategy_exposure",
                RejectReason.RISK_GATE,
                f"prospective strategy exposure {portfolio.exposure() + cost} exceeds "
                f"max_strategy_exposure_pct cap {max_strategy_exposure}",
            )

        # --- Category exposure cap. `category_exposures` is caller-supplied (typically
        # aggregated by a portfolio-wide layer outside this broker); we only add this
        # trade's prospective cost to whatever the caller already measured. -----------
        max_category_exposure = cfg.max_category_exposure_pct * cfg.initial_capital
        existing_category = category_exposures.get(market.category, ZERO)
        if existing_category + cost > max_category_exposure:
            return self._reject(
                "category_exposure",
                RejectReason.RISK_GATE,
                f"prospective category exposure {existing_category + cost} for "
                f"{market.category} exceeds max_category_exposure_pct cap {max_category_exposure}",
            )

        # --- Correlated cluster cap. No explicit "cluster id" field exists on
        # NormalizedMarket; we use `event_id` as the cluster key (markets that share an
        # event are the most obviously correlated group we can identify without an
        # external correlation graph). ------------------------------------------------
        max_cluster_exposure = cfg.max_correlated_cluster_pct * cfg.initial_capital
        existing_cluster = cluster_exposures.get(market.event_id, ZERO)
        if existing_cluster + cost > max_cluster_exposure:
            return self._reject(
                "cluster_exposure",
                RejectReason.RISK_GATE,
                f"prospective cluster exposure {existing_cluster + cost} for event "
                f"{market.event_id} exceeds max_correlated_cluster_pct cap {max_cluster_exposure}",
            )

        # --- Martingale detection. ----------------------------------------------------
        if not cfg.martingale and intent.action is Action.BUY:
            pos = portfolio.positions.get(Portfolio.key(intent.canonical_id, intent.side))
            if pos is not None and pos.quantity > 0:
                mark = book.mid if book is not None else pos.average_price
                if intent.side is Side.NO and book is not None and book.mid is not None:
                    mark = ONE - book.mid
                underwater = mark is not None and mark < pos.average_price
                if underwater and pos.cost_basis > ZERO:
                    increase_pct = cost / pos.cost_basis
                    if increase_pct > MARTINGALE_INCREASE_THRESHOLD_PCT:
                        return self._reject(
                            "martingale",
                            RejectReason.RISK_GATE,
                            f"adding {cost} ({increase_pct:.0%}) to an underwater position "
                            f"(avg {pos.average_price}, mark {mark}) exceeds the "
                            f"{MARTINGALE_INCREASE_THRESHOLD_PCT:.0%} martingale threshold",
                        )

        self.approved_count += 1
        return RiskDecision(approved=True)
