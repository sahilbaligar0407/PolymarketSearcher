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

import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

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
from marketlab.settings import DATA_DIR, RiskConfig

#: The global kill switch. While this file exists every risk-increasing order is refused,
#: by every strategy, in every mode. `marketlab kill` / STOP_TRADING.bat create it;
#: `marketlab unkill` removes it. A file, not a flag in memory, so it survives restarts and
#: can be pulled by hand from Explorer if everything else is wedged.
KILL_SWITCH_PATH = DATA_DIR / "KILL_SWITCH"

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


def cluster_key(market: NormalizedMarket) -> str:
    """The correlated cluster a market belongs to: its series (the event ticker up to
    the first ``-``), so every hour of KXBTCD - or every game of one league series -
    shares one cap. Event-level concentration is the per-event cap's job."""
    event = market.event_id or market.canonical_id
    return event.split("-", 1)[0].upper()


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

    def __init__(self, config: RiskConfig, kill_switch_path: Path | None = None) -> None:
        self.config = config
        self.kill_switch_path = kill_switch_path or KILL_SWITCH_PATH
        self._kill_checked_at = float("-inf")
        self._kill_engaged = False
        #: Per-check rejection counters, keyed by a short check name (not RejectReason,
        #: since several checks share RISK_GATE - this is what actually distinguishes them
        #: for metrics/dashboards).
        self.reject_counts: dict[str, int] = {}
        self.approved_count = 0

    def kill_switch_engaged(self) -> bool:
        # One stat() per second at most: evaluate() runs for every intent of every sleeve.
        now = time.monotonic()
        if now - self._kill_checked_at >= 1.0:
            self._kill_engaged = self.kill_switch_path.exists()
            self._kill_checked_at = now
        return self._kill_engaged

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
        *,
        marks: dict[str, Decimal] | None = None,
        open_order_exposure: Decimal = ZERO,
        event_exposure: Decimal | None = None,
    ) -> RiskDecision:
        """``category_exposures`` / ``cluster_exposures`` / ``event_exposure`` include the
        sleeve's resting buy orders; ``open_order_exposure`` is their total, reserved
        against cash and the strategy cap. Every cap scales with the sleeve's own
        ``initial_capital`` (FINDINGS 58)."""
        cfg = self.config
        capital = portfolio.initial_capital if portfolio.initial_capital > ZERO else cfg.initial_capital

        # --- Global kill switch: overrides every strategy, model and config. -----
        if self.kill_switch_engaged() and not _is_reducing(intent, portfolio):
            return self._reject(
                "kill_switch",
                RejectReason.RISK_GATE,
                f"global kill switch engaged ({self.kill_switch_path.name} present); no new risk",
            )

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
        daily_loss_floor = -(cfg.daily_loss_pause_pct * capital)
        if not reducing and daily_pnl <= daily_loss_floor:
            return self._reject(
                "daily_loss_pause",
                RejectReason.RISK_GATE,
                f"daily P&L {daily_pnl} breaches daily_loss_pause_pct "
                f"({cfg.daily_loss_pause_pct}); only reducing orders are allowed",
            )
        # Measured as loss of CAPITAL, not peak-to-trough. `max_drawdown` on Portfolio is
        # a high-water-mark measure, and pausing on it permanently halted sleeves that
        # were actually in profit: one sat at $56.57 on a $50 bankroll - up 13% - and was
        # frozen because it had once been higher. The PRD states this limit alongside
        # `initial_capital` and `daily_loss_pause_pct`, i.e. as a capital-preservation
        # rule, and permanently retiring a profitable experiment is the opposite of that.
        #
        # Peak-to-trough drawdown is still recorded and still reported - it is a headline
        # research metric - it just is not what trips the trading halt.
        equity = portfolio.equity(marks)
        capital_drawdown = (
            (capital - equity) / capital if capital > ZERO and equity < capital else ZERO
        )
        if not reducing and capital_drawdown >= cfg.total_drawdown_pause_pct:
            return self._reject(
                "drawdown_pause",
                RejectReason.RISK_GATE,
                f"equity {equity} is {capital_drawdown:.1%} below initial capital "
                f"{capital}, breaching total_drawdown_pause_pct "
                f"({cfg.total_drawdown_pause_pct}); only reducing orders are allowed",
            )

        # A reducing order skips every exposure/cash/martingale check below: it can only
        # shrink risk, never add it, and pausing must never trap a sleeve in a losing
        # position it wants to exit.
        if reducing:
            self.approved_count += 1
            return RiskDecision(approved=True)

        # --- No leverage, no borrowing. ---------------------------------------------
        free_cash = portfolio.cash - open_order_exposure
        if not cfg.leverage and not cfg.borrowing and intent.action is Action.BUY and cost > free_cash:
            return self._reject(
                "insufficient_cash",
                RejectReason.INSUFFICIENT_CASH,
                f"cost {cost} exceeds available cash {free_cash} "
                f"({portfolio.cash} less {open_order_exposure} reserved by resting orders)",
            )

        # --- Per-event loss cap (approximated at the single-market level: Portfolio has
        # no event_id grouping across canonical_ids, and evaluate() is not handed a
        # market graph, so "event" here means this one canonical_id's own two-sided
        # exposure, not a cross-market aggregation). ---------------------------------
        existing_market_exposure = sum(
            (p.cost_basis for k, p in portfolio.positions.items() if p.canonical_id == market.canonical_id and p.quantity > 0),
            ZERO,
        )
        # When the caller can see the whole event (every market sharing event_id, e.g. an
        # hour's BTC strike ladder or both teams of a game, plus resting orders), the cap
        # applies to that; it used to be one canonical_id, so a sleeve held $19.79 across
        # 7 strikes of one hour against a $4 "per-event" cap.
        if event_exposure is not None:
            existing_market_exposure = event_exposure
        max_event_loss = cfg.max_single_event_loss_pct * capital
        if existing_market_exposure + cost > max_event_loss:
            return self._reject(
                "single_event_loss",
                RejectReason.RISK_GATE,
                f"prospective exposure {existing_market_exposure + cost} on "
                f"{market.canonical_id} exceeds max_single_event_loss_pct cap {max_event_loss}",
            )

        # --- Strategy-level exposure cap. --------------------------------------------
        max_strategy_exposure = cfg.max_strategy_exposure_pct * capital
        committed = portfolio.exposure() + open_order_exposure
        if committed + cost > max_strategy_exposure:
            return self._reject(
                "strategy_exposure",
                RejectReason.RISK_GATE,
                f"prospective strategy exposure {committed + cost} (incl. {open_order_exposure} "
                f"resting) exceeds max_strategy_exposure_pct cap {max_strategy_exposure}",
            )

        # --- Category exposure cap. `category_exposures` is caller-supplied (typically
        # aggregated by a portfolio-wide layer outside this broker); we only add this
        # trade's prospective cost to whatever the caller already measured. -----------
        max_category_exposure = cfg.max_category_exposure_pct * capital
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
        max_cluster_exposure = cfg.max_correlated_cluster_pct * capital
        existing_cluster = cluster_exposures.get(
            cluster_key(market), cluster_exposures.get(market.event_id, ZERO)
        )
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
