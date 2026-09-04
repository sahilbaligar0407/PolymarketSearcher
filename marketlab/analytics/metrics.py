"""Every honest metric the PRD requires, computed from orders, fills, forecasts and
portfolio snapshots.

Nothing here calls ``datetime.now()`` — every timestamp is supplied by the caller, taken
from an injected :class:`~marketlab.clock.Clock` elsewhere in the system.  Money stays in
``Decimal``; ratios, rates and statistics are ``float`` (``Decimal`` sqrt/variance is
painful and buys nothing here).

**The empty-sample rule**: every metric here returns ``None`` (or a dataclass whose
fields are ``None``/zero-count) when there is not enough data to compute it honestly.
Never fabricate a ``0.0`` that a reader could mistake for a real measurement of "no
edge" — an empty sample means "unknown", not "zero".
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from marketlab.core.instruments import ZERO, Side
from marketlab.core.orders import Fill, Order, OrderStatus, RejectReason

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeRecord:
    """One resolved (or still-open) round-trip trade, built by the caller from fills and
    settlements.  ``pnl`` is net of fees for this trade (matches the convention used by
    :class:`marketlab.core.portfolio.Position`, whose ``realized_pnl`` is already fee-net).

    This is an analytics-domain object, not a frozen core contract — there is no
    "Trade" concept upstream of Position/Fill, so callers assemble these however their
    execution log allows (e.g. one ``TradeRecord`` per market per sleeve, opened at first
    fill and closed at settlement or final flat fill).
    """

    trade_id: str
    strategy_id: str = ""
    canonical_id: str = ""
    pnl: Decimal = ZERO
    fee: Decimal = ZERO
    #: Price-only P&L before fees/slippage were subtracted, if known. Falls back to pnl.
    gross_pnl: Decimal | None = None
    #: Fill price minus reference/decision price, signed adverse, if known.
    slippage_cost: Decimal | None = None
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    #: Capital placed at risk (cost basis) for this trade.
    exposure: Decimal | None = None
    is_resolved: bool = True

    @property
    def holding_period_seconds(self) -> float | None:
        if self.entry_time is None or self.exit_time is None:
            return None
        return (self.exit_time - self.entry_time).total_seconds()


def _resolved_pnls(trades: Sequence[TradeRecord]) -> list[Decimal]:
    return [t.pnl for t in trades if t.is_resolved]


# ---------------------------------------------------------------------------
# Small pure metrics (independently testable, composed into TradingMetrics below)
# ---------------------------------------------------------------------------


def win_rate(pnls: Sequence[Decimal]) -> Decimal | None:
    if not pnls:
        return None
    wins = sum(1 for p in pnls if p > ZERO)
    return Decimal(wins) / Decimal(len(pnls))


def average_winner(pnls: Sequence[Decimal]) -> Decimal | None:
    winners = [p for p in pnls if p > ZERO]
    if not winners:
        return None
    return sum(winners, ZERO) / Decimal(len(winners))


def average_loser(pnls: Sequence[Decimal]) -> Decimal | None:
    losers = [p for p in pnls if p < ZERO]
    if not losers:
        return None
    return sum(losers, ZERO) / Decimal(len(losers))


def expectancy_per_trade(pnls: Sequence[Decimal]) -> Decimal | None:
    if not pnls:
        return None
    return sum(pnls, ZERO) / Decimal(len(pnls))


def profit_factor(pnls: Sequence[Decimal]) -> Decimal | None:
    """Gross winnings / gross losses. ``None`` when undefined (no trades, or no losers to
    divide by — an all-winners record has no denominator, not an infinite profit factor).
    """
    if not pnls:
        return None
    gross_win = sum((p for p in pnls if p > ZERO), ZERO)
    gross_loss = sum((-p for p in pnls if p < ZERO), ZERO)
    if gross_loss == ZERO:
        return None
    return gross_win / gross_loss


def largest_single_win(pnls: Sequence[Decimal]) -> Decimal | None:
    return max(pnls) if pnls else None


def largest_single_loss(pnls: Sequence[Decimal]) -> Decimal | None:
    return min(pnls) if pnls else None


def average_holding_period_seconds(trades: Sequence[TradeRecord]) -> float | None:
    periods = [t.holding_period_seconds for t in trades if t.holding_period_seconds is not None]
    if not periods:
        return None
    return sum(periods) / len(periods)


@dataclass(frozen=True)
class DrawdownResult:
    max_drawdown: Decimal | None
    #: Wall-clock duration of the drawdown episode containing the deepest point: from the
    #: prior equity peak to the moment equity recovers back to it, or to the last point in
    #: the curve if it never recovers (an "ongoing" drawdown — flagged via ``recovered``).
    duration_seconds: float | None
    recovered: bool | None = None


def max_drawdown_from_curve(equity_curve: Sequence[tuple[datetime, Decimal]]) -> DrawdownResult:
    if len(equity_curve) < 2:
        return DrawdownResult(None, None, None)
    curve = sorted(equity_curve, key=lambda x: x[0])
    peak_value = curve[0][1]
    peak_time = curve[0][0]
    worst_dd = ZERO
    worst_peak_time = peak_time
    worst_trough_time = peak_time
    worst_trough_value = peak_value
    for ts, eq in curve:
        if eq > peak_value:
            peak_value = eq
            peak_time = ts
        if peak_value > ZERO:
            dd = (peak_value - eq) / peak_value
            if dd > worst_dd:
                worst_dd = dd
                worst_peak_time = peak_time
                worst_trough_time = ts
                worst_trough_value = eq
    if worst_dd == ZERO:
        return DrawdownResult(ZERO, 0.0, True)
    # Find recovery: first point at/after the trough whose equity >= the peak that preceded it.
    recovered_time: datetime | None = None
    for ts, eq in curve:
        if ts <= worst_trough_time:
            continue
        peak_for_this_dd = next(
            (p for t, p in curve if t == worst_peak_time), worst_trough_value
        )
        if eq >= peak_for_this_dd:
            recovered_time = ts
            break
    end_time = recovered_time if recovered_time is not None else curve[-1][0]
    duration = (end_time - worst_peak_time).total_seconds()
    return DrawdownResult(worst_dd, duration, recovered_time is not None)


def average_exposure(exposure_curve: Sequence[tuple[datetime, Decimal]]) -> Decimal | None:
    if not exposure_curve:
        return None
    values = [v for _, v in exposure_curve]
    return sum(values, ZERO) / Decimal(len(values))


def maximum_exposure(exposure_curve: Sequence[tuple[datetime, Decimal]]) -> Decimal | None:
    if not exposure_curve:
        return None
    return max(v for _, v in exposure_curve)


def fill_ratio(orders: Sequence[Order]) -> Decimal | None:
    """Filled quantity / requested quantity, across all orders given."""
    if not orders:
        return None
    requested = sum(o.quantity for o in orders)
    if requested == 0:
        return None
    filled = sum(o.filled_quantity for o in orders)
    return Decimal(filled) / Decimal(requested)


def maker_fill_ratio(fills: Sequence[Fill]) -> Decimal | None:
    if not fills:
        return None
    total_qty = sum(f.quantity for f in fills)
    if total_qty == 0:
        return None
    maker_qty = sum(f.quantity for f in fills if f.is_maker)
    return Decimal(maker_qty) / Decimal(total_qty)


def cancel_ratio(orders: Sequence[Order]) -> Decimal | None:
    if not orders:
        return None
    canceled = sum(1 for o in orders if o.status is OrderStatus.CANCELED)
    return Decimal(canceled) / Decimal(len(orders))


def rejected_order_count(orders: Sequence[Order]) -> int:
    return sum(1 for o in orders if o.status is OrderStatus.REJECTED)


def stale_data_skip_count(orders: Sequence[Order]) -> int:
    return sum(1 for o in orders if o.reject_reason is RejectReason.STALE_DATA)


def risk_gate_skip_count(orders: Sequence[Order]) -> int:
    return sum(1 for o in orders if o.reject_reason is RejectReason.RISK_GATE)


def turnover(fills: Sequence[Fill], starting_bankroll: Decimal) -> Decimal | None:
    """Total notional traded (both sides) divided by starting bankroll.

    A turnover of 3.0 means the sleeve cycled its starting capital through trades three
    times over. Chosen over "notional / average equity" because starting bankroll is
    always known and stable; average equity requires the same equity curve this function
    doesn't otherwise need.
    """
    if not fills or starting_bankroll <= ZERO:
        return None
    notional = sum((f.notional for f in fills), ZERO)
    return notional / starting_bankroll


# ---------------------------------------------------------------------------
# TradingMetrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradingMetrics:
    starting_bankroll: Decimal
    ending_bankroll: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    net_pnl: Decimal
    gross_pnl: Decimal | None
    fees: Decimal
    estimated_slippage: Decimal | None
    turnover: Decimal | None
    n_trades: int
    n_resolved_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal | None
    average_winner: Decimal | None
    average_loser: Decimal | None
    expectancy_per_trade: Decimal | None
    profit_factor: Decimal | None
    max_drawdown: Decimal | None
    drawdown_duration_seconds: float | None
    largest_single_loss: Decimal | None
    largest_single_win: Decimal | None
    average_exposure: Decimal | None
    maximum_exposure: Decimal | None
    average_holding_period_seconds: float | None
    fill_ratio: Decimal | None
    maker_fill_ratio: Decimal | None
    cancel_ratio: Decimal | None
    rejected_order_count: int
    stale_data_skip_count: int
    risk_gate_skip_count: int


def compute_trading_metrics(
    *,
    starting_bankroll: Decimal,
    ending_bankroll: Decimal,
    realized_pnl: Decimal,
    unrealized_pnl: Decimal = ZERO,
    trades: Sequence[TradeRecord] = (),
    orders: Sequence[Order] = (),
    fills: Sequence[Fill] = (),
    equity_curve: Sequence[tuple[datetime, Decimal]] = (),
    exposure_curve: Sequence[tuple[datetime, Decimal]] = (),
    fees: Decimal | None = None,
    estimated_slippage: Decimal | None = None,
) -> TradingMetrics:
    """Assemble every :class:`TradingMetrics` field from raw execution records.

    ``net_pnl`` is defined as ``realized_pnl + unrealized_pnl`` — exactly how
    :class:`marketlab.core.portfolio.Portfolio` already accounts for it (fees are folded
    into ``realized_pnl`` per fill, see ``Position.apply``). ``gross_pnl`` adds fees and
    estimated slippage back on top of ``net_pnl`` so a reader can see what the trades
    would have made ignoring the cost of trading — it is a display decomposition, not a
    second independent P&L calculation.
    """
    resolved = [t for t in trades if t.is_resolved]
    pnls = _resolved_pnls(trades)

    fees_total = fees if fees is not None else sum((t.fee for t in trades), ZERO)
    slippage_total = estimated_slippage
    if slippage_total is None:
        known = [t.slippage_cost for t in trades if t.slippage_cost is not None]
        slippage_total = sum(known, ZERO) if known else None

    net_pnl = realized_pnl + unrealized_pnl
    gross_pnl = net_pnl + fees_total + (slippage_total or ZERO) if trades or fees is not None else None

    dd = max_drawdown_from_curve(equity_curve)

    return TradingMetrics(
        starting_bankroll=starting_bankroll,
        ending_bankroll=ending_bankroll,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        net_pnl=net_pnl,
        gross_pnl=gross_pnl,
        fees=fees_total,
        estimated_slippage=slippage_total,
        turnover=turnover(fills, starting_bankroll) if fills else None,
        n_trades=len(trades),
        n_resolved_trades=len(resolved),
        winning_trades=sum(1 for p in pnls if p > ZERO),
        losing_trades=sum(1 for p in pnls if p < ZERO),
        win_rate=win_rate(pnls),
        average_winner=average_winner(pnls),
        average_loser=average_loser(pnls),
        expectancy_per_trade=expectancy_per_trade(pnls),
        profit_factor=profit_factor(pnls),
        max_drawdown=dd.max_drawdown,
        drawdown_duration_seconds=dd.duration_seconds,
        largest_single_loss=largest_single_loss(pnls),
        largest_single_win=largest_single_win(pnls),
        average_exposure=average_exposure(exposure_curve),
        maximum_exposure=maximum_exposure(exposure_curve),
        average_holding_period_seconds=average_holding_period_seconds(trades),
        fill_ratio=fill_ratio(orders) if orders else None,
        maker_fill_ratio=maker_fill_ratio(fills) if fills else None,
        cancel_ratio=cancel_ratio(orders) if orders else None,
        rejected_order_count=rejected_order_count(orders),
        stale_data_skip_count=stale_data_skip_count(orders),
        risk_gate_skip_count=risk_gate_skip_count(orders),
    )


# ---------------------------------------------------------------------------
# ExecutionMetrics (high-frequency arms)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionMetrics:
    quoted_edge: Decimal | None
    edge_at_arrival: Decimal | None
    edge_after_1s: Decimal | None
    edge_after_5s: Decimal | None
    realized_edge: Decimal | None
    #: How much the price moved against a passive fill in the seconds after it.
    adverse_selection_loss: Decimal | None
    fill_adjusted_return: Decimal | None


@dataclass(frozen=True)
class PassiveFillObservation:
    """One passive (maker) fill plus the mid-price observed at each latency horizon."""

    fill_price: Decimal
    side: Side
    quoted_mid_at_post: Decimal | None = None
    mid_at_arrival: Decimal | None = None
    mid_after_1s: Decimal | None = None
    mid_after_5s: Decimal | None = None
    settlement_value: Decimal | None = None
    fee: Decimal = ZERO
    quantity: int = 1


def _signed_move(entry: Decimal, later: Decimal, side: Side) -> Decimal:
    """Positive = moved in the position's favor."""
    move = later - entry
    return move if side is Side.YES else -move


def compute_execution_metrics(observations: Sequence[PassiveFillObservation]) -> ExecutionMetrics:
    if not observations:
        return ExecutionMetrics(None, None, None, None, None, None, None)

    def _avg(getter) -> Decimal | None:
        vals = [getter(o) for o in observations]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return sum(vals, ZERO) / Decimal(len(vals))

    quoted_edge = _avg(
        lambda o: _signed_move(o.fill_price, o.quoted_mid_at_post, o.side)
        if o.quoted_mid_at_post is not None
        else None
    )
    edge_at_arrival = _avg(
        lambda o: _signed_move(o.fill_price, o.mid_at_arrival, o.side)
        if o.mid_at_arrival is not None
        else None
    )
    edge_after_1s = _avg(
        lambda o: _signed_move(o.fill_price, o.mid_after_1s, o.side)
        if o.mid_after_1s is not None
        else None
    )
    edge_after_5s = _avg(
        lambda o: _signed_move(o.fill_price, o.mid_after_5s, o.side)
        if o.mid_after_5s is not None
        else None
    )
    realized_edge = _avg(
        lambda o: _signed_move(o.fill_price, o.settlement_value, o.side)
        if o.settlement_value is not None
        else None
    )
    # Adverse selection: the passive quoter got picked off if the market moved AGAINST
    # the side just filled, in the seconds immediately after. Reported as a positive loss
    # magnitude (0 or better means no measurable adverse selection).
    adverse = _avg(
        lambda o: max(ZERO, -_signed_move(o.fill_price, o.mid_after_5s, o.side))
        if o.mid_after_5s is not None
        else None
    )
    fill_adjusted = None
    fee_and_edge = [
        (_signed_move(o.fill_price, o.settlement_value, o.side) - o.fee)
        for o in observations
        if o.settlement_value is not None
    ]
    if fee_and_edge:
        fill_adjusted = sum(fee_and_edge, ZERO) / Decimal(len(fee_and_edge))

    return ExecutionMetrics(
        quoted_edge=quoted_edge,
        edge_at_arrival=edge_at_arrival,
        edge_after_1s=edge_after_1s,
        edge_after_5s=edge_after_5s,
        realized_edge=realized_edge,
        adverse_selection_loss=adverse,
        fill_adjusted_return=fill_adjusted,
    )


# ---------------------------------------------------------------------------
# CopyMetrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CopyMetrics:
    source_trader_pnl: Decimal | None
    #: Zero-delay diagnostic: what a follower would have made with instantaneous copying.
    theoretical_follower_pnl: Decimal | None
    follower_pnl_1s: Decimal | None
    follower_pnl_5s: Decimal | None
    follower_pnl_30s: Decimal | None
    follower_pnl_2m: Decimal | None
    actual_fill_rate: Decimal | None
    #: Follower entry price minus source entry price (positive = followed at a worse price).
    price_disadvantage: Decimal | None


@dataclass(frozen=True)
class CopyObservation:
    source_entry_price: Decimal
    side: Side
    source_pnl: Decimal | None = None
    #: Price available to the follower after each delay, None if the market moved away
    #: or there was no liquidity (counts against actual_fill_rate, not against price).
    follower_price_1s: Decimal | None = None
    follower_price_5s: Decimal | None = None
    follower_price_30s: Decimal | None = None
    follower_price_2m: Decimal | None = None
    settlement_value: Decimal | None = None
    attempted: bool = True
    filled: bool = True


def compute_copy_metrics(observations: Sequence[CopyObservation]) -> CopyMetrics:
    if not observations:
        return CopyMetrics(None, None, None, None, None, None, None, None)

    def _pnl_at(getter) -> Decimal | None:
        vals: list[Decimal] = []
        for o in observations:
            price = getter(o)
            if price is None or o.settlement_value is None:
                continue
            vals.append(_signed_move(price, o.settlement_value, o.side))
        return sum(vals, ZERO) / Decimal(len(vals)) if vals else None

    theoretical = _pnl_at(lambda o: o.source_entry_price)
    source_vals = [o.source_pnl for o in observations if o.source_pnl is not None]
    source_pnl = sum(source_vals, ZERO) / Decimal(len(source_vals)) if source_vals else None

    attempted = [o for o in observations if o.attempted]
    fill_rate = (
        Decimal(sum(1 for o in attempted if o.filled)) / Decimal(len(attempted))
        if attempted
        else None
    )

    disadvantages = [
        (getter(o) - o.source_entry_price)
        for o in observations
        for getter in (lambda x: x.follower_price_1s,)
        if getter(o) is not None
    ]
    price_disadvantage = (
        sum(disadvantages, ZERO) / Decimal(len(disadvantages)) if disadvantages else None
    )

    return CopyMetrics(
        source_trader_pnl=source_pnl,
        theoretical_follower_pnl=theoretical,
        follower_pnl_1s=_pnl_at(lambda o: o.follower_price_1s),
        follower_pnl_5s=_pnl_at(lambda o: o.follower_price_5s),
        follower_pnl_30s=_pnl_at(lambda o: o.follower_price_30s),
        follower_pnl_2m=_pnl_at(lambda o: o.follower_price_2m),
        actual_fill_rate=fill_rate,
        price_disadvantage=price_disadvantage,
    )


# ---------------------------------------------------------------------------
# CrossMarketMetrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossMarketMetrics:
    #: The headline number: how much of the apparent gap evaporates before it is
    #: executable. Reported so it cannot be buried — see module docstring in
    #: docs/CONTRACTS.md's cross_venue strategy.
    displayed_discrepancy: Decimal | None
    executable_discrepancy: Decimal | None
    post_fee_discrepancy: Decimal | None
    captured_discrepancy: Decimal | None
    leg_failure_rate: Decimal | None


@dataclass(frozen=True)
class CrossMarketObservation:
    #: Discrepancy read off the top-of-book quotes (best bid/ask) at signal time.
    displayed_discrepancy: Decimal
    #: Discrepancy actually available walking the book to the needed size.
    executable_discrepancy: Decimal | None = None
    fees: Decimal = ZERO
    #: What was actually captured once both legs (or the single Kalshi leg) settled.
    captured: Decimal | None = None
    #: True if any required leg failed to fill (no-liquidity, rejected, etc).
    leg_failed: bool = False


def compute_cross_market_metrics(
    observations: Sequence[CrossMarketObservation],
) -> CrossMarketMetrics:
    if not observations:
        return CrossMarketMetrics(None, None, None, None, None)

    def _avg(vals: list[Decimal]) -> Decimal | None:
        return sum(vals, ZERO) / Decimal(len(vals)) if vals else None

    displayed = _avg([o.displayed_discrepancy for o in observations])
    executable_vals = [o.executable_discrepancy for o in observations if o.executable_discrepancy is not None]
    executable = _avg(executable_vals)
    post_fee_vals = [
        o.executable_discrepancy - o.fees
        for o in observations
        if o.executable_discrepancy is not None
    ]
    post_fee = _avg(post_fee_vals)
    captured_vals = [o.captured for o in observations if o.captured is not None]
    captured = _avg(captured_vals)
    leg_failure_rate = Decimal(sum(1 for o in observations if o.leg_failed)) / Decimal(
        len(observations)
    )

    return CrossMarketMetrics(
        displayed_discrepancy=displayed,
        executable_discrepancy=executable,
        post_fee_discrepancy=post_fee,
        captured_discrepancy=captured,
        leg_failure_rate=leg_failure_rate,
    )


# ---------------------------------------------------------------------------
# NewsSignalMetrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NewsSignalMetrics:
    source_latency: float | None
    classifier_latency: float | None
    llm_latency: float | None
    signal_latency: float | None
    entry_latency: float | None
    topic: str | None
    novelty: Decimal | None
    confidence: Decimal | None
    #: Mapping of horizon label (e.g. "5m", "1h") to average forward return.
    forward_returns: dict[str, Decimal | None] = field(default_factory=dict)


@dataclass(frozen=True)
class NewsSignalObservation:
    source_latency: float
    classifier_latency: float
    llm_latency: float
    entry_latency: float
    topic: str
    novelty: Decimal
    confidence: Decimal
    #: horizon label -> forward return (price move after the signal), None if not yet known.
    forward_returns: dict[str, Decimal | None] = field(default_factory=dict)

    @property
    def signal_latency(self) -> float:
        return self.source_latency + self.classifier_latency + self.llm_latency


def compute_news_signal_metrics(
    observations: Sequence[NewsSignalObservation],
) -> NewsSignalMetrics:
    if not observations:
        return NewsSignalMetrics(None, None, None, None, None, None, None, None, {})

    def _favg(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    def _davg(vals: list[Decimal]) -> Decimal | None:
        return sum(vals, ZERO) / Decimal(len(vals)) if vals else None

    horizons: set[str] = set()
    for o in observations:
        horizons |= set(o.forward_returns)
    forward: dict[str, Decimal | None] = {}
    for h in sorted(horizons):
        vals = [v for o in observations if (v := o.forward_returns.get(h)) is not None]
        forward[h] = _davg(vals)

    topics = [o.topic for o in observations]
    most_common_topic = statistics.mode(topics) if topics else None

    return NewsSignalMetrics(
        source_latency=_favg([o.source_latency for o in observations]),
        classifier_latency=_favg([o.classifier_latency for o in observations]),
        llm_latency=_favg([o.llm_latency for o in observations]),
        signal_latency=_favg([o.signal_latency for o in observations]),
        entry_latency=_favg([o.entry_latency for o in observations]),
        topic=most_common_topic,
        novelty=_davg([o.novelty for o in observations]),
        confidence=_davg([o.confidence for o in observations]),
        forward_returns=forward,
    )


# ---------------------------------------------------------------------------
# RiskAdjustedMetrics + normalized-risk ranking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskAdjustedMetrics:
    #: Mean per-trade return / stdev per-trade return. Not annualized (trades are not
    #: evenly spaced in time), so this is comparable only across strategies, not to a
    #: textbook annualized Sharpe.
    sharpe_like: float | None
    sortino: float | None
    #: net P&L return (fraction of starting bankroll) / max drawdown (fraction).
    calmar: float | None
    #: net P&L / maximum capital ever placed at risk. The metric normalized_risk_ranking
    #: sorts on.
    return_per_max_exposure: float | None


def _to_float_returns(pnls: Sequence[Decimal], exposures: Sequence[Decimal]) -> list[float]:
    out = []
    for pnl, exp in zip(pnls, exposures, strict=True):
        if exp and exp > ZERO:
            out.append(float(pnl / exp))
    return out


def compute_risk_adjusted_metrics(
    *,
    trade_returns: Sequence[float] = (),
    net_pnl: Decimal | None = None,
    starting_bankroll: Decimal | None = None,
    max_drawdown: Decimal | None = None,
    maximum_exposure: Decimal | None = None,
) -> RiskAdjustedMetrics:
    sharpe_like: float | None = None
    sortino: float | None = None
    if len(trade_returns) >= 2:
        mean = statistics.fmean(trade_returns)
        stdev = statistics.pstdev(trade_returns)
        sharpe_like = mean / stdev if stdev > 0 else None
        downside = [r for r in trade_returns if r < 0]
        downside_std = statistics.pstdev(downside) if len(downside) >= 2 else None
        sortino = mean / downside_std if downside_std and downside_std > 0 else None

    calmar: float | None = None
    if (
        net_pnl is not None
        and starting_bankroll
        and starting_bankroll > ZERO
        and max_drawdown is not None
        and max_drawdown > ZERO
    ):
        period_return = float(net_pnl / starting_bankroll)
        calmar = period_return / float(max_drawdown)

    return_per_exposure: float | None = None
    if net_pnl is not None and maximum_exposure and maximum_exposure > ZERO:
        return_per_exposure = float(net_pnl / maximum_exposure)

    return RiskAdjustedMetrics(
        sharpe_like=sharpe_like,
        sortino=sortino,
        calmar=calmar,
        return_per_max_exposure=return_per_exposure,
    )


@dataclass(frozen=True)
class StrategyRiskEntry:
    strategy_id: str
    net_pnl: Decimal
    maximum_exposure: Decimal


@dataclass(frozen=True)
class RankedStrategy:
    strategy_id: str
    net_pnl: Decimal
    maximum_exposure: Decimal
    #: net_pnl / maximum_exposure. None if maximum_exposure is unknown or zero.
    return_per_max_exposure: float | None
    rank: int


def normalized_risk_ranking(entries: Sequence[StrategyRiskEntry]) -> list[RankedStrategy]:
    """Rank strategies by return earned per dollar of capital ever placed at risk.

    This exists so a strategy cannot win a leaderboard merely by risking its entire
    bankroll on one bet: two strategies with identical raw net P&L are told apart by how
    much capital each one exposed to get there. Entries with unknown/zero exposure sort
    last (their ratio is undefined, not zero — zero would rank them as "terrible" instead
    of "unmeasurable").
    """
    scored: list[tuple[StrategyRiskEntry, float | None]] = []
    for e in entries:
        ratio = float(e.net_pnl / e.maximum_exposure) if e.maximum_exposure > ZERO else None
        scored.append((e, ratio))
    scored.sort(key=lambda x: (x[1] is None, -(x[1] or 0.0)))
    return [
        RankedStrategy(
            strategy_id=e.strategy_id,
            net_pnl=e.net_pnl,
            maximum_exposure=e.maximum_exposure,
            return_per_max_exposure=ratio,
            rank=i + 1,
        )
        for i, (e, ratio) in enumerate(scored)
    ]


# ---------------------------------------------------------------------------
# Closing-price diagnostic (signal quality, independent of settlement P&L)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClosingPriceEntry:
    canonical_id: str
    #: Side bought. Predicted direction is "price rises" for YES, "price falls" for NO.
    side: Side
    entry_price: Decimal
    entry_time: datetime


@dataclass(frozen=True)
class ClosingPriceDiagnostic:
    """A DIAGNOSTIC of directional signal quality, not a P&L measurement.

    Realized P&L still comes only from settlement. A strategy can score well here and
    still lose money (bad sizing, fees, adverse settlement variance), or score poorly
    here and still make money (lucky settlements) — this number exists to separate
    "did the model read direction correctly" from "did the bet pay off."
    """

    label: str = (
        "DIAGNOSTIC ONLY: pre-close price movement in the predicted direction, "
        "independent of binary settlement outcome"
    )
    n: int = 0
    fraction_favorable: float | None = None
    average_favorable_move: Decimal | None = None


def closing_price_diagnostic(
    entries: Sequence[ClosingPriceEntry],
    subsequent_prices: dict[str, Decimal],
) -> ClosingPriceDiagnostic:
    moves: list[Decimal] = []
    for e in entries:
        later = subsequent_prices.get(e.canonical_id)
        if later is None:
            continue
        moves.append(_signed_move(e.entry_price, later, e.side))
    if not moves:
        return ClosingPriceDiagnostic(n=0, fraction_favorable=None, average_favorable_move=None)
    favorable = sum(1 for m in moves if m > ZERO)
    return ClosingPriceDiagnostic(
        n=len(moves),
        fraction_favorable=favorable / len(moves),
        average_favorable_move=sum(moves, ZERO) / Decimal(len(moves)),
    )
