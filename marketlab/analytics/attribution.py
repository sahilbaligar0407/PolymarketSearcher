"""Where the money came from, and where it went.

Every function here is a decomposition: the pieces must sum back to the total they came
from, and :mod:`tests.unit.test_metrics` asserts that for :func:`pnl_attribution`
directly. A decomposition that doesn't reconcile is worse than no decomposition — it
invites the reader to trust a number that is quietly wrong.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from marketlab.core.instruments import ZERO, Side

# ---------------------------------------------------------------------------
# pnl_attribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FillAttributionInput:
    """One fill plus the context needed to attribute its contribution to net P&L."""

    canonical_id: str
    side: Side
    action: str  # "buy" / "sell", matches marketlab.core.orders.Action.value
    price: Decimal
    quantity: int
    fee: Decimal
    #: Mid/mark price at decision time -- the reference the signal component is measured
    #: against ("did price move our way after we decided to trade").
    reference_price: Decimal | None = None
    #: Mid/mark price shortly after the fill, for adverse-selection detection.
    post_fill_price: Decimal | None = None


@dataclass(frozen=True)
class SettlementAttributionInput:
    canonical_id: str
    side: Side
    quantity: int
    average_price: Decimal
    won: bool | None  # None = voided


@dataclass(frozen=True)
class PnlAttribution:
    net_pnl: Decimal
    #: Price moved in our favor between decision and fill (captured directional edge).
    signal: Decimal
    fees: Decimal
    #: Fill price vs reference/decision price -- cost of not getting the reference price.
    slippage: Decimal
    #: Price moved against a resting fill in the moments after it (adverse selection).
    adverse_selection: Decimal
    #: Whatever is left once every other term is subtracted -- realized settlement
    #: variance the other terms cannot explain (binary outcome noise around the model's
    #: assessed probability).
    settlement_variance: Decimal

    def reconstructed_total(self) -> Decimal:
        return (
            self.signal
            - self.fees
            - self.slippage
            - self.adverse_selection
            + self.settlement_variance
        )


def _cost_per_share(action: str, reference: Decimal, actual: Decimal) -> Decimal:
    """Slippage cost of transacting at ``actual`` instead of the ``reference`` price we
    expected to pay/receive. Positive = worse than reference. Depends only on buy/sell
    direction: a fill's price is already denominated in the traded side's own
    probability terms (see marketlab.core.instruments.OrderBook convention), so YES/NO
    does not change the sign here -- only whether we were paying (buy) or receiving
    (sell) does."""
    return (actual - reference) if action == "buy" else (reference - actual)


def _adverse_move_cost(action: str, fill_price: Decimal, post_fill_price: Decimal) -> Decimal:
    """Cost of the market continuing to move against the side we just took. Positive =
    bad. A buyer is hurt when price falls after the fill; a seller is hurt when price
    rises after the fill -- the opposite direction from slippage, which compares the fill
    to what we expected *before* trading, not to what happened *after*."""
    return (fill_price - post_fill_price) if action == "buy" else (post_fill_price - fill_price)


def pnl_attribution(
    fills: Sequence[FillAttributionInput],
    settlements: Sequence[SettlementAttributionInput],
) -> PnlAttribution:
    """Decompose net P&L into signal, fees, slippage, adverse selection and settlement
    variance. The components are constructed to sum exactly back to ``net_pnl`` by
    construction: ``settlement_variance`` absorbs whatever the other four terms do not
    explain, rather than being an independently measured fifth quantity.
    """
    fees = sum((f.fee for f in fills), ZERO)

    slippage = ZERO
    adverse_selection = ZERO
    for f in fills:
        qty = Decimal(f.quantity)
        if f.reference_price is not None:
            # Slippage: cost of the fill price being worse than the reference/decision price.
            slippage += _cost_per_share(f.action, f.reference_price, f.price) * qty
        if f.post_fill_price is not None:
            # Adverse selection: the price kept moving against the side we just took.
            move_cost = _adverse_move_cost(f.action, f.price, f.post_fill_price)
            adverse_selection += max(ZERO, move_cost) * qty

    # Signal: net directional P&L on settled positions, valued at entry vs settlement,
    # BEFORE fees/slippage/adverse-selection are peeled off (those are already-counted
    # costs of getting into the position, not part of "did the model call it right").
    realized_from_settlement = ZERO
    for s in settlements:
        payout = Decimal(1) if s.won else ZERO if s.won is False else s.average_price
        realized_from_settlement += (payout - s.average_price) * Decimal(s.quantity)

    net_pnl = realized_from_settlement - fees

    # signal is defined as the raw directional P&L before fees/slippage/adverse-selection.
    signal = realized_from_settlement

    # settlement_variance absorbs the reconciliation gap so the identity holds exactly:
    # net_pnl = signal - fees - slippage - adverse_selection + settlement_variance
    settlement_variance = net_pnl - (signal - fees - slippage - adverse_selection)

    return PnlAttribution(
        net_pnl=net_pnl,
        signal=signal,
        fees=fees,
        slippage=slippage,
        adverse_selection=adverse_selection,
        settlement_variance=settlement_variance,
    )


# ---------------------------------------------------------------------------
# attribution_by(dimension)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttributedRecord:
    """One resolved trade tagged with every dimension it can be grouped by."""

    trade_id: str
    pnl: Decimal
    category: str = ""
    universe: str = ""
    venue: str = ""
    strategy_id: str = ""
    hour_of_day: int | None = None
    liquidity_bucket: str = ""
    holding_period_bucket: str = ""


@dataclass(frozen=True)
class GroupAttribution:
    key: str
    n_trades: int
    net_pnl: Decimal
    average_pnl: Decimal


_DIMENSION_GETTERS: dict[str, Callable[[AttributedRecord], str]] = {
    "category": lambda r: r.category,
    "universe": lambda r: r.universe,
    "venue": lambda r: r.venue,
    "strategy": lambda r: r.strategy_id,
    "time_of_day": lambda r: "" if r.hour_of_day is None else f"{r.hour_of_day:02d}:00",
    "liquidity_bucket": lambda r: r.liquidity_bucket,
    "holding_period_bucket": lambda r: r.holding_period_bucket,
}


def attribution_by(dimension: str, records: Sequence[AttributedRecord]) -> list[GroupAttribution]:
    """Group net P&L by ``dimension`` (one of ``_DIMENSION_GETTERS``).

    Sorted by net P&L descending so the best- and worst-performing groups are at the
    ends, ready for direct rendering.
    """
    if dimension not in _DIMENSION_GETTERS:
        raise ValueError(f"unknown attribution dimension: {dimension!r}, choose one of {sorted(_DIMENSION_GETTERS)}")
    getter = _DIMENSION_GETTERS[dimension]
    groups: dict[str, list[Decimal]] = defaultdict(list)
    for r in records:
        groups[getter(r)].append(r.pnl)
    out = [
        GroupAttribution(
            key=key or "(unknown)",
            n_trades=len(pnls),
            net_pnl=sum(pnls, ZERO),
            average_pnl=sum(pnls, ZERO) / Decimal(len(pnls)),
        )
        for key, pnls in groups.items()
    ]
    out.sort(key=lambda g: g.net_pnl, reverse=True)
    return out


# ---------------------------------------------------------------------------
# latency_sensitivity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencySensitivity:
    #: latency_ms -> net P&L (or return) observed at that arm.
    by_latency_ms: dict[int, Decimal]
    #: Interpolated latency (ms) at which P&L crosses zero, None if it never does within
    #: the tested range (either always positive or always negative).
    breakeven_latency_ms: float | None
    #: True once P&L at the *fastest tested* latency is materially better than at the
    #: slowest, AND the strategy goes non-positive somewhere in the tested range -- a
    #: strategy whose edge is gone past ~500ms is not viable outside a colocated setup.
    laptop_infeasible: bool


def latency_sensitivity(results_by_latency_arm: dict[int, Decimal]) -> LatencySensitivity:
    """How fast P&L decays as simulated latency rises, and the break-even latency.

    ``results_by_latency_arm`` maps a latency arm in milliseconds (see
    ``configs/strategies.yaml: latency_arms_ms``) to that arm's net P&L. A strategy whose
    edge vanishes past 500ms is labeled laptop-infeasible: it cannot be run competitively
    without colocated/low-latency infrastructure this deployment does not have.
    """
    if not results_by_latency_arm:
        return LatencySensitivity(by_latency_ms={}, breakeven_latency_ms=None, laptop_infeasible=False)

    ordered = sorted(results_by_latency_arm.items())
    breakeven: float | None = None
    for (lat_a, pnl_a), (lat_b, pnl_b) in zip(ordered, ordered[1:], strict=False):
        if (pnl_a > ZERO) != (pnl_b > ZERO):
            # linear interpolation for the zero-crossing
            span = float(pnl_a - pnl_b)
            if span != 0:
                frac = float(pnl_a) / span
                breakeven = lat_a + frac * (lat_b - lat_a)
            break

    laptop_infeasible = breakeven is not None and breakeven <= 500.0

    return LatencySensitivity(
        by_latency_ms=dict(ordered),
        breakeven_latency_ms=breakeven,
        laptop_infeasible=laptop_infeasible,
    )


# ---------------------------------------------------------------------------
# overfit_score
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverfitScore:
    #: How many variants were tried, recorded so a winner from 200 grids reads honestly.
    n_variants: int
    best_result: float
    median_result: float
    #: (best - median) / abs(median or best), scaled by log(n_variants). Larger = more
    #: suspicious that the "winner" is a product of searching, not of real edge.
    score: float | None


def overfit_score(variant_results: Sequence[float]) -> OverfitScore:
    """A simple, honest overfitting heuristic.

    Not a p-value or a formal multiple-comparisons correction -- just a scaled gap between
    the best variant and the pack, so "we searched 200 grids and one came back +40%" reads
    as a warning rather than a discovery. ``score`` grows with both the gap and the number
    of variants searched.
    """
    if not variant_results:
        return OverfitScore(n_variants=0, best_result=0.0, median_result=0.0, score=None)
    n = len(variant_results)
    best = max(variant_results)
    median = statistics.median(variant_results)
    denom = abs(median) if median != 0 else (abs(best) if best != 0 else 1.0)
    gap = (best - median) / denom
    scale = np.log(n + 1)
    score = float(gap * scale) if n > 1 else 0.0
    return OverfitScore(n_variants=n, best_result=best, median_result=median, score=score)


# ---------------------------------------------------------------------------
# correlation_matrix / cluster_strategies
# ---------------------------------------------------------------------------


def correlation_matrix(strategy_returns: dict[str, Sequence[float]]) -> tuple[list[str], np.ndarray]:
    """Pairwise Pearson correlation of per-period returns across strategies.

    Series of different lengths are truncated to the shortest common length (aligned from
    the start) so the matrix is always well-defined; callers who need proper time
    alignment should pre-align their series before calling this.
    """
    names = list(strategy_returns)
    if not names:
        return [], np.zeros((0, 0))
    min_len = min(len(v) for v in strategy_returns.values())
    if min_len < 2:
        n = len(names)
        return names, np.full((n, n), np.nan)
    matrix = np.array([list(strategy_returns[name])[:min_len] for name in names], dtype=float)
    return names, np.corrcoef(matrix)


def cluster_strategies(
    strategy_returns: dict[str, Sequence[float]], threshold: float = 0.8
) -> list[list[str]]:
    """Group strategies whose returns are highly correlated (likely "the same trade").

    Simple connected-components clustering on the correlation matrix thresholded at
    ``threshold`` -- deliberately not a fitted clustering model, since the ensemble's
    correlation cap only needs "are these two effectively duplicates," not a taxonomy.
    """
    names, corr = correlation_matrix(strategy_returns)
    n = len(names)
    if n == 0:
        return []
    visited = [False] * n
    clusters: list[list[str]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        component = [i]
        while stack:
            cur = stack.pop()
            for j in range(n):
                if not visited[j] and j != cur and not np.isnan(corr[cur, j]) and corr[cur, j] >= threshold:
                    visited[j] = True
                    component.append(j)
                    stack.append(j)
        clusters.append(sorted(names[k] for k in component))
    return clusters
