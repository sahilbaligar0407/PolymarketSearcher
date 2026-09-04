"""Copy-trading signal layer: trader statistics, scoring, and discovery.

Everything below the module-level ``Protocol`` definitions is pure: it takes lists of
observed actions/positions and a clock reading, and returns numbers. No network I/O, no
``datetime.now()``. This is what makes ``test_trader_stats.py`` fast and deterministic.

Two hard rules the PRD is explicit about, both enforced here:

1. **Never rank by raw win rate alone.** A 90% win rate with asymmetric losses (nine
   small wins, one catastrophic loss) is unprofitable. ``score_trader`` does not even
   read ``resolved_win_rate`` - it scores on realized expectancy, and separately
   penalizes the "one giant trade explains all the P&L" and "the largest loss dwarfs
   the largest win" shapes that a naive win-rate ranking would reward. See
   ``test_trader_stats.py::test_asymmetric_losses_score_below_positive_expectancy``.
2. **Discovery never backtests.** ``TraderDiscoveryService`` only ever writes
   candidates with status ``DISCOVERED``; forward tracking of a wallet's performance
   starts at the moment of discovery. Nothing here computes stats over a wallet's
   pre-discovery history and calls it a track record for ranking purposes - the
   leaderboard-reported pnl figures are carried through as informational context
   (what Polymarket itself reports), never as a stat this module derives or scores.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from marketlab.core.events import TraderActionEvent
from marketlab.core.instruments import Category

ZERO = Decimal(0)

# ---------------------------------------------------------------------------
# Local data model for positions.
#
# There is no frozen `TraderPosition`/`TraderPositionEvent` type in `core/`; positions
# are a Polymarket Data-API-specific concept (open/closed exposure with running P&L),
# not a cross-venue event type. We define a minimal model here rather than in
# `adapters/polymarket_global/normalize.py`, since it belongs to the *signal* layer
# (stats computation), not the adapter's payload-shape normalization.
# ---------------------------------------------------------------------------


class TraderPosition(BaseModel):
    """A snapshot of one wallet's exposure in one market, from the Data API."""

    model_config = ConfigDict(frozen=True)

    condition_id: str
    canonical_id: str = ""
    title: str = ""
    category: Category = Category.OTHER
    size: Decimal = ZERO
    avg_price: Decimal = ZERO
    cur_price: Decimal = ZERO
    #: `initialValue` on the Data API - USD cost basis for this position.
    cost_basis: Decimal = ZERO
    #: `currentValue` on the Data API.
    current_value: Decimal = ZERO
    #: `cashPnl` on the Data API - total pnl (realized + unrealized) at snapshot time.
    cash_pnl: Decimal = ZERO
    #: `realizedPnl` on the Data API.
    realized_pnl: Decimal = ZERO
    #: True once the market has resolved and this position is settled/redeemable.
    resolved: bool = False
    #: When the position resolved, if known - required to place this position on a
    #: chronological P&L curve for drawdown/slope calculations.
    resolution_time: datetime | None = None
    #: First observed trade time in this market - required for holding-time calc.
    opened_time: datetime | None = None


def position_from_data_api(raw: Mapping[str, Any]) -> TraderPosition:
    """Parse one row of `GET /positions` (see `data_api.DataApiAdapter.get_positions`)."""

    def _dec(key: str) -> Decimal:
        value = raw.get(key)
        if value is None or value == "":
            return ZERO
        try:
            return Decimal(str(value))
        except Exception:
            return ZERO

    cur_price = _dec("curPrice")
    redeemable = bool(raw.get("redeemable", False))
    resolved = redeemable or cur_price in (Decimal(0), Decimal(1))
    return TraderPosition(
        condition_id=str(raw.get("conditionId", "")),
        canonical_id=f"poly:{raw.get('conditionId', '')}",
        title=str(raw.get("title", "")),
        size=_dec("size"),
        avg_price=_dec("avgPrice"),
        cur_price=cur_price,
        cost_basis=_dec("initialValue"),
        current_value=_dec("currentValue"),
        cash_pnl=_dec("cashPnl"),
        realized_pnl=_dec("realizedPnl"),
        resolved=resolved,
    )


# ---------------------------------------------------------------------------
# Stats + scoring
# ---------------------------------------------------------------------------


class TraderStats(BaseModel):
    """Everything the PRD asks the discovery/scoring pipeline to compute about one wallet.

    Split conceptually into two families, though both live on one model:

    * *Reported* (``discovery_date``, ``rank_at_discovery``, ``category``, ``day_pnl``,
      ``week_pnl``, ``month_pnl``, ``all_time_pnl``, ``reported_volume``) - what
      Polymarket's own leaderboard said about the wallet at the moment it was
      discovered. Informational only; never used to backtest the wallet's past
      (see module docstring).
    * *Observed* (everything else) - computed purely from actions/positions seen
      **after** discovery, via ``compute_trader_stats``.
    """

    model_config = ConfigDict(frozen=True)

    wallet: str = ""

    # --- reported at discovery (informational, not backtested) ---
    discovery_date: datetime
    rank_at_discovery: int
    category: Category = Category.OTHER
    day_pnl: Decimal | None = None
    week_pnl: Decimal | None = None
    month_pnl: Decimal | None = None
    all_time_pnl: Decimal | None = None
    reported_volume: Decimal | None = None

    # --- observed, forward from discovery ---
    number_of_markets: int = 0
    number_of_observed_trades: int = 0
    median_trade_size: Decimal = ZERO
    #: Herfindahl-Hirschman Index over per-market exposure share, in [1/N, 1].
    #: 1.0 means all exposure is in a single market; near 0 means highly diversified.
    position_concentration: float = 0.0
    #: Fraction of *resolved* positions with positive realized P&L. None if none
    #: have resolved yet. Deliberately not read by `score_trader` alone - see module
    #: docstring rule 1.
    resolved_win_rate: float | None = None
    realized_pnl: Decimal = ZERO
    #: realized_pnl / total cost basis, if cost basis is known and positive.
    estimated_roi: float | None = None
    largest_loss: Decimal = ZERO
    largest_win: Decimal = ZERO
    #: Magnitude (>= 0) of the largest peak-to-trough decline on the observed
    #: cumulative realized-P&L curve. 0 if fewer than two resolved, dated positions.
    max_observed_drawdown: Decimal = ZERO
    category_specialization: Category = Category.OTHER
    #: Median seconds between opening and resolving a position. None if unavailable.
    median_holding_time: float | None = None
    #: Total traded USD / average cost basis - how many times capital was "turned over".
    turnover: float = 0.0
    #: OLS slope of cumulative realized P&L vs. time over the most recent window.
    #: Negative means performance is currently declining.
    recent_performance_slope: float = 0.0


#: Only TRADE-type actions count as "observed trades"; reward/yield/rebate rows in the
#: Data API activity feed are not trader intent (see `data_api.iter_trader_actions`).
_TRADE_ACTION_TYPES = {"TRADE"}


def _is_trade_action(action: TraderActionEvent) -> bool:
    # `action.action` carries either the order side (BUY/SELL, once normalize_activity
    # has already filtered to TRADE rows) or the raw activity `type` when unfiltered.
    # Treat BUY/SELL as trades too so this function is robust to either upstream path.
    return action.action.upper() in _TRADE_ACTION_TYPES or action.action.upper() in {"BUY", "SELL"}


def _market_key(canonical_id: str, condition_id: str) -> str:
    return canonical_id or (f"poly:{condition_id}" if condition_id else "")


def _herfindahl_index(exposures: Mapping[str, Decimal]) -> float:
    total = sum((abs(v) for v in exposures.values()), ZERO)
    if total <= ZERO:
        return 0.0
    shares = [float(abs(v) / total) for v in exposures.values()]
    return sum(s * s for s in shares)


def _ols_slope(points: Sequence[tuple[float, float]]) -> float:
    """Slope of the least-squares line through ``points`` (x, y). 0.0 if < 2 points or
    all x are identical (no numpy/statsmodels dependency needed for one line fit)."""
    n = len(points)
    if n < 2:
        return 0.0
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in points)
    den = sum((x - mean_x) ** 2 for x, y in points)
    if den == 0:
        return 0.0
    return num / den


def _build_realized_pnl_series(positions: Sequence[TraderPosition]) -> list[tuple[datetime, Decimal]]:
    dated = [
        (p.resolution_time, p.realized_pnl)
        for p in positions
        if p.resolved and p.resolution_time is not None
    ]
    dated.sort(key=lambda t: t[0])
    series: list[tuple[datetime, Decimal]] = []
    running = ZERO
    for ts, pnl in dated:
        running += pnl
        series.append((ts, running))
    return series


def _max_drawdown(series: Sequence[tuple[datetime, Decimal]]) -> Decimal:
    if not series:
        return ZERO
    peak = series[0][1]
    worst = ZERO
    for _, value in series:
        peak = max(peak, value)
        worst = max(worst, peak - value)
    return worst


def compute_trader_stats(
    actions: Sequence[TraderActionEvent],
    positions: Sequence[TraderPosition],
    now: datetime,
    *,
    discovery_date: datetime | None = None,
    rank_at_discovery: int = 0,
    category: Category = Category.OTHER,
    day_pnl: Decimal | None = None,
    week_pnl: Decimal | None = None,
    month_pnl: Decimal | None = None,
    all_time_pnl: Decimal | None = None,
    reported_volume: Decimal | None = None,
    wallet: str = "",
    recent_window: int = 20,
) -> TraderStats:
    """Compute observed statistics for one wallet from actions/positions seen so far.

    ``now`` is the caller's injected ``Clock`` reading (never `datetime.now()`), used
    only as the fallback ``discovery_date`` when the caller does not supply one - it is
    not read for anything else, since every other timestamp-derived stat here is
    computed from the actions'/positions' own timestamps.
    """
    trades = [a for a in actions if _is_trade_action(a)]

    market_keys = {
        _market_key(a.canonical_id, a.poly_condition_id) for a in trades if a.canonical_id or a.poly_condition_id
    }
    market_keys |= {p.canonical_id or p.condition_id for p in positions if p.canonical_id or p.condition_id}
    market_keys.discard("")

    trade_sizes = [a.usd_size for a in trades if a.usd_size is not None]
    median_trade_size = (
        statistics.median(trade_sizes) if trade_sizes else ZERO  # statistics.median handles Decimal
    )

    exposures: dict[str, Decimal] = defaultdict(lambda: ZERO)
    if positions:
        for p in positions:
            key = p.canonical_id or p.condition_id
            if key:
                exposures[key] += abs(p.current_value if p.current_value else p.cost_basis)
    else:
        for a in trades:
            key = _market_key(a.canonical_id, a.poly_condition_id)
            if key and a.usd_size is not None:
                exposures[key] += abs(a.usd_size)
    position_concentration = _herfindahl_index(exposures)

    resolved = [p for p in positions if p.resolved]
    wins = sum(1 for p in resolved if p.realized_pnl > ZERO)
    resolved_win_rate = (wins / len(resolved)) if resolved else None

    realized_pnl = sum((p.realized_pnl for p in positions), ZERO)
    total_cost_basis = sum((p.cost_basis for p in positions), ZERO)
    estimated_roi = float(realized_pnl / total_cost_basis) if total_cost_basis > ZERO else None

    position_pnls = [p.cash_pnl for p in positions]
    largest_win = max(position_pnls) if position_pnls else ZERO
    largest_loss = min(position_pnls) if position_pnls else ZERO

    holding_times = [
        (p.resolution_time - p.opened_time).total_seconds()
        for p in positions
        if p.resolved and p.resolution_time is not None and p.opened_time is not None
    ]
    median_holding_time = statistics.median(holding_times) if holding_times else None

    total_traded_usd = sum((a.usd_size for a in trades if a.usd_size is not None), ZERO)
    avg_cost_basis = (total_cost_basis / len(positions)) if positions else ZERO
    turnover = float(total_traded_usd / avg_cost_basis) if avg_cost_basis > ZERO else 0.0

    pnl_series = _build_realized_pnl_series(positions)
    max_drawdown = _max_drawdown(pnl_series)
    recent_points = pnl_series[-recent_window:]
    epoch = recent_points[0][0] if recent_points else now
    slope_points = [((ts - epoch).total_seconds(), float(pnl)) for ts, pnl in recent_points]
    recent_performance_slope = _ols_slope(slope_points)

    specialization = classify_specialization(trades)

    return TraderStats(
        wallet=wallet,
        discovery_date=discovery_date or now,
        rank_at_discovery=rank_at_discovery,
        category=category,
        day_pnl=day_pnl,
        week_pnl=week_pnl,
        month_pnl=month_pnl,
        all_time_pnl=all_time_pnl,
        reported_volume=reported_volume,
        number_of_markets=len(market_keys),
        number_of_observed_trades=len(trades),
        median_trade_size=median_trade_size,
        position_concentration=position_concentration,
        resolved_win_rate=resolved_win_rate,
        realized_pnl=realized_pnl,
        estimated_roi=estimated_roi,
        largest_loss=largest_loss,
        largest_win=largest_win,
        max_observed_drawdown=max_drawdown,
        category_specialization=specialization,
        median_holding_time=median_holding_time,
        turnover=turnover,
        recent_performance_slope=recent_performance_slope,
    )


def classify_specialization(actions: Sequence[TraderActionEvent]) -> Category:
    """Dominant category by observed trade count. ``Category.OTHER`` if empty or tied
    with no clear plurality."""
    if not actions:
        return Category.OTHER
    counts: dict[Category, int] = defaultdict(int)
    for a in actions:
        counts[a.category] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Tunable thresholds, named and grouped here (rather than scattered magic numbers) so
# the penalty structure in `score_trader` is auditable at a glance.
MIN_TRADES_FOR_CONFIDENCE = 15
SHORT_HISTORY_PENALTY_PER_TRADE = 1.5
LUCKY_TRADE_SHARE_THRESHOLD = 0.5  # one position explaining > 50% of total realized pnl
LUCKY_TRADE_PENALTY_SCALE = 80.0
CONCENTRATION_THRESHOLD = 0.35  # HHI above this = meaningfully concentrated
CONCENTRATION_PENALTY_SCALE = 60.0
SLOPE_PENALTY_SCALE = 3.0
SLOPE_PENALTY_CAP = 25.0
DRAWDOWN_PENALTY_SCALE = 0.05  # per dollar of drawdown relative to |realized_pnl|
DRAWDOWN_PENALTY_CAP = 20.0
ASYMMETRY_THRESHOLD = 2.0  # largest loss more than 2x the largest win = asymmetric risk
ASYMMETRY_PENALTY_SCALE = 15.0
ASYMMETRY_PENALTY_CAP = 40.0
LOW_LIQUIDITY_PENALTY = 15.0
EXPECTANCY_SCALE = 40.0  # how many score points one full stake of expectancy is worth


class TraderScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    wallet: str = ""
    score: float
    penalties: dict[str, float] = Field(default_factory=dict)
    reasons: tuple[str, ...] = ()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def score_trader(
    stats: TraderStats,
    *,
    market_liquidity_usd: Decimal | None = None,
    min_liquidity_usd: Decimal = Decimal(1000),
) -> TraderScore:
    """Score a trader for copy-worthiness on a roughly 0-100 scale (unclamped at the
    low end - a sufficiently bad trader can and should score negative).

    Deliberately does **not** read ``stats.resolved_win_rate``: win rate alone rewards
    "many small wins, one account-destroying loss" shapes. Instead the base score comes
    from *expectancy* (realized P&L per trade, relative to typical stake), and the
    penalty structure explicitly targets the failure modes a win-rate-only ranking
    would miss: short history, one lucky trade carrying the whole track record,
    over-concentration in one market, a currently-collapsing P&L trend, a large
    peak-to-trough drawdown, loss/win asymmetry, and (optionally) thin liquidity in the
    markets actually traded.
    """
    penalties: dict[str, float] = {}
    reasons: list[str] = []
    score = 50.0  # neutral prior before any evidence

    n_trades = max(stats.number_of_observed_trades, 1)
    typical_stake = stats.median_trade_size if stats.median_trade_size > ZERO else Decimal(1)
    expectancy_per_trade = stats.realized_pnl / Decimal(n_trades)
    normalized_expectancy = float(expectancy_per_trade / typical_stake)
    expectancy_component = _clamp(normalized_expectancy * EXPECTANCY_SCALE, -EXPECTANCY_SCALE, EXPECTANCY_SCALE)
    score += expectancy_component
    penalties["expectancy_component"] = expectancy_component

    if stats.number_of_observed_trades < MIN_TRADES_FOR_CONFIDENCE:
        deficit = MIN_TRADES_FOR_CONFIDENCE - stats.number_of_observed_trades
        penalty = deficit * SHORT_HISTORY_PENALTY_PER_TRADE
        score -= penalty
        penalties["short_history"] = penalty
        reasons.append(f"short history: only {stats.number_of_observed_trades} observed trades")

    if stats.largest_win > ZERO and stats.realized_pnl > ZERO:
        share = float(stats.largest_win / stats.realized_pnl)
        if share > LUCKY_TRADE_SHARE_THRESHOLD:
            penalty = (share - LUCKY_TRADE_SHARE_THRESHOLD) * LUCKY_TRADE_PENALTY_SCALE
            score -= penalty
            penalties["one_lucky_trade"] = penalty
            reasons.append(f"one position is {share:.0%} of total realized pnl")

    if stats.position_concentration > CONCENTRATION_THRESHOLD:
        penalty = (stats.position_concentration - CONCENTRATION_THRESHOLD) * CONCENTRATION_PENALTY_SCALE
        score -= penalty
        penalties["concentration"] = penalty
        reasons.append(f"HHI {stats.position_concentration:.2f} exceeds concentration threshold")

    if stats.recent_performance_slope < 0:
        penalty = min(abs(stats.recent_performance_slope) * SLOPE_PENALTY_SCALE, SLOPE_PENALTY_CAP)
        score -= penalty
        penalties["recent_collapse"] = penalty
        reasons.append("recent cumulative pnl trend is negative")

    if stats.realized_pnl != ZERO and stats.max_observed_drawdown > ZERO:
        relative_drawdown = float(stats.max_observed_drawdown / abs(stats.realized_pnl))
        penalty = min(relative_drawdown * DRAWDOWN_PENALTY_SCALE * 100, DRAWDOWN_PENALTY_CAP)
        score -= penalty
        penalties["drawdown"] = penalty

    if stats.largest_loss < ZERO:
        loss_magnitude = abs(stats.largest_loss)
        win_reference = stats.largest_win if stats.largest_win > ZERO else typical_stake
        ratio = float(loss_magnitude / win_reference) if win_reference > ZERO else 0.0
        if ratio > ASYMMETRY_THRESHOLD:
            penalty = min((ratio - ASYMMETRY_THRESHOLD) * ASYMMETRY_PENALTY_SCALE, ASYMMETRY_PENALTY_CAP)
            score -= penalty
            penalties["loss_asymmetry"] = penalty
            reasons.append(
                f"largest loss is {ratio:.1f}x the largest win - asymmetric risk profile"
            )

    if market_liquidity_usd is not None and market_liquidity_usd < min_liquidity_usd:
        score -= LOW_LIQUIDITY_PENALTY
        penalties["low_liquidity"] = LOW_LIQUIDITY_PENALTY
        reasons.append(f"median market liquidity ${market_liquidity_usd} below ${min_liquidity_usd} floor")

    return TraderScore(wallet=stats.wallet, score=score, penalties=penalties, reasons=tuple(reasons))


# ---------------------------------------------------------------------------
# Discovery service
# ---------------------------------------------------------------------------

#: Status a freshly-discovered candidate is written with. This module never writes any
#: other status - promotion to an actually-copyable state is a separate, later decision
#: made elsewhere, after forward-tracked performance has been observed.
DISCOVERED_STATUS = "DISCOVERED"


@runtime_checkable
class LeaderboardRowLike(Protocol):
    """Structural subset of `leaderboard.LeaderboardRow` this module needs.

    Defined locally (rather than importing `adapters.polymarket_global.leaderboard`)
    so the discovery service has no hard import-time dependency on the adapters
    package - it only needs something shaped like a leaderboard row.
    """

    wallet: str
    rank: int
    category: Category
    period: str
    metric: str
    pnl: Decimal | None
    volume: Decimal | None


@runtime_checkable
class SnapshotSource(Protocol):
    async def snapshot_all(self) -> Sequence[LeaderboardRowLike]: ...


@runtime_checkable
class WalletHistorySource(Protocol):
    async def get_positions(self, wallet: str, **kwargs: Any) -> Sequence[Mapping[str, Any]]: ...

    def iter_trader_actions(self, wallet: str, **kwargs: Any) -> Any:
        """Async-iterable of `TraderActionEvent` for `wallet` (see `DataApiAdapter`)."""
        ...


@runtime_checkable
class ClockLike(Protocol):
    def now(self) -> datetime: ...


@runtime_checkable
class TraderStoreProtocol(Protocol):
    """Duck-typed subset of `marketlab.storage.state.StateStore` this service needs.

    Defined locally so `copy_trader.py` does not import the (separately owned)
    storage module - callers pass in whatever object satisfies this shape.
    """

    def upsert_trader(self, wallet: str, stats: TraderStats, status: str) -> None: ...

    def save_leaderboard_snapshot(self, rows: Sequence[LeaderboardRowLike]) -> None: ...

    def save_trader_action(self, action: TraderActionEvent) -> None: ...


class TraderCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    wallet: str
    stats: TraderStats
    status: str = DISCOVERED_STATUS


class TraderDiscoveryService:
    """Turns a leaderboard snapshot into `DISCOVERED` trader candidates.

    Never marks anyone as copyable - see module docstring rule 2. Forward tracking
    (subsequent `iter_trader_actions` calls feeding `save_trader_action`, and later
    re-runs of `compute_trader_stats` over the growing action history) is what
    eventually justifies promoting a candidate; that promotion decision lives outside
    this service entirely.
    """

    def __init__(
        self,
        leaderboard: SnapshotSource,
        data_api: WalletHistorySource,
        clock: ClockLike,
        store: TraderStoreProtocol,
        *,
        max_wallets: int | None = None,
    ) -> None:
        self._leaderboard = leaderboard
        self._data_api = data_api
        self._clock = clock
        self._store = store
        self._max_wallets = max_wallets

    async def discover(self) -> list[TraderCandidate]:
        rows = list(await self._leaderboard.snapshot_all())
        self._store.save_leaderboard_snapshot(rows)

        best_by_wallet: dict[str, LeaderboardRowLike] = {}
        for row in rows:
            if not row.wallet:
                continue
            current = best_by_wallet.get(row.wallet)
            if current is None or row.rank < current.rank:
                best_by_wallet[row.wallet] = row

        wallets = list(best_by_wallet)
        if self._max_wallets is not None:
            wallets = wallets[: self._max_wallets]

        candidates: list[TraderCandidate] = []
        now = self._clock.now()
        for wallet in wallets:
            row = best_by_wallet[wallet]
            positions_raw = await self._data_api.get_positions(wallet)
            positions = [position_from_data_api(p) for p in positions_raw]

            actions: list[TraderActionEvent] = []
            async for action in self._data_api.iter_trader_actions(wallet):
                actions.append(action)
                self._store.save_trader_action(action)

            period_pnls = {r.period: r.pnl for r in rows if r.wallet == wallet}
            stats = compute_trader_stats(
                actions,
                positions,
                now,
                discovery_date=now,
                rank_at_discovery=row.rank,
                category=row.category,
                day_pnl=period_pnls.get("day"),
                week_pnl=period_pnls.get("week"),
                month_pnl=period_pnls.get("month"),
                all_time_pnl=period_pnls.get("all", row.pnl),
                reported_volume=row.volume,
                wallet=wallet,
            )
            self._store.upsert_trader(wallet, stats, DISCOVERED_STATUS)
            candidates.append(TraderCandidate(wallet=wallet, stats=stats, status=DISCOVERED_STATUS))

        return candidates
