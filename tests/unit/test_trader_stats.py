from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from marketlab.core.events import TraderActionEvent
from marketlab.core.instruments import Category
from marketlab.signals.copy_trader import (
    TraderPosition,
    TraderStats,
    _herfindahl_index,
    _ols_slope,
    classify_specialization,
    compute_trader_stats,
    position_from_data_api,
    score_trader,
)

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _stats(**overrides: object) -> TraderStats:
    base: dict = dict(
        discovery_date=EPOCH,
        rank_at_discovery=10,
        number_of_observed_trades=30,
        median_trade_size=Decimal(50),
        position_concentration=0.1,
        realized_pnl=Decimal(0),
        largest_win=Decimal(0),
        largest_loss=Decimal(0),
        max_observed_drawdown=Decimal(0),
        recent_performance_slope=0.0,
    )
    base.update(overrides)
    return TraderStats(**base)


# ---------------------------------------------------------------------------
# The headline requirement: never rank by raw win rate alone.
# ---------------------------------------------------------------------------


def test_asymmetric_losses_score_below_positive_expectancy() -> None:
    """A 90%-win-rate trader whose one loss dwarfs all nine wins put together (net
    losing) must score below a 55%-win-rate trader with smaller, positive-expectancy
    losses (net winning) - the canonical case a naive win-rate ranking gets backwards.
    """
    lucky_but_toxic = _stats(
        realized_pnl=Decimal("-410"),  # 9 * (+10) + 1 * (-500)
        largest_win=Decimal("10"),
        largest_loss=Decimal("-500"),
        resolved_win_rate=0.90,
        median_trade_size=Decimal(10),
    )
    steady_positive = _stats(
        realized_pnl=Decimal("280"),  # 11 * (+50) + 9 * (-30) roughly
        largest_win=Decimal("50"),
        largest_loss=Decimal("-30"),
        resolved_win_rate=0.55,
        median_trade_size=Decimal(30),
    )

    score_bad = score_trader(lucky_but_toxic).score
    score_good = score_trader(steady_positive).score

    assert score_bad < score_good
    assert score_bad < 50.0  # net losing trader should score below the neutral prior
    assert score_good > 50.0  # net winning trader should score above it


def test_never_uses_win_rate_directly() -> None:
    """Two otherwise-identical stats differing only in resolved_win_rate must score
    identically - proof score_trader is not a function of win rate at all."""
    high_win_rate = _stats(realized_pnl=Decimal(100), largest_win=Decimal(20), resolved_win_rate=0.9)
    low_win_rate = _stats(realized_pnl=Decimal(100), largest_win=Decimal(20), resolved_win_rate=0.1)
    assert score_trader(high_win_rate).score == score_trader(low_win_rate).score


# ---------------------------------------------------------------------------
# Individual penalties
# ---------------------------------------------------------------------------


def test_concentration_penalty_applies_above_threshold() -> None:
    diversified = _stats(position_concentration=0.1)
    concentrated = _stats(position_concentration=0.9)
    assert score_trader(concentrated).score < score_trader(diversified).score
    assert "concentration" in score_trader(concentrated).penalties


def test_one_lucky_trade_penalty() -> None:
    diversified = _stats(realized_pnl=Decimal(100), largest_win=Decimal(20))
    one_lucky_trade = _stats(realized_pnl=Decimal(100), largest_win=Decimal(90))
    assert score_trader(one_lucky_trade).score < score_trader(diversified).score
    assert "one_lucky_trade" in score_trader(one_lucky_trade).penalties
    assert "one_lucky_trade" not in score_trader(diversified).penalties


def test_short_history_penalty() -> None:
    short = _stats(number_of_observed_trades=2)
    long = _stats(number_of_observed_trades=50)
    assert score_trader(short).score < score_trader(long).score
    assert "short_history" in score_trader(short).penalties


def test_recent_collapse_penalty() -> None:
    declining = _stats(recent_performance_slope=-5.0)
    flat = _stats(recent_performance_slope=0.0)
    assert score_trader(declining).score < score_trader(flat).score
    assert "recent_collapse" in score_trader(declining).penalties


def test_low_liquidity_penalty_is_optional_context() -> None:
    stats = _stats()
    without_liquidity_context = score_trader(stats)
    with_thin_liquidity = score_trader(stats, market_liquidity_usd=Decimal(10))
    assert with_thin_liquidity.score < without_liquidity_context.score
    assert "low_liquidity" in with_thin_liquidity.penalties


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def test_herfindahl_index_bounds() -> None:
    assert _herfindahl_index({"a": Decimal(100)}) == pytest.approx(1.0)
    assert _herfindahl_index({"a": Decimal(50), "b": Decimal(50)}) == pytest.approx(0.5)
    assert _herfindahl_index({}) == 0.0


def test_ols_slope_declining_series_is_negative() -> None:
    points = [(0.0, 100.0), (1.0, 80.0), (2.0, 50.0), (3.0, 10.0)]
    assert _ols_slope(points) < 0


def test_ols_slope_rising_series_is_positive() -> None:
    points = [(0.0, 10.0), (1.0, 50.0), (2.0, 90.0)]
    assert _ols_slope(points) > 0


def test_ols_slope_needs_at_least_two_points() -> None:
    assert _ols_slope([]) == 0.0
    assert _ols_slope([(0.0, 5.0)]) == 0.0


# ---------------------------------------------------------------------------
# classify_specialization
# ---------------------------------------------------------------------------


def _action(category: Category, wallet: str = "0x1") -> TraderActionEvent:
    return TraderActionEvent(
        event_time=EPOCH, first_seen_time=EPOCH, wallet=wallet, category=category, action="BUY"
    )


def test_classify_specialization_majority_category() -> None:
    actions = [_action(Category.SPORTS), _action(Category.SPORTS), _action(Category.CRYPTO)]
    assert classify_specialization(actions) == Category.SPORTS


def test_classify_specialization_empty() -> None:
    assert classify_specialization([]) == Category.OTHER


# ---------------------------------------------------------------------------
# compute_trader_stats
# ---------------------------------------------------------------------------


def _trade_action(
    now: datetime,
    *,
    condition_id: str = "0xcond",
    usd_size: Decimal = Decimal(50),
    category: Category = Category.SPORTS,
) -> TraderActionEvent:
    return TraderActionEvent(
        event_time=now,
        first_seen_time=now,
        wallet="0x1",
        poly_condition_id=condition_id,
        canonical_id=f"poly:{condition_id}",
        action="BUY",
        usd_size=usd_size,
        category=category,
    )


def test_compute_trader_stats_median_trade_size_and_market_count() -> None:
    actions = [
        _trade_action(EPOCH, condition_id="0xa", usd_size=Decimal(10)),
        _trade_action(EPOCH, condition_id="0xa", usd_size=Decimal(20)),
        _trade_action(EPOCH, condition_id="0xb", usd_size=Decimal(30)),
    ]
    stats = compute_trader_stats(actions, [], EPOCH)
    assert stats.number_of_observed_trades == 3
    assert stats.number_of_markets == 2
    assert stats.median_trade_size == Decimal(20)


def test_compute_trader_stats_position_concentration_from_positions() -> None:
    positions = [
        TraderPosition(condition_id="0xa", canonical_id="poly:0xa", current_value=Decimal(100)),
        TraderPosition(condition_id="0xb", canonical_id="poly:0xb", current_value=Decimal(100)),
    ]
    stats = compute_trader_stats([], positions, EPOCH)
    assert stats.position_concentration == pytest.approx(0.5)


def test_compute_trader_stats_median_holding_time_and_declining_slope() -> None:
    positions = [
        TraderPosition(
            condition_id="0xa", canonical_id="poly:0xa", realized_pnl=Decimal(100),
            resolved=True, opened_time=EPOCH, resolution_time=EPOCH + timedelta(hours=1),
        ),
        TraderPosition(
            condition_id="0xb", canonical_id="poly:0xb", realized_pnl=Decimal(-50),
            resolved=True, opened_time=EPOCH, resolution_time=EPOCH + timedelta(hours=3),
        ),
        TraderPosition(
            condition_id="0xc", canonical_id="poly:0xc", realized_pnl=Decimal(-80),
            resolved=True, opened_time=EPOCH, resolution_time=EPOCH + timedelta(hours=5),
        ),
    ]
    stats = compute_trader_stats([], positions, EPOCH)
    assert stats.median_holding_time == timedelta(hours=3).total_seconds()
    # Cumulative realized pnl goes 100 -> 50 -> -30: a declining series.
    assert stats.recent_performance_slope < 0
    assert stats.max_observed_drawdown == Decimal(130)  # peak 100 -> trough -30


def test_compute_trader_stats_realized_pnl_and_roi() -> None:
    positions = [
        TraderPosition(
            condition_id="0xa", canonical_id="poly:0xa", realized_pnl=Decimal(50),
            cost_basis=Decimal(100), resolved=True,
        ),
        TraderPosition(
            condition_id="0xb", canonical_id="poly:0xb", realized_pnl=Decimal(-10),
            cost_basis=Decimal(100), resolved=True,
        ),
    ]
    stats = compute_trader_stats([], positions, EPOCH)
    assert stats.realized_pnl == Decimal(40)
    assert stats.estimated_roi == pytest.approx(40 / 200)
    assert stats.resolved_win_rate == pytest.approx(0.5)


def test_compute_trader_stats_discovery_never_backtests_reported_fields() -> None:
    """Reported (leaderboard) fields pass through as informational context only - they
    are not derived from, or blended with, the observed actions/positions."""
    stats = compute_trader_stats(
        [], [], EPOCH,
        discovery_date=EPOCH,
        rank_at_discovery=3,
        all_time_pnl=Decimal("999999"),
    )
    assert stats.rank_at_discovery == 3
    assert stats.all_time_pnl == Decimal("999999")
    # No observed history yet -> all observed stats are at their empty defaults.
    assert stats.number_of_observed_trades == 0
    assert stats.realized_pnl == Decimal(0)


# ---------------------------------------------------------------------------
# position_from_data_api
# ---------------------------------------------------------------------------


def test_position_from_data_api_parses_resolved_flag() -> None:
    raw = {
        "conditionId": "0xa",
        "curPrice": 1.0,
        "redeemable": True,
        "realizedPnl": 5,
        "initialValue": 10,
    }
    pos = position_from_data_api(raw)
    assert pos.resolved is True
    assert pos.realized_pnl == Decimal("5")
    assert pos.cost_basis == Decimal("10")


def test_position_from_data_api_unresolved_open_position() -> None:
    raw = {"conditionId": "0xb", "curPrice": 0.42, "redeemable": False, "realizedPnl": 0}
    pos = position_from_data_api(raw)
    assert pos.resolved is False
