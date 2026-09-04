"""Tests for marketlab.analytics.metrics (and the pnl_attribution reconciliation)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from marketlab.analytics.attribution import (
    FillAttributionInput,
    SettlementAttributionInput,
    pnl_attribution,
)
from marketlab.analytics.metrics import (
    StrategyRiskEntry,
    TradeRecord,
    average_loser,
    average_winner,
    compute_trading_metrics,
    expectancy_per_trade,
    max_drawdown_from_curve,
    normalized_risk_ranking,
    profit_factor,
    win_rate,
)
from marketlab.core.instruments import Side

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _at(hours: int) -> datetime:
    return T0 + timedelta(hours=hours)


# ---------------------------------------------------------------------------
# hand-computed profit factor / expectancy / win rate / avg winner-loser
# ---------------------------------------------------------------------------


def test_win_rate_and_average_winner_loser_hand_computed():
    pnls = [Decimal("10"), Decimal("-5"), Decimal("20"), Decimal("-15"), Decimal("30")]
    assert win_rate(pnls) == Decimal("0.6")
    assert average_winner(pnls) == Decimal("20")  # (10+20+30)/3
    assert average_loser(pnls) == Decimal("-10")  # (-5-15)/2
    assert expectancy_per_trade(pnls) == Decimal("8")  # 40/5
    assert profit_factor(pnls) == Decimal("3")  # 60/20


def test_profit_factor_hand_computed_simple():
    # gross win 100, gross loss 40 -> profit factor 2.5
    pnls = [Decimal("60"), Decimal("40"), Decimal("-40")]
    assert profit_factor(pnls) == Decimal("2.5")


# ---------------------------------------------------------------------------
# the 90%-win-rate-one-catastrophic-loss case
# ---------------------------------------------------------------------------


def test_high_win_rate_with_catastrophic_loss_shows_profit_factor_below_one():
    pnls = [Decimal("1")] * 9 + [Decimal("-50")]
    assert win_rate(pnls) == Decimal("0.9")
    pf = profit_factor(pnls)
    assert pf is not None
    assert pf < Decimal("1")
    assert pf == Decimal("9") / Decimal("50")


# ---------------------------------------------------------------------------
# max drawdown + duration on a known equity curve
# ---------------------------------------------------------------------------


def test_max_drawdown_and_duration_hand_computed():
    curve = [
        (_at(0), Decimal("100")),
        (_at(1), Decimal("120")),  # peak
        (_at(2), Decimal("90")),  # trough: dd = (120-90)/120 = 0.25
        (_at(3), Decimal("110")),  # still under water
        (_at(4), Decimal("130")),  # recovers past the 120 peak
    ]
    result = max_drawdown_from_curve(curve)
    assert result.max_drawdown == Decimal("0.25")
    assert result.recovered is True
    # peak at hour 1, recovery at hour 4 -> 3 hours = 10800 seconds
    assert result.duration_seconds == 3 * 3600


def test_max_drawdown_ongoing_never_recovers():
    curve = [
        (_at(0), Decimal("100")),
        (_at(1), Decimal("120")),
        (_at(2), Decimal("60")),  # dd = 0.5, never recovers
    ]
    result = max_drawdown_from_curve(curve)
    assert result.max_drawdown == Decimal("0.5")
    assert result.recovered is False
    assert result.duration_seconds == 3600  # peak (hour 1) to last point (hour 2)


# ---------------------------------------------------------------------------
# empty inputs -> None, never a fabricated 0.0
# ---------------------------------------------------------------------------


def test_empty_inputs_return_none_not_zero():
    assert win_rate([]) is None
    assert average_winner([]) is None
    assert average_loser([]) is None
    assert expectancy_per_trade([]) is None
    assert profit_factor([]) is None
    dd = max_drawdown_from_curve([])
    assert dd.max_drawdown is None
    assert dd.duration_seconds is None

    metrics = compute_trading_metrics(
        starting_bankroll=Decimal("50"),
        ending_bankroll=Decimal("50"),
        realized_pnl=Decimal("0"),
        unrealized_pnl=Decimal("0"),
        trades=(),
        orders=(),
        fills=(),
        equity_curve=(),
        exposure_curve=(),
    )
    assert metrics.n_trades == 0
    assert metrics.win_rate is None
    assert metrics.profit_factor is None
    assert metrics.max_drawdown is None
    assert metrics.gross_pnl is None
    assert metrics.turnover is None
    assert metrics.fill_ratio is None
    assert metrics.maker_fill_ratio is None
    assert metrics.cancel_ratio is None
    # These are real counts over an empty list -- 0 is the honest answer, not a fabrication.
    assert metrics.rejected_order_count == 0


def test_profit_factor_none_when_no_losers():
    # An all-winners record has no denominator: undefined, not "infinite".
    assert profit_factor([Decimal("5"), Decimal("10")]) is None


# ---------------------------------------------------------------------------
# normalized-risk ranking demotes an all-in strategy vs. a steadier one with the
# same raw return
# ---------------------------------------------------------------------------


def test_normalized_risk_ranking_demotes_all_in_strategy():
    entries = [
        StrategyRiskEntry(
            strategy_id="ALL_IN", net_pnl=Decimal("10"), maximum_exposure=Decimal("50")
        ),
        StrategyRiskEntry(
            strategy_id="STEADY", net_pnl=Decimal("10"), maximum_exposure=Decimal("10")
        ),
    ]
    ranked = normalized_risk_ranking(entries)
    assert [r.strategy_id for r in ranked] == ["STEADY", "ALL_IN"]
    assert ranked[0].rank == 1
    assert ranked[0].return_per_max_exposure == 1.0
    assert ranked[1].return_per_max_exposure == 0.2


def test_normalized_risk_ranking_unknown_exposure_sorts_last():
    entries = [
        StrategyRiskEntry(strategy_id="KNOWN", net_pnl=Decimal("1"), maximum_exposure=Decimal("10")),
        StrategyRiskEntry(strategy_id="UNKNOWN", net_pnl=Decimal("1"), maximum_exposure=Decimal("0")),
    ]
    ranked = normalized_risk_ranking(entries)
    assert ranked[0].strategy_id == "KNOWN"
    assert ranked[1].strategy_id == "UNKNOWN"
    assert ranked[1].return_per_max_exposure is None


# ---------------------------------------------------------------------------
# compute_trading_metrics wiring, using TradeRecord
# ---------------------------------------------------------------------------


def test_compute_trading_metrics_wires_trade_records():
    trades = [
        TradeRecord(trade_id="t1", pnl=Decimal("10"), fee=Decimal("0.10"), is_resolved=True),
        TradeRecord(trade_id="t2", pnl=Decimal("-4"), fee=Decimal("0.10"), is_resolved=True),
        TradeRecord(trade_id="t3", pnl=Decimal("6"), fee=Decimal("0.10"), is_resolved=True),
    ]
    metrics = compute_trading_metrics(
        starting_bankroll=Decimal("50"),
        ending_bankroll=Decimal("62"),
        realized_pnl=Decimal("12"),
        unrealized_pnl=Decimal("0"),
        trades=trades,
    )
    assert metrics.n_trades == 3
    assert metrics.n_resolved_trades == 3
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 1
    assert metrics.net_pnl == Decimal("12")
    assert metrics.fees == Decimal("0.30")
    # gross_pnl adds fees (and unknown slippage, which stays 0) back onto net_pnl
    assert metrics.gross_pnl == Decimal("12.30")


# ---------------------------------------------------------------------------
# pnl_attribution: components must sum back to net P&L
# ---------------------------------------------------------------------------


def test_pnl_attribution_components_sum_to_net_pnl():
    fills = [
        FillAttributionInput(
            canonical_id="M1",
            side=Side.YES,
            action="buy",
            price=Decimal("0.40"),
            quantity=10,
            fee=Decimal("0.10"),
            reference_price=Decimal("0.38"),
            post_fill_price=None,
        )
    ]
    settlements = [
        SettlementAttributionInput(
            canonical_id="M1",
            side=Side.YES,
            quantity=10,
            average_price=Decimal("0.40"),
            won=True,
        )
    ]
    result = pnl_attribution(fills, settlements)

    assert result.net_pnl == Decimal("5.90")  # (1-0.40)*10 - 0.10 fee
    assert result.fees == Decimal("0.10")
    assert result.slippage == Decimal("0.20")  # paid 0.02/share more than reference, x10
    assert result.adverse_selection == Decimal("0")  # no post-fill price observed

    # The identity the honest scorekeeper depends on: components reconstruct net P&L.
    assert result.reconstructed_total() == result.net_pnl


def test_pnl_attribution_with_loss_and_adverse_selection_still_reconciles():
    fills = [
        FillAttributionInput(
            canonical_id="M2",
            side=Side.NO,
            action="buy",
            price=Decimal("0.55"),
            quantity=5,
            fee=Decimal("0.05"),
            reference_price=Decimal("0.50"),
            post_fill_price=Decimal("0.45"),  # price dropped after we bought -> adverse
        )
    ]
    settlements = [
        SettlementAttributionInput(
            canonical_id="M2", side=Side.NO, quantity=5, average_price=Decimal("0.55"), won=False
        )
    ]
    result = pnl_attribution(fills, settlements)
    assert result.reconstructed_total() == result.net_pnl
    assert result.adverse_selection > Decimal("0")
