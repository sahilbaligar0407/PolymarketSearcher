"""TraderScore: realized-track-record scoring and its traps."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from marketlab.signals.trader_analytics import analyze_closed_positions

NOW = datetime(2026, 10, 4, tzinfo=UTC)


def _pos(pnl: float, price: float = 0.5, shares: float = 100.0, days_ago: float = 10.0, slug: str = "mlb-a-b") -> dict:
    return {
        "realizedPnl": pnl, "avgPrice": price, "totalBought": shares, "slug": slug,
        "timestamp": int((NOW - timedelta(days=days_ago)).timestamp()),
    }


def test_steady_edge_qualifies():
    # 60 bets of $50 stake: 36 wins of +$40, 24 losses of -$50 -> ROI +8%
    rows = [_pos(40 if i % 5 < 3 else -50, days_ago=60 - i) for i in range(60)]
    a = analyze_closed_positions("0xgood", rows, NOW)
    assert a.resolved_positions == 60
    assert a.roi > 0.05 and a.profit_factor and a.profit_factor > 1
    assert a.status == "QUALIFIED", a.reasons


def test_one_lucky_trade_is_not_trusted():
    rows = [_pos(-5, days_ago=50 - i) for i in range(40)] + [_pos(5000, days_ago=5)]
    a = analyze_closed_positions("0xlucky", rows, NOW)
    assert a.largest_win_share > 0.9
    assert a.status != "QUALIFIED"
    assert any("luck" in r for r in a.reasons)


def test_short_history_is_not_trusted():
    a = analyze_closed_positions("0xnew", [_pos(100, days_ago=d) for d in range(5)], NOW)
    assert a.status != "QUALIFIED"
    assert any("short history" in r for r in a.reasons)


def test_favorite_farming_is_penalized():
    # Buying at 0.97: tiny wins, the occasional wipeout. High win rate, poor ROI.
    rows = [_pos(3, price=0.97, days_ago=80 - i) for i in range(57)] + [_pos(-97, price=0.97, days_ago=i) for i in range(3)]
    a = analyze_closed_positions("0xfav", rows, NOW)
    assert a.win_rate > 0.9
    assert a.favorite_share > 0.9
    assert a.status != "QUALIFIED"


def test_losing_wallet_is_rejected():
    rows = [_pos(-30 if i % 3 else 20, days_ago=60 - i) for i in range(60)]
    assert analyze_closed_positions("0xbad", rows, NOW).status == "REJECTED"


def test_future_records_are_ignored():
    rows = [_pos(50, days_ago=-1)]  # "closed" tomorrow
    assert analyze_closed_positions("0xf", rows, NOW).resolved_positions == 0


def test_stake_is_shares_times_price():
    a = analyze_closed_positions("0x", [_pos(72.94, price=0.9526, shares=1539.79)], NOW)
    assert abs(a.total_staked - 1539.79 * 0.9526) < 0.01


def test_one_hot_streak_is_not_consistent_edge():
    # 40 old bets losing steadily, then 40 recent bets winning big: profitable overall,
    # but only the recent half earns money.
    rows = [_pos(-20, days_ago=200 - i) for i in range(40)] + [_pos(60, days_ago=40 - i * 0.5) for i in range(40)]
    a = analyze_closed_positions("0xstreak", rows, NOW)
    assert a.roi > 0
    assert a.status != "QUALIFIED"
    assert any("inconsistent halves" in r for r in a.reasons)
