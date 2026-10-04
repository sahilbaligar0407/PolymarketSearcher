"""The read-only dashboard builds a full snapshot from a real store."""

from __future__ import annotations

import sqlite3

import pytest

from marketlab.clock import SimulatedClock
from marketlab.dashboard.server import DashboardData
from tests.unit.test_runner import _GREEDY_LOSS_PATH, T0, _book_event, _make_runner, _variant


async def test_summary_explains_trades_and_ranks_families(tmp_path) -> None:
    db_path = tmp_path / "marketlab.db"
    clock = SimulatedClock(T0)
    runner = _make_runner(db_path, {"KXBTC-TEST": {"uni_1"}}, clock)
    await runner.load_or_create_sleeves([_variant("greedy_loss", _GREEDY_LOSS_PATH)])
    await runner.dispatch(_book_event("KXBTC-TEST", T0))
    # The real PaperBroker persists its orders and fills; the test fake does not.
    for order in runner.broker._orders.values():
        runner.store.save_order(order, order.experiment_id)
    await runner.snapshot()
    runner.store.close()

    data = DashboardData(db_path, tmp_path).summary()
    assert "error" not in data
    assert data["portfolio"]["traded"] == 1
    assert [f["strategy"] for f in data["families"]] == ["greedy_loss"]
    trade = data["recent_trades"][0]
    assert trade["rationale"] == "test: buy nearly everything"
    assert trade["outcome"] == "open"
    assert data["status"]["running"] is False


def test_dashboard_connection_is_read_only(tmp_path) -> None:
    db_path = tmp_path / "x.db"
    sqlite3.connect(db_path).execute("CREATE TABLE t (a)").connection.commit()
    conn = DashboardData(db_path, tmp_path)._connect()
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO t VALUES (1)")
