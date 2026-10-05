"""Tests for the ``marketlab.storage`` package: schema, StateStore, Parquet, DuckDB."""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from marketlab.core.events import Alert
from marketlab.core.instruments import (
    Fees,
    MarketStatus,
    NormalizedMarket,
    OutcomeType,
    Side,
    Venue,
)
from marketlab.core.orders import (
    Action,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from marketlab.core.portfolio import Portfolio, SleeveStatus
from marketlab.storage.analytics_db import AnalyticsDB
from marketlab.storage.parquet import ParquetWriter
from marketlab.storage.schema import connect, run_migrations
from marketlab.storage.state import Experiment, ExperimentStatus, StateStore

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _mk_experiment(experiment_id: str = "exp_1", created_at: datetime = T0) -> Experiment:
    return Experiment(
        experiment_id=experiment_id,
        strategy_name="mean_revert_v1",
        strategy_version="1.2.0",
        git_commit="abc1234",
        parameter_hash="deadbeef",
        parameters={"lookback": 10, "threshold": "0.05"},
        market_universe="kalshi_politics",
        venue="kalshi",
        data_version="2026-01-01",
        execution_model_version="v3",
        feature_version="v1",
        cohort="cohort_a",
        starting_bankroll=Decimal("50.00"),
        status=ExperimentStatus.BACKTESTING,
        created_at=created_at,
        notes="unit test experiment",
    )


def _mk_market(canonical_id: str, i: int) -> NormalizedMarket:
    return NormalizedMarket(
        canonical_id=canonical_id,
        venue=Venue.KALSHI,
        venue_market_id=f"KX-{i}",
        event_id=f"EVT-{i}",
        title=f"Synthetic market {i}",
        category="crypto",
        outcome_type=OutcomeType.BINARY,
        status=MarketStatus.OPEN,
        tick_size=Decimal("0.01"),
        min_order=1,
        fees=Fees(),
        open_time=T0,
        close_time=T0 + timedelta(days=1),
        raw={"i": i},
    )


# ---------------------------------------------------------------------------
# schema / migrations
# ---------------------------------------------------------------------------


def test_migrations_are_idempotent(tmp_path: Path) -> None:
    conn = connect(tmp_path / "idempotent.db")
    try:
        first = run_migrations(conn)
        second = run_migrations(conn)
        assert first, "expected at least one migration to apply on a fresh db"
        assert second == []

        # And a totally fresh connection to the same file sees the schema and applies
        # nothing new either.
        conn2 = connect(tmp_path / "idempotent.db")
        try:
            assert run_migrations(conn2) == []
            tables = {
                r[0]
                for r in conn2.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            for expected in (
                "experiments",
                "orders",
                "fills",
                "positions",
                "balances",
                "strategy_state",
                "trader_registry",
                "trader_leaderboard_snapshots",
                "trader_actions",
                "market_registry",
                "market_matches",
                "forecasts",
                "settlements",
                "alerts",
                "service_checkpoints",
                "social_challenge_candidates",
                "schema_migrations",
            ):
                assert expected in tables
        finally:
            conn2.close()
    finally:
        conn.close()


def test_pragmas_are_applied(tmp_path: Path) -> None:
    conn = connect(tmp_path / "pragmas.db")
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------


def test_create_and_read_back_experiment(tmp_path: Path) -> None:
    with StateStore(tmp_path / "state.db") as store:
        created = store.create_experiment(_mk_experiment())
        fetched = store.get_experiment("exp_1")

        assert fetched is not None
        assert fetched == created
        assert fetched.parameters == {"lookback": 10, "threshold": "0.05"}
        assert fetched.starting_bankroll == Decimal("50.00")
        assert fetched.status is ExperimentStatus.BACKTESTING

        store.update_experiment_status("exp_1", ExperimentStatus.QUALIFIED)
        assert store.get_experiment("exp_1").status is ExperimentStatus.QUALIFIED

        store.create_experiment(_mk_experiment("exp_2", created_at=T0 + timedelta(hours=1)))
        assert {e.experiment_id for e in store.list_experiments()} == {"exp_1", "exp_2"}
        assert [e.experiment_id for e in store.list_experiments(status="QUALIFIED")] == ["exp_1"]
        assert len(store.list_experiments(strategy="mean_revert_v1")) == 2


# ---------------------------------------------------------------------------
# orders / fills
# ---------------------------------------------------------------------------


def test_save_order_with_fills_preserves_latency_timestamps(tmp_path: Path) -> None:
    with StateStore(tmp_path / "state.db") as store:
        store.create_experiment(_mk_experiment())

        fill = Fill(
            order_id="ord_shared",
            canonical_id="KXBTC-1",
            venue=Venue.KALSHI,
            side=Side.YES,
            action=Action.BUY,
            price=Decimal("0.63"),
            quantity=10,
            fee=Decimal("0.02"),
            timestamp=T0 + timedelta(seconds=2),
            is_maker=False,
            book_timestamp_used=T0 + timedelta(milliseconds=900),
            level_breakdown=((Decimal("0.62"), 6), (Decimal("0.63"), 4)),
        )
        order = Order(
            order_id="ord_shared",
            intent_id="int_1",
            strategy_id="strat_1",
            experiment_id="exp_1",
            canonical_id="KXBTC-1",
            venue=Venue.KALSHI,
            side=Side.YES,
            action=Action.BUY,
            order_type=OrderType.LIMIT,
            quantity=10,
            limit_price=Decimal("0.65"),
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.FILLED,
            filled_quantity=10,
            average_fill_price=Decimal("0.626"),
            worst_fill_price=Decimal("0.63"),
            fees_paid=Decimal("0.02"),
            decision_timestamp=T0,
            simulated_network_send_timestamp=T0 + timedelta(milliseconds=100),
            simulated_exchange_arrival_timestamp=T0 + timedelta(milliseconds=175),
            book_timestamp_used=T0 + timedelta(milliseconds=900),
            reference_price=Decimal("0.60"),
            venue_order_id=None,
            fills=(fill,),
        )

        store.save_order(order, "exp_1")

        fetched = store.get_order("ord_shared")
        assert fetched is not None
        assert fetched.decision_timestamp == T0
        assert fetched.simulated_network_send_timestamp == T0 + timedelta(milliseconds=100)
        assert fetched.simulated_exchange_arrival_timestamp == T0 + timedelta(milliseconds=175)
        assert fetched.book_timestamp_used == T0 + timedelta(milliseconds=900)
        assert fetched.reference_price == Decimal("0.60")
        assert fetched.average_fill_price == Decimal("0.626")
        assert fetched.slippage == Decimal("0.026")
        assert len(fetched.fills) == 1
        got_fill = fetched.fills[0]
        assert got_fill.price == Decimal("0.63")
        assert got_fill.fee == Decimal("0.02")
        assert got_fill.level_breakdown == ((Decimal("0.62"), 6), (Decimal("0.63"), 4))
        assert got_fill.book_timestamp_used == T0 + timedelta(milliseconds=900)

        assert store.orders_for_experiment("exp_1") == [fetched]
        assert store.fills_for_experiment("exp_1") == [got_fill]

        # Update to CANCELED and confirm it drops out of open_orders / stays terminal.
        canceled = order.model_copy(update={"status": OrderStatus.FILLED})
        store.update_order(canceled)
        assert store.open_orders("exp_1") == []


# ---------------------------------------------------------------------------
# portfolio crash-recovery round trip
# ---------------------------------------------------------------------------


def test_portfolio_round_trip_with_open_positions_both_sides(tmp_path: Path) -> None:
    with StateStore(tmp_path / "state.db") as store:
        store.create_experiment(_mk_experiment())

        portfolio = Portfolio(
            experiment_id="exp_1",
            strategy_id="strat_1",
            initial_capital=Decimal("50.00"),
            cash=Decimal("12.34"),
            status=SleeveStatus.ACTIVE,
            created_at=T0,
            high_water_mark=Decimal("55.00"),
            max_drawdown=Decimal("0.15"),
            trade_count=7,
            resolved_trade_count=2,
        )

        fills = [
            Fill(
                order_id="ord_a",
                canonical_id="MKT-A",
                venue=Venue.KALSHI,
                side=Side.YES,
                action=Action.BUY,
                price=Decimal("0.40"),
                quantity=20,
                fee=Decimal("0.05"),
                timestamp=T0 + timedelta(minutes=1),
            ),
            Fill(
                order_id="ord_b",
                canonical_id="MKT-A",
                venue=Venue.KALSHI,
                side=Side.NO,
                action=Action.BUY,
                price=Decimal("0.35"),
                quantity=15,
                fee=Decimal("0.03"),
                timestamp=T0 + timedelta(minutes=2),
            ),
            Fill(
                order_id="ord_c",
                canonical_id="MKT-B",
                venue=Venue.KALSHI,
                side=Side.YES,
                action=Action.BUY,
                price=Decimal("0.55"),
                quantity=8,
                fee=Decimal("0.01"),
                timestamp=T0 + timedelta(minutes=3),
            ),
            Fill(
                order_id="ord_d",
                canonical_id="MKT-B",
                venue=Venue.KALSHI,
                side=Side.NO,
                action=Action.BUY,
                price=Decimal("0.42"),
                quantity=12,
                fee=Decimal("0.02"),
                timestamp=T0 + timedelta(minutes=4),
            ),
        ]
        for f in fills:
            portfolio.apply_fill(f)

        # Give position on MKT-A/YES some realized P&L and a partial close so the
        # round trip also covers a nonzero-realized, reduced-quantity position.
        sell = Fill(
            order_id="ord_e",
            canonical_id="MKT-A",
            venue=Venue.KALSHI,
            side=Side.YES,
            action=Action.SELL,
            price=Decimal("0.50"),
            quantity=5,
            fee=Decimal("0.01"),
            timestamp=T0 + timedelta(minutes=5),
        )
        portfolio.apply_fill(sell)
        portfolio.mark({})

        assert portfolio.realized_pnl != Decimal(0)
        assert portfolio.max_drawdown >= Decimal(0)

        store.save_portfolio(portfolio, as_of=T0 + timedelta(minutes=6))
        restored = store.load_portfolio("exp_1")

        assert restored is not None
        assert restored.experiment_id == portfolio.experiment_id
        assert restored.strategy_id == portfolio.strategy_id
        assert restored.initial_capital == portfolio.initial_capital
        assert restored.cash == portfolio.cash
        assert restored.realized_pnl == portfolio.realized_pnl
        assert restored.fees_paid == portfolio.fees_paid
        assert restored.status == portfolio.status
        assert restored.created_at == portfolio.created_at
        assert restored.high_water_mark == portfolio.high_water_mark
        assert restored.max_drawdown == portfolio.max_drawdown
        assert restored.trade_count == portfolio.trade_count
        assert restored.resolved_trade_count == portfolio.resolved_trade_count

        assert set(restored.positions.keys()) == set(portfolio.positions.keys())
        for key, pos in portfolio.positions.items():
            got = restored.positions[key]
            assert got.canonical_id == pos.canonical_id
            assert got.side == pos.side
            assert got.quantity == pos.quantity
            assert got.average_price == pos.average_price
            assert got.realized_pnl == pos.realized_pnl
            assert got.fees_paid == pos.fees_paid
            assert got.gross_bought == pos.gross_bought
            assert got.gross_sold == pos.gross_sold
            assert got.opened_at == pos.opened_at
            assert got.last_update == pos.last_update

        # Never a fresh bankroll on "restart": equity must match, not reset to $50.
        assert restored.equity() == portfolio.equity()

        # Both YES and NO of both markets are present as open positions.
        for market in ("MKT-A", "MKT-B"):
            for side in (Side.YES, Side.NO):
                key = Portfolio.key(market, side)
                assert key in restored.positions
                assert restored.positions[key].quantity > 0


def test_load_portfolio_missing_experiment_returns_none(tmp_path: Path) -> None:
    with StateStore(tmp_path / "state.db") as store:
        assert store.load_portfolio("does_not_exist") is None


# ---------------------------------------------------------------------------
# strategy state
# ---------------------------------------------------------------------------


def test_strategy_state_round_trip(tmp_path: Path) -> None:
    with StateStore(tmp_path / "state.db") as store:
        store.create_experiment(_mk_experiment())
        store.save_strategy_state("exp_1", "last_seen_seq", 42)
        store.save_strategy_state("exp_1", "cooldowns", {"MKT-A": "2026-01-01T00:00:00+00:00"})
        state = store.load_strategy_state("exp_1")
        assert state["last_seen_seq"] == 42
        assert state["cooldowns"] == {"MKT-A": "2026-01-01T00:00:00+00:00"}


# ---------------------------------------------------------------------------
# markets
# ---------------------------------------------------------------------------


def test_bulk_upsert_markets_is_fast_and_idempotent(tmp_path: Path) -> None:
    markets = [_mk_market(f"MKT-{i}", i) for i in range(500)]
    with StateStore(tmp_path / "state.db") as store:
        start = time.monotonic()
        store.bulk_upsert_markets(markets)
        elapsed_first = time.monotonic() - start
        assert elapsed_first < 5.0, f"bulk_upsert_markets(500) took {elapsed_first:.2f}s"

        start = time.monotonic()
        store.bulk_upsert_markets(markets)
        elapsed_second = time.monotonic() - start
        assert elapsed_second < 5.0

        all_markets = store.list_markets()
        assert len(all_markets) == 500

        one = store.get_market("MKT-250")
        assert one is not None
        assert one.canonical_id == "MKT-250"
        assert one.venue_market_id == "KX-250"
        assert one.tick_size == Decimal("0.01")

        assert len(store.list_markets(venue="kalshi")) == 500
        assert len(store.list_markets(category="crypto", limit=10)) == 10
        assert len(store.list_markets(venue="poly-global")) == 0


# ---------------------------------------------------------------------------
# alerts
# ---------------------------------------------------------------------------


def test_alerts_round_trip(tmp_path: Path) -> None:
    with StateStore(tmp_path / "state.db") as store:
        store.save_alert(
            Alert(
                timestamp=T0,
                severity="warning",
                component="daemon",
                message="feed lag",
                detail={"lag_seconds": 12.5},
            )
        )
        recent = store.recent_alerts(10)
        assert len(recent) == 1
        assert recent[0].message == "feed lag"
        assert recent[0].detail == {"lag_seconds": 12.5}


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------


def test_checkpoint_set_and_get(tmp_path: Path) -> None:
    with StateStore(tmp_path / "state.db") as store:
        assert store.get_checkpoint("kalshi_ws", "last_seq") is None
        store.set_checkpoint("kalshi_ws", "last_seq", "12345")
        assert store.get_checkpoint("kalshi_ws", "last_seq") == "12345"
        store.set_checkpoint("kalshi_ws", "last_seq", "12399")
        assert store.get_checkpoint("kalshi_ws", "last_seq") == "12399"


# ---------------------------------------------------------------------------
# parquet
# ---------------------------------------------------------------------------


def test_parquet_writer_creates_partitioned_file_and_round_trips(tmp_path: Path) -> None:
    import polars as pl

    writer = ParquetWriter(tmp_path / "parquet", flush_threshold=500)
    rows = [
        {
            "canonical_id": f"MKT-{i}",
            "venue": "kalshi",
            "timestamp": T0 + timedelta(seconds=i),
            "trade_id": f"t{i}",
            "aggressor": "yes",
            "size": 10,
            "price": Decimal("0.5") + Decimal(i) / Decimal(1000),
        }
        for i in range(120)
    ]
    writer.write("trades", rows)
    # Not yet flushed (below threshold).
    assert not list((tmp_path / "parquet" / "trades").glob("date=*/*.parquet"))

    writer.flush_all()
    files = list((tmp_path / "parquet" / "trades").glob("date=*/*.parquet"))
    assert len(files) == 1
    assert files[0].parent.name == f"date={T0.date().isoformat()}"

    table = pl.read_parquet(files[0])
    assert table.height == 120
    assert table["price_exact"][0] == "0.5"
    assert abs(table["price"][0] - 0.5) < 1e-9

    # A second write+flush lands a new part file rather than clobbering the first.
    writer.write("trades", rows[:5])
    writer.flush_all()
    files_after = list((tmp_path / "parquet" / "trades").glob("date=*/*.parquet"))
    assert len(files_after) == 2
    total_rows = sum(pl.read_parquet(f).height for f in files_after)
    assert total_rows == 125

    writer.close()


def test_parquet_writer_rejects_unknown_dataset(tmp_path: Path) -> None:
    writer = ParquetWriter(tmp_path / "parquet")
    with pytest.raises(ValueError, match="unknown parquet dataset"):
        writer.write("not_a_real_dataset", [{"a": 1}])


# ---------------------------------------------------------------------------
# analytics db
# ---------------------------------------------------------------------------


def test_analytics_db_degrades_gracefully_with_no_data(tmp_path: Path) -> None:
    db = AnalyticsDB(tmp_path / "parquet")
    try:
        db.register_parquet("trades")
        empty = db.query("SELECT * FROM trades")
        assert empty.height == 0
        assert "canonical_id" in empty.columns

        db.attach_sqlite(tmp_path / "does_not_exist.db")
        lb = db.leaderboard()
        assert lb.height == 0
        pnl = db.strategy_pnl()
        assert pnl.height == 0
    finally:
        db.close()


def test_analytics_db_reads_sqlite_and_parquet_together(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with StateStore(db_path) as store:
        store.create_experiment(_mk_experiment())
        portfolio = Portfolio(experiment_id="exp_1", strategy_id="strat_1", created_at=T0)
        store.save_portfolio(portfolio, as_of=T0)

    writer = ParquetWriter(tmp_path / "parquet")
    writer.write(
        "trades",
        [
            {
                "canonical_id": "MKT-1",
                "venue": "kalshi",
                "timestamp": T0,
                "trade_id": "t1",
                "aggressor": "yes",
                "size": 3,
                "price": Decimal("0.42"),
            }
        ],
    )
    writer.flush_all()

    db = AnalyticsDB(tmp_path / "parquet")
    try:
        db.register_parquet("trades")
        db.attach_sqlite(db_path)

        trades = db.query("SELECT * FROM trades")
        assert trades.height == 1

        experiments = db.query("SELECT experiment_id FROM experiments")
        assert experiments["experiment_id"].to_list() == ["exp_1"]

        pnl = db.strategy_pnl()
        assert pnl.height == 1
        assert pnl["experiment_id"][0] == "exp_1"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# concurrent-safety sanity: sqlite3 module import used only for the pragma assert.
# ---------------------------------------------------------------------------


def test_state_store_context_manager_closes(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    store.create_experiment(_mk_experiment())
    store.close()
    with pytest.raises(sqlite3.ProgrammingError):
        store.get_experiment("exp_1")


def test_a_hard_kill_loses_no_fills_the_live_checkpoint_wins(tmp_path) -> None:
    """2026-10-05: balances are written every 5 min; fills since then were lost on a kill."""
    from datetime import UTC, datetime, timedelta
    from decimal import Decimal

    from marketlab.core.instruments import Side, Venue
    from marketlab.core.orders import Action, Fill
    from marketlab.core.portfolio import Portfolio
    from marketlab.storage.state import StateStore

    t0 = datetime(2026, 10, 5, 4, 0, tzinfo=UTC)
    store = StateStore(tmp_path / "live.db")
    store.create_experiment(_mk_experiment())
    p = Portfolio(experiment_id="exp_1", strategy_id="s", initial_capital=Decimal("100"), cash=Decimal("100"))
    store.save_portfolio(p, as_of=t0)  # the 5-minute snapshot
    p.apply_fill(Fill(order_id="o", canonical_id="kalshi:x", venue=Venue.KALSHI, side=Side.YES,
                      action=Action.BUY, price=Decimal("0.40"), quantity=10, fee=Decimal("0.02"),
                      timestamp=t0 + timedelta(minutes=2)))
    store.save_portfolio_live(p, t0 + timedelta(minutes=2))  # what the broker writes per fill
    store.close()

    reloaded = StateStore(tmp_path / "live.db").load_portfolio("exp_1")
    assert reloaded is not None
    assert reloaded.cash == Decimal("95.98")
    assert reloaded.positions[Portfolio.key("kalshi:x", Side.YES)].quantity == 10


def test_a_newer_snapshot_supersedes_an_older_checkpoint(tmp_path) -> None:
    from datetime import UTC, datetime, timedelta
    from decimal import Decimal

    from marketlab.core.portfolio import Portfolio
    from marketlab.storage.state import StateStore

    t0 = datetime(2026, 10, 5, 4, 0, tzinfo=UTC)
    store = StateStore(tmp_path / "live2.db")
    store.create_experiment(_mk_experiment())
    p = Portfolio(experiment_id="exp_1", strategy_id="s", initial_capital=Decimal("100"), cash=Decimal("90"))
    store.save_portfolio_live(p, t0)
    p.cash = Decimal("80")
    store.save_portfolio(p, as_of=t0 + timedelta(minutes=5))
    assert store.load_portfolio("exp_1").cash == Decimal("80")
