"""DuckDB-backed analytics layer over the Parquet lake and the SQLite operational store.

DuckDB is used purely as a read-side query engine: nothing here writes back to either
the Parquet files or the SQLite database.  Two data sources are exposed as views in the
same DuckDB catalog so a single SQL statement can join live order/fill history against
book/trade history:

* ``register_parquet(dataset)`` creates a view over
  ``<parquet_dir>/<dataset>/date=*/*.parquet`` for one of the datasets in
  :data:`marketlab.storage.parquet.SCHEMAS`.
* ``attach_sqlite(db_path)`` tries DuckDB's ``sqlite_scanner`` extension first
  (``ATTACH ... (TYPE sqlite)``); if that extension can't be installed (e.g. this
  machine is offline and it isn't already cached), it degrades gracefully by reading
  each table with the stdlib :mod:`sqlite3` module and registering the resulting
  Polars frame under the same view name.  Either way the caller sees the same table
  names and never sees an exception from a missing extension.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb
import polars as pl
import pyarrow as pa

from marketlab.storage.parquet import SCHEMAS as PARQUET_SCHEMAS

#: SQLite tables exposed as DuckDB views by attach_sqlite().
SQLITE_TABLES: tuple[str, ...] = (
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
)

_LEADERBOARD_COLUMNS = (
    "snapshot_time",
    "category",
    "period",
    "metric",
    "rank",
    "wallet",
    "username",
    "pnl",
    "volume",
)

_STRATEGY_PNL_COLUMNS = (
    "experiment_id",
    "strategy_id",
    "realized_pnl",
    "equity",
    "max_drawdown",
    "trade_count",
)


def _empty_frame(columns: tuple[str, ...]) -> pl.DataFrame:
    """An empty Polars frame with exactly ``columns``, for callers with no data yet."""
    return pl.DataFrame({c: [] for c in columns})


class AnalyticsDB:
    """Query façade combining the Parquet lake and the SQLite operational store."""

    def __init__(self, parquet_dir: str | Path) -> None:
        self.parquet_dir = Path(parquet_dir)
        self._conn = duckdb.connect(":memory:")
        self._sqlite_mode: str | None = None

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> AnalyticsDB:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # generic query
    # ------------------------------------------------------------------

    def query(self, sql: str) -> pl.DataFrame:
        """Run arbitrary SQL against the DuckDB catalog and return a Polars frame."""
        return self._conn.execute(sql).pl()

    # ------------------------------------------------------------------
    # parquet lake
    # ------------------------------------------------------------------

    def register_parquet(self, dataset: str) -> None:
        """Create/replace a DuckDB view named ``dataset`` over its Parquet partitions.

        If no files have been written yet, the view is still created (empty, with the
        dataset's declared schema) so downstream SQL never has to special-case "no
        data yet".
        """
        if dataset not in PARQUET_SCHEMAS:
            raise ValueError(f"unknown parquet dataset: {dataset!r}")
        matches = list(self.parquet_dir.glob(f"{dataset}/date=*/*.parquet"))
        if matches:
            pattern = (self.parquet_dir / dataset / "**" / "*.parquet").as_posix()
            self._conn.execute(
                f"CREATE OR REPLACE VIEW {dataset} AS "
                f"SELECT * FROM read_parquet('{pattern}', union_by_name=true)"
            )
        else:
            empty = pa.Table.from_pylist([], schema=PARQUET_SCHEMAS[dataset])
            self._conn.register(f"_{dataset}_empty_arrow", empty)
            self._conn.execute(
                f"CREATE OR REPLACE VIEW {dataset} AS SELECT * FROM _{dataset}_empty_arrow"
            )

    # ------------------------------------------------------------------
    # sqlite operational store
    # ------------------------------------------------------------------

    def attach_sqlite(self, db_path: str | Path) -> None:
        """Expose every SQLite table in ``SQLITE_TABLES`` as a same-named DuckDB view."""
        path = Path(db_path)
        try:
            self._conn.execute("INSTALL sqlite")
            self._conn.execute("LOAD sqlite")
            self._conn.execute(f"ATTACH '{path.as_posix()}' AS sqlite_src (TYPE sqlite)")
            for table in SQLITE_TABLES:
                self._conn.execute(
                    f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM sqlite_src.{table}"
                )
            self._sqlite_mode = "sqlite_scanner"
        except Exception:
            # No internet to fetch the extension, or it isn't cached locally: degrade
            # to reading the tables directly and registering them as Polars frames.
            # This must never raise - a fresh checkout with no cached extension is a
            # completely normal state, not an error.
            self._attach_sqlite_fallback(path)
            self._sqlite_mode = "python_fallback"

    def _attach_sqlite_fallback(self, path: Path) -> None:
        if not path.exists():
            for table in SQLITE_TABLES:
                self._conn.register(table, _empty_frame(()))
            return
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        try:
            for table in SQLITE_TABLES:
                try:
                    cur = conn.execute(f"SELECT * FROM {table}")
                except sqlite3.OperationalError:
                    continue
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
                data: dict[str, list[object]] = {c: [] for c in cols}
                for row in rows:
                    for c in cols:
                        data[c].append(row[c])
                self._conn.register(table, pl.DataFrame(data))
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # convenience views
    # ------------------------------------------------------------------

    def leaderboard(self) -> pl.DataFrame:
        """Every leaderboard snapshot row, most recent and best-ranked first."""
        try:
            return self.query(
                "SELECT snapshot_time, category, period, metric, rank, wallet, "
                "username, pnl, volume FROM trader_leaderboard_snapshots "
                "ORDER BY snapshot_time DESC, rank ASC"
            )
        except duckdb.Error:
            return _empty_frame(_LEADERBOARD_COLUMNS)

    def strategy_pnl(self) -> pl.DataFrame:
        """Latest balance snapshot per experiment, ranked by realized P&L.

        ``realized_pnl`` / ``equity`` / ``max_drawdown`` stay as text (the SQLite
        Decimal-as-TEXT convention, see ``schema.py``); cast them in SQL if you need
        numeric ordering beyond what's already applied here via a text-safe cast.
        """
        try:
            return self.query(
                """
                SELECT experiment_id, strategy_id, realized_pnl, equity,
                       max_drawdown, trade_count
                FROM balances
                WHERE id IN (SELECT MAX(id) FROM balances GROUP BY experiment_id)
                ORDER BY CAST(realized_pnl AS DOUBLE) DESC
                """
            )
        except duckdb.Error:
            return _empty_frame(_STRATEGY_PNL_COLUMNS)
