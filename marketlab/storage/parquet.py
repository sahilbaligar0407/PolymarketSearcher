"""Append-only, date-partitioned Parquet writers for the raw/normalized data lake.

Layout: ``<parquet_dir>/<dataset>/date=YYYY-MM-DD/part-<n>.parquet``, one Hive-style
partition per UTC calendar date, files numbered so a process restart never overwrites
an existing part.  Each dataset has an explicit :mod:`pyarrow` schema (below) so that
every file written for that dataset - today or a year from now - has identical column
types and can be read back as one table with :func:`polars.read_parquet` /
DuckDB's ``read_parquet('.../**/*.parquet')`` glob.

Decimal convention (documented once, applies to every dataset here): a value that
must round-trip exactly (a price, a size, money) is stored twice - once as
``pa.float64()`` for fast numeric work, and once as ``pa.string()`` holding
``str(Decimal(...))`` for lossless reconstruction. Column names follow the pattern
``<field>`` (float64) and ``<field>_exact`` (string). This is simpler and more
portable across polars/duckdb/pyarrow versions than ``pa.decimal128`` and avoids
overflow surprises when a caller hands in a value with unexpected scale.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

#: Fields that get the float64 + "<field>_exact" string treatment, per dataset.
_DECIMAL_FIELDS: dict[str, tuple[str, ...]] = {
    "books": ("bid_price", "ask_price"),
    "trades": ("price",),
    "prices": ("price",),
    "market_metadata": ("tick_size",),
    "news": (),
    "social": (),
    "external_prices": ("price", "implied_probability"),
    "forecasts": ("p_yes", "confidence", "market_probability"),
    "trader_actions": ("price", "size", "usd_size"),
    "holdings": ("size", "avg_price", "cur_price", "usd_value"),
}


def _decimal_pair_fields(base: tuple[str, ...]) -> list[pa.Field]:
    fields: list[pa.Field] = []
    for name in base:
        fields.append(pa.field(name, pa.float64()))
        fields.append(pa.field(f"{name}_exact", pa.string()))
    return fields


#: Explicit pyarrow schema per dataset. Every writable dataset must appear here.
SCHEMAS: dict[str, pa.Schema] = {
    "books": pa.schema(
        [
            pa.field("canonical_id", pa.string()),
            pa.field("venue", pa.string()),
            pa.field("timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("venue_timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("sequence", pa.int64()),
            pa.field("side", pa.string()),
            pa.field("level", pa.int32()),
            *_decimal_pair_fields(_DECIMAL_FIELDS["books"]),
            pa.field("bid_size", pa.int64()),
            pa.field("ask_size", pa.int64()),
        ]
    ),
    "trades": pa.schema(
        [
            pa.field("canonical_id", pa.string()),
            pa.field("venue", pa.string()),
            pa.field("timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("trade_id", pa.string()),
            pa.field("aggressor", pa.string()),
            pa.field("size", pa.int64()),
            *_decimal_pair_fields(_DECIMAL_FIELDS["trades"]),
        ]
    ),
    "prices": pa.schema(
        [
            pa.field("canonical_id", pa.string()),
            pa.field("venue", pa.string()),
            pa.field("timestamp", pa.timestamp("us", tz="UTC")),
            *_decimal_pair_fields(_DECIMAL_FIELDS["prices"]),
        ]
    ),
    "market_metadata": pa.schema(
        [
            pa.field("canonical_id", pa.string()),
            pa.field("venue", pa.string()),
            pa.field("venue_market_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("category", pa.string()),
            pa.field("status", pa.string()),
            pa.field("open_time", pa.timestamp("us", tz="UTC")),
            pa.field("close_time", pa.timestamp("us", tz="UTC")),
            *_decimal_pair_fields(_DECIMAL_FIELDS["market_metadata"]),
            pa.field("first_seen_time", pa.timestamp("us", tz="UTC")),
        ]
    ),
    "news": pa.schema(
        [
            pa.field("news_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("url", pa.string()),
            pa.field("body_hash", pa.string()),
            pa.field("source", pa.string()),
            pa.field("source_class", pa.string()),
            pa.field("event_time", pa.timestamp("us", tz="UTC")),
            pa.field("published_time", pa.timestamp("us", tz="UTC")),
            pa.field("first_seen_time", pa.timestamp("us", tz="UTC")),
            pa.field("tickers", pa.list_(pa.string())),
        ]
    ),
    "social": pa.schema(
        [
            pa.field("post_id", pa.string()),
            pa.field("platform", pa.string()),
            pa.field("author", pa.string()),
            pa.field("text_hash", pa.string()),
            pa.field("action_type", pa.string()),
            pa.field("policy_topic", pa.string()),
            pa.field("event_time", pa.timestamp("us", tz="UTC")),
            pa.field("first_seen_time", pa.timestamp("us", tz="UTC")),
            pa.field("mentioned_tickers", pa.list_(pa.string())),
        ]
    ),
    "external_prices": pa.schema(
        [
            pa.field("symbol", pa.string()),
            pa.field("venue", pa.string()),
            pa.field("timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("first_seen_time", pa.timestamp("us", tz="UTC")),
            *_decimal_pair_fields(_DECIMAL_FIELDS["external_prices"]),
        ]
    ),
    "forecasts": pa.schema(
        [
            pa.field("experiment_id", pa.string()),
            pa.field("strategy_id", pa.string()),
            pa.field("canonical_id", pa.string()),
            pa.field("as_of", pa.timestamp("us", tz="UTC")),
            *_decimal_pair_fields(_DECIMAL_FIELDS["forecasts"]),
            pa.field("abstain", pa.bool_()),
            pa.field("rationale", pa.string()),
        ]
    ),
    # One row per (snapshot, wallet, open position) for the top leaderboard wallets: the
    # raw material of the holdings-consensus strategies, kept so any consensus rule can be
    # replayed later against what the wallets actually held at each snapshot.
    "holdings": pa.schema(
        [
            pa.field("snapshot_time", pa.timestamp("us", tz="UTC")),
            pa.field("wallet", pa.string()),
            pa.field("boards", pa.string()),
            pa.field("condition_id", pa.string()),
            pa.field("outcome_index", pa.int32()),
            pa.field("outcome", pa.string()),
            pa.field("title", pa.string()),
            *_decimal_pair_fields(_DECIMAL_FIELDS["holdings"]),
            pa.field("end_date", pa.string()),
        ]
    ),
    "trader_actions": pa.schema(
        [
            pa.field("wallet", pa.string()),
            pa.field("canonical_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("side", pa.string()),
            pa.field("action", pa.string()),
            *_decimal_pair_fields(_DECIMAL_FIELDS["trader_actions"]),
            pa.field("category", pa.string()),
            pa.field("event_time", pa.timestamp("us", tz="UTC")),
            pa.field("first_seen_time", pa.timestamp("us", tz="UTC")),
        ]
    ),
}


def _coerce_row(dataset: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Fill in the ``<field>_exact`` string twin for every decimal field."""
    out = dict(row)
    for name in _DECIMAL_FIELDS.get(dataset, ()):
        raw = out.get(name)
        exact_key = f"{name}_exact"
        if raw is None:
            out.setdefault(name, None)
            out.setdefault(exact_key, None)
            continue
        d = raw if isinstance(raw, Decimal) else Decimal(str(raw))
        out[name] = float(d)
        out[exact_key] = str(d)
    return out


class ParquetWriter:
    """Buffers rows per dataset and flushes them to date-partitioned Parquet files.

    Safe to use from asyncio: every method that touches disk acquires a lock and, in
    the async wrapper below, runs in a worker thread.
    """

    def __init__(self, root: str | Path, flush_threshold: int = 500) -> None:
        self.root = Path(root)
        self.flush_threshold = flush_threshold
        self._lock = threading.RLock()
        self._buffers: dict[str, list[dict[str, Any]]] = {}

    def _dataset_dir(self, dataset: str, day: date) -> Path:
        return self.root / dataset / f"date={day.isoformat()}"

    def _next_part_path(self, dataset: str, day: date) -> Path:
        ddir = self._dataset_dir(dataset, day)
        ddir.mkdir(parents=True, exist_ok=True)
        existing = sorted(ddir.glob("part-*.parquet"))
        n = 0
        if existing:
            nums = []
            for p in existing:
                try:
                    nums.append(int(p.stem.split("-")[1]))
                except (IndexError, ValueError):
                    continue
            n = max(nums, default=-1) + 1
        return ddir / f"part-{n}.parquet"

    def write(self, dataset: str, rows: list[dict[str, Any]]) -> None:
        """Buffer ``rows`` for ``dataset``, flushing automatically past the threshold.

        Each row must include a ``date`` key (a :class:`datetime.date`, or a
        ``datetime`` - its ``.date()`` is used) that decides which partition it lands
        in; if absent, the row is partitioned under the date derived from its first
        timestamp-like field.
        """
        if dataset not in SCHEMAS:
            raise ValueError(f"unknown parquet dataset: {dataset!r}")
        if not rows:
            return
        with self._lock:
            buf = self._buffers.setdefault(dataset, [])
            buf.extend(_coerce_row(dataset, r) for r in rows)
            if len(buf) >= self.flush_threshold:
                self._flush_dataset_locked(dataset)

    @staticmethod
    def _partition_date(row: dict[str, Any]) -> date:
        explicit = row.get("date")
        if isinstance(explicit, date):
            return explicit
        for value in row.values():
            if hasattr(value, "date") and callable(value.date):
                return value.date()
        raise ValueError("row has no date/datetime field to partition on")

    def _flush_dataset_locked(self, dataset: str) -> None:
        buf = self._buffers.get(dataset)
        if not buf:
            return
        schema = SCHEMAS[dataset]
        by_day: dict[date, list[dict[str, Any]]] = {}
        for row in buf:
            day = self._partition_date(row)
            clean = {k: v for k, v in row.items() if k in schema.names}
            for name in schema.names:
                clean.setdefault(name, None)
            by_day.setdefault(day, []).append(clean)
        for day, day_rows in by_day.items():
            table = pa.Table.from_pylist(day_rows, schema=schema)
            path = self._next_part_path(dataset, day)
            pq.write_table(table, path)
        self._buffers[dataset] = []

    def flush_all(self) -> None:
        with self._lock:
            for dataset in list(self._buffers):
                self._flush_dataset_locked(dataset)

    def close(self) -> None:
        self.flush_all()

    def __enter__(self) -> ParquetWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class AsyncParquetWriter:
    """Async facade running every :class:`ParquetWriter` call in a worker thread."""

    def __init__(self, writer: ParquetWriter) -> None:
        self._writer = writer

    async def write(self, dataset: str, rows: list[dict[str, Any]]) -> None:
        await asyncio.to_thread(self._writer.write, dataset, rows)

    async def flush_all(self) -> None:
        await asyncio.to_thread(self._writer.flush_all)

    async def close(self) -> None:
        await asyncio.to_thread(self._writer.close)
