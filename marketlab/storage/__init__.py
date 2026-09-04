"""Persistence layer: SQLite operational state, the Parquet lake, and DuckDB analytics.

Public surface:

* :mod:`marketlab.storage.schema` - connection setup, pragmas, migration runner.
* :mod:`marketlab.storage.state` - :class:`StateStore` / :class:`AsyncStateStore`.
* :mod:`marketlab.storage.parquet` - :class:`ParquetWriter` / :class:`AsyncParquetWriter`.
* :mod:`marketlab.storage.analytics_db` - :class:`AnalyticsDB`.
"""

from __future__ import annotations

from marketlab.storage.analytics_db import AnalyticsDB
from marketlab.storage.parquet import AsyncParquetWriter, ParquetWriter
from marketlab.storage.schema import connect, run_migrations
from marketlab.storage.state import AsyncStateStore, Experiment, StateStore

__all__ = [
    "AnalyticsDB",
    "AsyncParquetWriter",
    "AsyncStateStore",
    "Experiment",
    "ParquetWriter",
    "StateStore",
    "connect",
    "run_migrations",
]
