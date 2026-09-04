"""SQLite connection setup and the migration runner.

Storage-wide conventions, documented once here because every other module in this
package depends on them:

* All money and probability values are ``Decimal`` in Python.  SQLite has no native
  ``Decimal`` type, so every such column is stored as ``TEXT`` holding the exact
  ``str(Decimal(...))`` representation and converted back with ``Decimal(text)`` on
  read.  This is lossless (unlike ``REAL``/float) and is the single money-storage
  convention used across ``state.py``.
* All timestamps are stored as ISO-8601 strings (``datetime.isoformat()``) produced
  from timezone-aware UTC ``datetime`` objects, and parsed back with
  ``datetime.fromisoformat``.
* Booleans are stored as ``INTEGER`` (0/1).
* Nothing in this package calls ``datetime.now()``.  Business timestamps are always
  supplied by the caller (who gets them from an injected ``Clock``).  A handful of
  purely-administrative bookkeeping columns that have no corresponding field on any
  frozen core model (e.g. ``schema_migrations.applied_at``, ``market_registry.last_seen``)
  are stamped using SQLite's own ``strftime('%Y-%m-%dT%H:%M:%fZ','now')`` instead -
  that is SQLite's clock, not Python's, and keeps the "no wall clock in application
  code" rule intact while still recording *something* useful for operators.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

#: SQL expression that stamps the current UTC instant using SQLite's own clock.
#: Used only for administrative bookkeeping columns with no business meaning.
SQL_NOW = "strftime('%Y-%m-%dT%H:%M:%fZ','now')"

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_VERSION_RE = re.compile(r"^(\d+)_.*\.sql$")


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with the pragmas this project requires.

    WAL mode, ``synchronous=NORMAL`` and a 5s busy timeout make the daemon (a single
    writer, possibly several readers such as the analytics CLI) resilient to lock
    contention without sacrificing much durability.  ``foreign_keys=ON`` is not the
    SQLite default and must be set per-connection.
    """
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def run_migrations(conn: sqlite3.Connection, migrations_dir: Path | None = None) -> list[int]:
    """Apply every numbered ``.sql`` file in ``migrations_dir`` not yet recorded.

    Idempotent: a version already present in ``schema_migrations`` is skipped, and
    every migration's DDL is written defensively (``IF NOT EXISTS``) so re-running the
    whole set twice is always safe. Returns the list of newly-applied version numbers.
    """
    mdir = migrations_dir or MIGRATIONS_DIR
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    newly_applied: list[int] = []
    for path in sorted(mdir.glob("*.sql")):
        match = _VERSION_RE.match(path.name)
        if not match:
            continue
        version = int(match.group(1))
        if version in applied:
            continue
        sql = path.read_text(encoding="utf-8")
        conn.executescript(sql)
        conn.execute(
            f"INSERT INTO schema_migrations (version, applied_at) VALUES (?, {SQL_NOW})",
            (version,),
        )
        conn.commit()
        newly_applied.append(version)
    return newly_applied
