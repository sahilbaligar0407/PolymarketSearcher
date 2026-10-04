"""Replay a recorded day through the strategy tournament.

The daemon writes every Kalshi order-book snapshot it observes to
``data/parquet/books/date=YYYY-MM-DD``. This module feeds one such day back through the
*same* runner, risk gateway and :class:`~marketlab.execution.paper_broker.PaperBroker`
the live daemon uses, on a :class:`~marketlab.clock.SimulatedClock`, into a throwaway
database under ``data/replay/``.

No look-ahead: events are ordered by when MarketLab *observed* them (the snapshot's own
fetch timestamp, and a settlement's ``first_seen_time``), and a strategy only ever sees
the book that existed at that instant. What replay cannot reproduce: AI sleeves (model
calls are not recorded, so they are excluded), and Polymarket-driven signals (only the
Kalshi books are replayed), so copy-trading and cross-venue arms sit idle in a replay.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from marketlab.clock import SimulatedClock
from marketlab.core.broker import Mode
from marketlab.core.events import BookUpdateEvent, MarketUpdateEvent, SettlementEvent
from marketlab.core.instruments import (
    BookLevel,
    Category,
    MarketStatus,
    NormalizedMarket,
    OrderBook,
    Side,
    Venue,
)
from marketlab.logging import get_logger

log = get_logger(__name__)


@dataclass
class ReplayResult:
    day: str
    db_path: Path
    events: int
    book_snapshots: int
    settlements: int
    sleeves: int
    leaderboard: list[dict[str, Any]]


def available_dates(parquet_dir: Path) -> list[str]:
    root = parquet_dir / "books"
    if not root.exists():
        return []
    return sorted(p.name.removeprefix("date=") for p in root.iterdir() if p.is_dir() and p.name.startswith("date="))


def _category(value: str | None) -> Category:
    try:
        return Category(value or "other")
    except ValueError:
        return Category.OTHER


def load_markets(parquet_dir: Path, day: date) -> dict[str, NormalizedMarket]:
    """Latest known metadata for every Kalshi market first seen on or before ``day``."""
    import polars as pl

    root = parquet_dir / "market_metadata"
    parts = [p for p in root.glob("date=*") if p.name.removeprefix("date=") <= day.isoformat()]
    if not parts:
        return {}
    frame = (
        pl.scan_parquet([str(p / "*.parquet") for p in parts])
        .filter(pl.col("venue") == "kalshi")
        .sort("first_seen_time")
        .group_by("canonical_id")
        .last()
        .collect()
    )
    out: dict[str, NormalizedMarket] = {}
    for row in frame.iter_rows(named=True):
        cid = row["canonical_id"]
        out[cid] = NormalizedMarket(
            canonical_id=cid,
            venue=Venue.KALSHI,
            venue_market_id=row["venue_market_id"] or cid.split(":", 1)[-1].upper(),
            event_id=(row["venue_market_id"] or "").rsplit("-", 1)[0],
            title=row["title"] or "",
            category=_category(row.get("category")),
            status=MarketStatus.OPEN,
            open_time=row.get("open_time"),
            close_time=row.get("close_time"),
            tick_size=Decimal(row.get("tick_size_exact") or "0.01"),
        )
    return out


def iter_books(parquet_dir: Path, day: date, markets: set[str]) -> Iterator[OrderBook]:
    """Reconstruct order-book snapshots from the level rows, in observation order."""
    import polars as pl

    frame = (
        pl.scan_parquet(str(parquet_dir / "books" / f"date={day.isoformat()}" / "*.parquet"))
        .filter((pl.col("venue") == "kalshi") & pl.col("canonical_id").is_in(list(markets)))
        .select("canonical_id", "timestamp", "level", "bid_price_exact", "bid_size", "ask_price_exact", "ask_size")
        .sort("timestamp", "canonical_id", "level")
        .collect()
    )
    key: tuple[str, datetime] | None = None
    bids: list[BookLevel] = []
    asks: list[BookLevel] = []

    def emit() -> OrderBook | None:
        if key is None:
            return None
        return OrderBook(
            canonical_id=key[0], venue=Venue.KALSHI, timestamp=key[1],
            bids=tuple(sorted(bids, key=lambda lv: lv.price, reverse=True)),
            asks=tuple(sorted(asks, key=lambda lv: lv.price)),
        )

    for cid, ts, _level, bid_px, bid_sz, ask_px, ask_sz in frame.iter_rows():
        if key != (cid, ts):
            book = emit()
            if book is not None:
                yield book
            key, bids, asks = (cid, ts), [], []
        if bid_px is not None and bid_sz:
            bids.append(BookLevel(price=Decimal(bid_px), size=int(bid_sz)))
        if ask_px is not None and ask_sz:
            asks.append(BookLevel(price=Decimal(ask_px), size=int(ask_sz)))
    book = emit()
    if book is not None:
        yield book


def load_settlements(db_path: Path, day: date, markets: set[str]) -> list[SettlementEvent]:
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    end = start + timedelta(days=1)
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=30)
        rows = conn.execute(
            "SELECT canonical_id, winning_side, voided, first_seen_time FROM settlements "
            "WHERE first_seen_time >= ? AND first_seen_time < ?",
            (start.isoformat(), end.isoformat()),
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return []
    out = []
    for cid, side, voided, seen in rows:
        if cid not in markets:
            continue
        seen_at = datetime.fromisoformat(seen)
        winning = None if voided or side is None else Side(side)
        out.append(SettlementEvent(
            event_time=seen_at, first_seen_time=seen_at, source="replay",
            canonical_id=cid, venue=Venue.KALSHI, winning_side=winning,
        ))
    return out


async def run_replay(
    settings: Any,
    day_str: str,
    *,
    strategies: set[str] | None = None,
    tick_seconds: float = 10.0,
    out_dir: Path | None = None,
) -> ReplayResult:
    from marketlab.daemon.registry import BookRegistry, MarketRegistry, PortfolioRegistry
    from marketlab.daemon.supervisor import _PaperBrokerStoreBridge
    from marketlab.execution.fill_models import build_limit_fill_model
    from marketlab.execution.latency import LatencyModel
    from marketlab.execution.paper_broker import KalshiFeeCalculator, PaperBroker
    from marketlab.execution.risk_gateway import RiskGateway
    from marketlab.experiments.runner import ExperimentRunner
    from marketlab.experiments.sweep import generate_variants
    from marketlab.storage.state import StateStore

    day = date.fromisoformat(day_str)
    parquet_dir = Path(settings.parquet_dir)
    markets = load_markets(parquet_dir, day)
    if not markets:
        raise ValueError(f"no recorded Kalshi market metadata on or before {day_str}")

    out_dir = out_dir or Path(settings.data_dir) / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / f"replay_{day_str}.db"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)

    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    clock = SimulatedClock(start)
    store = StateStore.open(db_path)
    market_registry = MarketRegistry(max_tracked=len(markets) + 10, universes_cfg=settings.universes)
    for m in markets.values():
        market_registry.upsert(m)
    book_registry = BookRegistry()
    portfolios = PortfolioRegistry(default_bankroll=settings.paper.bankroll_per_strategy)
    broker = PaperBroker(
        mode=Mode.PAPER,
        clock=clock,
        latency_model=LatencyModel(
            signal_to_order_ms=settings.execution.signal_to_order_ms,
            network_latency_ms=settings.execution.network_latency_ms,
            processing_latency_ms=settings.execution.processing_latency_ms,
        ),
        limit_fill_model=build_limit_fill_model(settings.execution.limit_fill_model),
        fee_calculator=KalshiFeeCalculator(),
        book_provider=book_registry.get,
        market_provider=market_registry.get,
        portfolio_provider=portfolios.get_or_create,
        # Replay must not trip over a kill switch engaged on the live system.
        risk_gateway=RiskGateway(settings.risk, kill_switch_path=out_dir / "NO_KILL_SWITCH_IN_REPLAY"),
        settings=settings,
        store=_PaperBrokerStoreBridge(store),
    )
    runner = ExperimentRunner(
        settings, clock, store, broker, market_registry, book_registry,
        portfolio_registry=portfolios,
    )
    variants = [
        v for v in generate_variants(settings.strategies, settings.universes, ai_enabled=False)
        if strategies is None or v.strategy_name in strategies
    ]
    await runner.load_or_create_sleeves(variants)

    for m in markets.values():
        await runner.dispatch(MarketUpdateEvent(event_time=start, first_seen_time=start, source="replay", market=m))

    settlements = sorted(load_settlements(Path(settings.db_path), day, set(markets)), key=lambda e: e.first_seen_time)
    s_idx = 0
    n_books = n_events = 0
    next_tick = start
    for book in iter_books(parquet_dir, day, set(markets)):
        while s_idx < len(settlements) and settlements[s_idx].first_seen_time <= book.timestamp:
            ev = settlements[s_idx]
            if ev.first_seen_time > clock.now():
                clock.advance((ev.first_seen_time - clock.now()).total_seconds())
            for p in portfolios.all().values():
                if any(k.startswith(ev.canonical_id) for k in p.positions):
                    p.settle(ev.canonical_id, ev.winning_side)
            await runner.dispatch(ev)
            s_idx += 1
            n_events += 1
        if book.timestamp > clock.now():
            clock.advance((book.timestamp - clock.now()).total_seconds())
        book_registry.upsert(book)
        await broker.on_book_update(book)
        await runner.dispatch(BookUpdateEvent(event_time=book.timestamp, first_seen_time=book.timestamp, source="replay", book=book))
        n_books += 1
        n_events += 1
        if clock.now() >= next_tick:
            await runner.tick()
            next_tick = clock.now() + timedelta(seconds=tick_seconds)
    for ev in settlements[s_idx:]:
        await runner.dispatch(ev)
        n_events += 1

    await runner.snapshot()
    board = [
        {
            "experiment_id": r.experiment_id,
            "strategy": r.strategy_name,
            "universe": r.category,
            "equity": float(r.equity),
            "pnl": float(r.net_pnl),
            "trades": r.trade_count,
            "status": str(r.status),
        }
        for r in runner.leaderboard()
    ]
    store.close()
    return ReplayResult(day_str, db_path, n_events, n_books, len(settlements), len(runner._sleeves), board)


__all__ = ["ReplayResult", "available_dates", "iter_books", "load_markets", "run_replay"]
