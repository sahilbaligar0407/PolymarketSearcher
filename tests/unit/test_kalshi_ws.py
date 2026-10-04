"""Kalshi WS book reconstruction: seq is per subscription (sid), not per ticker."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal

from marketlab.adapters.kalshi.auth import KalshiAuth
from marketlab.adapters.kalshi.ws import KalshiWebSocketAdapter
from marketlab.settings import load_settings


class _FakeWs:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def _adapter(min_book_interval: float = 0.0) -> tuple[KalshiWebSocketAdapter, _FakeWs, asyncio.Queue]:
    queue: asyncio.Queue = asyncio.Queue()
    adapter = KalshiWebSocketAdapter(
        load_settings(), queue, auth=KalshiAuth(None, None), tickers=["A", "B"],
        min_book_interval=min_book_interval,
    )
    ws = _FakeWs()
    adapter._ws = ws  # type: ignore[assignment]
    return adapter, ws, queue


def _snapshot(sid: int, seq: int, ticker: str) -> dict:
    return {
        "type": "orderbook_snapshot",
        "sid": sid,
        "seq": seq,
        "msg": {"market_ticker": ticker, "yes": [[40, 10]], "no": [[55, 10]]},
    }


def _delta(sid: int, seq: int, ticker: str, delta: int = 5) -> dict:
    return {
        "type": "orderbook_delta",
        "sid": sid,
        "seq": seq,
        "msg": {"market_ticker": ticker, "price": 40, "delta": delta, "side": "yes"},
    }


async def test_interleaved_tickers_on_one_sid_are_not_gaps() -> None:
    adapter, ws, queue = _adapter()
    for msg in [
        _snapshot(1, 1, "A"),
        _snapshot(1, 2, "B"),
        _delta(1, 3, "A"),
        _delta(1, 4, "B"),
        _delta(1, 5, "A"),
    ]:
        await adapter._handle_message(msg)

    assert ws.sent == []
    assert queue.qsize() == 5
    assert adapter._books["A"].yes[min(adapter._books["A"].yes)] == 20


async def test_gap_resyncs_sid_once_and_drops_stale_messages() -> None:
    adapter, ws, queue = _adapter()
    await adapter._handle_message(_snapshot(1, 1, "A"))
    await adapter._handle_message(_snapshot(1, 2, "B"))

    await adapter._handle_message(_delta(1, 4, "A"))  # seq 3 lost
    for seq in range(5, 30):  # the old sid keeps streaming until Kalshi processes the unsub
        await adapter._handle_message(_delta(1, seq, "B"))

    assert ws.sent[0]["cmd"] == "unsubscribe"
    assert ws.sent[0]["params"] == {"sids": [1]}
    subs = [m for m in ws.sent if m["cmd"] == "subscribe"]
    assert len(subs) == 1
    assert subs[0]["params"]["market_tickers"] == ["A", "B"]
    assert adapter._books == {}
    assert queue.qsize() == 2  # only the two original snapshots

    # Fresh snapshots arrive on a new sid and the books resume.
    await adapter._handle_message(_snapshot(7, 1, "A"))
    await adapter._handle_message(_snapshot(7, 2, "B"))
    await adapter._handle_message(_delta(7, 3, "A"))
    assert adapter._resyncing == set()
    assert queue.qsize() == 5
    assert len(ws.sent) == 2


async def test_delta_before_snapshot_resubscribes_once() -> None:
    adapter, ws, _ = _adapter()
    for seq in range(1, 20):
        await adapter._handle_message(_delta(3, seq, "A"))

    subs = [m for m in ws.sent if m["cmd"] == "subscribe"]
    assert len(subs) == 1
    assert subs[0]["params"]["market_tickers"] == ["A"]


async def test_current_kalshi_field_names_build_a_real_book() -> None:
    """Shapes captured from the production feed on 2026-10-04."""
    adapter, ws, queue = _adapter()
    await adapter._handle_message({
        "type": "orderbook_snapshot", "sid": 1, "seq": 1,
        "msg": {
            "market_ticker": "A",
            "yes_dollars_fp": [["0.4400", "13815.66"], ["0.4500", "65.00"]],
            "no_dollars_fp": [["0.5300", "10.40"]],
        },
    })
    await adapter._handle_message({
        "type": "orderbook_delta", "sid": 1, "seq": 2,
        "msg": {"market_ticker": "A", "price_dollars": "0.4500", "delta_fp": "-33.68", "side": "yes"},
    })
    await adapter._handle_message({
        "type": "orderbook_delta", "sid": 1, "seq": 3,
        "msg": {"market_ticker": "A", "price_dollars": "0.4500", "delta_fp": "-31.32", "side": "yes"},
    })

    assert ws.sent == []
    book = None
    while not queue.empty():
        book = (await queue.get()).book
    assert [(lvl.price, lvl.size) for lvl in book.bids] == [(Decimal("0.44"), 13816)]
    assert [(lvl.price, lvl.size) for lvl in book.asks] == [(Decimal("0.47"), 10)]


async def test_lifecycle_v2_maps_event_types_and_ignores_non_status_events() -> None:
    adapter, _, queue = _adapter()
    await adapter._handle_message({
        "type": "market_lifecycle_v2", "sid": 4, "seq": 1,
        "msg": {"market_ticker": "A", "close_ts": 1791145007, "event_type": "close_date_updated"},
    })
    assert queue.empty()
    await adapter._handle_message({
        "type": "market_lifecycle_v2", "sid": 4, "seq": 2,
        "msg": {"market_ticker": "A", "result": "yes", "event_type": "determined"},
    })
    assert (await queue.get()).status == "closed"


async def test_command_acks_consume_seq_without_looking_like_a_gap() -> None:
    adapter, ws, queue = _adapter()
    await adapter._handle_message({"type": "subscribed", "id": 1, "msg": {"channel": "orderbook_delta", "sid": 1}})
    await adapter._handle_message(_snapshot(1, 1, "A"))
    await adapter._handle_message({"type": "ok", "id": 5, "sid": 1, "seq": 2, "msg": {"market_tickers": ["A"]}})
    await adapter._handle_message(_delta(1, 3, "A"))
    assert ws.sent == []
    assert queue.qsize() == 2


async def test_gap_on_a_trade_stream_does_not_touch_books() -> None:
    adapter, ws, _ = _adapter()
    await adapter._handle_message(_snapshot(1, 1, "A"))
    trade = {"trade_id": "t", "market_ticker": "A", "yes_price_dollars": "0.49", "count_fp": "2.00",
             "taker_side": "yes", "ts": 1791145065}
    await adapter._handle_message({"type": "trade", "sid": 3, "seq": 1, "msg": trade})
    await adapter._handle_message({"type": "trade", "sid": 3, "seq": 5, "msg": trade})
    await adapter._handle_message({"type": "trade", "sid": 3, "seq": 6, "msg": trade})
    assert ws.sent == []
    assert "A" in adapter._books


async def test_bursts_of_deltas_are_coalesced_into_one_book_per_interval() -> None:
    adapter, _, queue = _adapter(min_book_interval=60.0)
    await adapter._handle_message(_snapshot(1, 1, "A"))
    for seq in range(2, 52):
        await adapter._handle_message(_delta(1, seq, "A", delta=1))
    assert queue.qsize() == 1  # the snapshot; 50 deltas are pending
    assert adapter._dirty == {"A"}

    adapter._min_book_interval = 0.0
    await adapter._flush_dirty()
    assert queue.qsize() == 2
    await queue.get()
    latest = (await queue.get()).book
    assert latest.bids[0].size == 10 + 50  # every delta was applied, only the emit was folded
