"""Kalshi websocket adapter: live book/trade/lifecycle feed with local book reconstruction.

Connects to ``settings.sources.kalshi_ws`` (``/trade-api/ws/v2``) and subscribes to
``orderbook_delta``, ``ticker``, ``trade`` and ``market_lifecycle_v2`` for a caller-managed
set of tickers. A local order book is rebuilt per ticker from snapshot + deltas with
sequence-number gap detection. Kalshi numbers ``seq`` per subscription (``sid``), and one
subscription spans many tickers, so gaps are tracked per sid: a gap triggers one WARN log,
an unsubscribe of that sid and a single resubscribe of its tickers (which makes Kalshi push
fresh snapshots) rather than silently serving a stale/incorrect book.

Normalized events (``BookUpdateEvent``, ``TradeEvent``, ``MarketStatusEvent``) are pushed
onto an ``asyncio.Queue`` supplied by the caller - this module never assumes anything
about how the rest of the engine consumes them.

This adapter must never crash the process: connection failures, auth failures, and
malformed messages are all logged and retried/skipped, never raised out of the
background task.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal

import websockets

from marketlab.adapters.base import Adapter, SourceHealth, SourceStatus
from marketlab.adapters.kalshi.auth import KalshiAuth
from marketlab.adapters.kalshi.normalize import (
    _map_status,  # shared status table; both modules are owned by this team.
    make_canonical_id,
    normalize_trade,
)
from marketlab.clock import Clock, LiveClock
from marketlab.core.events import BookUpdateEvent, MarketStatusEvent, TradeEvent
from marketlab.core.instruments import (
    ONE,
    BookLevel,
    MarketStatus,
    OrderBook,
    Venue,
    cents_to_probability,
    to_probability,
)
from marketlab.logging import get_logger
from marketlab.settings import Settings

log = get_logger(__name__)

#: The subscription channels this adapter always asks for.
#: ``market_lifecycle`` (v1) is rejected with "Unknown channel name" (checked 2026-10-04).
_CHANNELS: tuple[str, ...] = ("orderbook_delta", "ticker", "trade", "market_lifecycle_v2")

#: ``market_lifecycle_v2`` ``event_type`` values that change whether a market can trade.
_LIFECYCLE_V2_STATUS: dict[str, MarketStatus] = {
    "activated": MarketStatus.OPEN,
    "deactivated": MarketStatus.PAUSED,
    "determined": MarketStatus.CLOSED,
    "settled": MarketStatus.SETTLED,
}

#: Kalshi caps the number of market tickers per subscribe message; chunk conservatively.
_MAX_TICKERS_PER_MSG = 100

#: WS auth is signed against this fixed path regardless of which channels are requested.
_WS_SIGNING_PATH = "/trade-api/ws/v2"


def _parse_price(value: object) -> Decimal:
    """Kalshi WS prices may arrive as legacy integer cents or modern dollar strings."""
    if isinstance(value, int):
        return cents_to_probability(value)
    return to_probability(Decimal(str(value)))


def _parse_qty(value: object) -> Decimal:
    """Kalshi now sends fractional contract counts as strings (``delta_fp: "-33.68"``)."""
    return Decimal(str(value))


def _parse_size(value: object) -> int:
    return int(Decimal(str(value)).to_integral_value(rounding=ROUND_HALF_UP))


def _parse_level_pair(pair: list) -> tuple[Decimal, Decimal]:
    return _parse_price(pair[0]), _parse_qty(pair[1])


def _first_present(body: dict, *keys: str) -> object:
    """Kalshi renamed its WS fields (``yes`` -> ``yes_dollars_fp``, ``price`` ->
    ``price_dollars``, ``delta`` -> ``delta_fp``); accept the legacy and current names."""
    for key in keys:
        if body.get(key) is not None:
            return body[key]
    return None


def _ws_trade_to_raw(body: dict) -> dict:
    """Adapt a WS trade payload to the REST-trade shape ``normalize_trade`` expects."""
    raw = dict(body)
    raw.setdefault("ticker", body.get("market_ticker") or body.get("ticker"))
    if "created_time" not in raw:
        ts = body.get("ts") or body.get("timestamp")
        if isinstance(ts, int | float):
            raw["created_time"] = datetime.fromtimestamp(ts, tz=UTC).isoformat()
    return raw


class _LocalBook:
    """One ticker's reconstructed book: raw YES/NO bid ladders, the subscription (sid)
    feeding it, and the last applied seq on that sid. Sizes are kept exact (Kalshi
    sends fractional counts) and rounded to whole contracts only on output."""

    __slots__ = ("yes", "no", "seq", "sid")

    def __init__(self) -> None:
        self.yes: dict[Decimal, Decimal] = {}
        self.no: dict[Decimal, Decimal] = {}
        self.seq: int | None = None
        self.sid: int | None = None

    def reset(self, yes_levels: list, no_levels: list, seq: int | None, sid: int | None) -> None:
        self.yes = {}
        self.no = {}
        for pair in yes_levels or []:
            p, s = _parse_level_pair(pair)
            if s > 0:
                self.yes[p] = s
        for pair in no_levels or []:
            p, s = _parse_level_pair(pair)
            if s > 0:
                self.no[p] = s
        self.seq = seq
        self.sid = sid

    def apply_delta(self, side: str, price_raw: object, delta: Decimal, seq: int | None) -> None:
        price = _parse_price(price_raw)
        side_book = self.yes if side == "yes" else self.no
        new_size = side_book.get(price, Decimal(0)) + delta
        if new_size <= 0:
            side_book.pop(price, None)
        else:
            side_book[price] = new_size
        self.seq = seq

    def to_orderbook(self, canonical_id: str, timestamp: datetime) -> OrderBook:
        bids = tuple(
            sorted(
                (BookLevel(price=p, size=_parse_size(s)) for p, s in self.yes.items()
                 if _parse_size(s) > 0),
                key=lambda lvl: lvl.price,
                reverse=True,
            )
        )
        asks = tuple(
            sorted(
                (BookLevel(price=to_probability(ONE - p), size=_parse_size(s))
                 for p, s in self.no.items() if _parse_size(s) > 0),
                key=lambda lvl: lvl.price,
            )
        )
        return OrderBook(
            canonical_id=canonical_id,
            venue=Venue.KALSHI,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            sequence=self.seq,
        )


class KalshiWebSocketAdapter(Adapter):
    """Maintains a live Kalshi websocket connection and reconstructed order books."""

    name = "kalshi_ws"

    def __init__(
        self,
        settings: Settings,
        out_queue: asyncio.Queue,
        *,
        clock: Clock | None = None,
        auth: KalshiAuth | None = None,
        tickers: Iterable[str] | None = None,
        max_queue_wait: float = 1.0,
        min_book_interval: float = 1.0,
    ) -> None:
        self._settings = settings
        self._queue = out_queue
        self._clock = clock or LiveClock()
        env = (settings.secrets.kalshi_environment or "production").strip().lower()
        self._ws_url = settings.sources.kalshi_demo_ws if env == "demo" else settings.sources.kalshi_ws
        self._auth = auth or KalshiAuth(
            settings.secrets.kalshi_api_key_id,
            settings.secrets.kalshi_private_key_path,
            environment=env,
        )
        self._max_queue_wait = max_queue_wait
        #: Emit at most one book per ticker per this many seconds. A live game can send
        #: dozens of deltas a second per market, and every emitted book fans out to every
        #: sleeve on that universe; deltas in between are folded into the next emit.
        self._min_book_interval = min_book_interval
        self._last_emit: dict[str, float] = {}
        self._dirty: set[str] = set()
        self._last_flush_scan = 0.0

        self._tickers: set[str] = set(tickers or ())
        self._books: dict[str, _LocalBook] = {}
        #: Last seq seen per subscription id; Kalshi's seq is per sid, not per ticker.
        self._sid_seq: dict[int, int] = {}
        #: Subscriptions abandoned after a gap; their in-flight messages are ignored.
        self._dead_sids: set[int] = set()
        #: Tickers with a resubscribe in flight, so each one is resubscribed only once.
        self._resyncing: set[str] = set()
        #: sids carrying orderbook_delta; only a gap on one of these invalidates books.
        self._book_sids: set[int] = set()

        self._status = SourceStatus.DISABLED
        self._last_message_at: datetime | None = None
        self._error_count = 0
        self._reconnect_count = 0
        self._detail = ""

        self._ws: websockets.ClientConnection | None = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._cmd_id = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin the background connect/read/reconnect loop, if not already running."""
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.ensure_future(self._run_forever())

    async def close(self) -> None:
        self._stop.set()
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def health(self) -> SourceHealth:
        return SourceHealth(
            name=self.name,
            status=self._status,
            last_message_at=self._last_message_at,
            error_count=self._error_count,
            reconnect_count=self._reconnect_count,
            detail=self._detail,
        )

    async def probe(self) -> SourceHealth:
        """If already running, report live health; otherwise try one short connection."""
        if self._task is not None and not self._task.done():
            return self.health()
        try:
            async with websockets.connect(
                self._ws_url, additional_headers=self._connect_headers(), open_timeout=5
            ):
                pass
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            text = str(exc)
            status = (
                SourceStatus.NO_CREDENTIALS
                if not self._auth.is_configured and ("401" in text or "403" in text)
                else SourceStatus.DOWN
            )
            return SourceHealth(name=self.name, status=status, error_count=1, detail=text)
        if not self._auth.is_configured:
            return SourceHealth(
                name=self.name,
                status=SourceStatus.NO_CREDENTIALS,
                detail="ws reachable; no credentials for authenticated channels",
            )
        return SourceHealth(name=self.name, status=SourceStatus.HEALTHY, detail="ws reachable")

    # ------------------------------------------------------------------
    # Subscription management (dynamic add/remove of tickers)
    # ------------------------------------------------------------------

    async def subscribe(self, tickers: Iterable[str]) -> None:
        new = set(tickers) - self._tickers
        self._tickers |= new
        if self._ws is not None and new:
            await self._send_subscriptions(sorted(new))

    async def unsubscribe(self, tickers: Iterable[str]) -> None:
        remove = set(tickers) & self._tickers
        self._tickers -= remove
        for t in remove:
            self._books.pop(t, None)
            self._resyncing.discard(t)
        if self._ws is not None and remove:
            await self._send(
                {
                    "id": self._next_id(),
                    "cmd": "unsubscribe",
                    "params": {"channels": list(_CHANNELS), "market_tickers": sorted(remove)},
                }
            )

    # ------------------------------------------------------------------
    # Connection loop
    # ------------------------------------------------------------------

    def _connect_headers(self) -> dict[str, str]:
        if not self._auth.is_configured:
            return {}
        return self._auth.headers("GET", _WS_SIGNING_PATH, base_url=self._ws_url)

    async def _run_forever(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - never let the feed crash the engine
                self._error_count += 1
                text = str(exc)
                self._status = (
                    SourceStatus.NO_CREDENTIALS
                    if not self._auth.is_configured and ("401" in text or "403" in text)
                    else SourceStatus.DOWN
                )
                self._detail = text
                log.warning("kalshi_ws_error", error=text, reconnect_count=self._reconnect_count)
            self._ws = None
            if self._stop.is_set():
                return
            self._reconnect_count += 1
            jitter = random.uniform(0, backoff * 0.25)
            sleep_for = min(backoff + jitter, 60.0)
            log.warning(
                "kalshi_ws_reconnecting",
                sleep_seconds=round(sleep_for, 1),
                reconnect_count=self._reconnect_count,
            )
            await self._clock.sleep(sleep_for)
            backoff = min(backoff * 2, 60.0)

    async def _run_once(self) -> None:
        async with websockets.connect(
            self._ws_url, additional_headers=self._connect_headers(), open_timeout=10
        ) as ws:
            self._ws = ws
            # sids and seqs are per connection; every book is rebuilt from fresh snapshots.
            self._books.clear()
            self._sid_seq.clear()
            self._dead_sids.clear()
            self._resyncing.clear()
            self._book_sids.clear()
            self._status = SourceStatus.HEALTHY
            self._detail = ""
            if self._tickers:
                await self._send_subscriptions(sorted(self._tickers))
            async for raw_msg in ws:
                if self._stop.is_set():
                    break
                self._last_message_at = self._clock.now()
                try:
                    msg = json.loads(raw_msg)
                except json.JSONDecodeError:
                    log.warning("kalshi_ws_bad_json", raw=raw_msg[:200])
                    continue
                await self._handle_message(msg)
                await self._flush_dirty()

    def _next_id(self) -> int:
        self._cmd_id += 1
        return self._cmd_id

    async def _send(self, payload: dict) -> None:
        if self._ws is None:
            return
        await self._ws.send(json.dumps(payload))

    async def _send_subscriptions(self, tickers: list[str]) -> None:
        for i in range(0, len(tickers), _MAX_TICKERS_PER_MSG):
            chunk = tickers[i : i + _MAX_TICKERS_PER_MSG]
            await self._send(
                {
                    "id": self._next_id(),
                    "cmd": "subscribe",
                    "params": {"channels": list(_CHANNELS), "market_tickers": chunk},
                }
            )

    def _sequence_ok(self, sid: int | None, seq: int | None) -> bool:
        """Advance ``sid``'s seq counter; False on a gap (counter left untouched)."""
        if sid is None or seq is None:
            return True
        last = self._sid_seq.get(sid)
        if last is not None and seq != last + 1:
            return False
        self._sid_seq[sid] = seq
        return True

    async def _resync_sid(self, sid: int | None, ticker: str | None) -> None:
        """Abandon subscription ``sid`` and resubscribe every ticker it was feeding, which
        makes Kalshi push fresh ``orderbook_snapshot`` messages on a new sid."""
        tickers = {t for t, b in self._books.items() if sid is not None and b.sid == sid}
        if ticker:
            tickers.add(ticker)
        tickers -= self._resyncing
        if sid is not None:
            self._dead_sids.add(sid)
            self._sid_seq.pop(sid, None)
            await self._send(
                {"id": self._next_id(), "cmd": "unsubscribe", "params": {"sids": [sid]}}
            )
        if not tickers:
            return
        for t in tickers:
            self._books.pop(t, None)
        self._resyncing |= tickers
        ordered = sorted(tickers)
        for i in range(0, len(ordered), _MAX_TICKERS_PER_MSG):
            await self._send(
                {
                    "id": self._next_id(),
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["orderbook_delta"],
                        "market_tickers": ordered[i : i + _MAX_TICKERS_PER_MSG],
                    },
                }
            )

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_message(self, msg: dict) -> None:
        msg_type = msg.get("type")
        sid = msg.get("sid")
        body = msg.get("msg") or {}
        if msg_type == "subscribed" and body.get("channel") == "orderbook_delta":
            self._book_sids.add(body.get("sid"))
        if msg_type in ("orderbook_snapshot", "orderbook_delta") and sid is not None:
            self._book_sids.add(sid)
        if sid is not None and sid in self._dead_sids:
            return
        # Every message on a sid - including command acks ("ok", "unsubscribed") - takes
        # a seq number, so the counter must advance on all of them, not just book data.
        if not self._sequence_ok(sid, msg.get("seq")):
            expected = self._sid_seq[sid] + 1
            if sid in self._book_sids:
                log.warning("kalshi_ws_sequence_gap", sid=sid, expected=expected, got=msg.get("seq"))
                await self._resync_sid(sid, body.get("market_ticker"))
                return
            # A trade/ticker/lifecycle stream has no state to rebuild: note it and move on.
            log.info("kalshi_ws_sequence_gap_stream", sid=sid, expected=expected, got=msg.get("seq"))
            self._sid_seq[sid] = msg["seq"]
        if msg_type in ("subscribed", "ok", "unsubscribed"):
            log.debug("kalshi_ws_ack", detail=msg)
            return
        if msg_type == "error":
            self._error_count += 1
            log.warning("kalshi_ws_error_message", detail=msg)
            return
        if msg_type in ("orderbook_snapshot", "orderbook_delta"):
            if msg_type == "orderbook_snapshot":
                await self._handle_snapshot(msg)
            else:
                await self._handle_delta(msg)
            return
        if msg_type == "trade":
            await self._handle_trade(msg)
            return
        if msg_type == "market_lifecycle" or msg_type == "market_lifecycle_v2":
            await self._handle_lifecycle(msg)
            return
        # Unrecognized/irrelevant message types (e.g. "ticker" summary quotes, "pong")
        # are logged at debug volume only; they don't map to a required event type.

    async def _handle_snapshot(self, msg: dict) -> None:
        body = msg.get("msg", {})
        ticker = body.get("market_ticker")
        if not ticker:
            return
        book = self._books.setdefault(ticker, _LocalBook())
        book.reset(
            _first_present(body, "yes_dollars_fp", "yes_dollars", "yes") or [],
            _first_present(body, "no_dollars_fp", "no_dollars", "no") or [],
            msg.get("seq"),
            msg.get("sid"),
        )
        self._resyncing.discard(ticker)
        await self._emit_book_update(ticker, book)

    async def _handle_delta(self, msg: dict) -> None:
        body = msg.get("msg", {})
        ticker = body.get("market_ticker")
        if not ticker:
            return
        sid = msg.get("sid")
        book = self._books.get(ticker)
        if book is None or book.sid != sid:
            if ticker in self._resyncing:
                return  # fresh snapshot already requested; drop deltas until it lands
            log.warning("kalshi_ws_delta_before_snapshot", ticker=ticker, sid=sid)
            await self._resync_sid(sid, ticker)
            return
        side = body.get("side", "")
        price = _first_present(body, "price_dollars", "price")
        delta = _first_present(body, "delta_fp", "delta")
        if price is None or delta is None:
            log.warning("kalshi_ws_bad_delta", ticker=ticker, body=body)
            return
        book.apply_delta(side, price, _parse_qty(delta), msg.get("seq"))
        await self._book_changed(ticker, book)

    async def _handle_trade(self, msg: dict) -> None:
        body = msg.get("msg", {})
        try:
            trade = normalize_trade(_ws_trade_to_raw(body))
        except (KeyError, ValueError) as exc:
            log.warning("kalshi_ws_bad_trade", error=str(exc), body=body)
            return
        now = self._clock.now()
        event = TradeEvent(
            event_time=trade.timestamp,
            first_seen_time=now,
            source=self.name,
            trade=trade,
        )
        await self._put(event)

    async def _handle_lifecycle(self, msg: dict) -> None:
        body = msg.get("msg", {})
        ticker = body.get("market_ticker")
        if not ticker:
            return
        event_type = str(body.get("event_type") or "").lower()
        if event_type:
            # market_lifecycle_v2: only these change tradability; close_date_updated,
            # metadata_updated, created etc. are not status changes.
            status = _LIFECYCLE_V2_STATUS.get(event_type)
            if status is None:
                return
        else:
            status = _map_status(body.get("status") or body.get("event") or "")
        now = self._clock.now()
        event = MarketStatusEvent(
            event_time=now,
            first_seen_time=now,
            source=self.name,
            canonical_id=make_canonical_id(Venue.KALSHI, ticker),
            venue=Venue.KALSHI,
            status=status,
        )
        await self._put(event)

    async def _book_changed(self, ticker: str, book: _LocalBook) -> None:
        last = self._last_emit.get(ticker)
        if last is None or time.monotonic() - last >= self._min_book_interval:
            await self._emit_book_update(ticker, book)
        else:
            self._dirty.add(ticker)

    async def _flush_dirty(self) -> None:
        """Emit coalesced books whose interval has passed (checked a few times a second)."""
        if not self._dirty:
            return
        now = time.monotonic()
        if now - self._last_flush_scan < min(0.25, self._min_book_interval):
            return
        self._last_flush_scan = now
        for ticker in list(self._dirty):
            book = self._books.get(ticker)
            if book is None:
                self._dirty.discard(ticker)
            elif now - self._last_emit.get(ticker, 0.0) >= self._min_book_interval:
                await self._emit_book_update(ticker, book)

    async def _emit_book_update(self, ticker: str, book: _LocalBook) -> None:
        self._last_emit[ticker] = time.monotonic()
        self._dirty.discard(ticker)
        now = self._clock.now()
        ob = book.to_orderbook(make_canonical_id(Venue.KALSHI, ticker), now)
        event = BookUpdateEvent(event_time=now, first_seen_time=now, source=self.name, book=ob)
        await self._put(event)

    async def _put(self, event: BookUpdateEvent | TradeEvent | MarketStatusEvent) -> None:
        try:
            await asyncio.wait_for(self._queue.put(event), timeout=self._max_queue_wait)
        except TimeoutError:
            self._error_count += 1
            log.warning("kalshi_ws_queue_full", event_type=event.event_type)
