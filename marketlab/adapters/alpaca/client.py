"""Alpaca Market Data v2 adapter: equity bars, news, and a news websocket.

Probed live on 2026-09-04 without credentials:

* ``GET data.alpaca.markets/v2/stocks/AAPL/bars?timeframe=1Min`` -> HTTP 401
  (``nginx`` 401 page), confirming the auth gate. This adapter checks that both
  ``APCA-API-KEY-ID``/``APCA-API-SECRET-KEY`` are present *before* making any request
  rather than relying on that 401 - ``probe()`` reports ``NO_CREDENTIALS`` directly.

With credentials, ``/v2/stocks/bars`` (multi-symbol), ``/v2/news``, and the
``wss://stream.data.alpaca.markets/v1beta1/news`` websocket are Alpaca's documented,
stable endpoints; not independently exercised here for lack of test credentials.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import websockets

from marketlab.adapters.base import Adapter, HttpAdapter, SourceHealth, SourceStatus
from marketlab.clock import Clock, LiveClock
from marketlab.core.events import NewsEvent, SourceClass
from marketlab.logging import get_logger

log = get_logger(__name__)

_MAX_BACKOFF_SECONDS = 60.0
_NEWS_WS_URL = "wss://stream.data.alpaca.markets/v1beta1/news"


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class AlpacaAdapter(Adapter):
    """Alpaca Market Data client. Optional: degrades cleanly with no API keys."""

    name = "alpaca"

    def __init__(self, *, base_url: str, api_key: str, secret_key: str, clock: Clock | None = None) -> None:
        self._clock = clock or LiveClock()
        self._api_key = (api_key or "").strip()
        self._secret_key = (secret_key or "").strip()
        headers = (
            {"APCA-API-KEY-ID": self._api_key, "APCA-API-SECRET-KEY": self._secret_key}
            if self.has_credentials
            else {}
        )
        self._http = HttpAdapter(base_url, name="alpaca", default_headers=headers, clock=self._clock)
        if not self.has_credentials:
            log.warning(
                "alpaca_no_credentials",
                detail="ALPACA_API_KEY/ALPACA_SECRET_KEY not set; Alpaca sources degrade to empty results.",
            )

    @property
    def has_credentials(self) -> bool:
        return bool(self._api_key and self._secret_key)

    async def probe(self) -> SourceHealth:
        if not self.has_credentials:
            return SourceHealth(name=self.name, status=SourceStatus.NO_CREDENTIALS, last_message_at=self._clock.now())
        try:
            await self._http.get_json("/v2/stocks/bars", params={"symbols": "AAPL", "timeframe": "1Day", "limit": 1})
            return SourceHealth(name=self.name, status=SourceStatus.HEALTHY, last_message_at=self._clock.now())
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            log.warning("alpaca_probe_failed", error=str(exc))
            return SourceHealth(name=self.name, status=SourceStatus.DOWN, detail=str(exc))

    async def close(self) -> None:
        await self._http.close()

    async def get_bars(
        self,
        symbols: list[str],
        timeframe: str = "1Min",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """``symbol -> list of bars``. Empty dict under NO_CREDENTIALS."""
        if not self.has_credentials:
            log.warning("alpaca_get_bars_no_credentials", symbols=symbols)
            return {}
        params: dict[str, Any] = {"symbols": ",".join(symbols), "timeframe": timeframe}
        if start is not None:
            params["start"] = start.isoformat()
        if end is not None:
            params["end"] = end.isoformat()
        if limit is not None:
            params["limit"] = limit
        data = await self._http.get_json("/v2/stocks/bars", params=params)
        return data.get("bars", {}) or {}

    def _news_item_to_event(self, item: dict[str, Any]) -> NewsEvent:
        now = self._clock.now()
        created_at = _parse_iso(item.get("created_at"))
        updated_at = _parse_iso(item.get("updated_at")) or created_at
        return NewsEvent(
            event_time=created_at or now,
            published_time=updated_at,
            first_seen_time=now,
            ingested_time=now,
            source="alpaca_news",
            source_class=SourceClass.SPECIALIST,
            news_id=str(item.get("id", "")),
            title=str(item.get("headline", "")),
            url=str(item.get("url", "")),
            body=str(item.get("summary", "")),
            tickers=tuple(item.get("symbols") or ()),
        )

    async def get_news(
        self,
        symbols: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 50,
    ) -> list[NewsEvent]:
        """``GET /v2/news``. Empty list under NO_CREDENTIALS."""
        if not self.has_credentials:
            log.warning("alpaca_get_news_no_credentials")
            return []
        params: dict[str, Any] = {"limit": limit}
        if symbols:
            params["symbols"] = ",".join(symbols)
        if start is not None:
            params["start"] = start.isoformat()
        if end is not None:
            params["end"] = end.isoformat()
        data = await self._http.get_json("/v2/news", params=params)
        return [self._news_item_to_event(item) for item in data.get("news", []) or []]

    async def stream_news(self, queue: asyncio.Queue[NewsEvent], clock: Clock | None = None) -> None:
        """Authenticate and subscribe to all-symbol news on Alpaca's websocket.

        No-ops (with a WARN) under NO_CREDENTIALS rather than opening a socket that can
        only fail. Reconnects with exponential backoff, like the other stream() methods
        in this team's adapters.
        """
        if not self.has_credentials:
            log.warning("alpaca_stream_news_no_credentials")
            return
        clock = clock or self._clock
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(_NEWS_WS_URL, open_timeout=10) as ws:
                    await ws.send(json.dumps({"action": "auth", "key": self._api_key, "secret": self._secret_key}))
                    await asyncio.wait_for(ws.recv(), timeout=10)  # auth ack; errors surface on next recv()
                    await ws.send(json.dumps({"action": "subscribe", "news": ["*"]}))
                    backoff = 1.0
                    async for raw in ws:
                        try:
                            payload = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        messages = payload if isinstance(payload, list) else [payload]
                        for msg in messages:
                            if not isinstance(msg, dict) or msg.get("T") != "n":
                                continue
                            await queue.put(self._news_item_to_event(msg))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect loop must never die
                log.warning("alpaca_news_ws_reconnect", error=str(exc), backoff_seconds=backoff)
                await clock.sleep(backoff)
                backoff = min(backoff * 2.0, _MAX_BACKOFF_SECONDS)
