"""Bluesky adapter: public Jetstream firehose + optional authenticated search.

Probed live on 2026-09-04:

* ``wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post``
  connects with **no auth**, and immediately streams one JSON message per event, e.g.::

      {"did": "did:plc:...", "time_us": 1788539450385938, "kind": "commit",
       "commit": {"rev": "...", "operation": "create", "collection": "app.bsky.feed.post",
                  "rkey": "...", "record": {"$type": "app.bsky.feed.post",
                  "createdAt": "2026-09-04T16:30:49.497145Z", "text": "...", "langs": ["en"]}}}

  This makes it the reliable, always-available social backbone the PRD wants -
  X requires a paid bearer token and a budget; Bluesky's public firehose requires nothing.

The firehose carries the author's **DID**, not their handle - resolving DID -> handle is
a separate network call (``app.bsky.actor.getProfile``) this module deliberately does not
make inline for every post, since that would turn a firehose into an N+1 problem.
``SocialEvent.author`` is the DID for firehose-sourced events.

Authenticated search (``BLUESKY_HANDLE``/``BLUESKY_APP_PASSWORD`` -> ``createSession`` ->
``app.bsky.feed.searchPosts``) was not independently probed here (no test credentials in
this environment); it follows the documented, stable AT Protocol XRPC contract and
degrades to an empty result with one WARN when credentials are absent.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import websockets

from marketlab.adapters.base import Adapter, HttpAdapter, SourceHealth, SourceStatus
from marketlab.clock import Clock, LiveClock
from marketlab.core.events import SocialEvent, SourceClass
from marketlab.logging import get_logger

log = get_logger(__name__)

_MAX_BACKOFF_SECONDS = 60.0
_BSKY_PDS_BASE = "https://bsky.social"


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _record_to_event(did: str, record: dict[str, Any], rkey: str, clock: Clock) -> SocialEvent | None:
    if record.get("$type") != "app.bsky.feed.post":
        return None
    text = str(record.get("text", ""))
    created_at = _parse_iso(record.get("createdAt"))
    now = clock.now()
    return SocialEvent(
        event_time=created_at or now,
        published_time=created_at,
        first_seen_time=now,
        ingested_time=now,
        source="bluesky_firehose",
        source_class=SourceClass.SOCIAL_UNVERIFIED,
        post_id=rkey,
        platform="bluesky",
        author=did,
        text=text,
        url=f"https://bsky.app/profile/{did}/post/{rkey}" if did and rkey else "",
    )


class BlueskyAdapter(Adapter):
    """Public Jetstream firehose (no credentials) + optional authenticated search."""

    name = "bluesky"

    def __init__(
        self,
        *,
        firehose_url: str,
        handle: str = "",
        app_password: str = "",
        clock: Clock | None = None,
    ) -> None:
        self._clock = clock or LiveClock()
        self._firehose_url = firehose_url
        self._handle = (handle or "").strip()
        self._app_password = (app_password or "").strip()
        self._session_jwt: str | None = None
        self._auth_http = HttpAdapter(
            _BSKY_PDS_BASE, name="bluesky_auth", default_headers={"User-Agent": "MarketLab research"}, clock=self._clock
        )
        if not (self._handle and self._app_password):
            log.warning(
                "bluesky_search_no_credentials",
                detail="BLUESKY_HANDLE/BLUESKY_APP_PASSWORD not set; authenticated search degrades to empty results.",
            )

    @property
    def has_search_credentials(self) -> bool:
        return bool(self._handle and self._app_password)

    async def probe(self) -> SourceHealth:
        """Opens the firehose briefly - the public stream needs no credential, so a
        successful connect+one-message is the whole health check."""
        try:
            async with websockets.connect(self._firehose_url, open_timeout=5) as ws:
                await asyncio.wait_for(ws.recv(), timeout=5)
            return SourceHealth(name=self.name, status=SourceStatus.HEALTHY, last_message_at=self._clock.now())
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            log.warning("bluesky_probe_failed", error=str(exc))
            return SourceHealth(name=self.name, status=SourceStatus.DOWN, detail=str(exc))

    async def close(self) -> None:
        await self._auth_http.close()

    async def stream(
        self,
        queue: asyncio.Queue[SocialEvent],
        clock: Clock | None = None,
        keywords: list[str] | None = None,
    ) -> None:
        """Consume the public firehose forever, filtering by keyword if given.

        Reconnects with exponential backoff (capped at 60s). Runs until the calling
        task cancels it, like :meth:`marketlab.adapters.crypto.spot.CryptoSpotAdapter.stream`.
        """
        clock = clock or self._clock
        kw_lower = [k.lower() for k in (keywords or [])]
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(
                    self._firehose_url, open_timeout=10, ping_interval=20, max_size=2**20
                ) as ws:
                    backoff = 1.0
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        if msg.get("kind") != "commit":
                            continue
                        commit = msg.get("commit") or {}
                        if commit.get("operation") != "create":
                            continue
                        record = commit.get("record") or {}
                        text = str(record.get("text", ""))
                        if kw_lower and not any(k in text.lower() for k in kw_lower):
                            continue
                        event = _record_to_event(str(msg.get("did", "")), record, str(commit.get("rkey", "")), clock)
                        if event is None:
                            continue
                        await queue.put(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect loop must never die
                log.warning("bluesky_ws_reconnect", error=str(exc), backoff_seconds=backoff)
                await clock.sleep(backoff)
                backoff = min(backoff * 2.0, _MAX_BACKOFF_SECONDS)

    async def _ensure_session(self) -> bool:
        if not self.has_search_credentials:
            return False
        if self._session_jwt:
            return True
        try:
            data = await self._auth_http.post_json(
                "/xrpc/com.atproto.server.createSession",
                json_body={"identifier": self._handle, "password": self._app_password},
            )
            self._session_jwt = data.get("accessJwt")
            return bool(self._session_jwt)
        except Exception as exc:  # noqa: BLE001 - degrade, never raise
            log.warning("bluesky_auth_failed", error=str(exc))
            return False

    async def search_posts(self, query: str, limit: int = 25) -> list[SocialEvent]:
        """``app.bsky.feed.searchPosts`` via an authenticated session. Empty list if
        no handle/app-password is configured or authentication fails."""
        if not await self._ensure_session():
            log.warning("bluesky_search_unavailable", query=query)
            return []
        try:
            data = await self._auth_http.get_json(
                "/xrpc/app.bsky.feed.searchPosts",
                params={"q": query, "limit": max(1, min(limit, 100))},
                headers={"Authorization": f"Bearer {self._session_jwt}"},
            )
        except Exception as exc:  # noqa: BLE001 - degrade, never raise
            log.warning("bluesky_search_failed", error=str(exc))
            return []

        now = self._clock.now()
        events: list[SocialEvent] = []
        for post in data.get("posts", []) or []:
            record = post.get("record") or {}
            author = (post.get("author") or {}).get("handle", "")
            uri = str(post.get("uri", ""))
            rkey = uri.rsplit("/", 1)[-1] if uri else ""
            created_at = _parse_iso(record.get("createdAt"))
            events.append(
                SocialEvent(
                    event_time=created_at or now,
                    published_time=created_at,
                    first_seen_time=now,
                    ingested_time=now,
                    source="bluesky_search",
                    source_class=SourceClass.SOCIAL_UNVERIFIED,
                    post_id=str(post.get("cid", "")) or rkey,
                    platform="bluesky",
                    author=author,
                    text=str(record.get("text", "")),
                    url=f"https://bsky.app/profile/{author}/post/{rkey}" if author and rkey else "",
                )
            )
        return events
