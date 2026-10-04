"""Google News RSS keyword search -> :class:`NewsEvent`.

Keyless and fast (~1s per query), unlike GDELT's DOC API, which throttles to the point of
returning nothing for most targeted queries (measured 2026-10-04: three 429s then an empty
200 for "federal reserve rates"; Google News returned 47 stories for the same query).

Point-in-time safety: ``first_seen_time`` is when MarketLab fetched the story, never the
publisher's timestamp, so replay can never see a story before it was actually observed.
"""

from __future__ import annotations

import calendar
import hashlib
from datetime import UTC, datetime

import feedparser
import httpx

from marketlab.clock import Clock
from marketlab.core.events import NewsEvent
from marketlab.logging import get_logger

log = get_logger(__name__)

SEARCH_URL = "https://news.google.com/rss/search"


class GoogleNewsSearch:
    def __init__(self, clock: Clock, client: httpx.AsyncClient | None = None) -> None:
        self._clock = clock
        self._client = client or httpx.AsyncClient(timeout=20, follow_redirects=True)

    async def search(self, query: str, *, window: str = "2d", limit: int = 25) -> list[NewsEvent]:
        """Never raises; an empty list on any failure."""
        try:
            resp = await self._client.get(
                SEARCH_URL, params={"q": f"{query} when:{window}", "hl": "en-US", "gl": "US", "ceid": "US:en"}
            )
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
        except Exception as exc:  # noqa: BLE001
            log.warning("google_news_failed", query=query, error=str(exc))
            return []
        now = self._clock.now()
        out: list[NewsEvent] = []
        for entry in feed.entries[:limit]:
            title = str(entry.get("title", "")).strip()
            link = str(entry.get("link", ""))
            if not title:
                continue
            published = None
            if entry.get("published_parsed"):
                published = datetime.fromtimestamp(calendar.timegm(entry.published_parsed), UTC)
            publisher = str((entry.get("source") or {}).get("title", "")) if entry.get("source") else ""
            out.append(
                NewsEvent(
                    event_time=min(published, now) if published else now,
                    published_time=published,
                    first_seen_time=now,
                    ingested_time=now,
                    source=f"gnews:{publisher}" if publisher else "gnews",
                    news_id=hashlib.sha1((link or title).encode("utf-8")).hexdigest(),
                    title=title,
                    url=link,
                    body=str(entry.get("summary", ""))[:600],
                )
            )
        return out

    async def close(self) -> None:
        await self._client.aclose()
