"""GDELT DOC 2.0 API client.

Probed live on 2026-09-04:

* ``GET api.gdeltproject.org/api/v2/doc/doc?query=...&mode=artlist&format=json``
  200 with a normal JSON ``{"articles": [...]}`` body when under the rate limit.
* Under load (observed repeatedly, including from this very probing session sharing an
  IP with other concurrent agents), GDELT returns **HTTP 429 with a plain-text body**
  ("Please limit requests to one every 5 seconds..."), not JSON, and with no
  ``Retry-After`` header. This is exactly the "sometimes returns HTML/malformed JSON on
  error" failure mode the spec warns about, confirmed live rather than assumed - every
  method here catches that (and any other transport/JSON failure) and returns an empty
  list with a WARN log rather than raising.
* ``GET api.gdeltproject.org/api/v2/context/context?query=...`` 200, returns
  sentence-level hits: ``{"articles": [{"url", "title", "seendate", "sentence",
  "context", "domain", ...}]}``.

GDELT's ``seendate`` field looks like ``"20260904T143000Z"`` - parsed here to a
tz-aware UTC ``datetime`` and used for both ``event_time`` and ``published_time``.
``first_seen_time`` always comes from the injected ``Clock``, never from ``seendate``,
since GDELT's crawl can lag the original publication by minutes to hours and a strategy
must only ever gate on when *this process* observed the article.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from marketlab.adapters.base import Adapter, HttpAdapter, SourceHealth, SourceStatus
from marketlab.adapters.ratelimit import RateLimiter
from marketlab.clock import Clock, LiveClock
from marketlab.core.events import NewsEvent
from marketlab.logging import get_logger
from marketlab.signals.news import canonicalize_url

log = get_logger(__name__)


def _root(url: str) -> str:
    """``https://api.gdeltproject.org/api/v2/doc/doc`` -> ``https://api.gdeltproject.org``.

    ``HttpAdapter`` needs *some* valid ``base_url`` for its ``httpx.AsyncClient``, but
    every request here passes the full configured endpoint URL (from
    ``settings.sources.gdelt_doc``/``gdelt_context``) as an absolute URL to ``get_json``
    - httpx uses an absolute request URL as-is regardless of the client's base_url, so
    this never risks silently hardcoding a path that drifts from configuration.
    """
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _parse_seendate(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def build_query(
    *terms: str,
    domain_is: str | None = None,
    source_lang: str | None = "eng",
    theme: str | None = None,
) -> str:
    """Compose a GDELT query string: quotes multi-word phrases, adds ``domainis:``/
    ``sourcelang:``/``theme:`` filters. GDELT ANDs space-separated clauses by default.
    """
    parts: list[str] = []
    for term in terms:
        t = term.strip()
        if not t:
            continue
        if " " in t and not (t.startswith('"') and t.endswith('"')):
            t = f'"{t}"'
        parts.append(t)
    if domain_is:
        parts.append(f"domainis:{domain_is}")
    if source_lang:
        parts.append(f"sourcelang:{source_lang}")
    if theme:
        parts.append(f"theme:{theme}")
    return " ".join(parts)


class GdeltAdapter(Adapter):
    """GDELT DOC 2.0 client. No API key; a shared-IP rate limit is the only gate."""

    name = "gdelt"

    #: GDELT returns a plain-text 429 asking callers to limit requests to one every five
    #: seconds (verified live 2026-09-04 - the body is prose, not JSON, and carries no
    #: Retry-After header). Measured behaviour is stricter than that stated limit when the
    #: egress IP is shared, so we budget well under it with a burst of 1: a burst is
    #: exactly what trips it, and `configs/sources.yaml` lists several standing queries
    #: that would otherwise fire back-to-back on every refresh.
    #:
    #: At the configured 900s refresh this costs nothing - five queries spaced 15s apart
    #: fit comfortably inside one cycle. GDELT is a best-effort source either way: a 429
    #: degrades to an empty result list and is never fatal.
    DEFAULT_MIN_INTERVAL_SECONDS: float = 15.0

    def __init__(
        self,
        *,
        doc_base: str,
        context_base: str,
        clock: Clock | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._clock = clock or LiveClock()
        self._doc_url = doc_base
        self._context_url = context_base
        headers = {"User-Agent": "MarketLab research"}
        limiter = rate_limiter or RateLimiter(
            rate=1.0 / self.DEFAULT_MIN_INTERVAL_SECONDS, burst=1.0
        )
        self._http = HttpAdapter(
            _root(doc_base),
            name="gdelt",
            default_headers=headers,
            clock=self._clock,
            rate_limiter=limiter,
        )

    async def probe(self) -> SourceHealth:
        try:
            events = await self.search("news", timespan="1h", maxrecords=1)
            status = SourceStatus.HEALTHY if events else SourceStatus.DEGRADED
            return SourceHealth(name=self.name, status=status, last_message_at=self._clock.now())
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            log.warning("gdelt_probe_failed", error=str(exc))
            return SourceHealth(name=self.name, status=SourceStatus.DOWN, detail=str(exc))

    async def close(self) -> None:
        await self._http.close()

    def _article_to_event(self, art: dict[str, Any], now: datetime) -> NewsEvent | None:
        url = str(art.get("url", ""))
        title = str(art.get("title", ""))
        if not url and not title:
            return None
        seen = _parse_seendate(art.get("seendate"))
        domain = str(art.get("domain", ""))
        news_id = hashlib.sha1((url or title).encode("utf-8")).hexdigest()
        language = str(art.get("language", "")).strip().lower()[:2] or "en"
        return NewsEvent(
            event_time=seen or now,
            published_time=seen,
            first_seen_time=now,
            ingested_time=now,
            source=f"gdelt:{domain}" if domain else "gdelt",
            news_id=news_id,
            title=title,
            url=url,
            canonical_url=canonicalize_url(url) if url else "",
            language=language,
        )

    async def search(
        self,
        query: str,
        *,
        timespan: str = "1h",
        maxrecords: int = 250,
        mode: str = "artlist",
        format: str = "json",  # noqa: A002 - matches GDELT's own query parameter name
        sort: str = "datedesc",
    ) -> list[NewsEvent]:
        """``doc/doc`` article search. Empty list (never a raised exception) on any
        transport failure, rate limit, or malformed response body."""
        params = {
            "query": query,
            "mode": mode,
            "maxrecords": maxrecords,
            "format": format,
            "sort": sort,
            "timespan": timespan,
        }
        now = self._clock.now()
        try:
            payload = await self._http.get_json(self._doc_url, params=params)
        except Exception as exc:  # noqa: BLE001 - GDELT 429s with a plain-text body; never crash on it
            log.warning("gdelt_search_failed", query=query, error=str(exc))
            return []
        if not isinstance(payload, dict):
            log.warning("gdelt_search_unexpected_payload", query=query, payload_type=type(payload).__name__)
            return []
        articles = payload.get("articles", []) or []
        events = [self._article_to_event(a, now) for a in articles if isinstance(a, dict)]
        return [e for e in events if e is not None]

    async def context(
        self,
        query: str,
        *,
        mode: str = "artlist",
        maxrecords: int = 250,
        format: str = "json",  # noqa: A002
    ) -> list[dict[str, Any]]:
        """``context/context`` sentence-level hits: raw rows with a matched ``sentence``
        and surrounding ``context`` field - kept as dicts since there's no single
        ``NewsEvent`` field for "the specific sentence that matched"; callers combine
        this with :meth:`search` when they need both the event shell and the excerpt.
        """
        params = {"query": query, "mode": mode, "maxrecords": maxrecords, "format": format}
        try:
            payload = await self._http.get_json(self._context_url, params=params)
        except Exception as exc:  # noqa: BLE001 - same rate-limit/malformed-body risk as search()
            log.warning("gdelt_context_failed", query=query, error=str(exc))
            return []
        if not isinstance(payload, dict):
            return []
        articles = payload.get("articles", []) or []
        return [a for a in articles if isinstance(a, dict)]
