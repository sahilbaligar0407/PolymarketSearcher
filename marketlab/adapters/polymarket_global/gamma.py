"""Gamma API adapter: Polymarket market/event discovery (read-only).

Gamma (``gamma-api.polymarket.com``) is Polymarket's market-metadata service - titles,
descriptions, tags, resolution rules, aggregate volume/liquidity. It is not the order
book (see ``clob.py``) and it is not authenticated; every call here is a plain GET.

Read-only by design: global Polymarket is close-only from the US (see
``geoblock.py``), so this adapter never needs, and never gains, a way to submit
anything.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from marketlab.adapters.base import Adapter, HttpAdapter, SourceHealth, SourceStatus
from marketlab.adapters.ratelimit import RateLimiter
from marketlab.clock import Clock
from marketlab.logging import get_logger

log = get_logger(__name__)


def _as_list(data: Any) -> list[dict[str, Any]]:
    """Gamma list endpoints return a bare JSON array; unwrap the ``{"data": [...]}``
    envelope some endpoints use just in case, and always hand back a list."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        inner = data.get("data")
        if isinstance(inner, list):
            return inner
    return []


class GammaAdapter(Adapter):
    """Read-only client for Gamma market/event discovery."""

    name = "poly_gamma"

    def __init__(
        self,
        base_url: str,
        clock: Clock,
        *,
        rate_limiter: RateLimiter | None = None,
        timeout: float = 10.0,
        http: HttpAdapter | None = None,
    ) -> None:
        self._clock = clock
        self._http = http or HttpAdapter(
            base_url, name=self.name, timeout=timeout, rate_limiter=rate_limiter, clock=clock
        )

    async def probe(self) -> SourceHealth:
        try:
            await self._http.get_json("/markets", params={"limit": 1})
        except Exception as exc:  # probe must never raise
            log.warning("poly_gamma_probe_failed", error=str(exc))
            health = self._http.health()
            return health.model_copy(update={"status": SourceStatus.DOWN, "detail": str(exc)})
        return self._http.health()

    async def close(self) -> None:
        await self._http.close()

    def health(self) -> SourceHealth:
        return self._http.health()

    async def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        closed: bool = False,
        order: str = "volume24hr",
        ascending: bool = False,
        extra_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "closed": str(closed).lower(),
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if extra_params:
            params.update(extra_params)
        data = await self._http.get_json("/markets", params=params)
        return _as_list(data)

    async def iter_markets(
        self,
        *,
        page_size: int = 100,
        closed: bool = False,
        max_pages: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Paginate Gamma markets via ``offset``, yielding one market dict at a time."""
        offset = 0
        pages = 0
        while True:
            batch = await self.get_markets(limit=page_size, offset=offset, closed=closed, **kwargs)
            if not batch:
                return
            for market in batch:
                yield market
            if len(batch) < page_size:
                return
            offset += page_size
            pages += 1
            if max_pages is not None and pages >= max_pages:
                return

    async def get_events(
        self, limit: int = 100, offset: int = 0, *, closed: bool = False
    ) -> list[dict[str, Any]]:
        params = {"limit": limit, "offset": offset, "closed": str(closed).lower()}
        data = await self._http.get_json("/events", params=params)
        return _as_list(data)

    async def get_market(self, market_id: str) -> dict[str, Any]:
        data = await self._http.get_json(f"/markets/{market_id}")
        return data if isinstance(data, dict) else {}

    async def search(self, query: str, *, limit_per_type: int = 20) -> dict[str, list[dict[str, Any]]]:
        """Gamma's ``/public-search`` endpoint. Degrades to empty results, not an
        exception, if the surface changes or is unavailable."""
        try:
            data = await self._http.get_json(
                "/public-search", params={"q": query, "limit_per_type": limit_per_type}
            )
        except Exception as exc:
            log.warning("poly_gamma_search_unavailable", error=str(exc))
            return {"events": [], "markets": []}
        if isinstance(data, dict):
            return {
                "events": _as_list(data.get("events")),
                "markets": _as_list(data.get("markets")),
            }
        return {"events": [], "markets": []}
