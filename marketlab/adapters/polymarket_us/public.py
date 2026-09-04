"""Polymarket US adapter: public/unauthenticated calls only.

Ed25519 request signing and live order placement are deliberately **out of scope** -
Kalshi is the only execution venue in this deployment (see ``docs/CONTRACTS.md``), and
Polymarket US credentials (`POLYMARKET_US_KEY_ID` / `POLYMARKET_US_SECRET_KEY`) are read
by ``Secrets`` only so `marketlab doctor` can report their presence; this adapter never
reads or sends them. If Polymarket US is ever activated as a second execution venue,
that requires its own explicitly-scoped adapter with a real signer - not an extension of
this file.

Live-probe finding (2026-09-04): every endpoint on ``api.polymarket.us`` - including
paths that would normally be public market/event discovery on other venues - returns
HTTP 401 with body ``"Missing required API key headers"``, even with no request body
and no auth headers sent. This is true for ``/``, ``/markets``, ``/events``,
``/v1/markets``, ``/v1/events``, ``/public/markets``, ``/health`` and ``/ping``. In other
words, Polymarket US does not currently expose *any* endpoint we can call without
credentials, contrary to what "public/unauthenticated" market data would normally mean
on a prediction-market API. Since we are intentionally not wiring credentials here, this
adapter's ``probe()`` always reports ``DOWN`` with that detail, and every accessor
degrades to an empty/``None`` result rather than raising - the adapter exists so the
engine has a place to plug in Polymarket US later, not because it is usable today.
"""

from __future__ import annotations

from typing import Any

from marketlab.adapters.base import Adapter, HttpAdapter, SourceHealth, SourceStatus
from marketlab.adapters.ratelimit import RateLimiter
from marketlab.clock import Clock
from marketlab.logging import get_logger

log = get_logger(__name__)

#: Recorded 2026-09-04: every probed path 401s with this exact message, unauthenticated.
_KNOWN_AUTH_WALL_DETAIL = (
    "api.polymarket.us returns 401 'Missing required API key headers' on all probed "
    "paths (including nominally public ones); no credentials are wired here by design "
    "since Kalshi is the only execution venue."
)


class PolymarketUsAdapter(Adapter):
    """Public/unauthenticated client for Polymarket US. No order code - see module docstring."""

    name = "poly_us_rest"

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
            await self._http.get_json("/events", params={"limit": 1})
        except Exception as exc:
            detail = str(exc)
            is_auth_wall = "401" in detail
            log.warning("poly_us_probe_failed", error=detail, is_known_auth_wall=is_auth_wall)
            health = self._http.health()
            return health.model_copy(
                update={
                    "status": SourceStatus.DOWN,
                    "detail": _KNOWN_AUTH_WALL_DETAIL if is_auth_wall else detail,
                }
            )
        # If Polymarket US ever opens up an unauthenticated surface, this branch starts
        # reporting HEALTHY automatically with no code change required.
        return self._http.health()

    async def close(self) -> None:
        await self._http.close()

    def health(self) -> SourceHealth:
        return self._http.health()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        try:
            data = await self._http.get_json(path, params=params)
        except Exception as exc:
            log.warning("poly_us_request_failed", path=path, error=str(exc))
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        return []

    async def get_events(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return await self._get("/events", {"limit": limit, "offset": offset})

    async def get_markets(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return await self._get("/markets", {"limit": limit, "offset": offset})

    async def get_book(self, market_id: str) -> dict[str, Any]:
        try:
            data = await self._http.get_json("/book", params={"market": market_id})
        except Exception as exc:
            log.warning("poly_us_book_failed", market_id=market_id, error=str(exc))
            return {}
        return data if isinstance(data, dict) else {}

    async def get_bbo(self, market_id: str) -> dict[str, Any]:
        try:
            data = await self._http.get_json("/bbo", params={"market": market_id})
        except Exception as exc:
            log.warning("poly_us_bbo_failed", market_id=market_id, error=str(exc))
            return {}
        return data if isinstance(data, dict) else {}

    async def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        return await self._get("/search", {"q": query, "limit": limit})

    async def get_series(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return await self._get("/series", {"limit": limit, "offset": offset})

    async def get_sports(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return await self._get("/sports", {"limit": limit, "offset": offset})
