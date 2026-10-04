"""Polymarket US adapter: public/unauthenticated calls only.

Ed25519 request signing and live order placement are deliberately **out of scope** -
Kalshi is the only execution venue in this deployment (see ``docs/CONTRACTS.md``), and
Polymarket US credentials (`POLYMARKET_US_KEY_ID` / `POLYMARKET_US_SECRET_KEY`) are read
by ``Secrets`` only so `marketlab doctor` can report their presence; this adapter never
reads or sends them. If Polymarket US is ever activated as a second execution venue,
that requires its own explicitly-scoped adapter with a real signer - not an extension of
this file.

Live-probe findings. On 2026-09-04 every path on ``api.polymarket.us`` returned HTTP 401
"Missing required API key headers". Re-probed 2026-10-04: the public, keyless surface
lives on ``gateway.polymarket.us`` under ``/v1``:

* ``GET /v1/markets?limit&offset&active&closed`` - markets, each carrying
  ``bestBidQuote`` / ``bestAskQuote``, so one page is a 200-market price snapshot
* ``GET /v1/events``, ``/v1/series``, ``/v1/sports``, ``/v1/search?q=``
* ``GET /v1/market/slug/<slug>`` - one market
* ``GET /v1/markets/<slug>/book`` and ``/v1/markets/<slug>/bbo`` - depth and top of book

(``api.polymarket.us/v1/markets`` also answers, but its siblings still 401.) This adapter
only ever calls those public paths and sends no credentials.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from marketlab.adapters.base import Adapter, HttpAdapter, SourceHealth, SourceStatus
from marketlab.adapters.ratelimit import RateLimiter
from marketlab.clock import Clock
from marketlab.core.instruments import Venue
from marketlab.logging import get_logger

log = get_logger(__name__)

#: Shown when a probe hits the authenticated host instead of the public gateway.
_KNOWN_AUTH_WALL_DETAIL = (
    "401 'Missing required API key headers': this host needs credentials; the keyless "
    "public API is https://gateway.polymarket.us/v1 (sources.poly_us_rest)."
)

#: The gateway caps a page at this many markets.
PAGE_SIZE = 200


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
            await self._http.get_json("/v1/markets", params={"limit": 1})
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
        return self._http.health()

    async def close(self) -> None:
        await self._http.close()

    def health(self) -> SourceHealth:
        return self._http.health()

    async def _get(
        self, path: str, key: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """GET a gateway list endpoint, which wraps its rows as ``{key: [...]}``."""
        try:
            data = await self._http.get_json(path, params=params)
        except Exception as exc:
            log.warning("poly_us_request_failed", path=path, error=str(exc))
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get(key), list):
            return data[key]
        return []

    async def _get_one(self, path: str, key: str) -> dict[str, Any]:
        try:
            data = await self._http.get_json(path)
        except Exception as exc:
            log.warning("poly_us_request_failed", path=path, error=str(exc))
            return {}
        inner = data.get(key) if isinstance(data, dict) else None
        return inner if isinstance(inner, dict) else {}

    async def get_markets(
        self, limit: int = PAGE_SIZE, offset: int = 0, *, open_only: bool = True
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if open_only:
            params.update({"active": "true", "closed": "false"})
        return await self._get("/v1/markets", "markets", params)

    async def iter_open_markets(self, max_markets: int) -> list[dict[str, Any]]:
        """Page through open markets until a short page or ``max_markets``."""
        out: list[dict[str, Any]] = []
        offset = 0
        while len(out) < max_markets:
            page = await self.get_markets(limit=PAGE_SIZE, offset=offset)
            out.extend(page)
            if len(page) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
        return out[:max_markets]

    async def get_events(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return await self._get("/v1/events", "events", {"limit": limit, "offset": offset})

    async def get_market(self, slug: str) -> dict[str, Any]:
        return await self._get_one(f"/v1/market/slug/{slug}", "market")

    async def get_book(self, slug: str) -> dict[str, Any]:
        return await self._get_one(f"/v1/markets/{slug}/book", "marketData")

    async def get_bbo(self, slug: str) -> dict[str, Any]:
        return await self._get_one(f"/v1/markets/{slug}/bbo", "marketData")

    async def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        return await self._get("/v1/search", "events", {"q": query, "limit": limit})

    async def get_series(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return await self._get("/v1/series", "series", {"limit": limit, "offset": offset})

    async def get_sports(self) -> list[dict[str, Any]]:
        return await self._get("/v1/sports", "sports")


# ---------------------------------------------------------------------------
# Normalization to the shared parquet datasets (read-only data collection)
# ---------------------------------------------------------------------------


def _quote(raw: dict[str, Any], key: str) -> Decimal | None:
    value = (raw.get(key) or {}).get("value") if isinstance(raw.get(key), dict) else None
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except ArithmeticError:
        return None


def _ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def canonical_id(slug: str) -> str:
    return f"{Venue.POLY_US.value}:{slug}"


def market_metadata_row(raw: dict[str, Any], first_seen: datetime) -> dict[str, Any]:
    """A ``market_metadata`` parquet row. These markets never enter the trading registry:
    Polymarket US is listed in ``EXECUTION_VENUES``, and this deployment only collects it."""
    title = raw.get("question") or ""
    if raw.get("title") and raw.get("title") != title:
        title = f"{title}: {raw['title']}"
    return {
        "canonical_id": canonical_id(str(raw.get("slug"))),
        "venue": Venue.POLY_US.value,
        "venue_market_id": str(raw.get("id") or ""),
        "title": title[:500],
        "category": str(raw.get("category") or ""),
        "status": str(raw.get("status") or ""),
        "open_time": _ts(raw.get("startDate")),
        "close_time": _ts(raw.get("endDate")),
        "tick_size": Decimal(str(raw.get("orderPriceMinTickSize") or "0.001")),
        "first_seen_time": first_seen,
        "date": first_seen.date(),
    }


def top_of_book_row(raw: dict[str, Any], now: datetime) -> dict[str, Any] | None:
    """A level-0 ``books`` parquet row from the list endpoint's best quotes, or None
    when the market has neither a bid nor an ask."""
    bid, ask = _quote(raw, "bestBidQuote"), _quote(raw, "bestAskQuote")
    if bid is None and ask is None:
        return None
    return {
        "canonical_id": canonical_id(str(raw.get("slug"))),
        "venue": Venue.POLY_US.value,
        "timestamp": now,
        "venue_timestamp": _ts(raw.get("updatedAt")),
        "sequence": None,
        "side": "both",
        "level": 0,
        "bid_price": bid,
        "ask_price": ask,
        "bid_size": None,
        "ask_size": None,
        "date": now.date(),
    }
