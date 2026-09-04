"""Kalshi REST adapter.

Covers every read endpoint MarketLab needs for market data, research, and portfolio
visibility. **Order placement (`create_order` / `cancel_order`) is intentionally not
implemented here** - live execution is owned by another team's
``marketlab/execution/kalshi_live.py``. This module only ever issues GET requests.

Public endpoints (markets, events, series, orderbook, trades, exchange status) work with
no credentials at all - verified against the live production API. Authenticated
endpoints (balance, positions, fills, orders) are only attempted when
``KalshiAuth.is_configured`` is true; callers that invoke them without credentials get a
clear ``KalshiAuthError`` rather than a confusing 401 deep in httpx.

On repeated failures against the primary host, requests fail over to
``settings.sources.kalshi_rest_fallback`` (only meaningful in the production
environment - the demo environment has no documented fallback host).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from marketlab.adapters.base import Adapter, HttpAdapter, SourceHealth, SourceStatus
from marketlab.adapters.kalshi.auth import KalshiAuth, KalshiAuthError
from marketlab.adapters.ratelimit import RateLimiter
from marketlab.clock import Clock, LiveClock
from marketlab.logging import get_logger
from marketlab.settings import Settings

log = get_logger(__name__)

#: Path prefix included in the RSA-PSS signed message but not in httpx's relative paths
#: (the httpx client's base_url already carries it).
_TRADE_API_PATH_PREFIX = "/trade-api/v2"

#: Consecutive request failures against the primary host before we fail over.
_FALLBACK_THRESHOLD = 3


class KalshiRestAdapter(Adapter):
    """Read-side Kalshi REST client: market data + authenticated portfolio reads."""

    name = "kalshi_rest"

    def __init__(
        self,
        settings: Settings,
        *,
        clock: Clock | None = None,
        auth: KalshiAuth | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock or LiveClock()
        env = (settings.secrets.kalshi_environment or "production").strip().lower()
        self._environment = env
        self._base_url = (
            settings.sources.kalshi_demo_rest if env == "demo" else settings.sources.kalshi_rest
        )
        # Kalshi documents no demo fallback host; only production has one.
        self._fallback_url = settings.sources.kalshi_rest_fallback if env != "demo" else None
        self._using_fallback = False
        self._consecutive_failures = 0

        self._auth = auth or KalshiAuth(
            settings.secrets.kalshi_api_key_id,
            settings.secrets.kalshi_private_key_path,
            environment=env,
        )
        # Kalshi's own documented default is roughly 10 req/s for basic tier; this is a
        # starting point only - update_capacity() lets a caller raise/lower it once the
        # account's actual tier is known, per adapters/ratelimit.py's design.
        self._rate_limiter = rate_limiter or RateLimiter(rate=10.0, burst=20.0)
        self._http = HttpAdapter(
            self._base_url, name=self.name, clock=self._clock, rate_limiter=self._rate_limiter
        )

    @property
    def current_base_url(self) -> str:
        return self._fallback_url if self._using_fallback and self._fallback_url else self._base_url

    def _require_auth(self) -> None:
        if not self._auth.is_configured:
            raise KalshiAuthError(
                "Kalshi credentials not configured; cannot call an authenticated endpoint"
            )

    def _auth_headers(self, method: str, path: str) -> dict[str, str] | None:
        if not self._auth.is_configured:
            return None
        signed_path = f"{_TRADE_API_PATH_PREFIX}{path}"
        return self._auth.headers(method, signed_path, base_url=self.current_base_url)

    async def _maybe_fail_over(self) -> None:
        if (
            self._fallback_url
            and not self._using_fallback
            and self._consecutive_failures >= _FALLBACK_THRESHOLD
        ):
            log.warning(
                "kalshi_rest_failover",
                from_url=self._base_url,
                to_url=self._fallback_url,
                failures=self._consecutive_failures,
            )
            old_http = self._http
            self._using_fallback = True
            self._http = HttpAdapter(
                self._fallback_url,
                name=f"{self.name}_fallback",
                clock=self._clock,
                rate_limiter=self._rate_limiter,
            )
            await old_http.close()

    async def _get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        cost: float = 1.0,
        auth: bool = False,
    ) -> dict[str, Any]:
        if auth:
            self._require_auth()
        headers = self._auth_headers("GET", path) if auth else None
        try:
            result = await self._http.get_json(path, params=params, headers=headers, cost=cost)
        except KalshiAuthError:
            raise
        except Exception:
            self._consecutive_failures += 1
            await self._maybe_fail_over()
            raise
        else:
            self._consecutive_failures = 0
            return result

    # ------------------------------------------------------------------
    # Public: markets / events / series
    # ------------------------------------------------------------------

    async def get_markets(
        self,
        limit: int = 200,
        cursor: str | None = None,
        status: str | None = None,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        min_close_ts: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if min_close_ts is not None:
            params["min_close_ts"] = min_close_ts
        return await self._get("/markets", params=params)

    async def iter_all_markets(
        self, max_pages: int = 100, **filters: Any
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield successive ``/markets`` pages, following ``cursor`` until it's exhausted."""
        cursor: str | None = filters.pop("cursor", None)
        seen: set[str] = set()
        for _ in range(max_pages):
            page = await self.get_markets(cursor=cursor, **filters)
            yield page
            next_cursor = page.get("cursor") or ""
            if not next_cursor or next_cursor == cursor or next_cursor in seen:
                return
            seen.add(next_cursor)
            cursor = next_cursor

    async def get_market(self, ticker: str) -> dict[str, Any]:
        return await self._get(f"/markets/{ticker}")

    async def get_events(
        self,
        limit: int = 200,
        cursor: str | None = None,
        status: str | None = None,
        series_ticker: str | None = None,
        with_nested_markets: bool | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if with_nested_markets is not None:
            params["with_nested_markets"] = with_nested_markets
        return await self._get("/events", params=params)

    async def get_series(self, series_ticker: str) -> dict[str, Any]:
        return await self._get(f"/series/{series_ticker}")

    # ------------------------------------------------------------------
    # Public: order book / trades / candlesticks
    # ------------------------------------------------------------------

    async def get_orderbook(self, ticker: str, depth: int = 10) -> dict[str, Any]:
        return await self._get(f"/markets/{ticker}/orderbook", params={"depth": depth})

    async def get_trades(
        self,
        ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        min_ts: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        if min_ts is not None:
            params["min_ts"] = min_ts
        return await self._get("/markets/trades", params=params)

    async def get_market_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
        period_interval: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if start_ts is not None:
            params["start_ts"] = start_ts
        if end_ts is not None:
            params["end_ts"] = end_ts
        if period_interval is not None:
            params["period_interval"] = period_interval
        return await self._get(
            f"/series/{series_ticker}/markets/{ticker}/candlesticks", params=params
        )

    async def get_exchange_status(self) -> dict[str, Any]:
        return await self._get("/exchange/status")

    # ------------------------------------------------------------------
    # Authenticated (read-only portfolio views; NOT order placement)
    # ------------------------------------------------------------------

    async def get_balance(self) -> dict[str, Any]:
        return await self._get("/portfolio/balance", auth=True)

    async def get_positions(self, **params: Any) -> dict[str, Any]:
        return await self._get("/portfolio/positions", params=params or None, auth=True)

    async def get_fills(self, **params: Any) -> dict[str, Any]:
        return await self._get("/portfolio/fills", params=params or None, auth=True)

    async def get_orders(self, **params: Any) -> dict[str, Any]:
        return await self._get("/portfolio/orders", params=params or None, auth=True)

    # NOTE: create_order / cancel_order deliberately do not live here. This adapter is
    # data + read-only portfolio visibility. Live order placement is
    # marketlab/execution/kalshi_live.py, owned by the execution team.

    # ------------------------------------------------------------------
    # Adapter interface
    # ------------------------------------------------------------------

    async def probe(self) -> SourceHealth:
        """Public connectivity check, then an authenticated check if credentials exist.

        A missing credential is reported as ``NO_CREDENTIALS`` even though public market
        data keeps flowing fine - `doctor` needs to know execution/portfolio access is
        unavailable, and this is the more informative signal than a blanket HEALTHY.
        """
        started = self._clock.now()
        try:
            await self.get_exchange_status()
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            return SourceHealth(
                name=self.name,
                status=SourceStatus.DOWN,
                last_message_at=None,
                latency_ms=None,
                error_count=1,
                detail=f"public endpoint unreachable: {exc}",
            )
        latency_ms = (self._clock.now() - started).total_seconds() * 1000

        if not self._auth.is_configured:
            return SourceHealth(
                name=self.name,
                status=SourceStatus.NO_CREDENTIALS,
                last_message_at=self._clock.now(),
                latency_ms=latency_ms,
                detail="public-only: no Kalshi API credentials configured",
            )

        try:
            await self.get_balance()
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            return SourceHealth(
                name=self.name,
                status=SourceStatus.DEGRADED,
                last_message_at=self._clock.now(),
                latency_ms=latency_ms,
                error_count=1,
                detail=f"public OK; authenticated check failed: {exc}",
            )
        return SourceHealth(
            name=self.name,
            status=SourceStatus.HEALTHY,
            last_message_at=self._clock.now(),
            latency_ms=latency_ms,
            detail="authenticated",
        )

    async def close(self) -> None:
        await self._http.close()
