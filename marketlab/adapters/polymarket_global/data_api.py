"""Data API adapter: public Polymarket trader intelligence (read-only).

``data-api.polymarket.com`` exposes wallet-level positions, activity and trade history
with no authentication required - this is the feed the copy-trading strategy family
watches. It is intelligence only: nothing here ever informs a Polymarket order, and the
follower simulation always trades the *equivalent Kalshi contract*.

``get_activity`` is the key feed for copy trading. Each row becomes a
:class:`~marketlab.core.events.TraderActionEvent` via ``normalize.normalize_activity``,
which stamps ``first_seen_time`` from the adapter's injected ``Clock`` - never from the
trade's own on-chain ``timestamp``. That distinction is the whole point of the
follower-latency experiment: a copy-trading strategy must only ever react to when *we*
observed a wallet's trade, not to the trade's true (and unknowable in real time,
ahead-of-observation) on-chain instant.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

from marketlab.adapters.base import Adapter, HttpAdapter, SourceHealth, SourceStatus
from marketlab.adapters.polymarket_global.normalize import normalize_activity
from marketlab.adapters.ratelimit import RateLimiter
from marketlab.clock import Clock
from marketlab.core.events import TraderActionEvent
from marketlab.logging import get_logger

log = get_logger(__name__)


def _as_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        inner = data.get("data")
        if isinstance(inner, list):
            return inner
    return []


class DataApiAdapter(Adapter):
    """Read-only client for Polymarket's public trader-intelligence Data API."""

    name = "poly_data"

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
            # An arbitrary, low-cost, always-answerable probe: value of a known wallet.
            await self._http.get_json("/positions", params={"user": "0x0", "limit": 1})
        except Exception as exc:
            log.warning("poly_data_api_probe_failed", error=str(exc))
            health = self._http.health()
            return health.model_copy(update={"status": SourceStatus.DOWN, "detail": str(exc)})
        return self._http.health()

    async def close(self) -> None:
        await self._http.close()

    def health(self) -> SourceHealth:
        return self._http.health()

    async def get_positions(
        self,
        wallet: str,
        *,
        limit: int = 100,
        offset: int = 0,
        size_threshold: Decimal | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"user": wallet, "limit": limit, "offset": offset}
        if size_threshold is not None:
            params["sizeThreshold"] = str(size_threshold)
        data = await self._http.get_json("/positions", params=params)
        return _as_list(data)

    async def get_closed_positions(self, wallet: str, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Resolved positions, newest first: realizedPnl, totalBought (shares), avgPrice,
        timestamp, slug. The realized track record TraderScore is computed from."""
        params = {"user": wallet, "limit": limit, "offset": offset, "sortBy": "TIMESTAMP", "sortDirection": "DESC"}
        data = await self._http.get_json("/closed-positions", params=params)
        return _as_list(data)

    async def get_activity(
        self,
        wallet: str,
        *,
        limit: int = 100,
        offset: int = 0,
        start_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"user": wallet, "limit": limit, "offset": offset}
        if start_ts is not None:
            params["start"] = start_ts
        data = await self._http.get_json("/activity", params=params)
        return _as_list(data)

    async def iter_trader_actions(
        self,
        wallet: str,
        *,
        page_size: int = 100,
        max_pages: int | None = None,
        trades_only: bool = True,
    ) -> AsyncIterator[TraderActionEvent]:
        """Page through ``/activity`` and yield normalized :class:`TraderActionEvent`.

        ``trades_only`` (default True) filters out non-trade activity rows (rewards,
        yield, maker/taker rebates) that the Data API mixes into the same feed - those
        are not trader intent, so the copy-trading strategy family should not see them
        as actions to follow.
        """
        offset = 0
        pages = 0
        while True:
            batch = await self.get_activity(wallet, limit=page_size, offset=offset)
            if not batch:
                return
            now = self._clock.now()
            for raw in batch:
                if trades_only and str(raw.get("type", "")).upper() != "TRADE":
                    continue
                yield normalize_activity(raw, now)
            if len(batch) < page_size:
                return
            offset += page_size
            pages += 1
            if max_pages is not None and pages >= max_pages:
                return

    async def get_user_trades(
        self, wallet: str, *, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        data = await self._http.get_json(
            "/trades", params={"user": wallet, "limit": limit, "offset": offset}
        )
        return _as_list(data)

    async def get_market_trades(
        self, condition_id: str, *, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        data = await self._http.get_json(
            "/trades", params={"market": condition_id, "limit": limit, "offset": offset}
        )
        return _as_list(data)

    async def get_value(self, wallet: str) -> Decimal | None:
        """Total portfolio value (USD) for ``wallet``, per ``GET /value?user=``."""
        data = await self._http.get_json("/value", params={"user": wallet})
        rows = _as_list(data)
        if rows and "value" in rows[0]:
            try:
                return Decimal(str(rows[0]["value"]))
            except Exception:
                return None
        if isinstance(data, dict) and "value" in data:
            try:
                return Decimal(str(data["value"]))
            except Exception:
                return None
        return None
