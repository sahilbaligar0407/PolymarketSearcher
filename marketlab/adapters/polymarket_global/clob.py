"""CLOB adapter: live Polymarket order-book state (read-only).

READ-ONLY BY DESIGN. Order creation (``POST /order``, signed EIP-712 payloads, API-key
provisioning) is intentionally absent from this client. Global Polymarket is close-only
from the United States (see ``geoblock.py`` - ``polymarket_global_execution`` can only
ever be False in this deployment), so there is nothing here that could submit an order
even by accident: no signer, no private key handling, no ``/order`` path. Polymarket
price/book data is consumed purely as a cross-venue signal to trade the *equivalent
Kalshi contract*.

Tick size and fees are fetched from the API rather than assumed: ``get_tick_size``
calls the CLOB's own ``/tick-size`` endpoint, and market objects returned by
``get_market``/``get_markets``/``get_sampling_markets`` carry ``maker_base_fee`` /
``taker_base_fee`` straight from the venue (see ``normalize.py`` for how those are
turned into a :class:`~marketlab.core.instruments.Fees`).

Endpoint notes from live probing (2026-09-04): ``/fee-rate-bps`` does not exist on the
current CLOB deployment (404) - fee data instead lives on the market object itself
(``maker_base_fee`` / ``taker_base_fee``, and on Gamma a richer ``feeSchedule`` block).
``/spreads`` (plural) is POST-only; ``/spread`` (singular, GET, single token) works.
"""

from __future__ import annotations

from typing import Any

from marketlab.adapters.base import Adapter, HttpAdapter, SourceHealth, SourceStatus
from marketlab.adapters.ratelimit import RateLimiter
from marketlab.clock import Clock
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


class ClobAdapter(Adapter):
    """Read-only client for the Polymarket Central Limit Order Book."""

    name = "poly_clob"

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
            await self._http.get_json("/markets", params={"next_cursor": ""})
        except Exception as exc:
            log.warning("poly_clob_probe_failed", error=str(exc))
            health = self._http.health()
            return health.model_copy(update={"status": SourceStatus.DOWN, "detail": str(exc)})
        return self._http.health()

    async def close(self) -> None:
        await self._http.close()

    def health(self) -> SourceHealth:
        return self._http.health()

    async def get_markets(self, *, next_cursor: str = "") -> dict[str, Any]:
        """One page of CLOB markets. Returns the raw envelope (``data`` + ``next_cursor``)
        since CLOB pagination is cursor-based, unlike Gamma's offset pagination."""
        data = await self._http.get_json("/markets", params={"next_cursor": next_cursor})
        return data if isinstance(data, dict) else {"data": _as_list(data), "next_cursor": ""}

    async def get_sampling_markets(self, *, next_cursor: str = "") -> dict[str, Any]:
        """Markets currently eligible for CLOB liquidity-mining/sampling - a convenient
        "markets with a real, active book" filter."""
        data = await self._http.get_json("/sampling-markets", params={"next_cursor": next_cursor})
        return data if isinstance(data, dict) else {"data": _as_list(data), "next_cursor": ""}

    async def get_book(self, token_id: str) -> dict[str, Any]:
        data = await self._http.get_json("/book", params={"token_id": token_id})
        return data if isinstance(data, dict) else {}

    async def get_books(self, token_ids: list[str]) -> list[dict[str, Any]]:
        """Batch book fetch via the CLOB's ``POST /books``.

        The CLOB requires a raw JSON *array* request body here, not an object -
        ``HttpAdapter.post_json`` is typed for a ``dict`` body (the common case for
        every other adapter), but it hands the value straight to ``httpx`` as ``json=``,
        which is happy with a list at runtime. Hence the explicit ``type: ignore``.
        """
        data = await self._http.post_json(
            "/books", json_body=[{"token_id": t} for t in token_ids]  # type: ignore[arg-type]
        )
        return _as_list(data)

    async def get_midpoint(self, token_id: str) -> Any | None:
        data = await self._http.get_json("/midpoint", params={"token_id": token_id})
        if isinstance(data, dict) and "mid" in data:
            return data["mid"]
        return None

    async def get_price(self, token_id: str, side: str) -> Any | None:
        """``side`` is ``"BUY"`` or ``"SELL"`` (the CLOB's own vocabulary, not our
        ``Side`` YES/NO enum - this is the price a taker would pay/receive)."""
        data = await self._http.get_json("/price", params={"token_id": token_id, "side": side})
        if isinstance(data, dict) and "price" in data:
            return data["price"]
        return None

    async def get_tick_size(self, token_id: str) -> Any | None:
        data = await self._http.get_json("/tick-size", params={"token_id": token_id})
        if isinstance(data, dict) and "minimum_tick_size" in data:
            return data["minimum_tick_size"]
        return None

    async def get_spread(self, token_id: str) -> Any | None:
        data = await self._http.get_json("/spread", params={"token_id": token_id})
        if isinstance(data, dict) and "spread" in data:
            return data["spread"]
        return None

    async def get_fee_rate_bps(self, token_id: str) -> Any | None:
        """The CLOB does not expose a standalone ``/fee-rate-bps`` endpoint as of this
        probe (confirmed 404 on 2026-09-04); fee data lives on the market object's
        ``maker_base_fee`` / ``taker_base_fee`` fields instead. This method fetches the
        market for ``token_id`` and reads those fields, so callers get a single
        "fetched from the API, not guessed" fee number regardless of which surface
        Polymarket happens to expose it on.
        """
        markets_page = await self.get_markets()
        for market in markets_page.get("data", []):
            for token in market.get("tokens", []):
                if token.get("token_id") == token_id:
                    return {
                        "maker_base_fee": market.get("maker_base_fee"),
                        "taker_base_fee": market.get("taker_base_fee"),
                    }
        return None
