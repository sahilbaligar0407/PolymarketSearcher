"""Integration test: hits the real, live public Kalshi API.

Skips gracefully (rather than failing the suite) if the network is unavailable, since
this is meant to run in CI/dev environments that may not have outbound access.
"""

from __future__ import annotations

from decimal import Decimal

import httpx
import pytest

from marketlab.adapters.kalshi.normalize import normalize_market, normalize_orderbook
from marketlab.adapters.kalshi.rest import KalshiRestAdapter
from marketlab.clock import LiveClock
from marketlab.settings import load_settings

pytestmark = pytest.mark.integration


def _network_available() -> bool:
    try:
        httpx.get("https://api.elections.kalshi.com/trade-api/v2/exchange/status", timeout=5.0)
        return True
    except httpx.HTTPError:
        return False


@pytest.mark.asyncio
async def test_live_markets_normalize_without_raising() -> None:
    if not _network_available():
        pytest.skip("network unavailable; skipping live Kalshi integration test")

    settings = load_settings()
    adapter = KalshiRestAdapter(settings, clock=LiveClock())
    try:
        page = await adapter.get_markets(limit=20)
        markets = page.get("markets", [])
        assert len(markets) > 0

        normalized = []
        for raw in markets:
            market = normalize_market(raw)
            assert Decimal("0") <= market.tick_size
            normalized.append(market)

        # Every normalized market must round-trip through NormalizedMarket cleanly and
        # not blow up on any status/category logic.
        assert len(normalized) == len(markets)

        # Fetch one order book and make sure the YES/NO -> bids/asks fold produces a
        # valid, well-ordered book with probabilities in [0, 1].
        ticker = markets[0]["ticker"]
        ob_raw = await adapter.get_orderbook(ticker, depth=10)
        now = LiveClock().now()
        book = normalize_orderbook(ticker, ob_raw, now)
        for level in (*book.bids, *book.asks):
            assert Decimal("0") <= level.price <= Decimal("1")
        if len(book.bids) > 1:
            assert all(
                book.bids[i].price >= book.bids[i + 1].price for i in range(len(book.bids) - 1)
            )
        if len(book.asks) > 1:
            assert all(
                book.asks[i].price <= book.asks[i + 1].price for i in range(len(book.asks) - 1)
            )
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_live_probe_reports_public_or_authenticated() -> None:
    if not _network_available():
        pytest.skip("network unavailable; skipping live Kalshi integration test")

    settings = load_settings()
    adapter = KalshiRestAdapter(settings, clock=LiveClock())
    try:
        health = await adapter.probe()
        assert health.status.value in {"healthy", "no_credentials", "degraded"}
    finally:
        await adapter.close()
