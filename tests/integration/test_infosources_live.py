"""Integration tests: hit the real, live public information-source APIs this team owns.

Each test skips gracefully (rather than failing the suite) when the network is
unavailable or the relevant credential is absent, since this is meant to run in
CI/dev environments that may not have outbound access or paid API keys configured.
"""

from __future__ import annotations

import httpx
import pytest

from marketlab.adapters.crypto.spot import CryptoSpotAdapter
from marketlab.adapters.sec.client import SecAdapter
from marketlab.adapters.weather.nws import NwsAdapter
from marketlab.clock import LiveClock
from marketlab.settings import load_settings

pytestmark = pytest.mark.integration

#: SEC's Akamai bot filter 403s any User-Agent that doesn't look like it has a real
#: contact email in it (confirmed live: a UA without "@" gets HTTP 403, one with "@"
#: gets 200). ``Secrets.sec_user_agent``'s own default placeholder has no "@" and will
#: always 403 - this test must not depend on ``.env`` being configured, so it falls
#: back to a concrete contact-bearing UA rather than the settings default.
_SEC_TEST_USER_AGENT = "MarketLab research (contact: sahilbaligar@gmail.com)"


def _network_available(url: str) -> bool:
    try:
        httpx.get(url, timeout=8.0, headers={"User-Agent": "MarketLab research (integration test)"})
        return True
    except httpx.HTTPError:
        return False


@pytest.mark.asyncio
async def test_gdelt_returns_articles() -> None:
    """GDELT's doc/doc search returns at least one article for a broad, always-topical query.

    GDELT enforces a strict client-side rate limit (observed: "one every 5 seconds")
    and returns a **plain-text** error body (not JSON) with HTTP 429 when exceeded -
    both handled here by treating any non-2xx or non-JSON response as "skip", since a
    shared-IP rate limit is an environment fact, not a bug in this adapter.
    """
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": "bitcoin",
        "mode": "artlist",
        "maxrecords": 5,
        "format": "json",
        "sort": "datedesc",
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, timeout=15.0)
    except httpx.HTTPError:
        pytest.skip("network unavailable; skipping live GDELT integration test")

    if resp.status_code != 200:
        pytest.skip(f"GDELT returned HTTP {resp.status_code} (likely rate-limited); skipping")
    try:
        data = resp.json()
    except ValueError:
        pytest.skip("GDELT returned a non-JSON body (rate-limit or transient error page); skipping")

    articles = data.get("articles", [])
    assert isinstance(articles, list)
    assert len(articles) > 0
    first = articles[0]
    assert "url" in first
    assert "title" in first
    print(f"GDELT: {len(articles)} articles, first title = {first.get('title')!r}")


@pytest.mark.asyncio
async def test_sec_ticker_map_loads() -> None:
    """SEC's ``company_tickers.json`` loads and contains well-known large-cap tickers."""
    settings = load_settings()
    if not _network_available("https://www.sec.gov/files/company_tickers.json"):
        pytest.skip("network unavailable; skipping live SEC integration test")

    adapter = SecAdapter(
        sec_base=settings.sources.sec_base,
        edgar_base=settings.sources.sec_edgar,
        user_agent=settings.secrets.sec_user_agent if "@" in settings.secrets.sec_user_agent else _SEC_TEST_USER_AGENT,
        clock=LiveClock(),
    )
    try:
        ticker_map = await adapter.get_ticker_map()
        assert len(ticker_map) > 1000
        assert "AAPL" in ticker_map
        assert len(ticker_map["AAPL"]) == 10  # zero-padded CIK
        print(f"SEC: ticker map has {len(ticker_map)} entries; AAPL CIK = {ticker_map['AAPL']}")
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_nws_returns_forecast() -> None:
    """NWS returns a real forecast for New York City's configured gridpoint."""
    settings = load_settings()
    if not _network_available("https://api.weather.gov/points/40.7128,-74.006"):
        pytest.skip("network unavailable; skipping live NWS integration test")

    adapter = NwsAdapter(
        base_url=settings.sources.nws_base,
        user_agent=settings.secrets.sec_user_agent if "@" in settings.secrets.sec_user_agent else _SEC_TEST_USER_AGENT,
        clock=LiveClock(),
    )
    try:
        events = await adapter.forecast_events("nyc")
        assert len(events) > 0
        first = events[0]
        assert first.station == "KNYC"
        assert first.value is not None
        print(f"NWS: {len(events)} forecast periods for NYC; first = {first.variable}={first.value}")
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_coinbase_returns_btc_price_above_zero() -> None:
    """Coinbase returns a positive BTC-USD spot price."""
    settings = load_settings()
    if not _network_available("https://api.exchange.coinbase.com/products/BTC-USD/ticker"):
        pytest.skip("network unavailable; skipping live Coinbase integration test")

    adapter = CryptoSpotAdapter(
        coinbase_base=settings.sources.coinbase_spot,
        binance_base=settings.sources.binance_spot,
        clock=LiveClock(),
    )
    try:
        event = await adapter.get_spot("BTC")
        assert event.price > 0
        assert event.symbol == "BTC-USD"
        print(f"Coinbase: BTC-USD spot = {event.price} (source={event.source})")
    finally:
        await adapter.close()
