"""Live smoke tests against real Polymarket endpoints. Marked `integration` and skipped
gracefully (not failed) when there is no network path to Polymarket's hosts."""

from __future__ import annotations

import socket
from decimal import Decimal

import pytest

from marketlab.adapters.polymarket_global.normalize import _parse_json_field
from marketlab.clock import LiveClock
from marketlab.core.instruments import to_probability
from marketlab.settings import load_settings

pytestmark = pytest.mark.integration


def _network_reachable(host: str, port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


_HAS_NETWORK = _network_reachable("gamma-api.polymarket.com")

skip_without_network = pytest.mark.skipif(not _HAS_NETWORK, reason="no network path to Polymarket")


@skip_without_network
async def test_leaderboard_snapshot_returns_wallets() -> None:
    from marketlab.adapters.polymarket_global.leaderboard import LeaderboardAdapter

    settings = load_settings(profile="paper")
    adapter = LeaderboardAdapter(
        settings.sources.poly_leaderboard, settings.sources.poly_lb_legacy, LiveClock()
    )
    try:
        rows = await adapter.get_official(category="overall", period="week", metric="pnl", limit=5)
        assert rows, "expected the official leaderboard to return at least one row"
        assert all("proxyWallet" in r and r["proxyWallet"] for r in rows)
    finally:
        await adapter.close()


@skip_without_network
async def test_gamma_markets_normalize_into_probability_bounds() -> None:
    from marketlab.adapters.polymarket_global.gamma import GammaAdapter
    from marketlab.adapters.polymarket_global.normalize import normalize_market

    settings = load_settings(profile="paper")
    adapter = GammaAdapter(settings.sources.poly_gamma, LiveClock())
    try:
        raw_markets = await adapter.get_markets(limit=5, closed=False)
        assert raw_markets, "expected Gamma to return at least one open market"
        for raw in raw_markets:
            market = normalize_market(raw)
            assert market.tick_size > Decimal("0")
            for price_str in _parse_json_field(raw.get("outcomePrices")):
                price = to_probability(Decimal(str(price_str)))
                assert Decimal("0") <= price <= Decimal("1")
    finally:
        await adapter.close()


@skip_without_network
async def test_geoblock_from_us_disables_global_execution() -> None:
    from marketlab.adapters.polymarket_global.geoblock import enforce

    settings = load_settings(profile="paper")
    result = await enforce(settings)
    # `enforce` must lock this off regardless of what the probe reports.
    assert settings.polymarket_global_execution is False
    # This deployment runs from a US IP, so we expect (and want to notice if this ever
    # changes) an actual block, not just the always-safe fallback.
    assert result.blocked is True


@skip_without_network
async def test_data_api_positions_for_top_leaderboard_wallet() -> None:
    from marketlab.adapters.polymarket_global.data_api import DataApiAdapter
    from marketlab.adapters.polymarket_global.leaderboard import LeaderboardAdapter

    settings = load_settings(profile="paper")
    clock = LiveClock()
    leaderboard = LeaderboardAdapter(
        settings.sources.poly_leaderboard, settings.sources.poly_lb_legacy, clock
    )
    data_api = DataApiAdapter(settings.sources.poly_data, clock)
    try:
        rows = await leaderboard.get_official(category="overall", period="week", metric="pnl", limit=1)
        assert rows
        wallet = rows[0]["proxyWallet"]
        positions = await data_api.get_positions(wallet, limit=5)
        # A top-ranked wallet may legitimately have zero *open* positions at any given
        # moment (fully resolved/cashed out); we only assert the call succeeds and
        # returns a well-shaped list.
        assert isinstance(positions, list)
        for p in positions:
            assert "conditionId" in p
    finally:
        await leaderboard.close()
        await data_api.close()
