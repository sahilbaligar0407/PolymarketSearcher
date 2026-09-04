"""Unit tests for marketlab.daemon.ingest.

Universe-expansion tests use fixture Kalshi market dicts run through the real
``normalize_market`` (no network). The ``run_once`` test injects fakes for every
adapter so it never touches the network either -- it exercises the "zero markets is a
failure, not a quiet no-op" contract in isolation.
"""

from __future__ import annotations

from datetime import UTC, datetime

from marketlab.adapters.kalshi.normalize import normalize_market
from marketlab.clock import SimulatedClock
from marketlab.daemon import ingest as ingest_module
from marketlab.daemon.ingest import (
    IngestService,
    market_update_event,
    select_tracked_markets,
    trade_event_from_raw,
)
from marketlab.settings import load_settings

START = datetime(2026, 9, 4, tzinfo=UTC)

UNIVERSES_CFG = {
    "defaults": {
        "exclude_series_prefixes": ["KXMVE"],
        "min_volume": 0,
        "min_liquidity": 0,
        "require_status": "open",
    },
    "universes": {
        "btc_15m": {
            "available": True,
            "kalshi_series": ["KXBTC15M"],
        },
    },
}


def _raw(ticker: str, volume: str = "100") -> dict:
    return {
        "ticker": ticker,
        "series_ticker": ticker.split("-")[0],
        "event_ticker": ticker.split("-")[0],
        "title": f"Market {ticker}",
        "status": "active",
        "volume_dollars": volume,
        "liquidity_dollars": volume,
    }


# ---------------------------------------------------------------------------
# Universe expansion
# ---------------------------------------------------------------------------


def test_universe_expansion_excludes_kxmve_and_respects_cap() -> None:
    raws = [_raw(f"KXMVECROSSCATEGORY-{i}") for i in range(5)]
    raws += [_raw(f"KXBTC15M-25SEP04{i:02d}") for i in range(6)]
    markets = [normalize_market(r) for r in raws]

    selected = select_tracked_markets(markets, UNIVERSES_CFG, max_tracked_markets=3)

    assert len(selected) == 3
    assert all(not m.venue_market_id.upper().startswith("KXMVE") for m in selected)
    assert all(m.venue_market_id.upper().startswith("KXBTC15M") for m in selected)


def test_universe_expansion_excludes_kxmve_when_uncapped() -> None:
    raws = [_raw("KXMVECROSSCATEGORY-1"), _raw("KXBTC15M-1"), _raw("KXBTC15M-2")]
    markets = [normalize_market(r) for r in raws]

    selected = select_tracked_markets(markets, UNIVERSES_CFG, max_tracked_markets=0)

    assert len(selected) == 2
    assert {m.venue_market_id for m in selected} == {"KXBTC15M-1", "KXBTC15M-2"}


def test_universe_expansion_respects_max_tracked_markets_alone() -> None:
    raws = [_raw(f"KXBTC15M-{i}", volume=str(1000 - i)) for i in range(10)]
    markets = [normalize_market(r) for r in raws]

    selected = select_tracked_markets(markets, UNIVERSES_CFG, max_tracked_markets=4)

    assert len(selected) == 4
    # Highest-volume markets win the cap.
    assert selected[0].venue_market_id == "KXBTC15M-0"


# ---------------------------------------------------------------------------
# first_seen_time discipline
# ---------------------------------------------------------------------------


def test_trade_first_seen_time_comes_from_clock_not_payload() -> None:
    clock = SimulatedClock(START)
    misleading_raw_trade = {
        "ticker": "KXBTC15M-1",
        "yes_price_dollars": "0.55",
        "count": 10,
        # A deliberately misleading timestamp far in the past -- this is the actual
        # trade time (event_time), never first_seen_time.
        "created_time": "2000-01-01T00:00:00Z",
        "trade_id": "t1",
    }

    event = trade_event_from_raw(misleading_raw_trade, clock, "test")

    assert event.first_seen_time == clock.now()
    assert event.event_time.year == 2000
    assert event.first_seen_time != event.event_time


def test_market_update_event_first_seen_time_tracks_clock_not_construction_order() -> None:
    clock = SimulatedClock(START)
    market = normalize_market(_raw("KXBTC15M-1"))

    first = market_update_event(market, clock, "test")
    assert first.first_seen_time == clock.now()

    clock.advance(500.0)
    second = market_update_event(market, clock, "test")
    assert second.first_seen_time == clock.now()
    assert second.first_seen_time != first.first_seen_time


# ---------------------------------------------------------------------------
# run_once: zero markets must surface as a failure
# ---------------------------------------------------------------------------


class _FakeKalshiRest:
    """Returns zero markets from every page, like a misconfigured or dead endpoint."""

    async def iter_all_markets(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return
        yield {}  # pragma: no cover - makes this an async generator with no items

    async def get_orderbook(self, ticker, depth=10):  # noqa: ANN001, ANN201
        return {}

    async def get_trades(self, ticker=None, limit=100, cursor=None, min_ts=None):  # noqa: ANN001
        return {"trades": []}

    async def close(self) -> None:
        return None


class _FakeGamma:
    async def get_markets(self, **kwargs):  # noqa: ANN003
        return []

    async def close(self) -> None:
        return None


class _FakeClob:
    async def get_book(self, token_id):  # noqa: ANN001
        return {}

    async def close(self) -> None:
        return None


class _FakeDataApi:
    async def iter_trader_actions(self, wallet, **kwargs):  # noqa: ANN001, ANN003
        return
        yield  # pragma: no cover

    async def close(self) -> None:
        return None


class _FakeLeaderboard:
    async def get_official(self, **kwargs):  # noqa: ANN003
        return []

    def _rows_from_official(self, raw_rows, **kwargs):  # noqa: ANN001, ANN003
        return []

    async def snapshot_all(self, **kwargs):  # noqa: ANN003
        return []

    async def close(self) -> None:
        return None


class _FakeGeoblockResult:
    blocked = True
    country = "US"


async def test_run_once_with_zero_markets_surfaces_failure(monkeypatch) -> None:  # noqa: ANN001
    async def fake_enforce(settings, *, http=None):  # noqa: ANN001
        return _FakeGeoblockResult()

    monkeypatch.setattr(ingest_module.poly_geoblock, "enforce", fake_enforce)

    settings = load_settings()
    clock = SimulatedClock(START)
    service = IngestService(settings, clock, kalshi_rest=_FakeKalshiRest())
    service.poly_gamma = _FakeGamma()
    service.poly_clob = _FakeClob()
    service.poly_data = _FakeDataApi()
    service.poly_leaderboard = _FakeLeaderboard()

    result = await service.run_once()

    assert result.kalshi_markets == 0
    assert result.poly_markets == 0
    assert result.total_markets == 0
    assert result.ok is False
    assert result.errors == []


async def test_run_once_reports_success_when_markets_present(monkeypatch) -> None:  # noqa: ANN001
    async def fake_enforce(settings, *, http=None):  # noqa: ANN001
        return _FakeGeoblockResult()

    monkeypatch.setattr(ingest_module.poly_geoblock, "enforce", fake_enforce)

    class _OneMarketKalshiRest(_FakeKalshiRest):
        async def iter_all_markets(self, *args, **kwargs):  # noqa: ANN002, ANN003
            yield {"markets": [_raw("KXBTC15M-1")]}

    settings = load_settings()
    clock = SimulatedClock(START)
    service = IngestService(settings, clock, kalshi_rest=_OneMarketKalshiRest())
    service.poly_gamma = _FakeGamma()
    service.poly_clob = _FakeClob()
    service.poly_data = _FakeDataApi()
    service.poly_leaderboard = _FakeLeaderboard()

    result = await service.run_once()

    assert result.kalshi_markets == 1
    assert result.ok is True
    assert len(service.markets) == 1
