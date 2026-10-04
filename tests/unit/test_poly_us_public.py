"""Polymarket US keyless gateway: payload shapes captured 2026-10-04."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from marketlab.adapters.polymarket_us.public import (
    PAGE_SIZE,
    PolymarketUsAdapter,
    market_metadata_row,
    top_of_book_row,
)
from marketlab.clock import SimulatedClock

NOW = datetime(2026, 10, 4, 22, 0, tzinfo=UTC)

RAW = {
    "id": "7900",
    "question": "National League Champion",
    "title": "Atlanta Braves",
    "slug": "tec-mlb-nlchamp-2026-09-27-atl",
    "category": "sports",
    "status": "MARKET_STATUS_OPEN",
    "startDate": "2026-03-19T14:32:23Z",
    "endDate": "2026-11-06T21:20:09Z",
    "updatedAt": "2026-10-04T22:11:14Z",
    "orderPriceMinTickSize": 0.001,
    "bestBidQuote": {"value": "0.0610", "currency": "USD"},
    "bestAskQuote": {"value": "0.0800", "currency": "USD"},
}


def test_rows_from_a_gateway_market() -> None:
    meta = market_metadata_row(RAW, NOW)
    assert meta["canonical_id"] == "poly-us:tec-mlb-nlchamp-2026-09-27-atl"
    assert meta["title"] == "National League Champion: Atlanta Braves"
    assert meta["close_time"] == datetime(2026, 11, 6, 21, 20, 9, tzinfo=UTC)
    book = top_of_book_row(RAW, NOW)
    assert book is not None
    assert (book["bid_price"], book["ask_price"]) == (Decimal("0.0610"), Decimal("0.0800"))


def test_unquoted_market_has_no_book_row() -> None:
    assert top_of_book_row({**RAW, "bestBidQuote": None, "bestAskQuote": {}}, NOW) is None


class _FakeHttp:
    def __init__(self, total: int) -> None:
        self.total = total
        self.calls: list[dict] = []

    async def get_json(self, path: str, params: dict | None = None) -> dict:
        self.calls.append({"path": path, **(params or {})})
        offset = (params or {}).get("offset", 0)
        n = max(0, min(PAGE_SIZE, self.total - offset))
        return {"markets": [{"slug": f"m{offset + i}"} for i in range(n)]}


async def test_iter_open_markets_pages_until_a_short_page() -> None:
    http = _FakeHttp(total=450)
    adapter = PolymarketUsAdapter("https://x", SimulatedClock(NOW), http=http)  # type: ignore[arg-type]
    rows = await adapter.iter_open_markets(10_000)
    assert len(rows) == 450
    assert [c["offset"] for c in http.calls] == [0, 200, 400]
    assert all(c["path"] == "/v1/markets" and c["active"] == "true" for c in http.calls)
