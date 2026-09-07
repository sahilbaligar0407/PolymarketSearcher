from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from marketlab.adapters.polymarket_global.normalize import (
    _parse_json_field,
    normalize_activity,
    normalize_book,
    normalize_market,
)
from marketlab.core.instruments import Category, MarketStatus, Side, Venue


def test_parse_json_field_handles_stringified_and_real_lists() -> None:
    assert _parse_json_field('["Yes", "No"]') == ["Yes", "No"]
    assert _parse_json_field(["Yes", "No"]) == ["Yes", "No"]
    assert _parse_json_field(None) == []
    assert _parse_json_field("") == []
    assert _parse_json_field("not json") == []


def _gamma_market(**overrides: object) -> dict:
    base: dict = {
        "id": "2252243",
        "conditionId": "0xabc123",
        "question": "Will the Fed cut rates?",
        "slug": "fed-cut",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.35", "0.65"]',
        "clobTokenIds": '["1111", "2222"]',
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "orderPriceMinTickSize": 0.01,
        "orderMinSize": 5,
        "volumeNum": 1000.0,
        "liquidityNum": 500.0,
        "startDate": "2026-01-01T00:00:00Z",
        "endDate": "2026-12-31T00:00:00Z",
        "makerBaseFee": 0,
        "takerBaseFee": 1000,
        "feeSchedule": {"exponent": 1, "rate": 0.02, "takerOnly": True},
        "events": [{"id": "9", "tags": [{"slug": "economics"}]}],
    }
    base.update(overrides)
    return base


def test_normalize_market_gamma_shape_with_stringified_fields() -> None:
    market = normalize_market(_gamma_market())
    assert market.canonical_id == "poly:0xabc123"
    assert market.venue == Venue.POLY_GLOBAL
    assert market.yes_symbol == "Yes"
    assert market.no_symbol == "No"
    assert market.status == MarketStatus.OPEN
    assert market.category == Category.ECONOMICS
    assert market.tick_size == Decimal("0.01")
    assert market.min_order == 5
    # feeSchedule is trusted over the raw fixed-point base-fee ints when present.
    assert market.fees.formula == "polymarket_fee_schedule"
    assert market.fees.taker_rate == Decimal("0.02")
    assert market.fees.maker_rate == Decimal("0")
    assert market.volume == Decimal("1000.0")
    assert market.raw["conditionId"] == "0xabc123"


def test_normalize_market_clob_shape() -> None:
    raw = {
        "condition_id": "0xdef456",
        "question": "Will X happen?",
        "market_slug": "x-happen",
        "tokens": [{"token_id": "1", "outcome": "Yes"}, {"token_id": "2", "outcome": "No"}],
        "active": True,
        "closed": False,
        "accepting_orders": True,
        "minimum_tick_size": 0.001,
        "minimum_order_size": 5,
        "maker_base_fee": 0,
        "taker_base_fee": 2000,
    }
    market = normalize_market(raw)
    assert market.canonical_id == "poly:0xdef456"
    assert market.tick_size == Decimal("0.001")
    assert market.status == MarketStatus.OPEN
    assert market.fees.formula == "polymarket_base_fee_fixed_point"
    assert market.fees.taker_rate == Decimal("2000") / Decimal(1_000_000)


def test_normalize_market_falls_back_to_slug_without_condition_id() -> None:
    raw = _gamma_market(conditionId="", slug="fallback-slug")
    market = normalize_market(raw)
    assert market.canonical_id == "poly:fallback-slug"


def test_normalize_market_closed_status() -> None:
    raw = _gamma_market(active=True, closed=True, acceptingOrders=False)
    market = normalize_market(raw)
    assert market.status == MarketStatus.CLOSED


def test_category_falls_back_to_slug_inference_when_tags_are_absent() -> None:
    """Live Gamma payloads carry no `tags` field at all (verified 2026-09-06).

    Classifying purely on tags therefore returned OTHER for 100% of real markets, which
    silently starved the cross-venue matcher: it buckets candidates by category, so every
    Kalshi sports/crypto market searched an empty bucket and produced zero candidate
    pairs. The slug is the reliable signal on this venue.
    """
    sports = normalize_market(_gamma_market(events=[], slug="cfb-lou-miss-2026-09-06"))
    assert sports.category == Category.SPORTS

    crypto = normalize_market(_gamma_market(events=[], slug="will-bitcoin-hit-100k"))
    assert crypto.category == Category.CRYPTO

    # Genuinely unclassifiable input still falls through to OTHER rather than guessing.
    unknown = normalize_market(_gamma_market(events=[], slug="zzz-unclassifiable-thing"))
    assert unknown.category == Category.OTHER


def test_book_folding_yes_and_no_token_agree() -> None:
    """A YES-token book and the economically equivalent NO-token book must fold into
    identical YES-probability-terms order books (see normalize.py module docstring for
    the price -> 1-price, bids<->asks derivation)."""
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    yes_raw = {
        "market": "0xcond",
        "asset_id": "yes-token",
        "timestamp": "1735689600000",
        "bids": [{"price": "0.60", "size": "100"}, {"price": "0.59", "size": "20"}],
        "asks": [{"price": "0.62", "size": "50"}, {"price": "0.63", "size": "30"}],
    }
    no_raw = {
        "market": "0xcond",
        "asset_id": "no-token",
        "timestamp": "1735689600000",
        # NO bid at q == YES ask at 1-q; NO ask at q == YES bid at 1-q.
        "bids": [{"price": "0.38", "size": "50"}, {"price": "0.37", "size": "30"}],
        "asks": [{"price": "0.40", "size": "100"}, {"price": "0.41", "size": "20"}],
    }

    yes_book = normalize_book("yes-token", yes_raw, ts)
    no_book = normalize_book("no-token", no_raw, ts, is_no_token=True)

    assert yes_book.canonical_id == no_book.canonical_id == "poly:0xcond"
    assert yes_book.best_bid == no_book.best_bid == Decimal("0.6000")
    assert yes_book.best_ask == no_book.best_ask == Decimal("0.6200")
    assert [lvl.price for lvl in yes_book.bids] == [lvl.price for lvl in no_book.bids]
    assert [lvl.size for lvl in yes_book.bids] == [lvl.size for lvl in no_book.bids]
    assert [lvl.price for lvl in yes_book.asks] == [lvl.price for lvl in no_book.asks]
    assert [lvl.size for lvl in yes_book.asks] == [lvl.size for lvl in no_book.asks]
    # Bids descending, asks ascending, regardless of raw order.
    assert yes_book.bids[0].price > yes_book.bids[1].price
    assert yes_book.asks[0].price < yes_book.asks[1].price


def test_book_levels_stay_within_probability_bounds() -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    raw = {
        "market": "0xcond",
        "bids": [{"price": "1.5", "size": "10"}, {"price": "0.5", "size": "5"}],
        "asks": [{"price": "-0.1", "size": "10"}],
    }
    book = normalize_book("tok", raw, ts)
    # Malformed out-of-[0,1] levels are dropped defensively rather than crashing.
    assert len(book.bids) == 1
    assert book.bids[0].price == Decimal("0.5000")
    assert len(book.asks) == 0
    for level in (*book.bids, *book.asks):
        assert Decimal("0") <= level.price <= Decimal("1")


def test_normalize_activity_sets_first_seen_from_clock_not_onchain_timestamp() -> None:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    raw = {
        "proxyWallet": "0xwallet",
        "timestamp": 1700000000,  # on-chain trade time, long before `now`
        "conditionId": "0xcond",
        "type": "TRADE",
        "side": "BUY",
        "size": 10,
        "usdcSize": 5.5,
        "price": "0.55",
        "asset": "tok1",
        "outcome": "Yes",
        "title": "Some market",
        "name": "trader1",
        "transactionHash": "0xhash",
    }
    event = normalize_activity(raw, now)
    assert event.first_seen_time == now
    assert event.event_time != now
    assert event.event_time < now
    assert event.side == Side.YES
    assert event.price == Decimal("0.5500")
    assert event.wallet == "0xwallet"
    assert event.username == "trader1"
    assert event.action == "BUY"
    assert event.canonical_id == "poly:0xcond"
    assert event.usd_size == Decimal("5.5")


def test_normalize_activity_handles_missing_price_gracefully() -> None:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    raw = {"proxyWallet": "0xw", "timestamp": 1700000000, "type": "REWARD", "outcome": ""}
    event = normalize_activity(raw, now)
    assert event.price is None
    assert event.side is None
    assert event.first_seen_time == now
