"""Unit tests for the Kalshi normalization layer.

Fixtures below are trimmed real payloads captured from the live production API
(``https://api.elections.kalshi.com/trade-api/v2``) on 2026-09-04 via curl - see the
adapter report for the full curl transcript. Notably the live API now uses
dollar-denominated string fields (``yes_bid_dollars``, ...) and fractional-size ``_fp``
fields, not the integer-cents shape the original spec assumed; both shapes are covered.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from marketlab.adapters.kalshi.normalize import (
    infer_category,
    make_canonical_id,
    normalize_market,
    normalize_orderbook,
    normalize_settlement,
    normalize_trade,
)
from marketlab.core.instruments import Category, MarketStatus, Side, Venue, cents_to_probability

# --- Modern (dollar-string) market fixture, trimmed from a live /markets row ---------
BTC_MARKET_MODERN = {
    "ticker": "KXBTCD-26SEP0413-T86799.99",
    "event_ticker": "KXBTCD-26SEP0413",
    "series_ticker": "KXBTCD",
    "title": "Bitcoin price on Sep 4, 2026?",
    "subtitle": "$86,800 or above",
    "yes_sub_title": "$86,800 or above",
    "no_sub_title": "$86,800 or above",
    "rules_primary": "If the simple average ... is above 86799.99 ... resolves Yes.",
    "rules_secondary": "Not all cryptocurrency price data is the same.",
    "status": "active",
    "open_time": "2026-09-04T16:00:00Z",
    "close_time": "2026-09-04T17:00:00Z",
    "expiration_time": "2026-09-11T17:00:00Z",
    "yes_bid_dollars": "0.0000",
    "yes_ask_dollars": "0.0100",
    "no_bid_dollars": "0.9900",
    "no_ask_dollars": "1.0000",
    "last_price_dollars": "0.0000",
    "liquidity_dollars": "125.5000",
    "volume_fp": "5000.00",
    "volume_24h_fp": "0.00",
    "open_interest_fp": "2500.00",
    "notional_value_dollars": "1.0000",
    "price_ranges": [
        {"start": "0.0000", "end": "0.0100", "step": "0.0001"},
        {"start": "0.0100", "end": "0.9900", "step": "0.0010"},
        {"start": "0.9900", "end": "1.0000", "step": "0.0001"},
    ],
    "result": "",
}

NFL_MARKET_MODERN = {
    "ticker": "KXNFLGAME-26SEP07KCBAL-KC",
    "event_ticker": "KXNFLGAME-26SEP07KCBAL",
    "series_ticker": "KXNFLGAME",
    "title": "Chiefs vs Ravens winner?",
    "subtitle": "Kansas City Chiefs",
    "yes_sub_title": "Kansas City Chiefs",
    "no_sub_title": "Baltimore Ravens",
    "rules_primary": "",
    "rules_secondary": "",
    "status": "initialized",
    "open_time": "2026-09-01T00:00:00Z",
    "close_time": "2026-09-07T20:00:00Z",
    "expiration_time": "2026-09-08T00:00:00Z",
    "yes_bid_dollars": "0.5500",
    "yes_ask_dollars": "0.5700",
    "no_bid_dollars": "0.4300",
    "no_ask_dollars": "0.4500",
    "liquidity_dollars": "0.0000",
    "volume_fp": "0.00",
    "notional_value_dollars": "1.0000",
    "result": "",
}

# A finalized (settled) market as returned live - status is "finalized", not "settled".
SETTLED_MARKET_MODERN = {
    "ticker": "KXBTC15M-26SEP041230-30",
    "event_ticker": "KXBTC15M-26SEP041230",
    "series_ticker": "KXBTC15M",
    "title": "Bitcoin above target at 12:30?",
    "status": "finalized",
    "result": "no",
    "settlement_ts": "2026-09-04T16:15:45Z",
    "settlement_value_dollars": "0.0000",
    "yes_bid_dollars": "0.0000",
    "yes_ask_dollars": "1.0000",
    "no_bid_dollars": "0.0000",
    "no_ask_dollars": "1.0000",
    "notional_value_dollars": "1.0000",
}

# --- Legacy (integer cents) shape, in case some endpoint still returns it -----------
LEGACY_MARKET = {
    "ticker": "KXHIGHNY-26SEP04-B75",
    "event_ticker": "KXHIGHNY-26SEP04",
    "series_ticker": "KXHIGHNY",
    "title": "Highest temperature in NYC today?",
    "status": "open",
    "yes_bid": 42,
    "yes_ask": 45,
    "no_bid": 55,
    "no_ask": 58,
    "last_price": 44,
    "liquidity": 1000,  # legacy cents
    "volume": 500,  # legacy cents (dollars = 5.00)
    "tick_size": 1,  # legacy: 1 cent
    "notional_value": 100,
}

ORDERBOOK_MODERN = {
    "orderbook_fp": {
        "no_dollars": [["0.4100", "1.00"], ["0.5000", "41.00"], ["0.9900", "28427.00"]],
        "yes_dollars": [["0.3000", "10.00"], ["0.2000", "5.00"]],
    }
}

ORDERBOOK_LEGACY = {
    "orderbook": {
        "yes": [[30, 10], [20, 5]],
        "no": [[41, 1], [50, 41], [99, 28427]],
    }
}

EMPTY_ORDERBOOK = {"orderbook_fp": {"yes_dollars": [], "no_dollars": None}}

TRADE_MODERN = {
    "ticker": "KXBTC15M-26SEP041230-30",
    "trade_id": "0721d9e2-bb39-8ebe-6b64-885eff1f73d0",
    "created_time": "2026-09-04T16:17:21.7293Z",
    "yes_price_dollars": "0.4300",
    "no_price_dollars": "0.5700",
    "count_fp": "10.00",
    "taker_side": "no",
}


def test_cents_to_probability_conversion() -> None:
    assert cents_to_probability(0) == Decimal("0.0000")
    assert cents_to_probability(1) == Decimal("0.0100")
    assert cents_to_probability(50) == Decimal("0.5000")
    assert cents_to_probability(100) == Decimal("1.0000")

    # Legacy market fixture: tick_size in cents (1) converts to Decimal("0.01").
    market = normalize_market(LEGACY_MARKET)
    assert market.tick_size == Decimal("0.01")


def test_normalize_market_modern_dollar_fields() -> None:
    market = normalize_market(BTC_MARKET_MODERN)
    assert market.canonical_id == "kalshi:kxbtcd-26sep0413-t86799.99"
    assert market.venue == Venue.KALSHI
    assert market.venue_market_id == "KXBTCD-26SEP0413-T86799.99"
    assert market.event_id == "KXBTCD-26SEP0413"
    assert market.status == MarketStatus.OPEN
    assert market.title == "Bitcoin price on Sep 4, 2026?"
    assert market.liquidity == Decimal("125.5000")
    assert market.volume == Decimal("5000.00")
    assert market.tick_size == Decimal("0.0001")  # finest step in price_ranges
    assert market.category == Category.CRYPTO


def test_normalize_market_legacy_cents_fields() -> None:
    market = normalize_market(LEGACY_MARKET)
    assert market.canonical_id == "kalshi:kxhighny-26sep04-b75"
    assert market.liquidity == Decimal("10.00")  # 1000 cents -> $10.00
    assert market.volume == Decimal("5.00")  # 500 cents -> $5.00
    assert market.category == Category.WEATHER


def test_status_mapping_table() -> None:
    assert normalize_market(NFL_MARKET_MODERN).status == MarketStatus.UNOPENED
    # "finalized" is what the live API actually returns for a resolved market.
    assert normalize_market(SETTLED_MARKET_MODERN).status == MarketStatus.SETTLED
    assert normalize_market(LEGACY_MARKET).status == MarketStatus.OPEN


def test_canonical_id_format_is_lowercase_venue_prefixed() -> None:
    assert make_canonical_id(Venue.KALSHI, "KXBTCD-26SEP0413-T86799.99") == (
        "kalshi:kxbtcd-26sep0413-t86799.99"
    )


def test_category_inference_crypto_and_sports() -> None:
    assert infer_category(BTC_MARKET_MODERN) == Category.CRYPTO
    assert infer_category(NFL_MARKET_MODERN) == Category.SPORTS


def test_category_inference_falls_back_to_other() -> None:
    assert infer_category({"ticker": "KXWEIRDTHING-1", "title": "Something unrelated"}) == (
        Category.OTHER
    )


def test_orderbook_no_bids_fold_into_yes_asks_modern() -> None:
    ts = datetime(2026, 9, 4, 16, 20, tzinfo=UTC)
    book = normalize_orderbook("KXBTCD-26SEP0413-T86799.99", ORDERBOOK_MODERN, ts)

    # YES bids: straightforward, sorted descending.
    assert [lvl.price for lvl in book.bids] == [Decimal("0.3000"), Decimal("0.2000")]
    assert [lvl.size for lvl in book.bids] == [10, 5]

    # NO bids at 0.41 / 0.50 / 0.99 become YES asks at 0.59 / 0.50 / 0.01, ascending.
    assert [lvl.price for lvl in book.asks] == [Decimal("0.0100"), Decimal("0.5000"), Decimal("0.5900")]
    assert [lvl.size for lvl in book.asks] == [28427, 41, 1]

    assert book.best_bid == Decimal("0.3000")
    assert book.best_ask == Decimal("0.0100")
    assert book.canonical_id == "kalshi:kxbtcd-26sep0413-t86799.99"
    assert book.venue == Venue.KALSHI


def test_orderbook_no_bids_fold_into_yes_asks_legacy_cents() -> None:
    ts = datetime(2026, 9, 4, 16, 20, tzinfo=UTC)
    book = normalize_orderbook("KXHIGHNY-26SEP04-B75", ORDERBOOK_LEGACY, ts)

    assert [lvl.price for lvl in book.bids] == [Decimal("0.3000"), Decimal("0.2000")]
    # NO bids at 41c/50c/99c -> YES asks at 0.59/0.50/0.01
    assert [lvl.price for lvl in book.asks] == [Decimal("0.0100"), Decimal("0.5000"), Decimal("0.5900")]
    assert [lvl.size for lvl in book.asks] == [28427, 41, 1]


def test_orderbook_empty_sides_yield_empty_tuples() -> None:
    ts = datetime(2026, 9, 4, 16, 20, tzinfo=UTC)
    book = normalize_orderbook("KXEMPTY-1", EMPTY_ORDERBOOK, ts)
    assert book.bids == ()
    assert book.asks == ()
    assert book.best_bid is None
    assert book.best_ask is None


def test_orderbook_missing_key_entirely_yields_empty_tuples() -> None:
    ts = datetime(2026, 9, 4, 16, 20, tzinfo=UTC)
    book = normalize_orderbook("KXEMPTY-2", {}, ts)
    assert book.bids == ()
    assert book.asks == ()


def test_normalize_trade_modern() -> None:
    trade = normalize_trade(TRADE_MODERN)
    assert trade.canonical_id == "kalshi:kxbtc15m-26sep041230-30"
    assert trade.price == Decimal("0.4300")
    assert trade.size == 10
    assert trade.aggressor == Side.NO
    assert trade.trade_id == "0721d9e2-bb39-8ebe-6b64-885eff1f73d0"


def test_normalize_settlement_yes_no_and_unsettled() -> None:
    assert normalize_settlement({"result": "yes"}) == (Side.YES, False)
    assert normalize_settlement({"result": "no"}) == (Side.NO, False)
    assert normalize_settlement({"result": ""}) == (None, False)
    assert normalize_settlement({"result": "void"}) == (None, True)
