"""Kalshi payload -> universal ``marketlab.core.instruments`` normalization.

**API-shape note (verified 2026-09-04 against the live production API):** the prompt
this adapter was built against assumed Kalshi quotes plain integer cents (``yes_bid``,
``no_bid``, ...) and that ``GET .../orderbook`` returns
``{"orderbook": {"yes": [[cents, size]], "no": [[cents, size]]}}``. The *current* live
API has moved to dollar-denominated decimal strings with a ``_dollars`` suffix
(``yes_bid_dollars``, ``last_price_dollars``, ...) and fractional contract sizes with a
``_fp`` suffix (``volume_fp``, ``yes_bid_size_fp``, ...); the orderbook endpoint now
returns ``{"orderbook_fp": {"yes_dollars": [[price_str, size_str]], "no_dollars": [...]}}``.
Every helper below tries the modern ``_dollars``/``_fp`` fields first and falls back to
the legacy integer-cents shape the prompt described, so this keeps working if Kalshi
reverts or if some other endpoint still returns the legacy shape. See the adapter
report for the full list of observed differences.

Kalshi's fundamental convention is unchanged: **both the ``yes`` and ``no`` arrays in an
orderbook are bids.** A NO bid at price ``p`` is economically a resting order willing to
buy NO at ``p``, which is equivalent to a standing YES *ask* at ``1 - p`` (someone selling
YES for ``1-p`` and someone buying NO at ``p`` are the same trade). We fold NO bids into
YES asks here so every ``OrderBook`` leaving this module is entirely in YES-probability
terms, per the shared contract in ``marketlab.core.instruments``.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal

from marketlab.adapters.kalshi.fees import fees_for_market
from marketlab.core.instruments import (
    ONE,
    ZERO,
    BookLevel,
    Category,
    MarketStatus,
    NormalizedMarket,
    OrderBook,
    OutcomeType,
    Side,
    Trade,
    Venue,
    cents_to_probability,
    to_probability,
)

# ---------------------------------------------------------------------------
# Canonical id
# ---------------------------------------------------------------------------


def make_canonical_id(venue: Venue, market_id: str) -> str:
    """Deterministic, stable canonical id. Other teams key their state off this."""
    return f"{venue.value}:{market_id.lower()}"


# ---------------------------------------------------------------------------
# Small parsing helpers - each tries the modern field shape, then the legacy one.
# ---------------------------------------------------------------------------


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    v = value.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _probability_field(raw: dict, base_name: str) -> Decimal | None:
    """A price-like field (yes_bid, yes_ask, last_price, ...) as a probability."""
    dollars_key = f"{base_name}_dollars"
    if raw.get(dollars_key) not in (None, ""):
        try:
            return to_probability(Decimal(str(raw[dollars_key])))
        except (ValueError, ArithmeticError):
            return None
    if raw.get(base_name) not in (None, ""):
        val = raw[base_name]
        try:
            if isinstance(val, int):
                return cents_to_probability(val)
            return to_probability(Decimal(str(val)))
        except (ValueError, ArithmeticError):
            return None
    return None


def _fractional_count(raw: dict, base_name: str) -> Decimal:
    """A fractional contract-count field such as ``open_interest``.

    Kept separate from :func:`_count_field`, which rounds to ``int`` because
    ``BookLevel.size`` and trade counts are integers.  Open interest is only ever read as
    a liquidity signal, so its fractional precision is worth preserving.  Also distinct
    from :func:`_money_field`: a legacy bare integer here is already a contract count and
    must not be divided by 100 the way a cents-denominated money field would be.
    """
    for key in (f"{base_name}_fp", base_name):
        value = raw.get(key)
        if value in (None, ""):
            continue
        try:
            return Decimal(str(value))
        except (ValueError, ArithmeticError):
            continue
    return ZERO


def _money_field(raw: dict, base_name: str) -> Decimal:
    """A dollar-valued (non-probability) field such as liquidity/volume."""
    for suffix in ("_dollars", "_fp"):
        key = f"{base_name}{suffix}"
        if raw.get(key) not in (None, ""):
            try:
                return Decimal(str(raw[key]))
            except (ValueError, ArithmeticError):
                pass
    if raw.get(base_name) not in (None, ""):
        val = raw[base_name]
        try:
            if isinstance(val, int):
                return Decimal(val) / Decimal(100)
            return Decimal(str(val))
        except (ValueError, ArithmeticError):
            pass
    return ZERO


def _count_field(raw: dict, base_name: str) -> int:
    """An integer contract count, tolerating the ``_fp`` fractional-string shape."""
    for suffix in ("_fp", ""):
        key = f"{base_name}{suffix}"
        if raw.get(key) not in (None, ""):
            try:
                return int(Decimal(str(raw[key])).to_integral_value(rounding=ROUND_HALF_UP))
            except (ValueError, ArithmeticError):
                pass
    return 0


def _infer_tick_size(raw: dict) -> Decimal:
    """Finest price increment. Modern payloads expose ``price_ranges`` (variable tick
    across price bands, e.g. penny ticks near 0/1 and dime ticks in the middle);
    legacy payloads expose a flat ``tick_size`` in cents."""
    ranges = raw.get("price_ranges")
    if isinstance(ranges, list) and ranges:
        steps = []
        for r in ranges:
            step = r.get("step") if isinstance(r, dict) else None
            if step is not None:
                with contextlib.suppress(ValueError, ArithmeticError):
                    steps.append(Decimal(str(step)))
        if steps:
            return min(steps)
    legacy = raw.get("tick_size")
    if legacy is not None:
        try:
            legacy_dec = Decimal(str(legacy))
            return legacy_dec / Decimal(100) if legacy_dec >= 1 else legacy_dec
        except (ValueError, ArithmeticError):
            pass
    return Decimal("0.01")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

#: Table-driven so a newly observed Kalshi status string is a one-line change.
_STATUS_MAP: dict[str, MarketStatus] = {
    "active": MarketStatus.OPEN,
    "open": MarketStatus.OPEN,
    "initialized": MarketStatus.UNOPENED,
    "unopened": MarketStatus.UNOPENED,
    "closed": MarketStatus.CLOSED,
    "settled": MarketStatus.SETTLED,
    # Observed on the live API 2026-09-04: settled markets report status "finalized",
    # not "settled" as one might assume from the settlement endpoint naming.
    "finalized": MarketStatus.SETTLED,
    "determined": MarketStatus.CLOSED,
    "paused": MarketStatus.PAUSED,
    "canceled": MarketStatus.CANCELED,
    "cancelled": MarketStatus.CANCELED,
}


def _map_status(raw_status: object) -> MarketStatus:
    return _STATUS_MAP.get(str(raw_status).strip().lower(), MarketStatus.UNKNOWN)


# ---------------------------------------------------------------------------
# Category inference - table-driven, ordered most-specific-first, easy to extend.
# ---------------------------------------------------------------------------

_TICKER_PREFIX_RULES: tuple[tuple[str, Category], ...] = (
    # Crypto
    ("KXBTC", Category.CRYPTO),
    ("KXETH", Category.CRYPTO),
    ("KXSOL", Category.CRYPTO),
    ("KXDOGE", Category.CRYPTO),
    ("KXXRP", Category.CRYPTO),
    ("KXBNB", Category.CRYPTO),
    ("KXNEAR", Category.CRYPTO),
    ("KXCRYPTO", Category.CRYPTO),
    # Sports (traditional + esports, which we treat as sports: competitive, scored, timed)
    ("KXNFL", Category.SPORTS),
    ("KXNBA", Category.SPORTS),
    ("KXMLB", Category.SPORTS),
    ("KXNHL", Category.SPORTS),
    ("KXNCAAF", Category.SPORTS),
    ("KXNCAAB", Category.SPORTS),
    ("KXATP", Category.SPORTS),
    ("KXWTA", Category.SPORTS),
    ("KXEPL", Category.SPORTS),
    ("KXBUNDESLIGA", Category.SPORTS),
    ("KXLALIGA", Category.SPORTS),
    ("KXLIGUE1", Category.SPORTS),
    ("KXSERIEA", Category.SPORTS),
    ("KXSAUDIPL", Category.SPORTS),
    ("KXELITESERIEN", Category.SPORTS),
    ("KXPERLIGA", Category.SPORTS),
    ("KXCS2", Category.SPORTS),
    ("KXVALORANT", Category.SPORTS),
    # Economics
    ("KXCPI", Category.ECONOMICS),
    ("KXGDP", Category.ECONOMICS),
    ("KXFED", Category.ECONOMICS),
    ("KXJOBS", Category.ECONOMICS),
    ("KXPAYROLL", Category.ECONOMICS),
    ("KXUNEMPLOYMENT", Category.ECONOMICS),
    ("KXPCE", Category.ECONOMICS),
    # Weather
    ("KXHIGH", Category.WEATHER),
    ("KXLOW", Category.WEATHER),
    ("KXWEATHER", Category.WEATHER),
    ("KXRAIN", Category.WEATHER),
    ("KXSNOW", Category.WEATHER),
    ("KXHURRICANE", Category.WEATHER),
    # Finance / commodities / indices (no dedicated Category, closest fit is FINANCE)
    ("KXGOLD", Category.FINANCE),
    ("KXSILVER", Category.FINANCE),
    ("KXCOPPER", Category.FINANCE),
    ("KXWTI", Category.FINANCE),
    ("KXOIL", Category.FINANCE),
    ("KXSPX", Category.FINANCE),
    ("KXNASDAQ", Category.FINANCE),
    # Politics
    ("KXPRES", Category.POLITICS),
    ("KXSPEAKER", Category.POLITICS),
    ("KXSENATE", Category.POLITICS),
    ("KXHOUSE", Category.POLITICS),
    ("KXGOV", Category.POLITICS),
    ("KXELECTION", Category.POLITICS),
    # Tech
    ("KXTECH", Category.TECH),
    ("KXAI", Category.TECH),
)

#: Fallback when the ticker prefix doesn't match: substring search on the title.
_TITLE_KEYWORD_RULES: tuple[tuple[str, Category], ...] = (
    ("bitcoin", Category.CRYPTO),
    ("ethereum", Category.CRYPTO),
    ("crypto", Category.CRYPTO),
    ("election", Category.POLITICS),
    ("president", Category.POLITICS),
    ("senate", Category.POLITICS),
    ("inflation", Category.ECONOMICS),
    ("cpi", Category.ECONOMICS),
    ("fed ", Category.ECONOMICS),
    ("temperature", Category.WEATHER),
    ("hurricane", Category.WEATHER),
    ("rain", Category.WEATHER),
)


def infer_category(raw: dict) -> Category:
    """Category from series/event ticker prefix, then title keywords, then OTHER."""
    candidates = (
        str(raw.get("series_ticker") or "").upper(),
        str(raw.get("event_ticker") or "").upper(),
        str(raw.get("ticker") or "").upper(),
    )
    for candidate in candidates:
        for prefix, category in _TICKER_PREFIX_RULES:
            if candidate.startswith(prefix):
                return category

    title = str(raw.get("title") or "").lower()
    for keyword, category in _TITLE_KEYWORD_RULES:
        if keyword in title:
            return category
    return Category.OTHER


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------


def normalize_market(raw: dict) -> NormalizedMarket:
    """Kalshi ``/markets`` row -> :class:`NormalizedMarket`."""
    ticker = raw["ticker"]
    event_ticker = raw.get("event_ticker", "") or ""
    rules = "\n\n".join(
        p for p in (raw.get("rules_primary") or "", raw.get("rules_secondary") or "") if p
    )
    max_payout = _money_field(raw, "notional_value")
    if max_payout == ZERO:
        max_payout = ONE

    return NormalizedMarket(
        canonical_id=make_canonical_id(Venue.KALSHI, ticker),
        venue=Venue.KALSHI,
        venue_market_id=ticker,
        event_id=event_ticker,
        title=str(raw.get("title") or ""),
        description=str(raw.get("subtitle") or ""),
        resolution_rules=rules,
        resolution_source="",
        category=infer_category(raw),
        subcategory=str(raw.get("series_ticker") or ""),
        outcome_type=OutcomeType.BINARY,
        yes_symbol=str(raw.get("yes_sub_title") or ""),
        no_symbol=str(raw.get("no_sub_title") or ""),
        open_time=_parse_dt(raw.get("open_time")),
        close_time=_parse_dt(raw.get("close_time")),
        expected_resolution_time=(
            _parse_dt(raw.get("expiration_time")) or _parse_dt(raw.get("expected_expiration_time"))
        ),
        timezone="UTC",
        tick_size=_infer_tick_size(raw),
        min_order=1,
        max_payout_per_contract=max_payout,
        status=_map_status(raw.get("status")),
        fees=fees_for_market(raw),
        liquidity=_money_field(raw, "liquidity"),
        volume=_money_field(raw, "volume"),
        open_interest=_fractional_count(raw, "open_interest"),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------


def normalize_orderbook(ticker: str, raw: dict, timestamp: datetime) -> OrderBook:
    """Kalshi orderbook response -> YES-probability-terms :class:`OrderBook`.

    Accepts either the modern ``{"orderbook_fp": {"yes_dollars": [...], "no_dollars":
    [...]}}`` shape or the legacy ``{"orderbook": {"yes": [...], "no": [...]}}`` shape
    (some endpoints, or a demo environment, may still return the latter). Both ``yes``
    and ``no`` arrays from Kalshi are BIDS; NO bids are folded into YES asks at
    ``1 - price`` here so the result is entirely in YES terms.
    """
    ob = raw.get("orderbook_fp")
    is_fp = ob is not None
    if ob is None:
        ob = raw.get("orderbook") or {}
    yes_key = "yes_dollars" if is_fp else "yes"
    no_key = "no_dollars" if is_fp else "no"
    yes_levels = ob.get(yes_key) or []
    no_levels = ob.get(no_key) or []

    def _level(pair: list) -> tuple[Decimal, int]:
        price_raw, size_raw = pair[0], pair[1]
        price = (
            to_probability(Decimal(str(price_raw)))
            if is_fp
            else cents_to_probability(int(price_raw))
        )
        size = int(Decimal(str(size_raw)).to_integral_value(rounding=ROUND_HALF_UP))
        return price, size

    bids = tuple(
        sorted(
            (BookLevel(price=p, size=s) for p, s in (_level(pair) for pair in yes_levels)),
            key=lambda lvl: lvl.price,
            reverse=True,
        )
    )
    asks = tuple(
        sorted(
            (
                BookLevel(price=to_probability(ONE - p), size=s)
                for p, s in (_level(pair) for pair in no_levels)
            ),
            key=lambda lvl: lvl.price,
        )
    )
    sequence = raw.get("sequence") if isinstance(raw.get("sequence"), int) else None
    return OrderBook(
        canonical_id=make_canonical_id(Venue.KALSHI, ticker),
        venue=Venue.KALSHI,
        timestamp=timestamp,
        bids=bids,
        asks=asks,
        venue_timestamp=None,
        sequence=sequence,
    )


# ---------------------------------------------------------------------------
# Trades / settlement
# ---------------------------------------------------------------------------


def normalize_trade(raw: dict) -> Trade:
    """Kalshi trade row -> :class:`Trade`, priced in YES-probability terms."""
    ticker = raw["ticker"]
    price = _probability_field(raw, "yes_price")
    if price is None:
        no_price = _probability_field(raw, "no_price")
        price = to_probability(ONE - no_price) if no_price is not None else ZERO

    taker_side = str(raw.get("taker_side") or "").strip().lower()
    aggressor = Side.YES if taker_side == "yes" else (Side.NO if taker_side == "no" else None)

    timestamp = _parse_dt(raw.get("created_time")) or _parse_dt(raw.get("ts"))
    if timestamp is None:
        raise ValueError(f"Kalshi trade for {ticker!r} has no parseable timestamp")

    return Trade(
        canonical_id=make_canonical_id(Venue.KALSHI, ticker),
        venue=Venue.KALSHI,
        timestamp=timestamp,
        price=price,
        size=_count_field(raw, "count"),
        aggressor=aggressor,
        trade_id=str(raw.get("trade_id") or ""),
    )


def normalize_settlement(raw: dict) -> tuple[Side | None, bool]:
    """A market's ``result`` field -> ``(winning_side, voided)``.

    ``result`` is ``"yes"``/``"no"`` once settled, ``""`` before settlement, and one of
    a few void/cancel spellings if the market was voided. Returns ``(None, False)`` for
    "not settled yet" and ``(None, True)`` for "settled void" - callers distinguish the
    two by whether the market's status is terminal.
    """
    result = str(raw.get("result") or "").strip().lower()
    if result == "yes":
        return Side.YES, False
    if result == "no":
        return Side.NO, False
    if result in ("void", "voided", "canceled", "cancelled"):
        return None, True
    return None, False
