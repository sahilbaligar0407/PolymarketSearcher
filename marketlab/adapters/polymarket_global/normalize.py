"""Polymarket (global) payload normalization.

Gamma and the CLOB return two different market shapes (camelCase JSON blobs with
stringified list fields vs. the CLOB's snake_case ``tokens`` array), and both need to
land on the same :class:`~marketlab.core.instruments.NormalizedMarket`.  This module is
pure data transformation - no network I/O - so it has no dependency on ``adapters.base``
and can be unit tested without a live adapter or a running event loop.

Order-book folding convention (see also ``core/instruments.py``): Polymarket books are
per-token - the YES outcome token and the NO outcome token each have their own book on
the CLOB.  Given the **YES token's** book, its bids are YES bids and its asks are YES
asks directly: no transformation needed.  Given the **NO token's** book, a NO bid at
price ``q`` is economically identical to a YES ask at ``1-q`` (buying NO == selling
YES), and a NO ask at price ``q`` is identical to a YES bid at ``1-q`` (selling NO ==
buying YES).  So folding a NO-token book inverts price (``p -> 1-p``) *and* swaps
bids/asks.  ``normalize_book(..., is_no_token=True)`` performs exactly that swap.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from marketlab.core.events import SourceClass, TraderActionEvent
from marketlab.core.instruments import (
    BookLevel,
    Category,
    Fees,
    MarketStatus,
    NormalizedMarket,
    OrderBook,
    OutcomeType,
    Side,
    Venue,
    to_probability,
)

ZERO = Decimal(0)
ONE = Decimal(1)

# Polymarket's on-chain base-fee fields (``makerBaseFee`` / ``takerBaseFee`` on Gamma,
# ``maker_base_fee`` / ``taker_base_fee`` on the CLOB) are fixed-point integers scaled by
# 1e6 (observed empirically: a market with ``feeSchedule.rate == 0.04`` reports
# ``takerBaseFee == 1000`` alongside it in one probed case, which is *not* 1000/1e6 -
# so wherever ``feeSchedule`` is present we trust it, since it names an explicit decimal
# ``rate``; the base-fee-int fallback is only used when no schedule is published, and is
# clearly labelled as a fixed-point-assumption fallback rather than an authoritative
# figure. Either way, this is fetched from the API response, never hand-typed.
_BASE_FEE_SCALE = Decimal(1_000_000)

# Keyword tables mapping Gamma/CLOB tag slugs to the shared Category enum.  Order
# matters: the first matching bucket wins, so more specific buckets are listed first.
_CATEGORY_KEYWORDS: tuple[tuple[Category, tuple[str, ...]], ...] = (
    (Category.CRYPTO, ("crypto", "bitcoin", "ethereum", "btc", "eth", "defi", "altcoin", "solana")),
    (
        Category.SPORTS,
        (
            "sports", "nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball",
            "tennis", "ufc", "mma", "boxing", "olympics", "world-cup", "golf", "cricket",
        ),
    ),
    (Category.POLITICS, ("politics", "elections", "election", "congress", "senate", "president", "geopolitics")),
    (Category.ECONOMICS, ("economics", "fed", "inflation", "recession", "economy", "gdp", "interest-rates", "fomc", "jobs-report")),
    (Category.FINANCE, ("finance", "business", "stocks", "markets", "earnings", "ipos", "ipo", "companies")),
    (Category.WEATHER, ("weather", "climate", "hurricane", "temperature")),
    (Category.TECH, ("tech", "technology", "ai", "science-tech", "space-tech")),
    (Category.ENTERTAINMENT, ("entertainment", "movies", "tv", "music", "awards", "pop-culture", "celebrity")),
    (Category.SCIENCE, ("science", "space", "nasa")),
)


def _parse_json_field(value: Any) -> list[Any]:
    """Gamma often returns list-typed fields (``outcomes``, ``outcomePrices``,
    ``clobTokenIds``) as JSON-encoded strings rather than actual JSON arrays.  Parse
    defensively: accept a real list as-is, parse a string, and fall back to ``[]``.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _to_decimal(value: Any, default: Decimal = ZERO) -> Decimal:
    if value is None or value == "":
        return default
    try:
        return Decimal(str(value))
    except Exception:
        return default


def _to_decimal_or_none(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _parse_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _parse_unix_seconds(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=UTC)
    except (TypeError, ValueError, OverflowError):
        return None


def _parse_unix_ms(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
    except (TypeError, ValueError, OverflowError):
        return None


def _extract_tags(raw: Mapping[str, Any]) -> list[str]:
    tags: list[str] = []
    for t in raw.get("tags") or []:
        if isinstance(t, dict) and t.get("slug"):
            tags.append(str(t["slug"]).lower())
    for ev in raw.get("events") or []:
        if not isinstance(ev, dict):
            continue
        for t in ev.get("tags") or []:
            if isinstance(t, dict) and t.get("slug"):
                tags.append(str(t["slug"]).lower())
        if ev.get("category"):
            tags.append(str(ev["category"]).lower())
    if raw.get("category"):
        tags.append(str(raw["category"]).lower())
    return tags


def _category_from_tags(tags: Sequence[str]) -> Category:
    tagset = set(tags)
    for category, keywords in _CATEGORY_KEYWORDS:
        if tagset & set(keywords):
            return category
        # Also allow substring matches (e.g. tag "nba-2026" contains "nba").
        for tag in tagset:
            if any(kw in tag for kw in keywords):
                return category
    return Category.OTHER


def _extract_fees(fee_schedule: Any, maker_base_fee: Any, taker_base_fee: Any) -> Fees:
    """Fees are always fetched from the venue payload, never guessed.

    Prefer the explicit ``feeSchedule`` block Gamma publishes on the market object
    (``{"rate": 0.04, "takerOnly": true, ...}``); fall back to the fixed-point base-fee
    integers when no schedule is present.
    """
    if isinstance(fee_schedule, dict) and "rate" in fee_schedule:
        rate = _to_decimal(fee_schedule.get("rate"), ZERO)
        taker_only = bool(fee_schedule.get("takerOnly", True))
        return Fees(
            formula="polymarket_fee_schedule",
            taker_rate=rate,
            maker_rate=ZERO if taker_only else rate,
        )
    maker_rate = _to_decimal(maker_base_fee, ZERO) / _BASE_FEE_SCALE
    taker_rate = _to_decimal(taker_base_fee, ZERO) / _BASE_FEE_SCALE
    return Fees(formula="polymarket_base_fee_fixed_point", maker_rate=maker_rate, taker_rate=taker_rate)


def _market_status(active: bool, closed: bool, accepting: bool) -> MarketStatus:
    if closed:
        return MarketStatus.CLOSED
    if active and accepting:
        return MarketStatus.OPEN
    if active:
        return MarketStatus.PAUSED
    return MarketStatus.UNOPENED


def normalize_market(raw: Mapping[str, Any], *, category_hint: Category | None = None) -> NormalizedMarket:
    """Normalize a Gamma or CLOB market payload into a :class:`NormalizedMarket`.

    ``canonical_id`` is ``poly:{condition_id}``, falling back to the slug when a
    condition id is unavailable (e.g. a not-yet-deployed market).
    """
    is_clob_shape = "condition_id" in raw and "conditionId" not in raw

    if is_clob_shape:
        condition_id = str(raw.get("condition_id", ""))
        question = str(raw.get("question", ""))
        slug = str(raw.get("market_slug", ""))
        tokens = raw.get("tokens") or []
        outcomes = [str(t.get("outcome", "")) for t in tokens if isinstance(t, dict)]
        description = str(raw.get("description", ""))
        active = bool(raw.get("active", False))
        closed = bool(raw.get("closed", False))
        accepting = bool(raw.get("accepting_orders", active))
        tick_size = _to_decimal(raw.get("minimum_tick_size"), Decimal("0.01"))
        min_order = int(_to_decimal(raw.get("minimum_order_size"), ONE))
        end_date = raw.get("end_date_iso")
        start_date = raw.get("game_start_time")
        fees = _extract_fees(None, raw.get("maker_base_fee", 0), raw.get("taker_base_fee", 0))
        volume = ZERO
        liquidity = ZERO
        event_id = condition_id
        venue_market_id = condition_id
    else:
        condition_id = str(raw.get("conditionId", ""))
        question = str(raw.get("question", ""))
        slug = str(raw.get("slug", ""))
        outcomes = [str(o) for o in _parse_json_field(raw.get("outcomes"))]
        description = str(raw.get("description", ""))
        active = bool(raw.get("active", False))
        closed = bool(raw.get("closed", False))
        accepting = bool(raw.get("acceptingOrders", active))
        tick_size = _to_decimal(raw.get("orderPriceMinTickSize"), Decimal("0.01"))
        min_order = int(_to_decimal(raw.get("orderMinSize"), ONE))
        end_date = raw.get("endDate") or raw.get("endDateIso")
        start_date = raw.get("startDate") or raw.get("startDateIso")
        fees = _extract_fees(raw.get("feeSchedule"), raw.get("makerBaseFee", 0), raw.get("takerBaseFee", 0))
        volume = _to_decimal(raw.get("volumeNum", raw.get("volume")), ZERO)
        liquidity = _to_decimal(raw.get("liquidityNum", raw.get("liquidity")), ZERO)
        events = raw.get("events") or []
        event_id = str(events[0].get("id", condition_id)) if events and isinstance(events[0], dict) else condition_id
        venue_market_id = str(raw.get("id", condition_id))

    canonical_id = f"poly:{condition_id or slug}"
    category = category_hint or _category_from_tags(_extract_tags(raw))
    status = _market_status(active, closed, accepting)
    yes_symbol = outcomes[0] if outcomes else "Yes"
    no_symbol = outcomes[1] if len(outcomes) > 1 else "No"

    return NormalizedMarket(
        canonical_id=canonical_id,
        venue=Venue.POLY_GLOBAL,
        venue_market_id=venue_market_id or canonical_id,
        event_id=event_id or canonical_id,
        title=question,
        description=description,
        category=category,
        outcome_type=OutcomeType.BINARY,
        yes_symbol=yes_symbol,
        no_symbol=no_symbol,
        open_time=_parse_dt(start_date),
        close_time=_parse_dt(end_date),
        tick_size=tick_size if tick_size > ZERO else Decimal("0.01"),
        min_order=max(min_order, 1),
        status=status,
        fees=fees,
        liquidity=liquidity,
        volume=volume,
        raw=dict(raw),
    )


def _parse_levels(levels: Any) -> list[tuple[Decimal, Decimal]]:
    out: list[tuple[Decimal, Decimal]] = []
    for lvl in levels or []:
        if not isinstance(lvl, Mapping):
            continue
        price = _to_decimal_or_none(lvl.get("price"))
        size = _to_decimal_or_none(lvl.get("size"))
        if price is None or size is None:
            continue
        out.append((price, size))
    return out


def _to_book_levels(levels: Sequence[tuple[Decimal, Decimal]], *, invert: bool) -> list[BookLevel]:
    out: list[BookLevel] = []
    for price, size in levels:
        p = (ONE - price) if invert else price
        if p < ZERO or p > ONE:
            continue  # a malformed level should not crash normalization
        prob = to_probability(p)
        # BookLevel.size is contracts (int) per the frozen core model; Polymarket sizes
        # are fractional share counts on-chain, so we round to the nearest whole share.
        qty = int(size.to_integral_value(rounding=ROUND_HALF_UP))
        out.append(BookLevel(price=prob, size=max(qty, 0)))
    return out


def normalize_book(
    token_id: str,
    raw: Mapping[str, Any],
    timestamp: datetime,
    *,
    is_no_token: bool = False,
    canonical_id: str | None = None,
) -> OrderBook:
    """Fold a CLOB per-token book into a YES-probability :class:`OrderBook`.

    ``raw`` is the CLOB ``/book`` (or one element of ``/books``) response:
    ``{"market": condition_id, "asset_id": token_id, "timestamp": "<unix ms>",
    "bids": [...], "asks": [...]}``.

    When ``is_no_token`` is False (the default - ``token_id`` is the YES outcome
    token), the raw bids/asks are already YES bids/asks; we only need to sort them
    into the canonical descending-bid / ascending-ask order the CLOB does not
    guarantee. When ``is_no_token`` is True, a NO bid at ``q`` becomes a YES ask at
    ``1-q`` and a NO ask at ``q`` becomes a YES bid at ``1-q`` - see the module
    docstring for the economic justification.
    """
    raw_bids = _parse_levels(raw.get("bids"))
    raw_asks = _parse_levels(raw.get("asks"))

    if is_no_token:
        yes_bid_source, yes_ask_source = raw_asks, raw_bids
    else:
        yes_bid_source, yes_ask_source = raw_bids, raw_asks

    bids = tuple(sorted(_to_book_levels(yes_bid_source, invert=is_no_token), key=lambda lvl: lvl.price, reverse=True))
    asks = tuple(sorted(_to_book_levels(yes_ask_source, invert=is_no_token), key=lambda lvl: lvl.price))

    cid = canonical_id or f"poly:{raw.get('market', token_id)}"
    venue_timestamp = _parse_unix_ms(raw.get("timestamp"))

    return OrderBook(
        canonical_id=cid,
        venue=Venue.POLY_GLOBAL,
        timestamp=timestamp,
        bids=bids,
        asks=asks,
        venue_timestamp=venue_timestamp,
    )


def normalize_activity(raw: Mapping[str, Any], now: datetime) -> TraderActionEvent:
    """Normalize one row of the Data API ``/activity`` feed into a ``TraderActionEvent``.

    ``first_seen_time`` is set to ``now`` - the caller's injected :class:`Clock` reading
    at ingestion time - and **never** to the trade's own on-chain ``timestamp``. That
    distinction is the entire point of the copy-trading follower-latency experiment: a
    strategy may only act on ``first_seen_time``, which reflects when *we* observed the
    trade, not when it happened on-chain.
    """
    event_time = _parse_unix_seconds(raw.get("timestamp")) or now
    outcome = str(raw.get("outcome", ""))
    side: Side | None
    if outcome.lower() == "yes":
        side = Side.YES
    elif outcome.lower() == "no":
        side = Side.NO
    else:
        side = None

    price = _to_decimal_or_none(raw.get("price"))
    if price is not None:
        try:
            price = to_probability(price)
        except ValueError:
            price = None

    condition_id = str(raw.get("conditionId", ""))
    return TraderActionEvent(
        event_time=event_time,
        published_time=event_time,
        first_seen_time=now,
        ingested_time=None,
        source="polymarket_data_api",
        source_class=SourceClass.SPECIALIST,
        wallet=str(raw.get("proxyWallet", "")),
        username=str(raw.get("name") or raw.get("pseudonym") or ""),
        canonical_id=f"poly:{condition_id}" if condition_id else "",
        poly_market_id=str(raw.get("asset", "")),
        poly_condition_id=condition_id,
        title=str(raw.get("title", "")),
        outcome=outcome,
        side=side,
        action=str(raw.get("side") or raw.get("type", "")),
        price=price,
        size=_to_decimal_or_none(raw.get("size")),
        usd_size=_to_decimal_or_none(raw.get("usdcSize")),
        transaction_hash=str(raw.get("transactionHash", "")),
    )
