"""Tests for `marketlab.ai.retrieval`, especially the point-in-time gate.

The core invariant under test: a bundle built "as of" T must never contain a record
whose ``first_seen_time`` is after T, no matter how small the gap.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from marketlab.ai.retrieval import RetrievalService
from marketlab.clock import SimulatedClock
from marketlab.core.events import NewsEvent, SourceClass
from marketlab.core.instruments import (
    Category,
    Fees,
    MarketStatus,
    NormalizedMarket,
    OutcomeType,
    Venue,
)

T0 = datetime(2026, 9, 4, 12, 0, 0, tzinfo=UTC)
CANONICAL_ID = "KALSHI:PIT-TEST"


def _market() -> NormalizedMarket:
    return NormalizedMarket(
        canonical_id=CANONICAL_ID,
        venue=Venue.KALSHI,
        venue_market_id="PIT-TEST",
        event_id="evt_pit",
        title="Point-in-time test market",
        resolution_rules="Resolves YES if X happens by the close time.",
        category=Category.OTHER,
        outcome_type=OutcomeType.BINARY,
        status=MarketStatus.OPEN,
        fees=Fees(),
    )


def _news(
    news_id: str,
    first_seen_time: datetime,
    title: str = "headline",
    body: str = "body text",
    tone: float | None = None,
) -> NewsEvent:
    return NewsEvent(
        event_time=first_seen_time,
        first_seen_time=first_seen_time,
        source="reuters.com",
        source_class=SourceClass.MAJOR_WIRE,
        news_id=news_id,
        title=title,
        body=body,
        url=f"https://reuters.com/{news_id}",
        canonical_url=f"https://reuters.com/{news_id}",
        tone=tone,
    )


class FakeStore:
    """A minimal duck-typed store: only the methods a given test needs are set."""

    def __init__(self, *, market: NormalizedMarket, news: list[NewsEvent] | None = None) -> None:
        self._market = market
        self._news = news or []

    def get_market(self, canonical_id: str) -> NormalizedMarket | None:
        return self._market if canonical_id == self._market.canonical_id else None

    def get_news(self, canonical_id: str) -> list[NewsEvent]:
        return list(self._news)

    # get_social / get_filings / get_external_prices / get_trader_activity /
    # get_prior_assessment are deliberately absent -- RetrievalService must degrade to
    # empty categories rather than raising AttributeError.


@pytest.mark.asyncio
async def test_point_in_time_gate_excludes_future_and_includes_past() -> None:
    market = _market()
    before = _news("n_before", T0 - timedelta(seconds=1), title="Before the gate")
    after = _news("n_after", T0 + timedelta(seconds=1), title="After the gate")
    store = FakeStore(market=market, news=[before, after])
    service = RetrievalService(store=store, clock=SimulatedClock(start=T0))

    bundle = await service.build_bundle(CANONICAL_ID, T0)

    headlines = {item.headline for item in bundle.news}
    assert "Before the gate" in headlines
    assert "After the gate" not in headlines
    assert all(item.timestamp <= T0 for item in bundle.news)


@pytest.mark.asyncio
async def test_bundle_degrades_gracefully_when_store_methods_are_missing() -> None:
    market = _market()
    store = FakeStore(market=market, news=[])
    service = RetrievalService(store=store, clock=SimulatedClock(start=T0))

    bundle = await service.build_bundle(CANONICAL_ID, T0)

    assert bundle.social == ()
    assert bundle.filings == ()
    assert bundle.external_prices == ()
    assert bundle.trader_activity == ()
    assert bundle.prior_assessment is None


@pytest.mark.asyncio
async def test_bundle_respects_size_cap() -> None:
    market = _market()
    many_items = [
        _news(f"n_{i}", T0 - timedelta(seconds=i + 1), title=f"headline {i}", body=f"distinct body content number {i} " * 20)
        for i in range(50)
    ]
    store = FakeStore(market=market, news=many_items)
    service = RetrievalService(store=store, clock=SimulatedClock(start=T0), max_items_per_category=8)

    bundle = await service.build_bundle(CANONICAL_ID, T0)

    assert len(bundle.news) <= 8
    context = bundle.to_prompt_context(max_chars=500)
    assert len(context) <= 500


@pytest.mark.asyncio
async def test_evidence_ids_are_stable_across_two_builds() -> None:
    market = _market()
    news = [_news("n_stable", T0 - timedelta(minutes=5), title="Stable headline", body="Stable body")]
    store = FakeStore(market=market, news=news)
    service = RetrievalService(store=store, clock=SimulatedClock(start=T0))

    bundle_a = await service.build_bundle(CANONICAL_ID, T0)
    bundle_b = await service.build_bundle(CANONICAL_ID, T0)

    ids_a = [item.evidence_id for item in bundle_a.news]
    ids_b = [item.evidence_id for item in bundle_b.news]
    assert ids_a == ids_b
    assert ids_a  # sanity: something was actually built
    assert bundle_a.market_evidence_id() == bundle_b.market_evidence_id() == f"market:{CANONICAL_ID}"


@pytest.mark.asyncio
async def test_counterevidence_opposes_market_price_direction() -> None:
    market = _market()
    # Market favors YES (price 0.7); a strongly negative-tone story is counterevidence.
    positive = _news(
        "n_pos", T0 - timedelta(minutes=10), title="Quarterly earnings beat analyst expectations",
        body="Strong earnings beat expectations across the board.", tone=0.8,
    )
    negative = _news(
        "n_neg", T0 - timedelta(minutes=5), title="Federal regulators open antitrust inquiry",
        body="Regulatory probe threatens the company's core business.", tone=-0.9,
    )
    store = FakeStore(market=market, news=[positive, negative])
    service = RetrievalService(store=store, clock=SimulatedClock(start=T0))

    from decimal import Decimal

    bundle = await service.build_bundle(CANONICAL_ID, T0, market_price=Decimal("0.7"))

    counter_headlines = {item.headline for item in bundle.counterevidence}
    assert "Federal regulators open antitrust inquiry" in counter_headlines
    assert "Quarterly earnings beat analyst expectations" not in counter_headlines
