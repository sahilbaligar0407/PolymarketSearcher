"""The tiered AI stack: arm resolution, evidence relevance, and inference sharing."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from marketlab.ai.provider import DisabledProvider
from marketlab.ai.stack import AIStack, AssessmentCache, EvidenceCache, market_keywords
from marketlab.clock import SimulatedClock
from marketlab.core.events import NewsEvent
from marketlab.core.instruments import MarketStatus, NormalizedMarket, Venue

TS = datetime(2026, 10, 1, tzinfo=UTC)


def _market(title: str) -> NormalizedMarket:
    return NormalizedMarket(
        canonical_id="kalshi:X", venue=Venue.KALSHI, venue_market_id="X", event_id="E",
        title=title, status=MarketStatus.OPEN,
    )


def _news(title: str, nid: str) -> NewsEvent:
    return NewsEvent(event_time=TS, first_seen_time=TS, source="test", news_id=nid, title=title)


def _stack(*tiers: str) -> AIStack:
    providers = {t: DisabledProvider() for t in tiers}
    return AIStack(providers, EvidenceCache(lambda _c: None), AssessmentCache(SimulatedClock(TS)))


def test_arms_need_every_tier():
    stack = _stack("local", "openai")
    assert stack.resolve("local") is not None
    resolved = stack.resolve("hybrid")
    assert resolved is not None and resolved[1] is not None
    assert stack.resolve("jev") is None
    assert stack.resolve("jev_hybrid") is None
    assert stack.resolve("nonsense") is None
    assert stack.available_stacks() == ["local", "hybrid"]
    assert _stack("local").available_stacks() == ["local"]


def test_keywords_drop_filler():
    kws = market_keywords(_market("Will the Fed cut rates at the December 2026 meeting?"))
    assert "fed" in kws and "december" in kws
    assert "will" not in kws and "the" not in kws and "2026" not in kws


def test_evidence_cache_returns_only_relevant_news():
    market = _market("Will Nvidia report revenue above $50B in Q3 earnings?")
    cache = EvidenceCache(lambda cid: market)
    cache.record(_news("Nvidia earnings preview: revenue expected to beat", "n1"))
    cache.record(_news("Local team wins championship", "n2"))
    cache.record(_news("Nvidia unveils new chip", "n3"))  # one keyword only: not enough
    got = [e.news_id for e in cache.get_news("kalshi:X")]
    assert got == ["n1"]
    assert cache.sizes()["news"] == 3


async def test_assessment_cache_shares_inflight_and_fresh_results():
    clock = SimulatedClock(TS)
    cache = AssessmentCache(clock)
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return "result"

    results = await asyncio.gather(*(cache.get_or_compute(("p", "m"), 60, compute) for _ in range(5)))
    assert results == ["result"] * 5
    assert calls == 1
    assert await cache.get_or_compute(("p", "m"), 60, compute) == "result"
    assert calls == 1
    clock.advance(61)
    await cache.get_or_compute(("p", "m"), 60, compute)
    assert calls == 2
