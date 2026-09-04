"""Integration-style tests for the matching pipeline:
``semantic_match`` -> ``extract`` -> ``resolution_rules`` -> ``cross_venue``.

These exercise the full ``CrossVenueMatcher.match()`` path against the corpus in
``tests/fixtures/match_pairs.py`` (so failures here point at wiring, not extraction
logic -- that's covered directly in ``test_resolution_rules.py``), plus the
structural invariant that an LLM-proposed candidate can never bypass the validator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from marketlab.clock import SimulatedClock
from marketlab.core.instruments import Venue
from marketlab.matching.cross_venue import (
    VALIDATOR_VERSION,
    CrossVenueMatcher,
    MarketMatch,
    approved_for_automation,
)
from marketlab.matching.semantic_match import build_candidates, jaccard, propose_candidates
from tests.fixtures.match_pairs import (
    BTC_REWORDED_SAME_TERMINAL,
    CORPUS,
    HEADLINE_VS_CORE_CPI,
    NFL_MONEYLINE_SAME_POSTPONEMENT_RULE,
)

T0 = datetime(2026, 9, 4, 0, 0, tzinfo=UTC)


class FakeStore:
    """Duck-typed stand-in for ``marketlab.storage.state.StateStore``."""

    def __init__(self) -> None:
        self.saved: list[Any] = []

    def save_match(self, match: Any) -> None:
        self.saved.append(match)


def _matches_by_pair(matches: list[MarketMatch]) -> dict[tuple[str, str], MarketMatch]:
    return {(m.canonical_id_a, m.canonical_id_b): m for m in matches}


@pytest.mark.asyncio
async def test_cross_venue_matcher_reproduces_corpus_labels() -> None:
    kalshi_markets = [fx.market_a for fx in CORPUS]
    poly_markets = [fx.market_b for fx in CORPUS]

    store = FakeStore()
    matcher = CrossVenueMatcher(clock=SimulatedClock(T0), store=store)
    matches = await matcher.match(kalshi_markets, poly_markets)
    by_pair = _matches_by_pair(matches)

    failures = []
    for fx in CORPUS:
        key = (fx.market_a.canonical_id, fx.market_b.canonical_id)
        match = by_pair.get(key)
        if match is None:
            failures.append(f"{fx.name}: no MarketMatch produced (candidate filtered out)")
            continue
        if match.same_outcome_boolean != fx.should_match:
            failures.append(
                f"{fx.name}: same_outcome_boolean={match.same_outcome_boolean} "
                f"expected {fx.should_match} ({fx.reason})"
            )
    assert not failures, "\n".join(failures)

    # Every match was persisted through the duck-typed store.
    assert len(store.saved) == len(matches)
    assert all(m.validator_version == VALIDATOR_VERSION for m in matches)


@pytest.mark.asyncio
async def test_approved_for_automation_matches_should_match_labels() -> None:
    kalshi_markets = [fx.market_a for fx in CORPUS]
    poly_markets = [fx.market_b for fx in CORPUS]

    matcher = CrossVenueMatcher(clock=SimulatedClock(T0), store=None)
    matches = await matcher.match(kalshi_markets, poly_markets)
    by_pair = _matches_by_pair(matches)

    for fx in CORPUS:
        key = (fx.market_a.canonical_id, fx.market_b.canonical_id)
        match = by_pair[key]
        approved = approved_for_automation(match)
        assert approved == fx.should_match, (
            f"{fx.name}: approved_for_automation={approved} expected {fx.should_match}"
        )


def test_approved_for_automation_false_whenever_same_outcome_boolean_is_false() -> None:
    """Even a suspiciously perfect confidence score can never overrule the gate."""
    match = MarketMatch(
        match_id="m1",
        canonical_id_a="A",
        canonical_id_b="B",
        match_confidence=Decimal("0.999"),
        same_outcome_boolean=False,
        rule_diff="measurement differs",
        time_diff="0.0",
        resolution_source_diff="False",
        human_review_required=False,
        created_at=T0,
    )
    assert approved_for_automation(match) is False
    assert approved_for_automation(match, min_confidence=Decimal("0.0")) is False


def test_approved_for_automation_false_when_same_outcome_boolean_is_none() -> None:
    match = MarketMatch(
        match_id="m2",
        canonical_id_a="A",
        canonical_id_b="B",
        match_confidence=Decimal("1.0"),
        same_outcome_boolean=None,
        rule_diff="",
        time_diff="",
        resolution_source_diff="False",
        human_review_required=False,
        created_at=T0,
    )
    assert approved_for_automation(match) is False


def test_approved_for_automation_false_when_human_review_required() -> None:
    match = MarketMatch(
        match_id="m3",
        canonical_id_a="A",
        canonical_id_b="B",
        match_confidence=Decimal("0.95"),
        same_outcome_boolean=True,
        rule_diff="",
        time_diff="0.0",
        resolution_source_diff="False",
        human_review_required=True,
        created_at=T0,
    )
    assert approved_for_automation(match) is False


def test_approved_for_automation_respects_min_confidence_threshold() -> None:
    match = MarketMatch(
        match_id="m4",
        canonical_id_a="A",
        canonical_id_b="B",
        match_confidence=Decimal("0.80"),
        same_outcome_boolean=True,
        rule_diff="",
        time_diff="0.0",
        resolution_source_diff="False",
        human_review_required=False,
        created_at=T0,
    )
    assert approved_for_automation(match, min_confidence=Decimal("0.90")) is False
    assert approved_for_automation(match, min_confidence=Decimal("0.75")) is True


# ---------------------------------------------------------------------------
# The structural invariant: an LLM proposal cannot become an approved match
# without passing resolution_rules.compare_claims.
# ---------------------------------------------------------------------------


@dataclass
class _FakeLLMResponse:
    parsed: dict[str, Any] | None


class _FakeLLMProvider:
    """Duck-typed ``LLMProvider`` that always suggests the (wrong) CPI/Core-CPI pair."""

    def __init__(self, suggested_id: str) -> None:
        self._suggested_id = suggested_id
        self.calls = 0

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        timeout: float = 90.0,  # noqa: ASYNC109
    ) -> _FakeLLMResponse:
        self.calls += 1
        # A real LLM might also (wrongly) claim high confidence here -- but propose_
        # candidates.CandidatePair has no field to carry that opinion downstream at all.
        return _FakeLLMResponse(parsed={"canonical_ids": [self._suggested_id]})


@pytest.mark.asyncio
async def test_llm_proposed_candidate_cannot_bypass_validator() -> None:
    """An LLM confidently proposing a bad pair must still be rejected by the validator."""
    kalshi_market = HEADLINE_VS_CORE_CPI.market_a  # "CPI YoY above 3.0%"
    wrong_pool_market = HEADLINE_VS_CORE_CPI.market_b  # "Core CPI YoY above 3.0%" -- NOT the same claim
    pool = [wrong_pool_market]

    llm = _FakeLLMProvider(suggested_id=wrong_pool_market.canonical_id)
    # Force the heuristic layer to propose nothing so the LLM's suggestion is the only
    # candidate on the table -- proving the *validator*, not the heuristic filter, is
    # what rejects it.
    candidates = await propose_candidates(
        kalshi_market, pool, llm, heuristic_candidates=[]
    )
    assert len(candidates) == 1
    assert candidates[0].source == "llm"
    assert llm.calls == 1

    store = FakeStore()
    matcher = CrossVenueMatcher(clock=SimulatedClock(T0), store=store, llm_provider=llm)
    matches = await matcher.match([kalshi_market], pool)

    assert len(matches) == 1
    match = matches[0]
    assert match.same_outcome_boolean is False
    assert approved_for_automation(match) is False
    # Persisted too -- an LLM-sourced candidate leaves the same audit trail as any other.
    assert store.saved == [match]


@pytest.mark.asyncio
async def test_llm_provider_failure_degrades_to_heuristic_candidates_only() -> None:
    class _BrokenLLMProvider:
        async def generate(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("local model unreachable")

    kalshi_market = BTC_REWORDED_SAME_TERMINAL.market_a
    pool = [BTC_REWORDED_SAME_TERMINAL.market_b]
    candidates = await propose_candidates(kalshi_market, pool, _BrokenLLMProvider())
    # Never raises; falls back to whatever the heuristic layer already found.
    assert candidates == build_candidates(kalshi_market, pool)


# ---------------------------------------------------------------------------
# semantic_match mechanics
# ---------------------------------------------------------------------------


def test_jaccard_similarity_basics() -> None:
    assert jaccard(frozenset(), frozenset()) == 0.0
    assert jaccard(frozenset({"a", "b"}), frozenset({"a", "b"})) == 1.0
    assert jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0
    assert jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == pytest.approx(1 / 3)


def test_build_candidates_excludes_markets_outside_close_time_gap() -> None:
    from tests.fixtures.match_pairs import _mk

    near = _mk("NEAR", "Will BTC be above $100,000 at 5 PM ET on September 4, 2026?", venue=Venue.POLY_GLOBAL)
    far = _mk(
        "FAR",
        "Will BTC be above $100,000 at 5 PM ET on September 4, 2026?",
        venue=Venue.POLY_GLOBAL,
        close_time=datetime(2026, 9, 5, 12, 0, tzinfo=UTC),  # ~15h later
    )
    market = _mk("KALSHI-BTC", "Will BTC be above $100,000 at 5 PM ET on September 4, 2026?")
    candidates = build_candidates(market, [near, far])
    ids = {c.market_b.canonical_id for c in candidates}
    assert "NEAR" in ids
    assert "FAR" not in ids


def test_build_candidates_excludes_different_category() -> None:
    from marketlab.core.instruments import Category
    from tests.fixtures.match_pairs import _mk

    market = _mk("KALSHI-BTC", "Will BTC be above $100,000?", category=Category.CRYPTO)
    other_category = _mk(
        "POLY-CPI", "Will BTC be above $100,000?", venue=Venue.POLY_GLOBAL, category=Category.ECONOMICS
    )
    candidates = build_candidates(market, [other_category])
    assert candidates == []


@pytest.mark.asyncio
async def test_matcher_match_id_is_stable_and_order_independent() -> None:
    matcher = CrossVenueMatcher(clock=SimulatedClock(T0))
    kalshi_markets = [NFL_MONEYLINE_SAME_POSTPONEMENT_RULE.market_a]
    poly_markets = [NFL_MONEYLINE_SAME_POSTPONEMENT_RULE.market_b]
    first = await matcher.match(kalshi_markets, poly_markets)
    second = await matcher.match(kalshi_markets, poly_markets)
    assert first[0].match_id == second[0].match_id
    assert first[0].validator_version == VALIDATOR_VERSION
