"""Unit tests for marketlab/strategies/news_probability.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from marketlab.ai.provider import DisabledProvider, LLMProvider, LLMResponse, ProviderHealth
from marketlab.clock import SimulatedClock
from marketlab.core.instruments import (
    BookLevel,
    Category,
    MarketStatus,
    NormalizedMarket,
    OrderBook,
    Venue,
)
from marketlab.core.strategy import StrategyContext
from marketlab.strategies.news_probability import NewsProbabilityStrategy

TS = datetime(2026, 1, 1, tzinfo=UTC)
CANONICAL_ID = "kalshi:cpi-above-3pct"


@dataclass
class FakeProvider(LLMProvider):
    """Deterministic stand-in for a local LLM: returns whatever payload the test configures."""

    payload: dict[str, Any] | None
    name: str = "fake"
    model: str = "fake-model-v1"
    call_count: int = field(default=0)

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        timeout: float = 90.0,  # noqa: ASYNC109
    ) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            text="" if self.payload is None else str(self.payload),
            parsed=self.payload,
            model=self.model,
            prompt_hash="hash1234567890ab",
            latency_ms=1.0,
        )

    async def probe(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider=self.name, model=self.model)

    async def close(self) -> None:
        return None


def _market() -> NormalizedMarket:
    return NormalizedMarket(
        canonical_id=CANONICAL_ID,
        venue=Venue.KALSHI,
        venue_market_id="KXCPI-TEST",
        event_id="cpi-evt-1",
        title="Will CPI print above 3%?",
        resolution_rules="",  # empty: interpretation-consistency check trivially passes
        category=Category.ECONOMICS,
        status=MarketStatus.OPEN,
        min_order=1,
    )


def _book(bid: str = "0.40", ask: str = "0.42") -> OrderBook:
    return OrderBook(
        canonical_id=CANONICAL_ID,
        venue=Venue.KALSHI,
        timestamp=TS,
        bids=(BookLevel(price=Decimal(bid), size=100),),
        asks=(BookLevel(price=Decimal(ask), size=100),),
    )


def _ctx() -> StrategyContext:
    clock = SimulatedClock(TS)
    market = _market()
    return StrategyContext(clock=clock, books={CANONICAL_ID: _book()}, markets={CANONICAL_ID: market}, marks={})


def _valid_payload(p_yes: str, confidence: str = "0.8") -> dict[str, Any]:
    return {
        "market_id": CANONICAL_ID,
        "as_of": TS.isoformat(),
        "question_interpretation": "Will CPI print above 3 percent this release?",
        "p_yes": p_yes,
        "confidence": confidence,
        "abstain": False,
        "evidence_ids": [],
        "supporting_facts": ["fake supporting fact"],
        "contradicting_facts": [],
        "missing_information": [],
        "resolution_rule_warning": False,
        "information_cutoff": TS.isoformat(),
    }


async def test_invalid_llm_assessment_causes_abstain_and_zero_intents() -> None:
    # Missing required fields (p_yes, confidence, ...) - parse_llm_output must fail.
    provider = FakeProvider(payload={"market_id": CANONICAL_ID, "not_a_real_field": True})
    ctx = _ctx()
    strat = NewsProbabilityStrategy(
        "s1", "e1", ctx, params={"llm_provider": provider, "entry_edge": "0.05", "min_confidence": "0.5"}
    )
    await strat.evaluate_market(CANONICAL_ID)

    forecasts = strat.drain_forecasts()
    assert len(forecasts) == 1
    assert forecasts[0].abstain is True
    assert strat.generate_intents() == []


async def test_valid_assessment_below_entry_edge_produces_nothing() -> None:
    # Market mid is ~0.41; a model p_yes of 0.43 clears no reasonable entry_edge once fees,
    # slippage and uncertainty are subtracted.
    provider = FakeProvider(payload=_valid_payload(p_yes="0.43"))
    ctx = _ctx()
    strat = NewsProbabilityStrategy(
        "s1", "e1", ctx, params={"llm_provider": provider, "entry_edge": "0.05", "min_confidence": "0.5"}
    )
    await strat.evaluate_market(CANONICAL_ID)

    forecasts = strat.drain_forecasts()
    assert len(forecasts) == 1
    assert forecasts[0].abstain is False
    assert strat.generate_intents() == []


async def test_valid_assessment_above_entry_edge_trades_with_model_and_prompt_hash() -> None:
    provider = FakeProvider(payload=_valid_payload(p_yes="0.75", confidence="0.9"))
    ctx = _ctx()
    strat = NewsProbabilityStrategy(
        "s1", "e1", ctx, params={"llm_provider": provider, "entry_edge": "0.05", "min_confidence": "0.5"}
    )
    await strat.evaluate_market(CANONICAL_ID)

    intents = strat.generate_intents()
    assert len(intents) == 1
    intent = intents[0]
    assert intent.venue is Venue.KALSHI
    assert intent.features["llm_model_id"] == "fake-model-v1"
    assert intent.features["prompt_hash"] == "hash1234567890ab"
    assert intent.expected_edge is not None and intent.expected_edge > 0


async def test_low_confidence_valid_assessment_does_not_trade() -> None:
    provider = FakeProvider(payload=_valid_payload(p_yes="0.75", confidence="0.3"))
    ctx = _ctx()
    strat = NewsProbabilityStrategy(
        "s1", "e1", ctx, params={"llm_provider": provider, "entry_edge": "0.05", "min_confidence": "0.7"}
    )
    await strat.evaluate_market(CANONICAL_ID)

    assert strat.generate_intents() == []
    forecasts = strat.drain_forecasts()
    assert len(forecasts) == 1
    assert forecasts[0].abstain is False


async def test_disabled_provider_runs_and_emits_nothing_cleanly() -> None:
    ctx = _ctx()
    strat = NewsProbabilityStrategy("s1", "e1", ctx, params={"llm_provider": DisabledProvider()})
    await strat.evaluate_market(CANONICAL_ID)

    forecasts = strat.drain_forecasts()
    assert len(forecasts) == 1
    assert forecasts[0].abstain is True
    assert strat.generate_intents() == []


async def test_no_provider_configured_defaults_to_disabled_and_is_clean() -> None:
    ctx = _ctx()
    strat = NewsProbabilityStrategy("s1", "e1", ctx, params={})
    await strat.evaluate_market(CANONICAL_ID)
    assert strat.generate_intents() == []
