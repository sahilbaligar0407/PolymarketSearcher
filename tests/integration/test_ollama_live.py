"""Live integration test against a real local Ollama server.

Skipped entirely if Ollama isn't reachable -- this must never fail CI on a machine
without a local model running. Uses ``qwen3:0.6b`` deliberately (not the default
``gpt-oss:20b``) so the test stays fast; see docs/CONTRACTS.md for the installed model
list.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest

from marketlab.ai.ollama import OllamaProvider
from marketlab.ai.prompts import MARKET_ASSESSMENT_SYSTEM, MARKET_ASSESSMENT_USER
from marketlab.ai.provider import DisabledProvider, detect_provider
from marketlab.ai.retrieval import EvidenceBundle, EvidenceItem
from marketlab.ai.schemas import MarketAssessment, json_schema_for
from marketlab.ai.validator import parse_llm_output, validate_assessment
from marketlab.core.instruments import (
    Category,
    Fees,
    MarketStatus,
    NormalizedMarket,
    OutcomeType,
    Venue,
)
from marketlab.settings import load_settings

pytestmark = pytest.mark.integration

FAST_MODEL = "qwen3:0.6b"
T0 = datetime(2026, 9, 4, 12, 0, 0, tzinfo=UTC)


def _ollama_reachable(base_url: str) -> bool:
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def _synthetic_market() -> NormalizedMarket:
    return NormalizedMarket(
        canonical_id="KALSHI:BTC-100K-DEC26",
        venue=Venue.KALSHI,
        venue_market_id="BTC-100K-DEC26",
        event_id="evt_btc",
        title="Will Bitcoin close above $100,000 on Dec 31, 2026?",
        resolution_rules=(
            "This market resolves YES if the Coinbase BTC-USD spot price at 2026-12-31 "
            "23:59 UTC is above $100,000."
        ),
        resolution_source="Coinbase",
        category=Category.CRYPTO,
        outcome_type=OutcomeType.BINARY,
        status=MarketStatus.OPEN,
        fees=Fees(),
    )


def _synthetic_bundle(market: NormalizedMarket) -> EvidenceBundle:
    news_item = EvidenceItem(
        evidence_id="news:btc_rally_2026",
        kind="news",
        timestamp=T0 - timedelta(hours=2),
        headline="Bitcoin rallies on ETF inflow news",
        text="Spot BTC-USD traded near $97,500 amid strong ETF inflows, up 8% this week.",
        source="reuters.com",
        tone=0.5,
    )
    counter_item = EvidenceItem(
        evidence_id="news:btc_regulatory_risk",
        kind="news",
        timestamp=T0 - timedelta(hours=1),
        headline="Regulators signal renewed scrutiny of crypto exchanges",
        text="A joint statement raised the prospect of tighter trading restrictions before year-end.",
        source="reuters.com",
        tone=-0.4,
    )
    return EvidenceBundle(
        market=market,
        rules=market.resolution_rules,
        news=(news_item,),
        social=(),
        filings=(),
        external_prices=(),
        trader_activity=(),
        prior_assessment=None,
        counterevidence=(counter_item,),
        as_of=T0,
        market_price=Decimal("0.42"),
    )


@pytest.mark.asyncio
async def test_detect_provider_finds_local_ollama() -> None:
    settings = load_settings(profile="paper")
    if not _ollama_reachable(settings.sources.ollama_base):
        pytest.skip("Ollama is not reachable on this machine")

    provider = await detect_provider(settings)
    try:
        assert not isinstance(provider, DisabledProvider)
        health = await provider.probe()
        assert health.ok, health.detail
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_real_structured_assessment_validates_or_abstains() -> None:
    settings = load_settings(profile="paper")
    if not _ollama_reachable(settings.sources.ollama_base):
        pytest.skip("Ollama is not reachable on this machine")

    provider = OllamaProvider(base_url=settings.sources.ollama_base, model=FAST_MODEL)
    try:
        health = await provider.probe()
        if not health.ok:
            pytest.skip(f"{FAST_MODEL} is not installed/reachable: {health.detail}")

        market = _synthetic_market()
        bundle = _synthetic_bundle(market)
        schema = json_schema_for(MarketAssessment)
        user_prompt = MARKET_ASSESSMENT_USER(market, bundle, T0)

        response = await provider.generate(
            prompt=user_prompt,
            schema=schema,
            system=MARKET_ASSESSMENT_SYSTEM,
            temperature=0.0,
            timeout=120.0,
        )

        assert response.model == FAST_MODEL
        assert response.prompt_hash  # non-empty: identity is always recorded

        assessment = parse_llm_output(response.parsed if response.parsed is not None else response.text)
        result = validate_assessment(assessment, market, bundle, decision_time=T0)

        # A tiny 0.6B model may reasonably abstain or even fail to produce parseable
        # JSON -- both are acceptable outcomes here. What must NEVER happen is the
        # validator raising, or a genuinely broken assessment being reported as valid.
        if result.valid:
            assert result.assessment_or_none is not None
            assert Decimal(0) <= result.assessment_or_none.p_yes <= Decimal(1)
        else:
            assert result.assessment_or_none is None
            assert result.failures
    finally:
        await provider.close()
