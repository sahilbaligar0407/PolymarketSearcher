"""Tests for the deterministic validator gate (`marketlab.ai.validator`).

The overriding rule under test: on any failure the gate abstains and returns
``assessment_or_none=None`` -- it never repairs a malformed value.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from marketlab.ai.retrieval import EvidenceBundle, EvidenceItem
from marketlab.ai.schemas import MarketAssessment
from marketlab.ai.validator import assessment_to_forecast, parse_llm_output, validate_assessment
from marketlab.core.instruments import (
    Category,
    Fees,
    MarketStatus,
    NormalizedMarket,
    OutcomeType,
    Venue,
)

T0 = datetime(2026, 9, 4, 12, 0, 0, tzinfo=UTC)
CANONICAL_ID = "KALSHI:APPLE-Q4-REV-100B"

RULES = (
    "This market resolves YES if Apple Inc. reports fiscal Q4 2026 revenue above "
    "$100 billion in its official earnings release."
)


def _market(canonical_id: str = CANONICAL_ID, rules: str = RULES) -> NormalizedMarket:
    return NormalizedMarket(
        canonical_id=canonical_id,
        venue=Venue.KALSHI,
        venue_market_id="APPLE-Q4-REV-100B",
        event_id="evt_1",
        title="Will Apple report Q4 2026 revenue above $100B?",
        resolution_rules=rules,
        category=Category.FINANCE,
        outcome_type=OutcomeType.BINARY,
        status=MarketStatus.OPEN,
        fees=Fees(),
    )


def _bundle(market: NormalizedMarket, as_of: datetime, extra_items: tuple[EvidenceItem, ...] = ()) -> EvidenceBundle:
    news_item = EvidenceItem(
        evidence_id="news:abc123456789",
        kind="news",
        timestamp=as_of - timedelta(hours=1),
        headline="Apple guides Q4 revenue higher",
        text="Apple Inc. raised guidance for fiscal Q4 2026 revenue to above $100 billion.",
        source="reuters.com",
        tone=0.6,
    )
    return EvidenceBundle(
        market=market,
        rules=market.resolution_rules,
        news=(news_item, *extra_items),
        social=(),
        filings=(),
        external_prices=(),
        trader_activity=(),
        prior_assessment=None,
        counterevidence=(),
        as_of=as_of,
        market_price=Decimal("0.55"),
    )


def _valid_assessment(market: NormalizedMarket, as_of: datetime, **overrides: object) -> MarketAssessment:
    fields = dict(
        market_id=market.canonical_id,
        as_of=as_of,
        question_interpretation=(
            "Does Apple Inc. report fiscal Q4 2026 revenue above $100 billion in its "
            "official earnings release?"
        ),
        p_yes=Decimal("0.62"),
        confidence=Decimal("0.7"),
        abstain=False,
        evidence_ids=["news:abc123456789"],
        supporting_facts=["Apple raised guidance above $100B [news:abc123456789]"],
        contradicting_facts=[],
        missing_information=[],
        resolution_rule_warning=False,
        information_cutoff=as_of - timedelta(hours=1),
    )
    fields.update(overrides)
    return MarketAssessment(**fields)  # type: ignore[arg-type]


def test_valid_assessment_passes() -> None:
    market = _market()
    bundle = _bundle(market, T0)
    assessment = _valid_assessment(market, T0)

    result = validate_assessment(assessment, market, bundle, decision_time=T0)

    assert result.valid is True
    assert result.assessment_or_none == assessment
    assert result.failures == []


def test_p_yes_out_of_range_is_invalid() -> None:
    market = _market()
    bundle = _bundle(market, T0)
    assessment = _valid_assessment(market, T0, p_yes=Decimal("1.4"))

    result = validate_assessment(assessment, market, bundle, decision_time=T0)

    assert result.valid is False
    assert result.assessment_or_none is None
    assert any("p_yes" in f for f in result.failures)


def test_confidence_out_of_range_is_invalid() -> None:
    market = _market()
    bundle = _bundle(market, T0)
    assessment = _valid_assessment(market, T0, confidence=Decimal("-0.1"))

    result = validate_assessment(assessment, market, bundle, decision_time=T0)

    assert result.valid is False
    assert result.assessment_or_none is None
    assert any("confidence" in f for f in result.failures)


def test_unknown_evidence_id_is_invalid() -> None:
    market = _market()
    bundle = _bundle(market, T0)
    assessment = _valid_assessment(market, T0, evidence_ids=["news:does-not-exist"])

    result = validate_assessment(assessment, market, bundle, decision_time=T0)

    assert result.valid is False
    assert result.assessment_or_none is None
    assert any("unknown evidence_id" in f for f in result.failures)


def test_look_ahead_evidence_is_invalid() -> None:
    """The core look-ahead test: an evidence item timestamped after decision_time must
    fail validation even if it was cited, because the model should never have been able
    to see it at decision time."""
    market = _market()
    future_item = EvidenceItem(
        evidence_id="news:futurefuture01",
        kind="news",
        timestamp=T0 + timedelta(seconds=1),
        headline="Something that hasn't happened yet",
        text="This is from the future relative to decision_time.",
        source="reuters.com",
    )
    bundle = _bundle(market, T0, extra_items=(future_item,))
    assessment = _valid_assessment(
        market, T0, evidence_ids=["news:abc123456789", "news:futurefuture01"]
    )

    result = validate_assessment(assessment, market, bundle, decision_time=T0)

    assert result.valid is False
    assert result.assessment_or_none is None
    assert any("look-ahead" in f for f in result.failures)


def test_information_cutoff_in_future_is_invalid() -> None:
    market = _market()
    bundle = _bundle(market, T0)
    assessment = _valid_assessment(market, T0, information_cutoff=T0 + timedelta(minutes=1))

    result = validate_assessment(assessment, market, bundle, decision_time=T0)

    assert result.valid is False
    assert result.assessment_or_none is None
    assert any("information_cutoff" in f for f in result.failures)


def test_market_id_mismatch_is_invalid() -> None:
    market = _market()
    bundle = _bundle(market, T0)
    assessment = _valid_assessment(market, T0, market_id="KALSHI:SOME-OTHER-MARKET")

    result = validate_assessment(assessment, market, bundle, decision_time=T0)

    assert result.valid is False
    assert result.assessment_or_none is None
    assert any("market_id mismatch" in f for f in result.failures)


def test_interpretation_inconsistent_with_rules_is_invalid() -> None:
    market = _market()
    bundle = _bundle(market, T0)
    # None of the rule's salient tokens (Apple, 2026, $100 billion) appear here.
    assessment = _valid_assessment(
        market, T0, question_interpretation="Will it rain in Chicago tomorrow?"
    )

    result = validate_assessment(assessment, market, bundle, decision_time=T0)

    assert result.valid is False
    assert result.assessment_or_none is None
    assert any("inconsistent" in f for f in result.failures)


def test_resolution_rule_warning_forces_abstain() -> None:
    market = _market()
    bundle = _bundle(market, T0)
    assessment = _valid_assessment(market, T0, resolution_rule_warning=True, abstain=False)

    result = validate_assessment(assessment, market, bundle, decision_time=T0)

    assert result.valid is True
    assert result.assessment_or_none is not None
    assert result.assessment_or_none.abstain is True
    # The rest of the assessment must be left alone -- only abstain is flipped.
    assert result.assessment_or_none.p_yes == assessment.p_yes


def test_malformed_json_blob_abstains_without_exception() -> None:
    """A malformed blob must never raise, and must never be 'repaired' into something
    that looks like a valid assessment."""
    market = _market()
    bundle = _bundle(market, T0)

    parsed = parse_llm_output("{this is not valid json at all")
    assert parsed is None

    result = validate_assessment(parsed, market, bundle, decision_time=T0)

    assert result.valid is False
    assert result.assessment_or_none is None
    assert result.failures  # some reason was recorded


def test_malformed_dict_missing_required_field_abstains_without_exception() -> None:
    market = _market()
    bundle = _bundle(market, T0)

    # Missing required fields (market_id, as_of, information_cutoff, ...).
    parsed = parse_llm_output({"p_yes": "0.6", "confidence": "0.7"})
    assert parsed is None

    result = validate_assessment(parsed, market, bundle, decision_time=T0)

    assert result.valid is False
    assert result.assessment_or_none is None


def test_assessment_to_forecast_maps_fields() -> None:
    market = _market()
    assessment = _valid_assessment(market, T0)

    forecast = assessment_to_forecast(
        assessment, strategy_id="ai_analyst", experiment_id="exp_1", market_probability=Decimal("0.55")
    )

    assert forecast.canonical_id == market.canonical_id
    assert forecast.p_yes == assessment.p_yes
    assert forecast.confidence == assessment.confidence
    assert forecast.market_probability == Decimal("0.55")
    assert forecast.abstain is False
    assert tuple(assessment.evidence_ids) == forecast.evidence_ids
