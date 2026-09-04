"""The deterministic gate between the local model and everything downstream.

This is the most important file in the AI layer. Nothing produced by
:mod:`marketlab.ai.ollama` (or any other provider) reaches a strategy, the risk
gateway, or the broker without passing through :func:`validate_assessment` first.

**On any failure: ABSTAIN. This module never "repairs" a malformed recommendation.**
An out-of-range ``p_yes``, an unknown ``evidence_id``, a look-ahead timestamp, or a
future ``information_cutoff`` is a reason to reject the *entire* assessment -- never to
clamp one field back into range and keep using the rest. The one exception is
``resolution_rule_warning``, where forcing ``abstain: true`` is the intentional,
documented behavior the model itself asked for, not a repair of a mistake.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from marketlab.ai.retrieval import EvidenceBundle
from marketlab.ai.schemas import MarketAssessment
from marketlab.core.instruments import NormalizedMarket
from marketlab.core.strategy import ProbabilityForecast
from marketlab.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """``valid=False`` always means ``assessment_or_none is None``: reject outright,
    nothing to record. ``valid=True`` means a usable assessment is returned -- possibly
    with ``abstain`` forced to True (see ``resolution_rule_warning`` handling below),
    which is a sanctioned transformation, not a repair.
    """

    valid: bool
    assessment_or_none: MarketAssessment | None
    failures: list[str] = field(default_factory=list)


def parse_llm_output(raw: str | dict[str, Any] | None) -> MarketAssessment | None:
    """Parse raw model output into a :class:`MarketAssessment`. Never raises.

    A parse failure -- malformed JSON, a missing required field, a wrong type -- returns
    ``None``. Callers must feed that ``None`` straight into :func:`validate_assessment`,
    which treats it as an automatic abstain. This function makes no attempt to coerce or
    salvage a partial payload into something plausible: a malformed blob is exactly the
    case the deterministic gate exists to catch.
    """
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            return MarketAssessment.model_validate_json(raw)
        return MarketAssessment.model_validate(raw)
    except (ValidationError, ValueError, TypeError) as exc:
        log.warning("ai.validator.parse_failed", error=str(exc))
        return None


#: Salient tokens worth checking for: dollar amounts, 4-digit years, percentages, and
#: capitalized multi-letter words (named entities, proper nouns).
_TOKEN_RE = re.compile(r"\$[0-9][0-9,.]*|\b\d{4}\b|\b\d+(?:\.\d+)?%|\b[A-Z][a-zA-Z]{2,}\b")


def _extract_rule_tokens(rules: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(rules)}


def _interpretation_consistent(interpretation: str, rules: str) -> bool:
    """Conservative consistency check between the model's restatement of the question
    and the stored resolution rules.

    We pull salient tokens (numbers, dollar thresholds, percentages, 4-digit years,
    capitalized entities) out of ``rules`` and require at least one of them to appear in
    ``interpretation``. This cannot prove the model understood the rules, but it can
    catch the case where it substituted a different entity, date, or threshold entirely.
    If ``rules`` is empty there is nothing to check against, so we pass (there is no
    warrant for a stronger claim); if ``rules`` is non-empty but yields tokens that never
    show up in the interpretation at all, we fail closed -- "when in doubt, fail" means
    an unverifiable interpretation is treated as inconsistent, not as innocent.
    """
    if not rules.strip():
        return True
    tokens = _extract_rule_tokens(rules)
    if not tokens:
        return True
    lowered = interpretation.lower()
    return any(tok in lowered for tok in tokens)


def validate_assessment(
    assessment: MarketAssessment | None,
    market: NormalizedMarket,
    bundle: EvidenceBundle,
    decision_time: datetime,
) -> ValidationResult:
    """Run every mandatory check. Any failure => ``valid=False``, ``assessment_or_none
    =None``. All checks run (rather than short-circuiting) so ``failures`` reports every
    problem found, not just the first.
    """
    if assessment is None:
        return ValidationResult(valid=False, assessment_or_none=None, failures=["malformed_or_missing_assessment"])

    failures: list[str] = []

    if not (Decimal(0) <= assessment.p_yes <= Decimal(1)):
        failures.append(f"p_yes out of range [0,1]: {assessment.p_yes}")
    if not (Decimal(0) <= assessment.confidence <= Decimal(1)):
        failures.append(f"confidence out of range [0,1]: {assessment.confidence}")

    evidence_index = bundle.index()
    for evidence_id in assessment.evidence_ids:
        item = evidence_index.get(evidence_id)
        if item is None:
            failures.append(f"unknown evidence_id: {evidence_id}")
            continue
        if item.timestamp > decision_time:
            failures.append(
                f"look-ahead evidence: {evidence_id} timestamped {item.timestamp.isoformat()} "
                f"is after decision_time {decision_time.isoformat()}"
            )

    if assessment.information_cutoff > decision_time:
        failures.append(
            f"information_cutoff {assessment.information_cutoff.isoformat()} is after "
            f"decision_time {decision_time.isoformat()}"
        )

    if assessment.market_id != market.canonical_id:
        failures.append(f"market_id mismatch: assessment={assessment.market_id} market={market.canonical_id}")
    if bundle.market.canonical_id != market.canonical_id:
        failures.append(
            f"bundle/market mismatch: bundle={bundle.market.canonical_id} market={market.canonical_id}"
        )

    if not _interpretation_consistent(assessment.question_interpretation, bundle.rules or market.resolution_rules):
        failures.append("question_interpretation inconsistent with stored resolution rules")

    if failures:
        return ValidationResult(valid=False, assessment_or_none=None, failures=failures)

    if assessment.resolution_rule_warning and not assessment.abstain:
        forced = assessment.model_copy(update={"abstain": True})
        return ValidationResult(
            valid=True,
            assessment_or_none=forced,
            failures=["forced_abstain: resolution_rule_warning was set"],
        )

    return ValidationResult(valid=True, assessment_or_none=assessment, failures=[])


def assessment_to_forecast(
    assessment: MarketAssessment,
    strategy_id: str,
    experiment_id: str,
    market_probability: Decimal | None,
) -> ProbabilityForecast:
    """Project a validated :class:`MarketAssessment` onto the shared
    :class:`~marketlab.core.strategy.ProbabilityForecast` record.

    Callers must only pass an assessment that already came back from
    :func:`validate_assessment` with ``valid=True`` -- this function performs no
    re-validation of its own.
    """
    rationale = (
        "; ".join(assessment.missing_information)
        if assessment.abstain
        else "; ".join(assessment.supporting_facts)
    )
    return ProbabilityForecast(
        strategy_id=strategy_id,
        experiment_id=experiment_id,
        canonical_id=assessment.market_id,
        as_of=assessment.as_of,
        p_yes=assessment.p_yes,
        confidence=assessment.confidence,
        market_probability=market_probability,
        abstain=assessment.abstain,
        evidence_ids=tuple(assessment.evidence_ids),
        features={
            "question_interpretation": assessment.question_interpretation,
            "resolution_rule_warning": assessment.resolution_rule_warning,
        },
        rationale=rationale,
    )
