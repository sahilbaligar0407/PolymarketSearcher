"""The cross-venue matching service: candidates -> claims -> rule validation -> persisted matches.

Kalshi is the only execution venue; Polymarket is read-only intelligence (see
``docs/CONTRACTS.md``). This module's ``match()`` answers, for every Kalshi contract,
"which Polymarket market (if any) is provably the identical bet" -- and it is the
*only* place that assembles the persisted ``MarketMatch`` record. It never assigns
``same_outcome_boolean`` itself; that always comes straight out of
``resolution_rules.compare_claims``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from marketlab.clock import Clock
from marketlab.core.instruments import NormalizedMarket
from marketlab.logging import get_logger
from marketlab.matching.extract import extract_claim
from marketlab.matching.resolution_rules import RuleComparison, compare_claims
from marketlab.matching.semantic_match import (
    CandidatePair,
    LLMProviderLike,
    build_candidates,
    propose_candidates,
)

log = get_logger(__name__)

#: Bumped whenever the comparison logic in ``resolution_rules`` changes. A re-match
#: under new logic is a new experiment -- this string is part of that identity, mirrored
#: verbatim into every persisted ``MarketMatch.validator_version``.
VALIDATOR_VERSION = "1.1.0"

#: Mirrors ``configs/strategies.yaml: cross_venue.variants.require_match_confidence``.
DEFAULT_MIN_MATCH_CONFIDENCE = Decimal("0.90")


@dataclass(frozen=True)
class MarketMatch:
    """Storage-facing record of one cross-venue link.

    Field names and types mirror ``marketlab.storage.state.MarketMatch`` exactly (duck-
    typed rather than imported -- storage is owned by another team) so that
    ``store.save_match(match)`` can bind every field straight into SQL without
    conversion: ``rule_diff``/``time_diff``/``resolution_source_diff`` are flattened to
    strings here precisely because that table stores them as TEXT.
    """

    match_id: str
    canonical_id_a: str
    canonical_id_b: str
    match_confidence: Decimal
    same_outcome_boolean: bool | None
    rule_diff: str
    time_diff: str
    resolution_source_diff: str
    human_review_required: bool
    created_at: datetime
    validator_version: str = VALIDATOR_VERSION


@runtime_checkable
class StoreLike(Protocol):
    """Duck-typed subset of ``marketlab.storage.state.StateStore`` this module needs."""

    def save_match(self, match: Any) -> None: ...


def _match_id(canonical_id_a: str, canonical_id_b: str, validator_version: str) -> str:
    # Order-independent and stable across re-runs; changes only when either market or
    # the validator version changes, per the module docstring.
    a, b = sorted((canonical_id_a, canonical_id_b))
    digest = hashlib.sha256(f"{a}|{b}|{validator_version}".encode()).hexdigest()[:16]
    return f"match_{digest}"


class CrossVenueMatcher:
    """Candidates -> claim extraction -> rule comparison -> persisted ``MarketMatch``.

    An LLM provider, if supplied, only ever influences which *candidates*
    ``semantic_match.propose_candidates`` returns. It never sees, sets, or overrides
    ``same_outcome_boolean`` -- that field is produced exclusively by
    ``resolution_rules.compare_claims`` inside ``_build_match``.
    """

    def __init__(
        self,
        clock: Clock,
        store: StoreLike | None = None,
        *,
        llm_provider: LLMProviderLike | None = None,
        time_tolerance_seconds: float = 60.0,
    ) -> None:
        self._clock = clock
        self._store = store
        self._llm_provider = llm_provider
        self._time_tolerance_seconds = time_tolerance_seconds
        self._index: dict[str, list[NormalizedMarket]] = {}

    def build_index(self, markets: list[NormalizedMarket]) -> dict[str, list[NormalizedMarket]]:
        """Bucket a market pool by category for fast candidate lookup.

        The finer-grained close-time and entity narrowing happens per-pair inside
        ``semantic_match.build_candidates``; this index only needs to cut down which
        markets are even considered before that per-pair filter runs.
        """
        index: dict[str, list[NormalizedMarket]] = {}
        for m in markets:
            index.setdefault(str(m.category), []).append(m)
        self._index = index
        return index

    async def match(
        self,
        kalshi_markets: list[NormalizedMarket],
        poly_markets: list[NormalizedMarket],
    ) -> list[MarketMatch]:
        """Full pipeline: candidates -> claim extraction -> rule comparison -> matches."""
        self.build_index(poly_markets)
        results: list[MarketMatch] = []
        for km in kalshi_markets:
            bucket = self._index.get(str(km.category), [])
            candidates: list[CandidatePair] = build_candidates(km, bucket)
            if self._llm_provider is not None:
                candidates = await propose_candidates(
                    km, bucket, self._llm_provider, heuristic_candidates=candidates
                )
            claim_a = extract_claim(km)
            for candidate in candidates:
                claim_b = extract_claim(candidate.market_b)
                comparison = compare_claims(
                    claim_a, claim_b, time_tolerance_seconds=self._time_tolerance_seconds
                )
                built = self._build_match(km, candidate.market_b, comparison)
                results.append(built)
                if self._store is not None:
                    self._store.save_match(built)
        return results

    def _build_match(
        self, a: NormalizedMarket, b: NormalizedMarket, comparison: RuleComparison
    ) -> MarketMatch:
        return MarketMatch(
            match_id=_match_id(a.canonical_id, b.canonical_id, VALIDATOR_VERSION),
            canonical_id_a=a.canonical_id,
            canonical_id_b=b.canonical_id,
            match_confidence=comparison.confidence,
            same_outcome_boolean=comparison.same_outcome_boolean,
            rule_diff="; ".join(comparison.rule_diff),
            time_diff=(
                "" if comparison.time_diff_seconds is None else f"{comparison.time_diff_seconds:.3f}"
            ),
            resolution_source_diff=str(comparison.resolution_source_diff),
            human_review_required=comparison.human_review_required,
            created_at=self._clock.now(),
            validator_version=VALIDATOR_VERSION,
        )


def approved_for_automation(
    match: MarketMatch, min_confidence: Decimal = DEFAULT_MIN_MATCH_CONFIDENCE
) -> bool:
    """True only when it is unambiguously safe to let a strategy trade on this match.

    Requires ``same_outcome_boolean is True`` **and** ``match_confidence >=
    min_confidence`` **and** ``not human_review_required``. No confidence score, however
    high, can compensate for ``same_outcome_boolean`` being False or unknown (None) --
    that field alone encodes whether the deterministic validator found the two claims to
    describe the identical bet.
    """
    if match.same_outcome_boolean is not True:
        return False
    if match.human_review_required:
        return False
    return match.match_confidence >= min_confidence
