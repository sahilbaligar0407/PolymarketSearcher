"""Cheap, dependency-free candidate *proposal* -- never approval.

This module's only job is to narrow an O(n*m) cross-venue pairing problem down to a
short candidate list worth running through the validator. It uses plain token-set
(Jaccard) similarity, entity overlap, and category/close-time bucketing -- no external
library, per house rules.

An optional LLM may be plugged in via ``propose_candidates(..., llm_provider=...)`` to
*suggest* additional candidates a keyword heuristic might miss (e.g. "Fed holds rates"
vs "FOMC keeps rates unchanged"). Its suggestions are appended to the same plain
``CandidatePair`` list heuristic candidates use.

**Invariant (enforced by construction, not convention): nothing in this module can ever
produce an approved match.** ``CandidatePair`` carries only market references and an
advisory score -- it has no ``same_outcome_boolean``, no ``confidence`` used for trading,
no approval flag of any kind. Every candidate this module proposes, heuristic or
LLM-sourced, must still be extracted into a ``MarketClaim`` and passed through
``marketlab.matching.resolution_rules.compare_claims`` before
``marketlab.matching.cross_venue.approved_for_automation`` can ever return True. See
``tests/unit/test_matching.py::test_llm_proposed_candidate_cannot_bypass_validator``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from marketlab.core.instruments import NormalizedMarket
from marketlab.logging import get_logger

log = get_logger(__name__)

#: Markets whose close times are further apart than this are never candidates -- they
#: cannot be settling on the same real-world instant.
DEFAULT_MAX_CLOSE_TIME_GAP_HOURS = 6.0
DEFAULT_MIN_JACCARD = 0.15

_STOPWORDS: frozenset[str] = frozenset(
    ["will", "be", "the", "a", "an", "at", "on", "in", "of", "to", "for", "and", "or", "is", "are", "does", "do", "than", "more", "by", "end", "before", "after", "this", "that", "what", "what's", "happens", "happen", "today", "tomorrow"]
)


@dataclass(frozen=True)
class CandidatePair:
    """An advisory suggestion that two markets *might* be the same bet.

    Carries no approval semantics whatsoever -- see the module docstring's invariant.
    """

    market_a: NormalizedMarket
    market_b: NormalizedMarket
    jaccard_score: float
    entity_overlap: float
    source: str  # "heuristic" | "llm"


@runtime_checkable
class LLMProviderLike(Protocol):
    """Duck-typed subset of ``marketlab.ai.provider.LLMProvider`` this module needs."""

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        timeout: float = 90.0,  # noqa: ASYNC109
    ) -> Any: ...


def _tokenize(title: str) -> frozenset[str]:
    words = re.findall(r"[a-zA-Z0-9]+", title.lower())
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 1)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _close_time_gap_hours(a: NormalizedMarket, b: NormalizedMarket) -> float | None:
    if a.close_time is None or b.close_time is None:
        return None
    return abs((a.close_time - b.close_time).total_seconds()) / 3600.0


def build_candidates(
    market: NormalizedMarket,
    pool: list[NormalizedMarket],
    *,
    same_category_only: bool = True,
    max_close_time_gap_hours: float = DEFAULT_MAX_CLOSE_TIME_GAP_HOURS,
    min_jaccard: float = DEFAULT_MIN_JACCARD,
) -> list[CandidatePair]:
    """Heuristic candidate generation: Jaccard title similarity within category/time buckets."""
    a_tokens = _tokenize(market.title)
    out: list[CandidatePair] = []
    for other in pool:
        if other.canonical_id == market.canonical_id:
            continue
        if same_category_only and market.category != other.category:
            continue
        gap = _close_time_gap_hours(market, other)
        if gap is not None and gap > max_close_time_gap_hours:
            continue
        b_tokens = _tokenize(other.title)
        score = jaccard(a_tokens, b_tokens)
        if score < min_jaccard:
            continue
        out.append(
            CandidatePair(
                market_a=market,
                market_b=other,
                jaccard_score=score,
                entity_overlap=score,
                source="heuristic",
            )
        )
    out.sort(key=lambda c: c.jaccard_score, reverse=True)
    return out


async def propose_candidates(
    market: NormalizedMarket,
    pool: list[NormalizedMarket],
    llm_provider: LLMProviderLike | None = None,
    *,
    heuristic_candidates: list[CandidatePair] | None = None,
    max_llm_suggestions: int = 5,
) -> list[CandidatePair]:
    """Propose candidates for ``market`` out of ``pool``. Advisory only -- see module docstring.

    If ``llm_provider`` is supplied it may suggest extra ``(market, pool_market)`` pairs a
    keyword heuristic missed, by canonical_id. A malformed, empty, or failing LLM response
    degrades silently to the heuristic-only list -- this function never raises on an LLM
    problem, and never lets the LLM emit anything but *which pool market to consider next*.
    """
    candidates = (
        heuristic_candidates
        if heuristic_candidates is not None
        else build_candidates(market, pool)
    )
    if llm_provider is None:
        return candidates

    already = {c.market_b.canonical_id for c in candidates}
    pool_by_id = {m.canonical_id: m for m in pool if m.canonical_id != market.canonical_id}
    if not pool_by_id:
        return candidates

    prompt = (
        "You are proposing CANDIDATE markets that might describe the same real-world bet "
        f"as: {market.title!r}. You are NOT deciding whether they match -- a separate "
        "deterministic validator does that. From the following pool markets, list the "
        "canonical_ids of up to "
        f"{max_llm_suggestions} that are plausibly about the same event.\n\nPool:\n"
        + "\n".join(f"- {m.canonical_id}: {m.title}" for m in pool_by_id.values())
    )
    try:
        response = await llm_provider.generate(
            prompt,
            schema={"type": "object", "properties": {"canonical_ids": {"type": "array"}}},
        )
    except Exception:  # noqa: BLE001 -- an LLM failure must never break candidate proposal.
        log.warning("semantic_match.llm_propose_failed", market=market.canonical_id)
        return candidates

    parsed = getattr(response, "parsed", None)
    suggested_ids: list[str] = []
    if isinstance(parsed, dict):
        raw_ids = parsed.get("canonical_ids", [])
        if isinstance(raw_ids, list):
            suggested_ids = [str(x) for x in raw_ids]

    for cid in suggested_ids[:max_llm_suggestions]:
        if cid in already or cid not in pool_by_id:
            continue
        candidates.append(
            CandidatePair(
                market_a=market,
                market_b=pool_by_id[cid],
                jaccard_score=jaccard(_tokenize(market.title), _tokenize(pool_by_id[cid].title)),
                entity_overlap=0.0,
                source="llm",
            )
        )
        already.add(cid)

    return candidates
