"""Canonical cross-venue market matching.

Kalshi is the only execution venue; Polymarket is read-only intelligence (see
``docs/CONTRACTS.md``). A match's job is: given a Polymarket market, which Kalshi
contract, if any, is provably the *identical* bet -- not merely a similar one.

Pipeline: :mod:`marketlab.matching.semantic_match` proposes cheap candidates (never an
approval) -> :mod:`marketlab.matching.extract` turns each side's title/rules into a
structured :class:`~marketlab.matching.extract.MarketClaim` -> :mod:`marketlab.matching.
resolution_rules` deterministically compares the two claims and is the only function
allowed to set ``same_outcome_boolean`` -> :mod:`marketlab.matching.cross_venue` wires
the whole pipeline together and persists the result.

When uncertain, refuse the match: a missed arbitrage costs nothing, a false one risks
real capital.
"""

from __future__ import annotations

from marketlab.matching.cross_venue import (
    VALIDATOR_VERSION,
    CrossVenueMatcher,
    MarketMatch,
    approved_for_automation,
)
from marketlab.matching.extract import MarketClaim, extract_claim, extract_claim_from_text
from marketlab.matching.resolution_rules import RuleComparison, compare_claims, detect_complement
from marketlab.matching.semantic_match import CandidatePair, build_candidates, propose_candidates

__all__ = [
    "VALIDATOR_VERSION",
    "CandidatePair",
    "CrossVenueMatcher",
    "MarketClaim",
    "MarketMatch",
    "RuleComparison",
    "approved_for_automation",
    "build_candidates",
    "compare_claims",
    "detect_complement",
    "extract_claim",
    "extract_claim_from_text",
    "propose_candidates",
]
