"""Match discovery: catalogue search proposes twins, Jev approves at most one per market."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from marketlab.matching.cross_venue import approved_for_automation
from marketlab.matching.discovery import (
    VALIDATOR_VERSION,
    KalshiCatalogIndex,
    kalshi_entries_from_events,
    poly_target_from_gamma,
    tokens,
    verdict_to_match,
    verify,
)

NOW = datetime(2026, 10, 5, tzinfo=UTC)

EVENTS = [
    {"title": "Los Angeles Mayoral Election", "category": "Elections", "markets": [
        {"ticker": "KXMAYORLA-26-KBAS", "title": "Who will win Los Angeles Mayoral Election?",
         "yes_sub_title": "Karen Bass", "status": "active", "close_time": "2026-11-04T00:00:00Z"},
        {"ticker": "KXMAYORLA-26-NRAM", "title": "Who will win Los Angeles Mayoral Election?",
         "yes_sub_title": "Nithya Raman", "status": "active", "close_time": "2026-11-04T00:00:00Z"},
    ]},
    {"title": "Fed decision", "category": "Economics", "markets": [
        {"ticker": "KXFED-26OCT-H25", "title": "Will the Federal Reserve hike rates by 25bps?",
         "yes_sub_title": "Hike 25bps", "status": "active", "close_time": "2026-10-29T00:00:00Z"},
    ]},
    {"title": "Parlay", "markets": [{"ticker": "KXMVESPORTS-1", "title": "x", "status": "active"}]},
]


def _gamma(question: str, outcomes: str = '["Yes", "No"]', cid: str = "0xbass") -> dict:
    return {"conditionId": cid, "question": question, "outcomes": outcomes,
            "description": "", "endDate": "2026-11-04T00:00:00Z"}


def test_tokens_normalise_parties_and_plurals() -> None:
    assert "democrat" in tokens("Will the Democrats win?")
    assert "democrat" in tokens("Democratic party")
    assert "republican" in tokens("GOP")
    assert "will" not in tokens("Will it")


def test_catalogue_skips_parlays_and_finds_the_named_candidate() -> None:
    entries = kalshi_entries_from_events(EVENTS)
    assert all(not e.ticker.startswith("KXMVE") for e in entries)
    index = KalshiCatalogIndex(entries)
    target = poly_target_from_gamma(_gamma("Will Karen Bass win the 2026 Los Angeles mayoral election?"))
    best, _ = index.candidates(target, k=3)[0]
    assert best.ticker == "KXMAYORLA-26-KBAS"


def test_targets_need_exactly_two_outcomes() -> None:
    assert poly_target_from_gamma(_gamma("q", outcomes='["A", "B", "C"]')) is None


class _Jev:
    def __init__(self, by_ticker: dict[str, tuple[float, float]]) -> None:
        self.by_ticker = by_ticker

    async def evaluate(self, key, state, questions):  # noqa: ANN001, ANN201
        ticker = key.rsplit(":", 1)[1]
        identical, yes = self.by_ticker.get(ticker, (0.0, 0.0))
        return {
            "relation": {"type": "choice", "probabilities": {
                "identical": identical, "related_not_identical": 1 - identical, "unrelated": 0.0}},
            "same_bet": {"type": "noul", "noul": yes},
        }


async def test_only_the_best_confident_candidate_is_approved() -> None:
    entries = kalshi_entries_from_events(EVENTS)
    target = poly_target_from_gamma(_gamma("Will Karen Bass win the 2026 Los Angeles mayoral election?"))
    jev = _Jev({"KXMAYORLA-26-KBAS": (0.95, 0.8), "KXMAYORLA-26-NRAM": (0.02, 0.05)})
    verdicts = await verify(jev, target, [(e, 1.0) for e in entries[:2]])
    approved = [v for v in verdicts if v.approved]
    assert [v.entry.ticker for v in approved] == ["KXMAYORLA-26-KBAS"]

    match = verdict_to_match(approved[0], NOW)
    assert match.validator_version == VALIDATOR_VERSION
    assert match.canonical_id_a == "kalshi:kxmayorla-26-kbas" and match.canonical_id_b == "poly:0xbass"
    assert approved_for_automation(match)  # 0.95 clears Jev's 0.80 bar, not just the 0.90 one
    rejected = verdict_to_match(next(v for v in verdicts if not v.approved), NOW)
    assert rejected.human_review_required and not approved_for_automation(rejected)


async def test_low_confidence_is_never_approved() -> None:
    entries = kalshi_entries_from_events(EVENTS)
    target = poly_target_from_gamma(_gamma("Will Karen Bass win the 2026 Los Angeles mayoral election?"))
    verdicts = await verify(_Jev({"KXMAYORLA-26-KBAS": (0.74, 0.9)}), target, [(entries[0], 1.0)])
    assert not any(v.approved for v in verdicts)


async def test_a_conflicting_threshold_vetoes_even_a_confident_jev() -> None:
    entries = kalshi_entries_from_events(EVENTS)
    fed = next(e for e in entries if e.ticker == "KXFED-26OCT-H25")
    target = poly_target_from_gamma(_gamma("Will the Fed hike rates by 50 bps in October?", cid="0xfed"))
    verdicts = await verify(_Jev({"KXFED-26OCT-H25": (0.95, 0.9)}), target, [(fed, 1.0)])
    assert verdicts[0].vetoes and not verdicts[0].approved
    assert Decimal("0.95") == verdict_to_match(verdicts[0], NOW).match_confidence
