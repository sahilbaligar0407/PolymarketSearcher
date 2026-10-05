"""Find the Kalshi twin of any Polymarket market: full-catalogue search, Jev verification.

Polymarket is where the signal is (its leaderboard and every wallet's holdings are
public); Kalshi is where we can trade. So the job is: for each Polymarket market the top
wallets hold, find the Kalshi contract that is the *same bet*. The older matchers could
not get there. The structural sports matcher only knows game winners, and the free-text
validator can only ever *flag* a pair for human review, so on 2026-10-05 only 5 of 182
held-in-common positions had an approved twin. Meanwhile we only ever downloaded the Kalshi
series our universes name (8.7k of ~114k open markets; 160 of ~2.8k politics).

This module works in three steps:

1. **Catalogue.** All open Kalshi events with nested markets (~12k events and ~114k
   markets, ~5 s to page through; multi-leg parlays excluded), held in memory for
   matching only.
2. **Candidates.** An inverted index over each Kalshi market's text (event title, market
   title, what YES means). For each Polymarket market it proposes the best few, scored
   by shared informative words weighted by rarity, times a closeness-of-dates factor.
3. **Verification.** TypeSafe's Jev is asked, in one call, how Kalshi YES relates to
   the Polymarket market resolving to its first outcome: *identical*, *related but not
   identical* (another stage, threshold, person or date) or *unrelated*; plus the same
   thing as a plain yes/no. Measured on hand-labelled pairs (2026-10-05) the choice
   separates cleanly - every true pair >= 0.81 "identical", every false one <= 0.21,
   including nomination-vs-election and party-vs-candidate - while the yes/no alone
   gave true pairs only 0.65-0.95. A pair is approved when it is the best candidate for
   that market, ``p(identical) >= min_identical`` and ``p(yes) >= min_yes``, and the one
   deterministic veto kept (a conflicting numeric threshold) does not fire. Every
   verdict, approved or not, is persisted, so a pair is asked once.

Orientation is fixed by construction. Polymarket outcome 0 must equal Kalshi YES ("Yes"
for a yes/no market; the first team for a game, whose own Kalshi market is the one
proposed). So an approved pair is always ``same_outcome_boolean=True`` and the existing
YES==YES consumers can use it unchanged.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from marketlab.logging import get_logger

log = get_logger(__name__)

VALIDATOR_VERSION = "jev-discovery-1.0"

_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "vs",
        "was",
        "what",
        "when",
        "which",
        "who",
        "will",
        "win",
        "wins",
        "winner",
        "with",
        "yes",
        "no",
        "market",
        "than",
        "over",
        "under",
        "more",
        "less",
        "least",
        "most",
        "any",
        "after",
        "during",
        "end",
        "close",
        "price",
        "above",
        "below",
        "between",
        "2025",
        "2026",
        "2027",
    ]
)
_SYNONYMS = {
    "democratic": "democrat",
    "democrats": "democrat",
    "dem": "democrat",
    "dems": "democrat",
    "republicans": "republican",
    "gop": "republican",
    "rep": "republican",
    "presidential": "president",
    "elected": "election",
    "elections": "election",
    "fed": "federal",
    "fomc": "federal",
    "rates": "rate",
    "bps": "bp",
}
#: Tokens on more Kalshi markets than this say nothing about which one is the twin.
_MAX_DF = 4000
#: The one validator diff code that vetoes a Jev-approved pair. The location, party and
#: comparator checks were tried and dropped: on live pairs they vetoed "Atlanta wins" vs
#: "Braves vs. Dodgers" and the Lions -3.5 spread (Jev 0.92), while Jev alone already
#: put every wrong candidate they caught at <= 0.1.
_VETO_PREFIXES = ("threshold_mismatch",)


def tokens(text: str) -> list[str]:
    out = []
    for raw in _WORD.findall(text.lower()):
        word = _SYNONYMS.get(raw, raw)
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = _SYNONYMS.get(word[:-1], word[:-1])
        if word not in _STOP and (len(word) > 2 or word.isdigit()):
            out.append(word)
    return out


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(frozen=True)
class KalshiEntry:
    ticker: str
    event_title: str
    title: str
    yes_means: str
    rules: str
    close_time: datetime | None
    category: str
    raw: dict

    @property
    def canonical_id(self) -> str:
        return f"kalshi:{self.ticker.lower()}"

    @property
    def text(self) -> str:
        return f"{self.event_title} {self.title} {self.yes_means}"


@dataclass(frozen=True)
class PolyTarget:
    condition_id: str
    question: str
    outcomes: tuple[str, ...]
    description: str
    end_time: datetime | None
    event_title: str
    raw: dict

    @property
    def canonical_id(self) -> str:
        return f"poly:{self.condition_id}"

    @property
    def text(self) -> str:
        first = self.outcomes[0] if self.outcomes else ""
        return f"{self.event_title} {self.question} {first if first.lower() != 'yes' else ''}"


def kalshi_entries_from_events(events: list[dict]) -> list[KalshiEntry]:
    out: list[KalshiEntry] = []
    for event in events:
        for m in event.get("markets") or []:
            ticker = str(m.get("ticker") or "")
            if (
                not ticker
                or ticker.startswith("KXMVE")
                or str(m.get("status", "active")) not in ("active", "open", "initialized")
            ):
                continue
            out.append(
                KalshiEntry(
                    ticker=ticker,
                    event_title=str(event.get("title") or ""),
                    title=str(m.get("title") or ""),
                    yes_means=str(m.get("yes_sub_title") or m.get("subtitle") or ""),
                    rules=str(m.get("rules_primary") or "")[:1500],
                    close_time=_parse_time(
                        m.get("expected_expiration_time") or m.get("close_time")
                    ),
                    category=str(event.get("category") or ""),
                    raw={**m, "event_title": event.get("title"), "category": event.get("category")},
                )
            )
    return out


def poly_target_from_gamma(raw: dict) -> PolyTarget | None:
    try:
        outcomes = raw.get("outcomes")
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        outcomes = tuple(str(o) for o in (outcomes or ()))
        if len(outcomes) != 2:
            return None
        events = raw.get("events") or []
        return PolyTarget(
            condition_id=str(raw["conditionId"]),
            question=str(raw.get("question") or ""),
            outcomes=outcomes,
            description=str(raw.get("description") or "")[:1500],
            end_time=_parse_time(raw.get("endDate")),
            event_title=str(events[0].get("title") or "")
            if events and isinstance(events[0], dict)
            else "",
            raw=raw,
        )
    except (KeyError, ValueError, TypeError):
        return None


class KalshiCatalogIndex:
    """Inverted index over the Kalshi catalogue for candidate generation."""

    def __init__(self, entries: list[KalshiEntry]) -> None:
        self.entries = entries
        postings: dict[str, list[int]] = defaultdict(list)
        for i, e in enumerate(entries):
            for tok in set(tokens(e.text)):
                postings[tok].append(i)
        n = max(len(entries), 1)
        self._postings = {t: ids for t, ids in postings.items() if len(ids) <= _MAX_DF}
        self._idf = {t: math.log(n / len(ids)) for t, ids in self._postings.items()}

    def candidates(self, target: PolyTarget, k: int = 3) -> list[tuple[KalshiEntry, float]]:
        scores: dict[int, float] = defaultdict(float)
        for tok in set(tokens(target.text)):
            idf = self._idf.get(tok)
            if idf is None:
                continue
            for i in self._postings[tok]:
                scores[i] += idf
        ranked: list[tuple[KalshiEntry, float]] = []
        for i, score in scores.items():
            entry = self.entries[i]
            if target.end_time is not None and entry.close_time is not None:
                days = abs((entry.close_time - target.end_time).total_seconds()) / 86400
                score /= 1 + days / 14
            ranked.append((entry, score))
        ranked.sort(key=lambda pair: -pair[1])
        return ranked[:k]


def _vetoed(target: PolyTarget, entry: KalshiEntry) -> list[str]:
    from marketlab.matching.extract import extract_claim_from_text
    from marketlab.matching.resolution_rules import compare_claims

    try:
        a = extract_claim_from_text(
            f"{entry.title} {entry.yes_means}", entry.rules, anchor_time=entry.close_time
        )
        b = extract_claim_from_text(
            f"{target.question} {target.outcomes[0]}",
            target.description,
            anchor_time=target.end_time,
        )
        reasons = compare_claims(a, b).blocking_reasons
    except Exception:  # noqa: BLE001 - the veto is a safety net, not a gate that can crash
        return []
    return [r for r in reasons if r.startswith(_VETO_PREFIXES)]


def jev_question(target: PolyTarget, entry: KalshiEntry) -> tuple[dict, dict]:
    first = target.outcomes[0]
    state = {
        "polymarket": {
            "event": target.event_title,
            "question": target.question,
            "outcomes": list(target.outcomes),
            "resolution_rules": target.description,
            "ends": target.end_time.isoformat() if target.end_time else None,
        },
        "kalshi": {
            "event": entry.event_title,
            "contract": entry.title,
            "yes_means": entry.yes_means,
            "resolution_rules": entry.rules,
            "closes": entry.close_time.isoformat() if entry.close_time else None,
        },
    }
    questions = {
        "relation": {
            "type": "choice",
            "instructions": f"How does Kalshi YES relate to the Polymarket market resolving '{first}'?",
            "criteria": {
                "identical": f"Same event, same condition: Kalshi YES happens exactly when Polymarket resolves '{first}'",
                "related_not_identical": (
                    "Same topic but a different condition, threshold, stage (e.g. nomination vs "
                    "election), person, party, outcome or date"
                ),
                "unrelated": "A different event",
            },
        },
        "same_bet": {
            "type": "noul",
            "instructions": (
                f"Does the Kalshi contract resolve YES in exactly the cases where the Polymarket "
                f"market resolves to '{first}'?"
            ),
        },
    }
    return state, questions


@dataclass
class Verdict:
    target: PolyTarget
    entry: KalshiEntry
    #: Jev's p("identical") from the relation choice; the ranking and approval score.
    probability: float
    p_yes: float
    approved: bool
    vetoes: list[str]
    search_score: float


def _choice_probability(answers: dict | None, question_id: str, option: str) -> float | None:
    answer = (answers or {}).get(question_id) or {}
    value = (answer.get("probabilities") or {}).get(option)
    return float(value) if isinstance(value, int | float) else None


async def verify(
    jev: Any,
    target: PolyTarget,
    candidates: list[tuple[KalshiEntry, float]],
    *,
    min_identical: float = 0.8,
    min_yes: float = 0.5,
) -> list[Verdict]:
    """Ask Jev about each candidate (concurrently); approve at most one, the best."""
    import asyncio

    from marketlab.ai.typesafe import noul

    async def ask(entry: KalshiEntry, score: float) -> Verdict | None:
        state, questions = jev_question(target, entry)
        answers = await jev.evaluate(
            f"match:{VALIDATOR_VERSION}:{target.condition_id}:{entry.ticker}", state, questions
        )
        identical = _choice_probability(answers, "relation", "identical")
        if identical is None:
            return None
        p_yes = noul(answers, "same_bet") or 0.0
        return Verdict(target, entry, identical, p_yes, False, _vetoed(target, entry), score)

    verdicts = [
        v for v in await asyncio.gather(*(ask(e, s) for e, s in candidates)) if v is not None
    ]
    best = max((v for v in verdicts if not v.vetoes), key=lambda v: v.probability, default=None)
    if best is not None and best.probability >= min_identical and best.p_yes >= min_yes:
        best.approved = True
    return verdicts


def verdict_to_match(verdict: Verdict, now: datetime) -> Any:
    from marketlab.storage.state import MarketMatch

    a, b = verdict.entry.canonical_id, verdict.target.canonical_id
    return MarketMatch(
        match_id=f"{a}|{b}|{VALIDATOR_VERSION}",
        canonical_id_a=a,
        canonical_id_b=b,
        match_confidence=Decimal(str(round(verdict.probability, 4))),
        same_outcome_boolean=verdict.approved,
        rule_diff="; ".join(verdict.vetoes)
        or f"jev p(identical)={verdict.probability:.3f} p(yes)={verdict.p_yes:.3f}",
        time_diff="",
        resolution_source_diff="",
        human_review_required=not verdict.approved,
        created_at=now,
        validator_version=VALIDATOR_VERSION,
    )


__all__ = [
    "VALIDATOR_VERSION",
    "KalshiCatalogIndex",
    "KalshiEntry",
    "PolyTarget",
    "Verdict",
    "jev_question",
    "kalshi_entries_from_events",
    "poly_target_from_gamma",
    "tokens",
    "verdict_to_match",
    "verify",
]
