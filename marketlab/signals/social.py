"""Public-statement classifier for tracked public figures.

**This module produces features for an event study, not a trade signal.** It answers
"what kind of statement was this, about what, mentioning whom, during which trading
session" - never "buy" or "sell". Sentiment is a descriptive float, not a direction; the
first strategy to *act* on any of this belongs to a strategy module, which will combine
it with price, liquidity and risk state that this module deliberately knows nothing
about. No function in this module returns a trade direction, an order side, or anything
resembling one - that is a hard invariant, checked by ``tests/unit/test_social_classifier.py``.

Configuration comes from ``configs/sources.yaml``'s ``public_figures:`` block (owned by
another team); sane defaults (Trump, first) are used if that file or block is absent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from marketlab.signals.news import (
    extract_entities,
    extract_tickers,
    minhash_signature,
    minhash_similarity,
    text_shingles,
)

_EASTERN = ZoneInfo("America/New_York")

#: The forward-return horizons the PRD requires for the event study. ``"close"`` is a
#: sentinel meaning "next scheduled session close", not a fixed timedelta - callers
#: resolve it against a market calendar rather than adding it to a timestamp directly.
EVENT_STUDY_HORIZONS: tuple[str, ...] = ("1m", "5m", "15m", "1h", "close", "1d", "3d", "5d")

_FIXED_HORIZONS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
    "3d": timedelta(days=3),
    "5d": timedelta(days=5),
}


def event_study_windows() -> dict[str, timedelta | None]:
    """Horizon name -> timedelta, with ``"close"`` mapped to ``None`` (session-close sentinel).

    Shared constant: other teams (analytics, AI analyst) import this so every event
    study in the codebase measures the same set of forward-return windows.
    """
    return {"close": None, **_FIXED_HORIZONS}


# ---------------------------------------------------------------------------
# Public-figure configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PublicFigureConfig:
    key: str
    enabled: bool = True
    sources: tuple[str, ...] = ()
    handles: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()


#: Used when ``configs/sources.yaml`` is absent or has no ``public_figures`` block.
#: Trump is the first configured subject per the PRD; kept generic so a config file
#: can add more without code changes.
DEFAULT_PUBLIC_FIGURES: dict[str, PublicFigureConfig] = {
    "trump": PublicFigureConfig(
        key="trump",
        enabled=True,
        sources=("x", "bluesky", "gdelt"),
        handles=("realDonaldTrump",),
        keywords=(
            "tariff", "china", "fed", "rates", "oil", "semiconductor", "auto",
            "defense", "pharma", "crypto", "immigration", "sanctions",
        ),
    ),
}


def load_public_figures(source_toggles: dict[str, Any] | None) -> dict[str, PublicFigureConfig]:
    """Parse the ``public_figures:`` block of ``configs/sources.yaml``.

    Falls back to :data:`DEFAULT_PUBLIC_FIGURES` when the file, or the block, is
    missing - another team owns that file, and this module must not fail to import or
    crash the engine just because it hasn't landed yet.
    """
    if not source_toggles:
        return dict(DEFAULT_PUBLIC_FIGURES)
    block = source_toggles.get("public_figures")
    if not isinstance(block, dict) or not block:
        return dict(DEFAULT_PUBLIC_FIGURES)

    out: dict[str, PublicFigureConfig] = {}
    for key, raw in block.items():
        if not isinstance(raw, dict):
            continue
        out[key] = PublicFigureConfig(
            key=key,
            enabled=bool(raw.get("enabled", True)),
            sources=tuple(raw.get("sources", ()) or ()),
            handles=tuple(raw.get("handles", ()) or ()),
            keywords=tuple(raw.get("keywords", ()) or ()),
        )
    return out or dict(DEFAULT_PUBLIC_FIGURES)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

#: Ordered most-specific-first: the first matching category wins so a sentence that
#: is both "tariff" and vaguely "critical" is filed under the concrete policy action.
_ACTION_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tariff", ("tariff", "tariffs", "import duty", "import duties")),
    ("sanctions", ("sanction", "sanctions", "blacklist", "export ban", "export controls")),
    ("antitrust", ("antitrust", "monopoly", "monopolies", "breakup", "break up the company")),
    ("subsidy", ("subsidy", "subsidies", "grant funding", "stimulus check")),
    ("tax", ("tax cut", "tax cuts", "tax increase", "tax hike", "corporate tax", "capital gains tax")),
    ("contract", ("awarded a contract", "contract with", "defense contract", "procurement deal")),
    ("regulatory", ("regulation", "regulatory", "deregulate", "deregulation", "new rule", "compliance rule")),
    ("executive_action", ("executive order", "signed an order", "signs an order")),
    ("foreign_policy", ("foreign policy", "nato", "treaty", "diplomatic", "embassy", "ceasefire", "summit")),
)

_PRAISE_WORDS = (
    "great job", "fantastic job", "doing a great", "amazing job", "proud of",
    "thank you", "tremendous", "incredible job", "very happy with", "respect for",
)
_CRITICISM_WORDS = (
    "disaster", "failing", "terrible job", "should be ashamed", "witch hunt",
    "very disappointed", "weak and pathetic", "fake news", "total failure",
)

_POLICY_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "trade": ("tariff", "import", "export", "trade deal", "trade war", "china"),
    "monetary_policy": ("fed", "federal reserve", "interest rate", "rates", "inflation", "powell"),
    "immigration": ("immigration", "border", "asylum", "deportation"),
    "energy": ("oil", "gas prices", "opec", "drilling", "pipeline", "energy"),
    "healthcare": ("drug prices", "pharma", "medicare", "medicaid", "insulin"),
    "technology": ("semiconductor", "chips act", "ai regulation", "tech companies"),
    "defense": ("defense", "military", "pentagon", "weapons"),
    "crypto": ("crypto", "bitcoin", "digital asset", "stablecoin"),
    "foreign_policy": ("nato", "ukraine", "israel", "treaty", "sanctions", "diplomatic"),
    "taxes": ("tax cut", "tax increase", "corporate tax", "capital gains"),
}

_SECTOR_BY_TICKER: dict[str, str] = {
    "AAPL": "technology", "MSFT": "technology", "NVDA": "technology", "GOOGL": "technology",
    "META": "technology", "AMZN": "technology",
    "F": "auto", "GM": "auto", "TSLA": "auto",
    "LMT": "defense", "RTX": "defense", "NOC": "defense", "GD": "defense",
    "PFE": "pharma", "MRK": "pharma", "JNJ": "pharma", "LLY": "pharma",
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    "JPM": "financials", "GS": "financials", "BAC": "financials",
}

_POSITIVE_WORDS = (
    "great", "fantastic", "amazing", "tremendous", "incredible", "proud", "thank",
    "strong", "success", "winning", "historic", "record",
)
_NEGATIVE_WORDS = (
    "disaster", "failing", "terrible", "weak", "pathetic", "fake", "disappointed",
    "ashamed", "crisis", "collapse", "witch hunt", "failure",
)

_WORD_RE = re.compile(r"[a-z']+")


@dataclass(frozen=True, slots=True)
class StatementClassification:
    """Descriptive features about one public statement. Never a trade direction."""

    action_type: str = "misc"
    policy_topic: str = ""
    mentioned_companies: tuple[str, ...] = ()
    mentioned_tickers: tuple[str, ...] = ()
    sector: str = ""
    sentiment: float = 0.0
    novelty: float = 1.0
    market_hours_state: str = "closed"


def _sentiment_score(lowered: str) -> float:
    words = _WORD_RE.findall(lowered)
    if not words:
        return 0.0
    # Count phrase-level hits rather than per-word double counting.
    pos_hits = sum(lowered.count(p) for p in _POSITIVE_WORDS)
    neg_hits = sum(lowered.count(n) for n in _NEGATIVE_WORDS)
    total = pos_hits + neg_hits
    if total == 0:
        return 0.0
    score = (pos_hits - neg_hits) / total
    return max(-1.0, min(1.0, score))


def _classify_action_type(lowered: str) -> str:
    for action, keywords in _ACTION_KEYWORDS:
        if any(kw in lowered for kw in keywords):
            return action
    if any(p in lowered for p in _PRAISE_WORDS):
        return "praise"
    if any(c in lowered for c in _CRITICISM_WORDS):
        return "criticism"
    return "misc"


def _classify_policy_topic(lowered: str) -> str:
    for topic, keywords in _POLICY_TOPIC_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return topic
    return ""


def _infer_sector(tickers: tuple[str, ...], lowered: str) -> str:
    for t in tickers:
        sector = _SECTOR_BY_TICKER.get(t)
        if sector:
            return sector
    # Fall back to the same keyword table used for policy_topic where it doubles as
    # a sector hint (energy, defense, technology, pharma).
    for sector in ("energy", "defense", "technology", "pharma"):
        keywords = _POLICY_TOPIC_KEYWORDS.get(sector, _POLICY_TOPIC_KEYWORDS.get("healthcare", ()))
        if sector == "pharma":
            keywords = _POLICY_TOPIC_KEYWORDS["healthcare"]
        if any(kw in lowered for kw in keywords):
            return sector
    return ""


def market_hours_state(timestamp: datetime) -> str:
    """premarket / open / afterhours / closed, evaluated in US/Eastern.

    Uses the standard NYSE-adjacent session bounds (4:00 premarket, 9:30 open, 16:00
    close, 20:00 end of extended hours) and correctly reflects the US/Eastern UTC
    offset across the DST boundary because ``zoneinfo`` carries the IANA rules, not a
    fixed offset. Market holidays are not modeled - this is a session-hours heuristic
    for event-study bucketing, not a trading calendar.
    """
    if timestamp.tzinfo is None:
        raise ValueError("market_hours_state requires a timezone-aware timestamp")
    local = timestamp.astimezone(_EASTERN)
    if local.weekday() >= 5:  # Saturday/Sunday
        return "closed"
    minutes = local.hour * 60 + local.minute
    if 4 * 60 <= minutes < 9 * 60 + 30:
        return "premarket"
    if 9 * 60 + 30 <= minutes < 16 * 60:
        return "open"
    if 16 * 60 <= minutes < 20 * 60:
        return "afterhours"
    return "closed"


def classify_statement(
    text: str,
    timestamp: datetime,
    ticker_map: dict[str, str] | None = None,
    recent_texts: tuple[str, ...] = (),
) -> StatementClassification:
    """Classify one public statement into event-study features.

    ``timestamp`` must be the statement's own tz-aware time (``first_seen_time`` or
    ``published_time`` from the originating event) - this function never reads the
    clock itself, it only interprets a timestamp it's handed.
    """
    if not text:
        return StatementClassification(market_hours_state=market_hours_state(timestamp))

    lowered = text.lower()
    action_type = _classify_action_type(lowered)
    policy_topic = _classify_policy_topic(lowered)
    entities = extract_entities(text)
    tickers = extract_tickers(text, ticker_map)
    sector = _infer_sector(tickers, lowered)
    sentiment = _sentiment_score(lowered)

    novelty_score = 1.0
    if recent_texts:
        sig = minhash_signature(text_shingles(text))
        max_sim = 0.0
        for other in recent_texts:
            other_sig = minhash_signature(text_shingles(other))
            max_sim = max(max_sim, minhash_similarity(sig, other_sig))
        novelty_score = max(0.0, 1.0 - max_sim)

    return StatementClassification(
        action_type=action_type,
        policy_topic=policy_topic,
        mentioned_companies=entities,
        mentioned_tickers=tickers,
        sector=sector,
        sentiment=sentiment,
        novelty=novelty_score,
        market_hours_state=market_hours_state(timestamp),
    )
