"""Deterministic structural extraction of a :class:`MarketClaim` from a market's
title, resolution rules text and structured metadata.

**No LLM is used here.** Everything is regex / gazetteer / stdlib-datetime based, so the
output is reproducible and auditable -- the validator in :mod:`marketlab.matching.
resolution_rules` depends on that determinism to make an automated-trading decision.

The single most important distinction this module must get right (the PRD's headline
test) is:

    "Will BTC be above $100k at 5 PM ET?"        -> measurement = "terminal"
    "Will BTC reach $100k before 5 PM ET?"       -> measurement = "barrier_touch"

These describe different stochastic events (the terminal value of a path vs. whether the
path ever crosses a barrier) and must never be treated as equivalent, even though the
English is nearly identical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from datetime import timezone as _tz
from decimal import Decimal, InvalidOperation

from marketlab.core.instruments import NormalizedMarket

# ---------------------------------------------------------------------------
# MarketClaim
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketClaim:
    """A structured claim extracted from one market's text + metadata.

    The first twelve fields are the PRD's minimum shape. The trailing three
    (``postponement_rule``, ``inclusion_exclusion``, ``settlement_timing``) are
    additive -- they carry the textual signals :mod:`resolution_rules` needs to
    validate cancellation and settlement-timing compatibility without re-parsing raw
    text at comparison time. All three default to "unspecified" values so a claim built
    from thin text degrades to "unknown", never to a false "compatible".
    """

    subject: str
    subject_tokens: frozenset[str]
    outcome: str | None
    comparator: str | None
    threshold: Decimal | None
    threshold_unit: str | None
    measurement: str | None
    time_window_start: datetime | None
    time_window_end: datetime | None
    timezone: str
    resolution_authority: str | None
    entities: frozenset[str]
    postponement_rule: str | None = None
    inclusion_exclusion: frozenset[str] = field(default_factory=frozenset)
    settlement_timing: str | None = None
    raw_title: str = ""
    #: US states / major cities named by the market ("TX", "MI", "CITY:CHICAGO"). A
    #: Texas race and a Michigan race are never the same bet, however alike the English.
    locations: frozenset[str] = field(default_factory=frozenset)
    #: Political parties the market is about ("R", "D", "L").
    parties: frozenset[str] = field(default_factory=frozenset)
    #: Capitalized proper names in a sentence-case title (people, teams, places the
    #: gazetteers do not know). Empty for title-cased titles, where capitals carry no
    #: signal.
    named_entities: frozenset[str] = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# gazetteers / normalization tables
# ---------------------------------------------------------------------------

#: Small alias table so differently-worded mentions of the same entity collapse to one
#: canonical token. A production system would load this from a maintained catalogue;
#: for deterministic matching what matters is that both sides of a candidate pair are
#: normalized through the *same* table.
_ENTITY_ALIASES: dict[str, str] = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "xrp": "XRP",
    "ripple": "XRP",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "chiefs": "KC",
    "kansas city chiefs": "KC",
    "kansas city": "KC",
    "bills": "BUF",
    "buffalo bills": "BUF",
    "buffalo": "BUF",
    "cpi": "CPI",
    "core cpi": "CORE_CPI",
    "core": "CORE",
    "headline": "HEADLINE",
    "fed": "FED",
    "fomc": "FED",
    "federal reserve": "FED",
    "nyc": "NYC",
    "new york city": "NYC",
    "new york": "NYC",
}

_US_STATES: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL",
    "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
    # Bare "washington" is deliberately absent: state, D.C. and several teams share it.
    "washington state": "WA", "washington dc": "DC", "washington d c": "DC",
    "district of columbia": "DC",
    # "New York" is ambiguous between city and state, so every spelling folds to one
    # token rather than letting "NYC" vs "New York" read as a conflict.
    "new york": "NY", "new york city": "NY", "new york state": "NY", "nyc": "NY",
}

_US_CITIES: dict[str, str] = {
    name: f"CITY:{name.upper().replace(' ', '_')}"
    for name in (
        "los angeles", "chicago", "houston", "phoenix", "philadelphia", "san antonio",
        "san diego", "dallas", "austin", "san francisco", "seattle", "denver", "boston",
        "miami", "atlanta", "detroit", "las vegas", "new orleans", "minneapolis",
        "kansas city", "oklahoma city", "salt lake city", "virginia beach",
    )
}

#: Longest phrase first, so "kansas city" is consumed before "kansas" and "west
#: virginia" before "virginia".
_LOCATION_PHRASES: tuple[tuple[str, str], ...] = tuple(
    sorted({**_US_STATES, **_US_CITIES}.items(), key=lambda kv: -len(kv[0]))
)

_PARTY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:republicans?|gop)\b", re.IGNORECASE), "R"),
    # "Democratic Republic of the Congo" is a country, not a party.
    (re.compile(r"\b(?:democrats?|democratic(?!\s+republic)|dems?)\b", re.IGNORECASE), "D"),
    (re.compile(r"\blibertarians?\b", re.IGNORECASE), "L"),
)

#: Capitalized words that are never a distinguishing proper name.
_NON_NAME_WORDS: frozenset[str] = frozenset(
    {
        "will", "who", "what", "which", "when", "how", "the", "senate", "house", "race",
        "election", "elections", "president", "presidential", "governor", "primary",
        "general", "special", "midterm", "midterms", "congress", "seat", "week", "game",
        "season", "series", "championship", "cup", "super", "bowl", "world", "finals",
        "win", "wins", "lose", "loses", "beat", "beats", "make", "makes", "get", "say",
        "yes", "no", "party", "control", "majority", "us", "u", "s", "united", "states",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    }
)

#: Tokens that carry no discriminating meaning for subject/entity comparison.
_STOPWORDS: frozenset[str] = frozenset(
    ["will", "be", "the", "a", "an", "at", "on", "in", "of", "to", "for", "and", "or", "is", "are", "does", "do", "than", "more", "than", "by", "end", "before", "after", "any", "point", "close", "closing", "high", "highest", "lowest", "low", "official", "result", "reaches", "reach", "above", "below", "over", "under", "this", "that"]
)

_MONTHS: dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

#: Standard-time (non-DST) UTC offsets for the four contiguous US zones.
_US_TZ_STD_OFFSET: dict[str, int] = {"ET": -5, "CT": -6, "MT": -7, "PT": -8}
#: Daylight-time UTC offsets for the same zones.
_US_TZ_DST_OFFSET: dict[str, int] = {"ET": -4, "CT": -5, "MT": -6, "PT": -7}
#: Explicit EST/EDT-style abbreviations resolve to a fixed offset regardless of date.
_US_TZ_EXPLICIT_OFFSET: dict[str, int] = {
    "EST": -5, "EDT": -4,
    "CST": -6, "CDT": -5,
    "MST": -7, "MDT": -6,
    "PST": -8, "PDT": -7,
}

_DATE_RE = re.compile(
    r"\b(?P<month>[A-Za-z]{3,9})\.?\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s*(?P<year>\d{4})?\b"
)
_TIME_RE = re.compile(
    r"\b(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>[APap]\.?[Mm]\.?)"
    r"\s*(?P<tz>E[SD]?T|C[SD]?T|M[SD]?T|P[SD]?T|UTC|GMT)?\b"
)
_QUARTER_RE = re.compile(r"\bQ([1-4])\s+(\d{4})\b", re.IGNORECASE)

_CURRENCY_RE = re.compile(
    r"\$?\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<mag>[kKmMbB])?\b"
)
_PERCENT_RE = re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*%")
_BP_RE = re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*(?:bp|bps|basis points?)\b", re.IGNORECASE)
_TEMP_RE = re.compile(r"(?P<num>-?\d+(?:\.\d+)?)\s*(?:°\s*F|deg(?:rees)?\s*F?|F\b)", re.IGNORECASE)
_STATION_RE = re.compile(r"\b[KP][A-Z]{3}\b")

#: Known non-station resolution-authority keywords, checked case-insensitively.
_AUTHORITY_KEYWORDS: tuple[str, ...] = (
    "Coinbase", "Binance", "Kraken", "Bitstamp",
    "BLS", "NWS", "FOMC", "Fed", "NFL", "NBA", "MLB", "NHL",
    "Associated Press", "AP", "CDC", "FDA", "SEC", "Treasury",
)

_BARRIER_KEYWORDS = ("before", "any point", "any time", "touches", "touch", "ever reaches", "at any moment")

_COMPARATOR_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\bat least\b|\bor (?:more|above|higher)\b|\b>=\b"), ">=", "above"),
    (re.compile(r"\bat most\b|\bor (?:less|below|lower)\b|\b<=\b"), "<=", "below"),
    (re.compile(r"\babove\b|\bexceeds?\b|\bmore than\b|\bgreater than\b|\bover\b|\breach(?:es)?\b"), ">", "above"),
    (re.compile(r"\bbelow\b|\bless than\b|\bunder\b|\bfewer than\b"), "<", "below"),
    (re.compile(r"\bexactly\b|\bequal to\b|\b==\b"), "==", "equals"),
)

_SPREAD_RE = re.compile(r"\bwin(?:s)? by more than\s+(?P<num>\d+(?:\.\d+)?)", re.IGNORECASE)
_WINS_RE = re.compile(r"\bwins?\b", re.IGNORECASE)
_LOSES_RE = re.compile(r"\bloses?\b|\bdefeated\b|\bloss\b", re.IGNORECASE)

_INCLUSION_EXCLUSION_KEYWORDS = (
    "excluding", "not including", "only if", "unless", "provided that",
)
_VOID_KEYWORDS = ("void", "voided", "no contest", "postponed and voided")
_RESOLVES_NO_KEYWORDS = ("resolves no", "resolve no", "settles no")
_SETTLEMENT_TIMING_RE = re.compile(
    r"\b(t\+\d+|same[- ]day|within\s+\d+\s*(?:hours?|business days?|days?))\b", re.IGNORECASE
)


def _tokenize(text: str) -> frozenset[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 1)


def _extract_entities(text: str) -> frozenset[str]:
    found: set[str] = set()
    lowered = text.lower()
    for alias, canonical in _ENTITY_ALIASES.items():
        # Word-boundary containment check; alias keys are already lowercase phrases.
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            found.add(canonical)
    # A bare 3-4 letter uppercase ticker not covered by the alias table (e.g. "SPY").
    _non_entity_tokens = {
        "YES", "NO", "ET", "CT", "MT", "PT", "UTC", "GMT", "USD",
        "AM", "PM", "EST", "EDT", "CST", "CDT", "MST", "MDT", "PST", "PDT",
        "NFL", "NBA", "MLB", "NHL", "CPI", "BLS", "GDP", "SEC", "FDA", "CDC",
        "NWS", "FOMC", "AP", "Q1", "Q2", "Q3", "Q4",
        # Index/oracle and settlement-source acronyms. These name the *authority* that
        # resolves a market, not its subject, and they are already compared separately
        # via `resolution_authority` (where a Coinbase-vs-Binance oracle difference is
        # correctly disqualifying). Left in the subject set they poisoned every
        # cross-venue comparison: Kalshi states its index in the rules text ("BRTI",
        # "CF Benchmarks") and Polymarket does not, so all 112 real candidate pairs were
        # rejected for "subject entities differ" before any genuine comparison happened.
        "BRTI", "BRR", "CF", "RTI", "CME", "ICE", "NYSE", "CBOE", "LSE",
        "TBD", "TBA", "N/A", "ID", "II", "III", "IV", "VS", "OU", "AND", "OR", "THE",
    }
    for m in re.finditer(r"\b[A-Z]{2,5}\b", text):
        token = m.group(0)
        # A 4-letter station code (KNYC, KLGA, ...) is resolution-authority information,
        # not a subject entity -- it is surfaced separately via resolution_authority so
        # it does not also masquerade as a subject mismatch.
        if token in _non_entity_tokens or _STATION_RE.fullmatch(token):
            continue
        found.add(token)
    return frozenset(found)


def _extract_locations(text: str) -> frozenset[str]:
    # Punctuation folds to spaces so "Washington, D.C." and "washington dc" both hit.
    remaining = f" {re.sub(r'[^a-z0-9]+', ' ', text.lower())} "
    found: set[str] = set()
    for phrase, code in _LOCATION_PHRASES:
        pattern = rf"\b{re.escape(phrase)}\b"
        if re.search(pattern, remaining):
            found.add(code)
            # Consume the span so "kansas city" does not also count as Kansas.
            remaining = re.sub(pattern, " ", remaining)
    return frozenset(found)


def _extract_parties(text: str) -> frozenset[str]:
    return frozenset(code for pattern, code in _PARTY_PATTERNS if pattern.search(text))


_LOCATION_WORDS: frozenset[str] = frozenset(
    w for phrase, _code in _LOCATION_PHRASES for w in phrase.split()
)
_PARTY_WORDS: frozenset[str] = frozenset(
    {"republican", "republicans", "gop", "democrat", "democrats", "democratic", "dem", "dems",
     "libertarian", "libertarians"}
)


def _extract_named_entities(title: str) -> frozenset[str]:
    """Capitalized proper names from a sentence-case title ("Trump", "Ossoff", "Lakers").

    A title-cased title ("Will The Democrats Win The Michigan Senate Race?") capitalizes
    every word, so its capitals say nothing; it yields the empty set (= no signal), never
    a guess. States, cities and parties are excluded here -- they have their own fields.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", title)
    content = [w for w in words[1:] if len(w) >= 4 and w.lower() not in _STOPWORDS]
    if content and all(w[0].isupper() for w in content):
        return frozenset()  # title case: not one ordinary word was left lowercase
    found: set[str] = set()
    for w in words:
        if not (w[0].isupper() and any(c.islower() for c in w)):
            continue  # lowercase word, or an all-caps acronym (handled by _extract_entities)
        token = w.lower().removesuffix("'s")
        if (
            len(token) < 3
            or token in _STOPWORDS
            or token in _NON_NAME_WORDS
            or token in _MONTHS
            or token in _LOCATION_WORDS
            or token in _PARTY_WORDS
            or token in _ENTITY_ALIASES  # already compared via `entities`
        ):
            continue
        found.add(token)
    return frozenset(found)


def _is_us_dst(d: date) -> bool:
    """US DST: 2nd Sunday of March through 1st Sunday of November (post-2007 rule)."""

    def _nth_sunday(year: int, month: int, n: int) -> date:
        d0 = date(year, month, 1)
        first_sunday = d0 + timedelta(days=(6 - d0.weekday()) % 7)
        return first_sunday + timedelta(weeks=n - 1)

    start = _nth_sunday(d.year, 3, 2)
    end = _nth_sunday(d.year, 11, 1)
    return start <= d < end


def _resolve_tz_offset_hours(abbrev: str, on_date: date) -> int:
    abbrev = abbrev.upper()
    if abbrev in ("UTC", "GMT"):
        return 0
    if abbrev in _US_TZ_EXPLICIT_OFFSET:
        return _US_TZ_EXPLICIT_OFFSET[abbrev]
    if abbrev in _US_TZ_STD_OFFSET:
        return _US_TZ_DST_OFFSET[abbrev] if _is_us_dst(on_date) else _US_TZ_STD_OFFSET[abbrev]
    return 0


def _parse_quarter_end(text: str) -> date | None:
    m = _QUARTER_RE.search(text)
    if not m:
        return None
    quarter, year = int(m.group(1)), int(m.group(2))
    end_month = quarter * 3
    end_day = 31 if end_month in (3, 12) else 30
    return date(year, end_month, end_day)


def _parse_time_window(
    text: str, anchor: datetime | None
) -> tuple[datetime | None, str]:
    """Parse an explicit "<time> <tz>" (+ optional date) mention out of free text.

    Falls back to ``anchor`` (typically the market's structured ``close_time``) for the
    calendar date, and to the anchor's own timezone label when no explicit tz token is
    present in the text. Returns ``(None, "UTC")`` when nothing at all can be resolved.
    """
    time_match = _TIME_RE.search(text)
    q_end = _parse_quarter_end(text)

    if time_match is None and q_end is None:
        return None, "UTC"

    if q_end is not None and time_match is None:
        dt = datetime(q_end.year, q_end.month, q_end.day, 23, 59, tzinfo=UTC)
        return dt, "UTC"

    assert time_match is not None
    hour = int(time_match.group("hour"))
    minute = int(time_match.group("minute") or 0)
    ampm = time_match.group("ampm").lower().replace(".", "")
    if ampm.startswith("p") and hour < 12:
        hour += 12
    if ampm.startswith("a") and hour == 12:
        hour = 0

    tz_token = time_match.group("tz")
    month_day_year: tuple[int, int, int] | None = None
    date_match = _DATE_RE.search(text)
    if date_match is not None:
        month = _MONTHS.get(date_match.group("month").lower())
        if month is not None:
            day = int(date_match.group("day"))
            year_str = date_match.group("year")
            if year_str:
                month_day_year = (int(year_str), month, day)
            elif anchor is not None:
                # No year in the text -- an explicit year always wins, but a bare
                # "Sept 4" needs some reference point, and this module never calls
                # datetime.now() to supply one.
                month_day_year = (anchor.year, month, day)
            # else: no year in the text and no anchor to infer one from -- give up on
            # this date rather than guess.

    if month_day_year is None:
        if anchor is None:
            return None, (tz_token or "UTC").upper()
        month_day_year = (anchor.year, anchor.month, anchor.day)

    year, month, day = month_day_year
    tz_label = tz_token or "ET"
    if not tz_token and anchor is not None:
        anchor_tz_name = getattr(anchor.tzinfo, "_name", None)
        if isinstance(anchor_tz_name, str) and anchor_tz_name:
            tz_label = anchor_tz_name
    tz_label = tz_label.upper()
    offset_hours = _resolve_tz_offset_hours(tz_label, date(year, month, day))
    try:
        local_dt = datetime(year, month, day, hour, minute, tzinfo=_tz(timedelta(hours=offset_hours)))
    except ValueError:
        # Malformed text ("13PM", "Feb 30"): an unparseable time is "unknown", never a
        # crash. One such title used to abort the entire matching pass every cycle.
        return None, tz_label
    return local_dt.astimezone(UTC), tz_label


def _parse_amount(text: str) -> tuple[Decimal | None, str | None]:
    """Currency / percent / basis-point / temperature amount, in that priority order."""
    m = _PERCENT_RE.search(text)
    if m:
        try:
            return Decimal(m.group("num")), "percent"
        except InvalidOperation:
            pass
    m = _BP_RE.search(text)
    if m:
        try:
            return Decimal(m.group("num")), "bp"
        except InvalidOperation:
            pass
    m = _TEMP_RE.search(text)
    if m:
        try:
            return Decimal(m.group("num")), "degF"
        except InvalidOperation:
            pass
    # Currency needs a $ sign or a k/m/b magnitude suffix to avoid false positives on
    # bare integers (e.g. a year, a point spread already handled separately).
    for m in _CURRENCY_RE.finditer(text):
        raw = text[max(0, m.start() - 1): m.start()]
        has_dollar = "$" in m.group(0) or (raw == "$")
        mag = m.group("mag")
        if not has_dollar and not mag:
            continue
        num_str = m.group("num").replace(",", "")
        try:
            value = Decimal(num_str)
        except InvalidOperation:
            continue
        if mag:
            multiplier = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}[mag.lower()]
            value *= multiplier
        return value, "USD"
    return None, None


def _extract_comparator_outcome(text: str) -> tuple[str | None, str | None]:
    for pattern, comparator, outcome in _COMPARATOR_PATTERNS:
        if pattern.search(text):
            return comparator, outcome
    return None, None


def _extract_measurement(text: str, comparator: str | None) -> str | None:
    lowered = text.lower()
    if any(k in lowered for k in _BARRIER_KEYWORDS):
        return "barrier_touch"
    if "high" in lowered or "highest" in lowered:
        return "high"
    if any(word in lowered for word in ("cpi", "jobs", "payroll", "gdp", "unemployment", "jolts")):
        return "official_result"
    if comparator is not None and ("at " in lowered or "close" in lowered or lowered.strip().endswith("?") is False):
        return "terminal"
    if comparator is not None:
        return "terminal"
    return None


def _extract_resolution_authority(*texts: str) -> str | None:
    combined = " ".join(t for t in texts if t)
    station = _STATION_RE.search(combined)
    if station:
        return station.group(0)
    for keyword in _AUTHORITY_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", combined, re.IGNORECASE):
            return keyword
    return None


def _extract_inclusion_exclusion(text: str) -> frozenset[str]:
    lowered = text.lower()
    return frozenset(kw for kw in _INCLUSION_EXCLUSION_KEYWORDS if kw in lowered)


def _extract_postponement_rule(text: str) -> str | None:
    lowered = text.lower()
    if any(kw in lowered for kw in _VOID_KEYWORDS):
        return "void_on_postponement"
    if any(kw in lowered for kw in _RESOLVES_NO_KEYWORDS):
        return "resolves_no_on_postponement"
    if "postpone" in lowered:
        return "postponement_mentioned_unspecified"
    return None


def _extract_settlement_timing(text: str) -> str | None:
    m = _SETTLEMENT_TIMING_RE.search(text)
    return m.group(0).lower() if m else None


def extract_claim_from_text(
    title: str,
    rules_text: str = "",
    *,
    resolution_source: str = "",
    anchor_time: datetime | None = None,
) -> MarketClaim:
    """Extract a :class:`MarketClaim` from raw text plus an optional structural anchor.

    ``anchor_time`` should be the market's own ``close_time`` (or similar) when
    available -- it grounds a bare "5 PM ET" mention to a calendar date and disambiguates
    an omitted year, without this module ever calling ``datetime.now()``.
    """
    combined = f"{title}\n{rules_text}"

    comparator, outcome = _extract_comparator_outcome(combined)

    spread_match = _SPREAD_RE.search(combined)
    if spread_match:
        comparator, outcome = ">", "wins_by_more_than"
        threshold = Decimal(spread_match.group("num"))
        threshold_unit: str | None = "points"
    elif _LOSES_RE.search(combined) and not _WINS_RE.search(combined):
        outcome, comparator = "loses", None
        threshold, threshold_unit = None, None
    elif _WINS_RE.search(combined) and comparator is None:
        outcome, comparator = "wins", None
        threshold, threshold_unit = None, None
    else:
        threshold, threshold_unit = _parse_amount(combined)

    measurement = _extract_measurement(combined, comparator)

    time_end, tz_label = _parse_time_window(combined, anchor_time)
    if time_end is None:
        time_end = anchor_time

    resolution_authority = resolution_source.strip() or _extract_resolution_authority(combined)

    entities = _extract_entities(combined)
    subject_tokens = _tokenize(title)
    subject = " ".join(sorted(entities)) or title.strip().lower()
    # Location and party come from the title when it names one; rules text is only a
    # fallback, because it routinely mentions the other party or a certifying state in
    # passing ("...if a candidate who caucuses with Republicans...").
    locations = _extract_locations(title) or _extract_locations(rules_text)
    parties = _extract_parties(title) or _extract_parties(rules_text)

    return MarketClaim(
        subject=subject,
        subject_tokens=subject_tokens,
        outcome=outcome,
        comparator=comparator,
        threshold=threshold,
        threshold_unit=threshold_unit,
        measurement=measurement,
        time_window_start=None,
        time_window_end=time_end,
        timezone=tz_label,
        resolution_authority=resolution_authority,
        entities=entities,
        postponement_rule=_extract_postponement_rule(combined),
        inclusion_exclusion=_extract_inclusion_exclusion(combined),
        settlement_timing=_extract_settlement_timing(combined),
        raw_title=title,
        locations=locations,
        parties=parties,
        named_entities=_extract_named_entities(title),
    )


def extract_claim(market: NormalizedMarket) -> MarketClaim:
    """Extract a :class:`MarketClaim` from a :class:`NormalizedMarket`.

    Uses the market's structured ``resolution_source`` and ``close_time`` as authoritative
    grounding, falling back to text parsing only where structure is silent.
    """
    return extract_claim_from_text(
        market.title,
        market.resolution_rules,
        resolution_source=market.resolution_source,
        anchor_time=market.close_time,
    )
