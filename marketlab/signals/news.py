"""News dedup, source weighting, and entity/ticker extraction.

GDELT (and any wire-following aggregator) will hand us the same Reuters story ten times
under ten different domains within a minute of publication.  Counting that as ten
independent confirmations would badly overweight one underlying fact, so everything here
exists to collapse republications to a single canonical story while still recording *that*
duplication happened (via ``duplicate_of``), never dropping the record outright.

Nothing here calls the network or the clock; it operates purely on already-ingested
``NewsEvent`` objects.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from marketlab.core.events import NewsEvent, SourceClass

# ---------------------------------------------------------------------------
# Dedup thresholds. Tuned to be generous about collapsing wire copies (which differ
# only in boilerplate/ads/tracking params) while never merging genuinely different
# stories about the same entity.
# ---------------------------------------------------------------------------
TITLE_SIM_THRESHOLD = 0.85
SHINGLE_SIM_THRESHOLD = 0.60
_SHINGLE_K = 4
_MINHASH_HASHES = 24

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "ref", "referrer", "cmpid", "ito", "amp", "outputType",
}

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def canonicalize_url(url: str) -> str:
    """Strip scheme noise, `www.`, trailing slash, and tracking query params.

    Two syndication copies of the same wire story are frequently byte-identical apart
    from a tracking query string; stripping it is the cheapest, highest-precision dedup
    signal available and is tried first.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip().lower()
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    query_pairs = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ]
    query_pairs.sort()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(("https", netloc, path, urlencode(query_pairs), ""))


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for similarity comparison."""
    t = title.lower().strip()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def token_set_ratio(a: str, b: str) -> float:
    """A dependency-free approximation of fuzzywuzzy's ``token_set_ratio``.

    Splits both strings into token sets, then compares the intersection against each
    side's remainder using a longest-common-subsequence ratio (via ``difflib``), and
    returns the best of the three comparisons. This is robust to reordered words and
    to one copy carrying an extra clause ("BREAKING: ...") that the other lacks.
    """
    import difflib

    tokens_a = set(_tokenize(a))
    tokens_b = set(_tokenize(b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    only_a = tokens_a - tokens_b
    only_b = tokens_b - tokens_a

    sorted_intersection = " ".join(sorted(intersection))
    combo_a = " ".join(sorted(intersection | only_a))
    combo_b = " ".join(sorted(intersection | only_b))

    def ratio(x: str, y: str) -> float:
        return difflib.SequenceMatcher(None, x, y).ratio()

    return max(
        ratio(sorted_intersection, combo_a),
        ratio(sorted_intersection, combo_b),
        ratio(combo_a, combo_b),
    )


def _shingles(text: str, k: int = _SHINGLE_K) -> set[str]:
    tokens = _tokenize(text)
    if not tokens:
        return set()
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def text_shingles(text: str, k: int = _SHINGLE_K) -> set[str]:
    """Public wrapper around the word-shingle extraction used for near-dup detection."""
    return _shingles(text, k)


def _stable_hash(value: str, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def minhash_signature(shingles: set[str], num_hashes: int = _MINHASH_HASHES) -> tuple[int, ...]:
    """A small MinHash signature for cheap near-duplicate detection on body text."""
    if not shingles:
        return tuple(0 for _ in range(num_hashes))
    return tuple(min(_stable_hash(s, seed) for s in shingles) for seed in range(num_hashes))


def minhash_similarity(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for x, y in zip(sig_a, sig_b, strict=True) if x == y)
    return matches / len(sig_a)


def body_hash(text: str) -> str:
    normalized = _WS_RE.sub(" ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def dedupe(events: list[NewsEvent]) -> list[NewsEvent]:
    """Collapse republications to one canonical story.

    Order of dedup signals, cheapest/highest-precision first:
      1. canonical URL match
      2. body hash match
      3. normalized-title token-set-ratio similarity
      4. shingle/MinHash near-duplicate similarity on title+body

    Duplicates are **never dropped** - they come back in the same order as the input,
    with ``duplicate_of`` set to the ``news_id`` of the first-seen original. This keeps
    "ten wire copies" collapsible to one source for counting purposes
    (``{e.duplicate_of or e.news_id for e in events}`` has size 1) while preserving every
    record for audit.
    """
    # Process in first-seen order so "the original" is always the earliest-observed copy,
    # never a later republication - but return results in the caller's original order.
    order = sorted(range(len(events)), key=lambda i: events[i].first_seen_time)

    canonical_index: dict[str, NewsEvent] = {}
    hash_index: dict[str, NewsEvent] = {}
    kept: list[NewsEvent] = []
    kept_sigs: list[tuple[int, ...]] = []
    dup_of_by_index: dict[int, str | None] = {}

    for i in order:
        event = events[i]
        dup_of: str | None = None

        curl = event.canonical_url or canonicalize_url(event.url)
        if curl and curl in canonical_index:
            dup_of = canonical_index[curl].news_id

        if dup_of is None and event.body_hash and event.body_hash in hash_index:
            dup_of = hash_index[event.body_hash].news_id

        if dup_of is None:
            computed_hash = body_hash(event.body) if event.body else ""
            if computed_hash and computed_hash in hash_index:
                dup_of = hash_index[computed_hash].news_id

        if dup_of is None and event.title:
            norm = normalize_title(event.title)
            for other in kept:
                if other.title and token_set_ratio(norm, normalize_title(other.title)) >= TITLE_SIM_THRESHOLD:
                    dup_of = other.news_id
                    break

        sig: tuple[int, ...] = ()
        if dup_of is None:
            text = f"{event.title} {event.body}".strip()
            sig = minhash_signature(_shingles(text))
            for other, other_sig in zip(kept, kept_sigs, strict=True):
                if minhash_similarity(sig, other_sig) >= SHINGLE_SIM_THRESHOLD:
                    dup_of = other.news_id
                    break

        dup_of_by_index[i] = dup_of
        if dup_of is None:
            kept.append(event)
            kept_sigs.append(sig or minhash_signature(_shingles(f"{event.title} {event.body}")))
            if curl:
                canonical_index[curl] = event
            h = event.body_hash or (body_hash(event.body) if event.body else "")
            if h:
                hash_index[h] = event

    result: list[NewsEvent] = []
    for i, event in enumerate(events):
        dup_of = dup_of_by_index[i]
        if dup_of is None:
            result.append(event)
        else:
            result.append(event.model_copy(update={"duplicate_of": dup_of}))
    return result


def effective_source_count(events: list[NewsEvent]) -> int:
    """Number of distinct underlying stories after dedup - what should drive confidence."""
    deduped = dedupe(events)
    ids = {e.duplicate_of or e.news_id for e in deduped}
    return len(ids)


# ---------------------------------------------------------------------------
# Source classification
# ---------------------------------------------------------------------------

_OFFICIAL_SUFFIXES = (".gov", "nfl.com", "nba.com", "mlb.com", "nhl.com", "fifa.com", "sec.gov")
_SOURCE_TABLE: dict[str, SourceClass] = {
    "reuters.com": SourceClass.MAJOR_WIRE,
    "apnews.com": SourceClass.MAJOR_WIRE,
    "ap.org": SourceClass.MAJOR_WIRE,
    "bloomberg.com": SourceClass.MAJOR_WIRE,
    "afp.com": SourceClass.MAJOR_WIRE,
    "nytimes.com": SourceClass.MAJOR_NEWS,
    "wsj.com": SourceClass.MAJOR_NEWS,
    "washingtonpost.com": SourceClass.MAJOR_NEWS,
    "cnn.com": SourceClass.MAJOR_NEWS,
    "foxnews.com": SourceClass.MAJOR_NEWS,
    "bbc.com": SourceClass.MAJOR_NEWS,
    "bbc.co.uk": SourceClass.MAJOR_NEWS,
    "npr.org": SourceClass.MAJOR_NEWS,
    "cnbc.com": SourceClass.MAJOR_NEWS,
    "abcnews.go.com": SourceClass.MAJOR_NEWS,
    "cbsnews.com": SourceClass.MAJOR_NEWS,
    "nbcnews.com": SourceClass.MAJOR_NEWS,
    "politico.com": SourceClass.MAJOR_NEWS,
    "axios.com": SourceClass.MAJOR_NEWS,
    "theguardian.com": SourceClass.MAJOR_NEWS,
    "coindesk.com": SourceClass.SPECIALIST,
    "theblock.co": SourceClass.SPECIALIST,
    "techcrunch.com": SourceClass.SPECIALIST,
    "espn.com": SourceClass.SPECIALIST,
    "cointelegraph.com": SourceClass.SPECIALIST,
    "marketwatch.com": SourceClass.SPECIALIST,
    "seekingalpha.com": SourceClass.SPECIALIST,
    "twitter.com": SourceClass.SOCIAL_UNVERIFIED,
    "x.com": SourceClass.SOCIAL_UNVERIFIED,
    "bsky.app": SourceClass.SOCIAL_UNVERIFIED,
}


def classify_source(domain: str) -> SourceClass:
    """Map a publisher domain to its evidentiary weight class.

    An explicit table wins; any ``*.gov`` (or listed official league) domain is
    ``OFFICIAL_PRIMARY`` regardless of table presence. Unknown domains abstain to
    ``UNKNOWN`` rather than guessing.
    """
    if not domain:
        return SourceClass.UNKNOWN
    d = domain.lower().strip()
    if d.startswith("www."):
        d = d[4:]
    if d.endswith(".gov") or d in {"sec.gov", "data.sec.gov", "whitehouse.gov"}:
        return SourceClass.OFFICIAL_PRIMARY
    for suffix in _OFFICIAL_SUFFIXES:
        if d == suffix or d.endswith("." + suffix.lstrip(".")):
            return SourceClass.OFFICIAL_PRIMARY
    if d in _SOURCE_TABLE:
        return _SOURCE_TABLE[d]
    # Strip a leading subdomain once (e.g. "www2.reuters.com", "amp.reuters.com").
    parts = d.split(".")
    if len(parts) > 2:
        base = ".".join(parts[-2:])
        if base in _SOURCE_TABLE:
            return _SOURCE_TABLE[base]
    return SourceClass.UNKNOWN


# ---------------------------------------------------------------------------
# Entity / ticker extraction with deliberate abstention on ambiguous terms
# ---------------------------------------------------------------------------

#: Bare uppercase tokens that look like tickers but are common words/acronyms.
#: These must never be extracted as tickers from bare text - only an explicit
#: ``$TICKER`` cashtag can confirm intent for these.
_AMBIGUOUS_BARE_TOKENS = {
    "ALL", "IT", "ON", "FOR", "ARE", "CEO", "CFO", "GDP", "USA", "NEW", "ONE",
    "TWO", "SIX", "SEE", "NOW", "WAS", "HAS", "CAN", "GET", "TOP", "BIG", "OWN",
    "PAY", "PUT", "RAN", "RUN", "SIT", "TRY", "USE", "WIN", "YES", "YOU", "ITS",
    "AI", "US", "UK", "EU", "A", "I",
}

#: Company names that are also common English words / generic terms. A bare mention
#: must not be treated as a ticker reference for these - require the full legal name
#: (with a corporate suffix) or an explicit cashtag.
_AMBIGUOUS_ENTITY_NAMES = {
    "apple", "target", "gap", "match", "chart", "block", "square", "live",
    "real", "first", "best", "under", "above", "shop", "spark", "root",
    "chime", "on", "figs", "duck", "workday",
}

_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})\b")
_BARE_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
_CORP_SUFFIX_RE = re.compile(
    r"\s+(inc\.?|incorporated|corp\.?|corporation|co\.?|company|ltd\.?|llc|plc|holdings|group)\s*$",
    re.IGNORECASE,
)
_CAPWORD_RE = re.compile(r"\b([A-Z][a-zA-Z&]+(?:\s+[A-Z][a-zA-Z&]+){0,3})\b")
_STOP_ENTITY_WORDS = {
    "The", "A", "An", "In", "On", "At", "By", "For", "With", "This", "That",
    "It", "Its", "They", "He", "She", "We", "You", "I",
}


def _strip_corp_suffix(name: str) -> str:
    return _CORP_SUFFIX_RE.sub("", name).strip()


def build_ticker_map(company_tickers: dict[str, dict[str, object]] | dict[str, object]) -> dict[str, str]:
    """Build ``ticker -> company name`` from SEC's ``company_tickers.json`` shape.

    Accepts the raw payload (``{"0": {"cik_str":.., "ticker":.., "title":..}, ...}``).
    """
    out: dict[str, str] = {}
    values = company_tickers.values() if isinstance(company_tickers, dict) else company_tickers
    for row in values:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker", "")).upper().strip()
        title = str(row.get("title", "")).strip()
        if ticker and title:
            out[ticker] = title
    return out


def extract_tickers(text: str, ticker_map: dict[str, str] | None = None) -> tuple[str, ...]:
    """Extract stock tickers, abstaining on ambiguous bare mentions.

    - ``$TICKER`` cashtags are always trusted (explicit syntax, even single letter).
    - Bare uppercase tokens are only accepted if 2-5 letters, present in
      ``ticker_map``, and not a known ambiguous acronym/common word.
    - Single-letter bare tickers always abstain (too many false positives: "I", "A").
    - Company names are matched against ``ticker_map`` values, but ambiguous
      generic-word company names require the full legal name (with a corporate
      suffix) to appear, not a bare mention.
    """
    if not text:
        return ()
    ticker_map = ticker_map or {}
    found: set[str] = set()

    for m in _CASHTAG_RE.finditer(text):
        found.add(m.group(1).upper())

    for m in _BARE_TICKER_RE.finditer(text):
        token = m.group(0)
        if token in _AMBIGUOUS_BARE_TOKENS:
            continue
        if token in ticker_map:
            found.add(token)

    name_to_ticker: dict[str, str] = {}
    for ticker, title in ticker_map.items():
        stripped = _strip_corp_suffix(title).lower()
        if stripped:
            name_to_ticker.setdefault(stripped, ticker)

    for m in _CAPWORD_RE.finditer(text):
        phrase = m.group(1)
        if phrase in _STOP_ENTITY_WORDS:
            continue
        stripped = _strip_corp_suffix(phrase)
        phrase_key = phrase.lower()
        stripped_key = stripped.lower()
        suffix_present = stripped_key != phrase_key

        matched_ticker = name_to_ticker.get(phrase_key) or name_to_ticker.get(stripped_key)
        if not matched_ticker:
            continue
        # Ambiguous generic-word company names (Apple, Gap, ...) only count when the
        # matched phrase itself carried a corporate suffix - a bare "Apple" abstains.
        if stripped_key in _AMBIGUOUS_ENTITY_NAMES and not suffix_present:
            continue
        found.add(matched_ticker)

    return tuple(sorted(found))


def extract_entities(text: str) -> tuple[str, ...]:
    """Heuristic proper-noun phrase extraction (capitalized word runs).

    Deliberately conservative: single common capitalized words at sentence starts are
    filtered via ``_STOP_ENTITY_WORDS``, but this is not a full NER model - it's a cheap
    first pass good enough to seed ticker matching and novelty comparisons.
    """
    if not text:
        return ()
    found: list[str] = []
    seen: set[str] = set()
    for m in _CAPWORD_RE.finditer(text):
        phrase = m.group(1).strip()
        if phrase in _STOP_ENTITY_WORDS or len(phrase) < 2:
            continue
        if phrase not in seen:
            seen.add(phrase)
            found.append(phrase)
    return tuple(found)


def novelty(event: NewsEvent, recent_events: list[NewsEvent]) -> float:
    """How much new information ``event`` carries relative to ``recent_events``.

    1.0 = nothing similar seen recently, 0.0 = an exact republication. Uses the same
    MinHash near-duplicate machinery as :func:`dedupe`, but returns a continuous score
    rather than a binary duplicate decision, since a strategy or the AI analyst may want
    to weight "mostly the same story, one new paragraph" partway between the extremes.
    """
    if not recent_events:
        return 1.0
    text = f"{event.title} {event.body}".strip()
    sig = minhash_signature(_shingles(text))
    max_sim = 0.0
    for other in recent_events:
        if other.news_id == event.news_id:
            continue
        other_text = f"{other.title} {other.body}".strip()
        other_sig = minhash_signature(_shingles(other_text))
        sim = minhash_similarity(sig, other_sig)
        max_sim = max(max_sim, sim)
    return max(0.0, 1.0 - max_sim)
