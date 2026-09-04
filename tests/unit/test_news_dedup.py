from __future__ import annotations

from datetime import UTC, datetime, timedelta

from marketlab.core.events import NewsEvent, SourceClass
from marketlab.signals.news import (
    build_ticker_map,
    classify_source,
    dedupe,
    effective_source_count,
    extract_entities,
    extract_tickers,
    novelty,
    token_set_ratio,
)

T0 = datetime(2026, 9, 4, 12, 0, 0, tzinfo=UTC)

_TICKER_MAP = build_ticker_map(
    {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
        "2": {"cik_str": 37996, "ticker": "F", "title": "FORD MOTOR CO"},
        "3": {"cik_str": 999, "ticker": "GAP", "title": "GAP INC"},
    }
)


def _news(i: int, title: str, body: str = "", url: str | None = None, **kw) -> NewsEvent:
    return NewsEvent(
        news_id=f"n{i}",
        title=title,
        body=body,
        url=url if url is not None else f"https://wire{i}.example.com/story-{i}",
        event_time=T0 + timedelta(seconds=i),
        first_seen_time=T0 + timedelta(seconds=i),
        **kw,
    )


def test_ten_near_identical_wire_copies_collapse_to_one():
    body = (
        "The Federal Reserve held interest rates steady on Wednesday, citing "
        "continued progress on inflation and a resilient labor market."
    )
    events = [
        _news(
            i,
            title=f"Fed holds rates steady{' - Reuters' if i % 2 else ''}",
            body=body + (f" (updated {i})" if i % 3 == 0 else ""),
            url=f"https://news{i}.example.com/fed-holds-rates?utm_source=twitter&id={i}",
        )
        for i in range(10)
    ]
    deduped = dedupe(events)
    assert len(deduped) == 10
    ids = {e.duplicate_of or e.news_id for e in deduped}
    assert len(ids) == 1, f"expected all ten copies to collapse to one source, got {ids}"
    assert effective_source_count(events) == 1

    # Exactly one record has no duplicate_of (the earliest-seen original); the rest point to it.
    originals = [e for e in deduped if e.duplicate_of is None]
    assert len(originals) == 1
    assert originals[0].news_id == "n0"
    for e in deduped:
        if e.news_id != "n0":
            assert e.duplicate_of == "n0"


def test_different_stories_do_not_merge():
    events = [
        _news(0, "Fed holds interest rates steady", "The FOMC voted to hold rates."),
        _news(1, "Apple unveils new iPhone lineup", "Apple announced three new iPhone models today."),
        _news(2, "Hurricane forms in the Atlantic", "The NHC is tracking a new tropical system."),
    ]
    deduped = dedupe(events)
    assert all(e.duplicate_of is None for e in deduped)
    assert effective_source_count(events) == 3


def test_dedupe_preserves_input_order_and_count():
    events = [_news(i, f"Story number {i}", f"Unique body content number {i} about topic {i}.") for i in range(5)]
    deduped = dedupe(events)
    assert [e.news_id for e in deduped] == [f"n{i}" for i in range(5)]


def test_canonical_url_dedup_ignores_tracking_params():
    a = _news(0, "Market rallies on Fed news", url="https://example.com/story?utm_source=x&id=1")
    b = _news(1, "Market rallies on Fed news!", url="https://www.example.com/story?utm_campaign=y&id=1")
    deduped = dedupe([a, b])
    assert deduped[1].duplicate_of == "n0"


def test_token_set_ratio_basic():
    assert token_set_ratio("fed holds rates steady", "fed holds rates steady") == 1.0
    assert token_set_ratio("fed holds rates steady", "hurricane forms in atlantic") < 0.5
    assert token_set_ratio("", "") == 1.0


# ---------------------------------------------------------------------------
# Ticker / entity extraction abstention
# ---------------------------------------------------------------------------


def test_ticker_extraction_finds_explicit_cashtag():
    assert extract_tickers("Shares of $AAPL rose today.", _TICKER_MAP) == ("AAPL",)


def test_ticker_extraction_finds_full_company_name():
    tickers = extract_tickers("NVIDIA CORP reported record revenue.", _TICKER_MAP)
    assert "NVDA" in tickers


def test_ticker_extraction_abstains_on_ambiguous_fruit_apple():
    # "Apple" the fruit / generic use, no corporate suffix, no cashtag -> abstain.
    text = "She picked an apple from the tree and ate it for lunch."
    assert extract_tickers(text, _TICKER_MAP) == ()


def test_ticker_extraction_accepts_apple_with_corporate_suffix():
    text = "Apple Inc. reported quarterly earnings above expectations."
    assert "AAPL" in extract_tickers(text, _TICKER_MAP)


def test_ticker_extraction_abstains_on_single_letter_ticker():
    text = "F is a common grade and also Ford's ticker."
    assert "F" not in extract_tickers(text, _TICKER_MAP)


def test_ticker_extraction_abstains_on_common_word_acronyms():
    text = "ALL of IT depends ON the outcome."
    assert extract_tickers(text, _TICKER_MAP) == ()


def test_ticker_extraction_abstains_on_ambiguous_gap_word():
    text = "There is a big gap between expectations and reality."
    assert extract_tickers(text, _TICKER_MAP) == ()


def test_ticker_extraction_accepts_gap_with_suffix():
    text = "GAP INC posted same-store sales growth."
    assert "GAP" in extract_tickers(text, _TICKER_MAP)


def test_extract_entities_filters_sentence_starters():
    entities = extract_entities("The Federal Reserve raised rates. Apple Inc. responded quickly.")
    assert "The" not in entities
    assert any("Federal Reserve" in e for e in entities)


# ---------------------------------------------------------------------------
# Source classification
# ---------------------------------------------------------------------------


def test_classify_source_table():
    assert classify_source("reuters.com") == SourceClass.MAJOR_WIRE
    assert classify_source("www.reuters.com") == SourceClass.MAJOR_WIRE
    assert classify_source("sec.gov") == SourceClass.OFFICIAL_PRIMARY
    assert classify_source("data.sec.gov") == SourceClass.OFFICIAL_PRIMARY
    assert classify_source("nytimes.com") == SourceClass.MAJOR_NEWS
    assert classify_source("coindesk.com") == SourceClass.SPECIALIST
    assert classify_source("some-random-blog.example") == SourceClass.UNKNOWN
    assert classify_source("") == SourceClass.UNKNOWN
    assert classify_source("whitehouse.gov") == SourceClass.OFFICIAL_PRIMARY


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------


def test_novelty_no_prior_events_is_maximal():
    e = _news(0, "Brand new story", "Something nobody has reported before.")
    assert novelty(e, []) == 1.0


def test_novelty_near_duplicate_is_low():
    a = _news(0, "Fed holds rates steady", "The FOMC voted unanimously to hold rates steady today.")
    b = _news(1, "Fed holds rates steady", "The FOMC voted unanimously to hold rates steady today.")
    assert novelty(b, [a]) < 0.3


def test_novelty_unrelated_story_is_high():
    a = _news(0, "Fed holds rates steady", "The FOMC voted unanimously to hold rates steady today.")
    b = _news(1, "Hurricane forms in Atlantic", "The NHC is tracking a new tropical depression.")
    assert novelty(b, [a]) > 0.7
