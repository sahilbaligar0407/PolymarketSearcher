"""Unit tests for the deterministic validator: ``resolution_rules.compare_claims`` and
``detect_complement``.

The corpus in ``tests/fixtures/match_pairs.py`` is the source of truth for the
accept/reject boundary; this file also probes specific mechanics (timezone
normalization, unit handling, complement detection) directly against
``MarketClaim`` so failures point at the exact field responsible.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from marketlab.matching.extract import extract_claim_from_text
from marketlab.matching.resolution_rules import (
    DEFAULT_TIME_TOLERANCE_SECONDS,
    compare_claims,
    detect_complement,
)
from tests.fixtures.match_pairs import CORPUS


def test_corpus_reproduces_every_label() -> None:
    """The whole corpus is the acceptance test: every label must be reproduced exactly."""
    failures: list[str] = []
    for fx in CORPUS:
        claim_a = extract_claim_from_text(
            fx.market_a.title,
            fx.market_a.resolution_rules,
            resolution_source=fx.market_a.resolution_source,
            anchor_time=fx.market_a.close_time,
        )
        claim_b = extract_claim_from_text(
            fx.market_b.title,
            fx.market_b.resolution_rules,
            resolution_source=fx.market_b.resolution_source,
            anchor_time=fx.market_b.close_time,
        )
        comparison = compare_claims(claim_a, claim_b)
        if comparison.same_outcome_boolean != fx.should_match:
            failures.append(
                f"{fx.name}: same_outcome_boolean={comparison.same_outcome_boolean} "
                f"expected {fx.should_match} ({fx.reason}) diffs={comparison.rule_diff}"
            )
        complement = detect_complement(claim_a, claim_b)
        if complement != fx.expected_complement:
            failures.append(
                f"{fx.name}: detect_complement={complement} expected {fx.expected_complement}"
            )
        if (
            fx.expected_human_review is not None
            and comparison.human_review_required != fx.expected_human_review
        ):
            failures.append(
                f"{fx.name}: human_review_required={comparison.human_review_required} "
                f"expected {fx.expected_human_review}"
            )
    assert not failures, "\n".join(failures)


def test_same_outcome_boolean_false_whenever_any_blocking_reason_present() -> None:
    for fx in CORPUS:
        claim_a = extract_claim_from_text(
            fx.market_a.title, fx.market_a.resolution_rules,
            resolution_source=fx.market_a.resolution_source, anchor_time=fx.market_a.close_time,
        )
        claim_b = extract_claim_from_text(
            fx.market_b.title, fx.market_b.resolution_rules,
            resolution_source=fx.market_b.resolution_source, anchor_time=fx.market_b.close_time,
        )
        comparison = compare_claims(claim_a, claim_b)
        assert comparison.same_outcome_boolean == (len(comparison.blocking_reasons) == 0), fx.name


def test_measurement_mismatch_is_disqualifying_even_with_everything_else_equal() -> None:
    base_kwargs = dict(resolution_source="Coinbase", anchor_time=datetime(2026, 9, 4, 21, 0, tzinfo=UTC))
    terminal = extract_claim_from_text(
        "Will BTC be above $100,000 at 5 PM ET on September 4, 2026?", **base_kwargs
    )
    barrier = extract_claim_from_text(
        "Will BTC reach $100,000 before 5 PM ET on September 4, 2026?", **base_kwargs
    )
    assert terminal.measurement == "terminal"
    assert barrier.measurement == "barrier_touch"
    comparison = compare_claims(terminal, barrier)
    assert comparison.same_outcome_boolean is False
    assert "measurement_mismatch" in comparison.blocking_reasons


def test_timezone_normalization_edt_vs_est_same_instant() -> None:
    # September is US daylight time: "5 PM ET" resolves to EDT (UTC-4).
    summer = extract_claim_from_text(
        "Will BTC be above $100,000 at 5 PM ET on September 4, 2026?",
        resolution_source="Coinbase",
    )
    summer_explicit = extract_claim_from_text(
        "Will BTC be above $100,000 at 5 PM EDT on September 4, 2026?",
        resolution_source="Coinbase",
    )
    assert summer.time_window_end == summer_explicit.time_window_end
    assert summer.time_window_end == datetime(2026, 9, 4, 21, 0, tzinfo=UTC)

    # January is standard time: the same clock-face "5 PM ET" is UTC-5 (EST), a
    # different UTC instant than the summer case above.
    winter = extract_claim_from_text(
        "Will BTC be above $100,000 at 5 PM ET on January 4, 2026?",
        resolution_source="Coinbase",
    )
    winter_explicit = extract_claim_from_text(
        "Will BTC be above $100,000 at 5 PM EST on January 4, 2026?",
        resolution_source="Coinbase",
    )
    assert winter.time_window_end == winter_explicit.time_window_end
    assert winter.time_window_end == datetime(2026, 1, 4, 22, 0, tzinfo=UTC)

    comparison = compare_claims(summer, summer_explicit)
    assert comparison.same_outcome_boolean is True
    assert comparison.time_diff_seconds == 0.0


def test_time_window_tolerance_is_tight_by_default() -> None:
    assert DEFAULT_TIME_TOLERANCE_SECONDS == 60.0
    base = dict(resolution_source="Coinbase")
    a = extract_claim_from_text(
        "Will BTC be above $100,000 at 5:00 PM ET on September 4, 2026?", **base
    )
    b = extract_claim_from_text(
        "Will BTC be above $100,000 at 5:02 PM ET on September 4, 2026?", **base
    )
    comparison = compare_claims(a, b)
    assert comparison.time_diff_seconds == 120.0
    assert comparison.same_outcome_boolean is False
    assert "time_window_mismatch" in comparison.blocking_reasons


def test_detect_complement_requires_matching_authority_and_time() -> None:
    a = extract_claim_from_text(
        "Will the Chiefs win their Week 1 game?", resolution_source="NFL",
        anchor_time=datetime(2026, 9, 8, 20, 0, tzinfo=UTC),
    )
    b_same_time = extract_claim_from_text(
        "Will the Chiefs lose their Week 1 game?", resolution_source="NFL",
        anchor_time=datetime(2026, 9, 8, 20, 0, tzinfo=UTC),
    )
    assert detect_complement(a, b_same_time) is True

    b_different_time = extract_claim_from_text(
        "Will the Chiefs lose their Week 1 game?", resolution_source="NFL",
        anchor_time=datetime(2026, 9, 8, 23, 0, tzinfo=UTC),
    )
    assert detect_complement(a, b_different_time) is False

    b_different_authority = extract_claim_from_text(
        "Will the Chiefs lose their Week 1 game?", resolution_source="ESPN",
        anchor_time=datetime(2026, 9, 8, 20, 0, tzinfo=UTC),
    )
    assert detect_complement(a, b_different_authority) is False


def test_comparator_mismatch_is_low_severity_but_still_blocking() -> None:
    base = dict(resolution_source="Coinbase")
    strict = extract_claim_from_text(
        "Will BTC be above $100,000 at 5 PM ET on September 4, 2026?", **base
    )
    inclusive = extract_claim_from_text(
        "Will BTC be at least $100,000 at 5 PM ET on September 4, 2026?", **base
    )
    comparison = compare_claims(strict, inclusive)
    assert comparison.same_outcome_boolean is False
    assert comparison.human_review_required is True
    assert comparison.confidence > Decimal("0.5"), "low-severity diff should not tank confidence like a real mismatch"
