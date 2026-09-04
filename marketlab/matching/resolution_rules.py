"""The deterministic validator -- the only gate that may authorize automated trading.

``compare_claims`` takes two :class:`~marketlab.matching.extract.MarketClaim` objects and
returns a :class:`RuleComparison`. Its ``same_outcome_boolean`` field is the single
field anywhere in the matching pipeline that a strategy may treat as "yes, these two
contracts are the identical bet." Every other signal in this package -- Jaccard
similarity, category buckets, an LLM's opinion -- is advisory only and must pass through
this function before it can become an approved match.

Design stance: **conservative by default.** Any field that cannot be confirmed
compatible is treated as a difference, not silently ignored. A missed match costs
nothing; a false match risks real capital on a bet that was never actually the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from marketlab.matching.extract import MarketClaim

#: Two claims resolving within this many seconds of each other are considered the
#: "same" time window. Tight by design -- cross-venue clock skew should never be
#: mistaken for a genuinely different settlement instant.
DEFAULT_TIME_TOLERANCE_SECONDS = 60.0

_LOW_SEVERITY = Decimal("0.15")
_MED_SEVERITY = Decimal("0.50")
_HIGH_SEVERITY = Decimal("0.90")

#: comparator pairs that are the exact logical negation of one another at an identical
#: threshold -- e.g. NOT(x > T) == (x <= T). Used by :func:`detect_complement`.
_NUMERIC_COMPLEMENT_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {(">", "<="), ("<=", ">"), ("<", ">="), (">=", "<")}
)
#: outcome pairs that are the logical negation of one another for a single-entity
#: categorical claim (e.g. a team's own win/lose binary).
_CATEGORICAL_COMPLEMENT_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {("wins", "loses"), ("loses", "wins")}
)


@dataclass(frozen=True)
class RuleComparison:
    """Result of comparing two claims. ``same_outcome_boolean`` is the trading gate."""

    same_outcome_boolean: bool
    confidence: Decimal
    rule_diff: list[str] = field(default_factory=list)
    time_diff_seconds: float | None = None
    resolution_source_diff: bool = False
    human_review_required: bool = False
    blocking_reasons: list[str] = field(default_factory=list)


def _diff_units_and_threshold(
    a: MarketClaim, b: MarketClaim
) -> tuple[str | None, str | None, Decimal]:
    """Returns (blocking_reason, message, severity) or (None, None, 0) if compatible."""
    if a.threshold_unit != b.threshold_unit:
        return (
            "units_incompatible",
            f"threshold units differ: {a.threshold_unit!r} vs {b.threshold_unit!r}",
            _HIGH_SEVERITY,
        )
    if a.threshold is None or b.threshold is None:
        # Neither side claims a numeric threshold at all (e.g. plain moneylines) -- that
        # is compatible, not a difference.
        if a.threshold is None and b.threshold is None:
            return None, None, Decimal(0)
        return (
            "threshold_mismatch",
            f"threshold present on only one side: {a.threshold!r} vs {b.threshold!r}",
            _HIGH_SEVERITY,
        )
    if a.threshold != b.threshold:
        return (
            "threshold_mismatch",
            f"threshold differs: {a.threshold} vs {b.threshold} {a.threshold_unit}",
            _HIGH_SEVERITY,
        )
    return None, None, Decimal(0)


def compare_claims(
    a: MarketClaim,
    b: MarketClaim,
    *,
    time_tolerance_seconds: float = DEFAULT_TIME_TOLERANCE_SECONDS,
) -> RuleComparison:
    """Compare two claims field by field. See module docstring for the trading gate."""
    diffs: list[tuple[str, str, Decimal]] = []
    review_notes: list[str] = []

    # 1. subject / entities -- entity-level, not string similarity.
    if a.entities and b.entities:
        if a.entities != b.entities:
            only_a = sorted(a.entities - b.entities)
            only_b = sorted(b.entities - a.entities)
            diffs.append(
                (
                    "subject_mismatch",
                    f"subject entities differ: only-a={only_a} only-b={only_b}",
                    _HIGH_SEVERITY,
                )
            )
    else:
        review_notes.append("subject entities could not be extracted on one or both sides")

    # 2. outcome direction.
    if a.outcome != b.outcome:
        diffs.append(
            ("outcome_mismatch", f"outcome differs: {a.outcome!r} vs {b.outcome!r}", _HIGH_SEVERITY)
        )

    # 3. comparator -- a real difference, but a low-severity one at an otherwise-equal
    #    threshold (">" vs ">=" only disagrees on the knife-edge value itself).
    if a.comparator != b.comparator:
        diffs.append(
            (
                "comparator_mismatch",
                f"comparator differs: {a.comparator!r} vs {b.comparator!r} (low severity)",
                _LOW_SEVERITY,
            )
        )

    # 4. units + threshold, exact Decimal comparison after unit normalization.
    reason, message, severity = _diff_units_and_threshold(a, b)
    if reason is not None:
        assert message is not None
        diffs.append((reason, message, severity))

    # 5. measurement -- terminal vs barrier-touch (or any other mismatch) is disqualifying.
    if a.measurement != b.measurement:
        diffs.append(
            (
                "measurement_mismatch",
                f"measurement differs: {a.measurement!r} vs {b.measurement!r} "
                "(terminal vs barrier-touch is never equivalent)",
                _HIGH_SEVERITY,
            )
        )

    # 6. time windows, within tolerance, already timezone-normalized to UTC by extract.py.
    time_diff_seconds: float | None = None
    a_t, b_t = a.time_window_end, b.time_window_end
    if a_t is not None and b_t is not None:
        time_diff_seconds = abs((a_t - b_t).total_seconds())
        if time_diff_seconds > time_tolerance_seconds:
            diffs.append(
                (
                    "time_window_mismatch",
                    f"time windows differ by {time_diff_seconds:.0f}s "
                    f"(tolerance {time_tolerance_seconds:.0f}s)",
                    _MED_SEVERITY,
                )
            )
    else:
        review_notes.append("time window missing on one or both sides")

    # 7. resolution authority -- different oracles/stations for the same nominal
    #    threshold is a real difference (Coinbase vs Binance, KNYC vs KLGA, ...).
    resolution_source_diff = False
    if a.resolution_authority and b.resolution_authority:
        if a.resolution_authority.strip().lower() != b.resolution_authority.strip().lower():
            resolution_source_diff = True
            diffs.append(
                (
                    "resolution_authority_mismatch",
                    f"resolution authority differs: {a.resolution_authority!r} vs "
                    f"{b.resolution_authority!r}",
                    _HIGH_SEVERITY,
                )
            )
    else:
        review_notes.append("resolution authority not specified on one or both sides")

    # 8. cancellation / postponement rules.
    if (a.postponement_rule or b.postponement_rule) and a.postponement_rule != b.postponement_rule:
        diffs.append(
            (
                "postponement_rule_mismatch",
                f"postponement/cancellation rule differs: {a.postponement_rule!r} vs "
                f"{b.postponement_rule!r}",
                _HIGH_SEVERITY,
            )
        )

    # 9. inclusion / exclusion language -- both sides must agree on the same qualifiers.
    if a.inclusion_exclusion != b.inclusion_exclusion:
        diffs.append(
            (
                "inclusion_exclusion_mismatch",
                f"inclusion/exclusion language differs: {sorted(a.inclusion_exclusion)} vs "
                f"{sorted(b.inclusion_exclusion)}",
                _MED_SEVERITY,
            )
        )

    # 10. settlement timing.
    if a.settlement_timing and b.settlement_timing and a.settlement_timing != b.settlement_timing:
        diffs.append(
            (
                "settlement_timing_mismatch",
                f"settlement timing differs: {a.settlement_timing!r} vs {b.settlement_timing!r}",
                _MED_SEVERITY,
            )
        )

    blocking_reasons = [code for code, _msg, _sev in diffs]
    rule_diff = [msg for _code, msg, _sev in diffs] + review_notes
    same_outcome_boolean = len(blocking_reasons) == 0

    confidence = Decimal("1.0")
    for _code, _msg, sev in diffs:
        confidence -= sev
    confidence -= Decimal("0.05") * len(review_notes)
    confidence = max(Decimal("0"), min(Decimal("1"), confidence)).quantize(Decimal("0.0001"))

    only_trivial_diffs = bool(diffs) and all(sev <= _LOW_SEVERITY for _c, _m, sev in diffs)
    human_review_required = bool(review_notes) or only_trivial_diffs

    return RuleComparison(
        same_outcome_boolean=same_outcome_boolean,
        confidence=confidence,
        rule_diff=rule_diff,
        time_diff_seconds=time_diff_seconds,
        resolution_source_diff=resolution_source_diff,
        human_review_required=human_review_required,
        blocking_reasons=blocking_reasons,
    )


def detect_complement(
    a: MarketClaim, b: MarketClaim, *, time_tolerance_seconds: float = DEFAULT_TIME_TOLERANCE_SECONDS
) -> bool:
    """Is ``b`` the logical NO of ``a``?

    Used by the binary-parity strategy: two claims about the *same* event, same
    resolution authority, same settlement instant, whose outcome/comparator are the
    exact logical negation of one another (never merely "different"). This is distinct
    from ``compare_claims`` agreeing they are the same bet -- a complement pair always
    has ``same_outcome_boolean is False`` there, because they pay off oppositely.
    """
    if not a.entities or a.entities != b.entities:
        return False
    if a.measurement != b.measurement:
        return False
    if (
        a.resolution_authority
        and b.resolution_authority
        and a.resolution_authority.strip().lower() != b.resolution_authority.strip().lower()
    ):
        return False
    a_t, b_t = a.time_window_end, b.time_window_end
    if (
        a_t is not None
        and b_t is not None
        and abs((a_t - b_t).total_seconds()) > time_tolerance_seconds
    ):
        return False

    if a.threshold is not None and b.threshold is not None:
        if a.threshold != b.threshold or a.threshold_unit != b.threshold_unit:
            return False
        return (a.comparator, b.comparator) in _NUMERIC_COMPLEMENT_PAIRS

    return (a.outcome, b.outcome) in _CATEGORICAL_COMPLEMENT_PAIRS
