"""The corpus of cross-venue match pairs that is the matching module's acceptance test.

Every pair is labeled ``should_match`` (what ``resolution_rules.compare_claims``'s
``same_outcome_boolean`` must be) with a human ``reason``. A future change that starts
approving a pair labeled ``should_match=False`` must fail ``tests/unit/test_matching.py``
loudly -- that is the entire point of keeping this corpus as the source of truth rather
than ad hoc asserts scattered through the test file.

Complement pairs (``expected_complement=True``) are deliberately *not* the same claim
(they pay off oppositely) -- ``should_match`` is False for those, and
``detect_complement`` is checked separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from marketlab.core.instruments import Category, MarketStatus, NormalizedMarket, OutcomeType, Venue

T_BTC = datetime(2026, 9, 4, 21, 0, tzinfo=UTC)
T_NFL = datetime(2026, 9, 8, 20, 0, tzinfo=UTC)
T_NFL2 = datetime(2026, 9, 15, 20, 0, tzinfo=UTC)
T_CPI = datetime(2026, 9, 10, 12, 30, tzinfo=UTC)
T_FED = datetime(2026, 9, 17, 18, 0, tzinfo=UTC)
T_WX = datetime(2026, 9, 4, 23, 59, tzinfo=UTC)


def _mk(
    canonical_id: str,
    title: str,
    *,
    venue: Venue = Venue.KALSHI,
    category: Category = Category.CRYPTO,
    close_time: datetime = T_BTC,
    resolution_source: str = "",
    resolution_rules: str = "",
) -> NormalizedMarket:
    return NormalizedMarket(
        canonical_id=canonical_id,
        venue=venue,
        venue_market_id=canonical_id,
        event_id=canonical_id,
        title=title,
        resolution_rules=resolution_rules,
        resolution_source=resolution_source,
        category=category,
        outcome_type=OutcomeType.BINARY,
        status=MarketStatus.OPEN,
        open_time=close_time,
        close_time=close_time,
    )


@dataclass(frozen=True)
class MatchPairFixture:
    name: str
    market_a: NormalizedMarket
    market_b: NormalizedMarket
    should_match: bool
    reason: str
    expected_complement: bool = False
    expected_human_review: bool | None = None


# ---------------------------------------------------------------------------
# REJECT: near-identical wording, genuinely different bets.
# ---------------------------------------------------------------------------

TERMINAL_VS_BARRIER = MatchPairFixture(
    name="terminal_vs_barrier_touch",
    market_a=_mk(
        "KALSHI-BTC-ABOVE-100K",
        "Will BTC be above $100,000 at 5 PM ET on September 4, 2026?",
        resolution_source="Coinbase",
    ),
    market_b=_mk(
        "POLY-BTC-REACH-100K",
        "Will BTC reach $100,000 before 5 PM ET on September 4, 2026?",
        venue=Venue.POLY_GLOBAL,
        resolution_source="Coinbase",
    ),
    should_match=False,
    reason=(
        "Terminal value at a fixed instant vs. a barrier-touch anywhere before that "
        "instant are different stochastic events even at an identical threshold/time."
    ),
)

DIFFERENT_ORACLE = MatchPairFixture(
    name="different_price_oracle",
    market_a=_mk(
        "KALSHI-BTC-ABOVE-100K-CB",
        "Will BTC be above $100,000 at 5 PM ET on September 4, 2026?",
        resolution_source="Coinbase",
    ),
    market_b=_mk(
        "POLY-BTC-ABOVE-100K-BN",
        "Will BTC be above $100,000 at 5 PM ET on September 4, 2026?",
        venue=Venue.POLY_GLOBAL,
        resolution_source="Binance",
    ),
    should_match=False,
    reason="Same threshold and instant, but Coinbase vs Binance can settle differently.",
)

DIFFERENT_WEATHER_STATION = MatchPairFixture(
    name="different_weather_station",
    market_a=_mk(
        "KALSHI-HIGHNY-75",
        "Highest temperature in NYC above 75°F?",
        category=Category.WEATHER,
        close_time=T_WX,
        resolution_rules=(
            "Settles based on the official high temperature recorded at NWS station "
            "KNYC (Central Park)."
        ),
    ),
    market_b=_mk(
        "POLY-HIGHNY-75",
        "Highest temperature in New York above 75°F?",
        venue=Venue.POLY_GLOBAL,
        category=Category.WEATHER,
        close_time=T_WX,
        resolution_rules=(
            "Settles based on the official high temperature recorded at NWS station "
            "KLGA (LaGuardia)."
        ),
    ),
    should_match=False,
    reason="Same city, same threshold, but KNYC and KLGA are different stations that can print different highs.",
)

MONEYLINE_VS_SPREAD = MatchPairFixture(
    name="moneyline_vs_spread",
    market_a=_mk(
        "KALSHI-CHIEFS-WIN",
        "Will the Chiefs win their Week 1 game?",
        category=Category.SPORTS,
        close_time=T_NFL,
        resolution_source="NFL",
    ),
    market_b=_mk(
        "POLY-CHIEFS-WIN-BY-3",
        "Will the Chiefs win by more than 3 points in their Week 1 game?",
        venue=Venue.POLY_GLOBAL,
        category=Category.SPORTS,
        close_time=T_NFL,
        resolution_source="NFL",
    ),
    should_match=False,
    reason="Moneyline (any win) vs. a spread (win by more than 3) are different bets.",
)

HEADLINE_VS_CORE_CPI = MatchPairFixture(
    name="headline_vs_core_cpi",
    market_a=_mk(
        "KALSHI-CPI-YOY-3",
        "Will CPI YoY be above 3.0%?",
        category=Category.ECONOMICS,
        close_time=T_CPI,
        resolution_source="BLS",
    ),
    market_b=_mk(
        "POLY-CORE-CPI-YOY-3",
        "Will Core CPI YoY be above 3.0%?",
        venue=Venue.POLY_GLOBAL,
        category=Category.ECONOMICS,
        close_time=T_CPI,
        resolution_source="BLS",
    ),
    should_match=False,
    reason="Headline CPI and core CPI (ex food & energy) are different series and routinely diverge.",
)

FED_CUT_VS_FED_CUT_50BP = MatchPairFixture(
    name="fed_cut_vs_fed_cut_50bp",
    market_a=_mk(
        "KALSHI-FED-CUT-SEP",
        "Will the Fed cut rates in September?",
        category=Category.ECONOMICS,
        close_time=T_FED,
        resolution_source="Fed",
    ),
    market_b=_mk(
        "POLY-FED-CUT-50BP-SEP",
        "Will the Fed cut rates by 50 basis points in September?",
        venue=Venue.POLY_GLOBAL,
        category=Category.ECONOMICS,
        close_time=T_FED,
        resolution_source="Fed",
    ),
    should_match=False,
    reason="Any cut vs. a specific 50bp magnitude are different claims; a 25bp cut would settle these oppositely.",
)

WINDOWS_SIX_HOURS_APART = MatchPairFixture(
    name="windows_six_hours_apart",
    market_a=_mk(
        "KALSHI-BTC-100K-CLOSE-A",
        "Will BTC be above $100,000 at market close?",
        close_time=datetime(2026, 9, 4, 20, 0, tzinfo=UTC),
        resolution_source="Coinbase",
    ),
    market_b=_mk(
        "POLY-BTC-100K-CLOSE-B",
        "Will BTC be above $100,000 at market close?",
        venue=Venue.POLY_GLOBAL,
        close_time=datetime(2026, 9, 5, 2, 0, tzinfo=UTC),
        resolution_source="Coinbase",
    ),
    should_match=False,
    reason="Otherwise-identical claim, but the settlement windows are six hours apart.",
)

POSTPONEMENT_VOID_VS_RESOLVES_NO = MatchPairFixture(
    name="postponement_void_vs_resolves_no",
    market_a=_mk(
        "KALSHI-CHIEFS-WIN-VOID",
        "Will the Chiefs win their Week 1 game?",
        category=Category.SPORTS,
        close_time=T_NFL,
        resolution_source="NFL",
        resolution_rules=(
            "If the game is postponed, this market voids and all wagers are refunded, "
            "to be rescheduled once the game is played."
        ),
    ),
    market_b=_mk(
        "POLY-CHIEFS-WIN-NO",
        "Will the Chiefs win their Week 1 game?",
        venue=Venue.POLY_GLOBAL,
        category=Category.SPORTS,
        close_time=T_NFL,
        resolution_source="NFL",
        resolution_rules=(
            "If the game is postponed, this market resolves NO regardless of the "
            "eventual outcome."
        ),
    ),
    should_match=False,
    reason="Identical question, but one venue voids on postponement and the other settles NO -- different payoffs in that scenario.",
)

# ---------------------------------------------------------------------------
# APPROVE: genuinely equivalent, or an exact logical complement.
# ---------------------------------------------------------------------------

BTC_REWORDED_SAME_TERMINAL = MatchPairFixture(
    name="btc_reworded_same_terminal",
    market_a=_mk(
        "KALSHI-BTC-100K-TERM-A",
        "Will BTC be above $100,000 at 5 PM ET on September 4, 2026?",
        resolution_source="Coinbase",
    ),
    market_b=_mk(
        "POLY-BTC-100K-TERM-B",
        "Will Bitcoin exceed $100,000 at 5 PM ET on September 4, 2026?",
        venue=Venue.POLY_GLOBAL,
        resolution_source="Coinbase",
    ),
    should_match=True,
    reason="Same threshold, same terminal instant (5 PM ET / Sept 4, 2026), same oracle -- just worded differently.",
)

NFL_MONEYLINE_SAME_POSTPONEMENT_RULE = MatchPairFixture(
    name="nfl_moneyline_same_postponement_rule",
    market_a=_mk(
        "KALSHI-CHIEFS-WIN-2",
        "Will the Chiefs win their Week 1 game?",
        category=Category.SPORTS,
        close_time=T_NFL2,
        resolution_source="NFL",
        resolution_rules="If the game is postponed, this market voids and all wagers are refunded.",
    ),
    market_b=_mk(
        "POLY-CHIEFS-WIN-2",
        "Chiefs to win Week 1?",
        venue=Venue.POLY_GLOBAL,
        category=Category.SPORTS,
        close_time=T_NFL2,
        resolution_source="NFL",
        resolution_rules="Should the game be postponed, this market voids with all wagers refunded.",
    ),
    should_match=True,
    reason="Same game, same moneyline, same void-on-postponement rule -- just worded differently on each venue.",
)

CHIEFS_WIN_VS_LOSE_COMPLEMENT = MatchPairFixture(
    name="chiefs_win_vs_lose_complement",
    market_a=_mk(
        "KALSHI-CHIEFS-WIN-3",
        "Will the Chiefs win their Week 1 game?",
        category=Category.SPORTS,
        close_time=T_NFL2,
        resolution_source="NFL",
    ),
    market_b=_mk(
        "POLY-CHIEFS-LOSE-3",
        "Will the Chiefs lose their Week 1 game?",
        venue=Venue.POLY_GLOBAL,
        category=Category.SPORTS,
        close_time=T_NFL2,
        resolution_source="NFL",
    ),
    should_match=False,
    reason=(
        "Not the same claim -- they pay off oppositely -- but they are an exact logical "
        "complement (YES on A == NO on B), which is exactly what the binary-parity "
        "strategy needs `detect_complement` for."
    ),
    expected_complement=True,
)

BTC_ABOVE_VS_AT_OR_BELOW_COMPLEMENT = MatchPairFixture(
    name="btc_above_vs_at_or_below_complement",
    market_a=_mk(
        "KALSHI-BTC-100K-ABOVE",
        "Will BTC be above $100,000 at 5 PM ET on September 4, 2026?",
        resolution_source="Coinbase",
    ),
    market_b=_mk(
        "POLY-BTC-100K-AT-OR-BELOW",
        "Will BTC be at or below $100,000 at 5 PM ET on September 4, 2026?",
        venue=Venue.POLY_GLOBAL,
        resolution_source="Coinbase",
    ),
    should_match=False,
    reason="NOT(x > T) == (x <= T) at an identical threshold/instant/oracle -- the numeric complement case.",
    expected_complement=True,
)

# ---------------------------------------------------------------------------
# Genuinely ambiguous / borderline: real difference, but low severity.
# ---------------------------------------------------------------------------

COMPARATOR_KNIFE_EDGE = MatchPairFixture(
    name="comparator_knife_edge",
    market_a=_mk(
        "KALSHI-BTC-100K-STRICT",
        "Will BTC be above $100,000 at 5 PM ET on September 4, 2026?",
        resolution_source="Coinbase",
    ),
    market_b=_mk(
        "POLY-BTC-100K-INCLUSIVE",
        "Will BTC be at least $100,000 at 5 PM ET on September 4, 2026?",
        venue=Venue.POLY_GLOBAL,
        resolution_source="Coinbase",
    ),
    should_match=False,
    reason=(
        "'>' vs '>=' at an identical threshold/instant/oracle only disagree in the "
        "knife-edge case of BTC settling at exactly $100,000.00 -- a real difference, "
        "but low severity, and flagged for human review rather than silently approved."
    ),
    expected_human_review=True,
)

CORPUS: tuple[MatchPairFixture, ...] = (
    TERMINAL_VS_BARRIER,
    DIFFERENT_ORACLE,
    DIFFERENT_WEATHER_STATION,
    MONEYLINE_VS_SPREAD,
    HEADLINE_VS_CORE_CPI,
    FED_CUT_VS_FED_CUT_50BP,
    WINDOWS_SIX_HOURS_APART,
    POSTPONEMENT_VOID_VS_RESOLVES_NO,
    BTC_REWORDED_SAME_TERMINAL,
    NFL_MONEYLINE_SAME_POSTPONEMENT_RULE,
    CHIEFS_WIN_VS_LOSE_COMPLEMENT,
    BTC_ABOVE_VS_AT_OR_BELOW_COMPLEMENT,
    COMPARATOR_KNIFE_EDGE,
)
