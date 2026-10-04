"""Deterministic game-winner matching, the match book, and trade translation."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from marketlab.core.events import TraderActionEvent
from marketlab.core.instruments import Category, MarketStatus, NormalizedMarket, Side, Venue
from marketlab.matching.book import MatchBook, MatchView
from marketlab.matching.cross_venue import approved_for_automation
from marketlab.matching.extract import extract_claim
from marketlab.matching.sports import codes_match, match_games, parse_kalshi_game, split_teams

NOW = datetime(2026, 10, 4, 12, tzinfo=UTC)


def _kalshi(event_id: str, suffix: str, title: str) -> NormalizedMarket:
    return NormalizedMarket(
        canonical_id=f"kalshi:{event_id.lower()}-{suffix}", venue=Venue.KALSHI,
        venue_market_id=f"{event_id}-{suffix.upper()}", event_id=event_id, title=title,
        category=Category.SPORTS, status=MarketStatus.OPEN,
        close_time=datetime(2026, 10, 6, tzinfo=UTC),
    )


def _poly(slug: str, outcomes: list[str], kind: str = "moneyline", cid: str = "0xabc") -> NormalizedMarket:
    return NormalizedMarket(
        canonical_id=f"poly:{cid}", venue=Venue.POLY_GLOBAL, venue_market_id=cid, event_id="1",
        title=" vs. ".join(outcomes), category=Category.SPORTS, status=MarketStatus.OPEN,
        yes_symbol=outcomes[0], no_symbol=outcomes[1] if len(outcomes) > 1 else "",
        close_time=datetime(2026, 10, 5, 20, tzinfo=UTC),
        raw={"slug": slug, "sportsMarketType": kind, "outcomes": str(outcomes).replace("'", '"')},
    )


def test_pairs_only_the_kalshi_market_whose_yes_is_poly_outcome_zero():
    kalshi = [
        _kalshi("KXMLBGAME-26OCT052000NYYTB", "nyy", "New York Y wins"),
        _kalshi("KXMLBGAME-26OCT052000NYYTB", "tb", "Tampa Bay wins"),
    ]
    poly = [_poly("mlb-nyy-tb-2026-10-05", ["New York Yankees", "Tampa Bay Rays"])]
    matches = match_games(kalshi, poly, NOW)
    assert [m.canonical_id_a for m in matches] == ["kalshi:kxmlbgame-26oct052000nyytb-nyy"]
    m = matches[0]
    assert m.same_outcome_boolean is True and not m.human_review_required
    assert approved_for_automation(m)


def test_team_order_may_differ_between_venues():
    kalshi = [_kalshi("KXNFLGAME-26OCT04LARPHI", "phi", "Philadelphia wins")]
    poly = [_poly("nfl-phi-la-2026-10-04", ["Eagles", "Rams"])]
    assert len(match_games(kalshi, poly, NOW)) == 1


def test_ambiguous_or_unprovable_pairs_are_never_matched():
    k = _kalshi("KXMLBGAME-26OCT052000NYYTB", "nyy", "New York Y wins")
    # wrong date, three-way market, non-moneyline, and a doubleheader game 2
    assert match_games([k], [_poly("mlb-nyy-tb-2026-10-08", ["Yankees", "Rays"])], NOW) == []
    assert match_games([k], [_poly("mlb-nyy-tb-2026-10-05", ["Yankees", "Rays", "Draw"])], NOW) == []
    assert match_games([k], [_poly("mlb-nyy-tb-2026-10-05", ["Yankees", "Rays"], kind="totals")], NOW) == []
    g2 = _kalshi("KXMLBGAME-26OCT051915DETCLEG2", "det", "Detroit wins")
    assert parse_kalshi_game(g2) is None
    # two Polymarket listings of the same game: do not guess
    dupes = [_poly("mlb-nyy-tb-2026-10-05", ["Yankees", "Rays"], cid="0x1"),
             _poly("mlb-nyy-tb-2026-10-05", ["Yankees", "Rays"], cid="0x2")]
    assert match_games([k], dupes, NOW) == []


def test_code_alignment():
    assert codes_match("nfl", "lar", "la")  # one-character prefix
    assert codes_match("nhl", "cgy", "cal")  # alias
    assert not codes_match("mlb", "cleg2", "cle")  # doubleheader suffix is not a prefix match
    assert split_teams("mlb", "NYYTB", ("tb", "nyy")) == ("tb", "nyy")
    assert split_teams("mlb", "NYYTB", ("bos", "nyy")) is None


def test_malformed_time_text_does_not_crash_claim_extraction():
    m = NormalizedMarket(
        canonical_id="kalshi:x", venue=Venue.KALSHI, venue_market_id="X", event_id="E",
        title="Will it happen by 13PM ET on Oct 5, 2026?", status=MarketStatus.OPEN,
    )
    extract_claim(m)  # used to raise "hour must be in 0..23" and abort the whole pass


def test_match_view_filters_by_universe_and_supports_both_lookup_shapes():
    kalshi = [_kalshi("KXMLBGAME-26OCT052000NYYTB", "nyy", "New York Y wins")]
    poly = [_poly("mlb-nyy-tb-2026-10-05", ["Yankees", "Rays"])]
    book = MatchBook()
    book.replace(match_games(kalshi, poly, NOW))
    universes = {"kalshi:kxmlbgame-26oct052000nyytb-nyy": {"sports_mlb"}}
    mlb = MatchView(book, "sports_mlb", lambda cid: universes.get(cid, set()))
    nfl = MatchView(book, "sports_nfl", lambda cid: universes.get(cid, set()))
    assert len(list(mlb)) == 1 and bool(mlb)
    assert not nfl and nfl.get("poly:0xabc") is None
    assert mlb.get("0xabc") is not None  # copy_trader looks up by raw condition id
    assert mlb.get("kalshi:kxmlbgame-26oct052000nyytb-nyy") is not None


async def test_runner_translates_a_matched_poly_buy_onto_kalshi(tmp_path):
    from marketlab.clock import SimulatedClock
    from tests.unit.test_runner import _make_runner

    runner = _make_runner(tmp_path / "m.db", {}, SimulatedClock(NOW))
    kalshi = [_kalshi("KXMLBGAME-26OCT052000NYYTB", "nyy", "New York Y wins")]
    poly = _poly("mlb-nyy-tb-2026-10-05", ["Yankees", "Rays"])
    runner.match_book.replace(match_games(kalshi, [poly], NOW))
    runner._markets[poly.canonical_id] = poly

    def action(outcome: str, act: str = "BUY") -> TraderActionEvent:
        return TraderActionEvent(
            event_time=NOW, first_seen_time=NOW, wallet="0xw", canonical_id=poly.canonical_id,
            poly_condition_id="0xabc", outcome=outcome, action=act, price=Decimal("0.6"),
        )

    yes = runner._translate_trader_action(action("Yankees"))
    assert yes.canonical_id == "kalshi:kxmlbgame-26oct052000nyytb-nyy" and yes.side is Side.YES
    no = runner._translate_trader_action(action("Rays"))
    assert no.side is Side.NO
    sell = runner._translate_trader_action(action("Yankees", "SELL"))
    assert sell.canonical_id == poly.canonical_id  # exits are not entry signals
