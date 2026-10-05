"""Deterministic Kalshi <-> Polymarket game-winner matching.

The general matcher compares free-text claims and, after a week of live running, had
approved **zero** of 2,372 candidate pairs: every pair lacked a stated resolution
authority, and most compared Kalshi's *event* title ("Bitcoin price range on Sep 7")
against a Polymarket threshold contract - genuinely different bets. Rejecting those was
correct, but it left copy trading and cross-venue relative value with nothing to trade.

Game-winner markets are the one large class where identity can be proven from structure
alone, without reading prose:

* Kalshi:     ``KXMLBGAME-26OCT052000NYYTB`` / market ``...-tb``
              -> league MLB, game date 2026-10-05, teams NYY+TB, YES = "TB wins"
* Polymarket: slug ``mlb-nyy-tb-2026-10-05``, ``sportsMarketType == "moneyline"``,
              outcomes ``["Yankees", "Rays"]`` (slug order) -> YES token = outcome[0]

Both settle on the official result of the same game. A Polymarket market is paired only
with the Kalshi market whose YES team is Polymarket's outcome[0], so every match is a
true YES==YES pair and no downstream code ever has to invert a price. Anything ambiguous
(no unique game on that date, a three-way market with a draw, codes that do not line
up) is simply not matched: a missed pair costs an opportunity, a wrong pair costs money.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from marketlab.core.instruments import NormalizedMarket
from marketlab.logging import get_logger
from marketlab.matching.cross_venue import MarketMatch

log = get_logger(__name__)

_EASTERN = ZoneInfo("America/New_York")

#: 1.1: offset-day pairs require the Eastern start date (FINDINGS 57); 1.0 rows are void.
VALIDATOR_VERSION = "sports-moneyline-1.1"
MATCH_CONFIDENCE = Decimal("0.97")

#: Kalshi game series -> Polymarket slug league prefix.
KALSHI_LEAGUES: dict[str, str] = {
    "KXMLBGAME": "mlb",
    "KXNFLGAME": "nfl",
    "KXNCAAFGAME": "cfb",
    "KXNHLGAME": "nhl",
    "KXNBAGAME": "nba",
    "KXWNBAGAME": "wnba",
    "KXNCAABGAME": "cbb",
    "KXMLSGAME": "mls",
}

#: Codes that neither equal nor prefix each other across venues (lower-case, per league).
ALIASES: dict[str, dict[str, str]] = {
    "nhl": {"cal": "cgy", "mon": "mtl", "nj": "njd", "tb": "tbl", "la": "lak", "sj": "sjs"},
    "nfl": {"la": "lar", "wsh": "was", "jac": "jax"},
    "mlb": {"wsh": "was", "az": "ari", "cws": "chw", "kc": "kcr", "sd": "sdp", "sf": "sfg", "tb": "tbr"},
    "nba": {"gs": "gsw", "no": "nop", "ny": "nyk", "sa": "sas", "uta": "utah"},
}

_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}
_KALSHI_EVENT_RE = re.compile(r"^(?P<series>KX[A-Z]+GAME)-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?P<hhmm>\d{4})?(?P<teams>[A-Z0-9]+)$")
_POLY_SLUG_RE = re.compile(r"^(?P<league>[a-z0-9]+)-(?P<a>[a-z0-9]+)-(?P<b>[a-z0-9]+)-(?P<date>\d{4}-\d{2}-\d{2})$")


@dataclass(frozen=True)
class KalshiGame:
    market: NormalizedMarket
    league: str
    game_date: date
    teams: str  # concatenated upper-case codes, e.g. "NYYTB"
    yes_code: str  # lower-case code of the team whose win is YES


@dataclass(frozen=True)
class PolyGame:
    market: NormalizedMarket
    league: str
    game_date: date
    codes: tuple[str, str]
    outcomes: tuple[str, str]
    #: The game's start (``gameStartTime``) as a US Eastern date - Kalshi's own dating.
    start_et: date | None = None


def parse_kalshi_game(market: NormalizedMarket) -> KalshiGame | None:
    m = _KALSHI_EVENT_RE.match(market.event_id or "")
    if m is None:
        return None
    league = KALSHI_LEAGUES.get(m["series"])
    month = _MONTHS.get(m["mon"])
    if league is None or month is None:
        return None
    try:
        game_date = date(2000 + int(m["yy"]), month, int(m["dd"]))
    except ValueError:
        return None
    teams = m["teams"]
    if re.search(r"G\d$", teams):
        # A doubleheader game: Polymarket rarely distinguishes G1 from G2 in the slug,
        # so the pair cannot be proven. Never guess which game a market is.
        return None
    yes_code = market.canonical_id.rsplit("-", 1)[-1].lower()
    if not yes_code or yes_code.upper() not in teams:
        return None
    return KalshiGame(market, league, game_date, teams, yes_code)


def _raw(market: NormalizedMarket) -> dict[str, Any]:
    raw = market.raw or {}
    # Markets reloaded from storage carry the venue payload one level down.
    inner = raw.get("raw")
    return inner if isinstance(inner, dict) else raw


def parse_poly_game(market: NormalizedMarket) -> PolyGame | None:
    raw = _raw(market)
    if raw.get("sportsMarketType") != "moneyline":
        return None
    m = _POLY_SLUG_RE.match(str(raw.get("slug") or ""))
    if m is None:
        return None  # includes 3-way soccer slugs ending in a team/draw suffix
    outcomes = raw.get("outcomes")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except ValueError:
            return None
    if not isinstance(outcomes, list) or len(outcomes) != 2:
        return None
    try:
        game_date = date.fromisoformat(m["date"])
    except ValueError:
        return None
    start_et: date | None = None
    with contextlib.suppress(ValueError, TypeError):
        start = datetime.fromisoformat(str(raw.get("gameStartTime")).replace("Z", "+00:00").replace(" ", "T"))
        if start.tzinfo is not None:
            start_et = start.astimezone(_EASTERN).date()
    return PolyGame(
        market, m["league"], game_date, (m["a"], m["b"]), (str(outcomes[0]), str(outcomes[1])), start_et
    )


def codes_match(league: str, kalshi_code: str, poly_code: str) -> bool:
    k, p = kalshi_code.lower(), poly_code.lower()
    if k == p:
        return True
    aliases = ALIASES.get(league, {})
    if aliases.get(p) == k or aliases.get(k) == p:
        return True
    shorter, longer = sorted((k, p), key=len)
    return len(shorter) >= 2 and longer.startswith(shorter) and len(longer) - len(shorter) == 1


def split_teams(league: str, teams: str, codes: tuple[str, str]) -> tuple[str, str] | None:
    """Split Kalshi's concatenated team string so each half matches one Poly code.

    Returns the Kalshi codes aligned to Polymarket's order (codes[0], codes[1]), or None
    when no split matches or more than one does.
    """
    found: list[tuple[str, str]] = []
    for i in range(2, len(teams) - 1):
        left, right = teams[:i].lower(), teams[i:].lower()
        if codes_match(league, left, codes[0]) and codes_match(league, right, codes[1]):
            found.append((left, right))
        if codes_match(league, right, codes[0]) and codes_match(league, left, codes[1]):
            found.append((right, left))
    unique = set(found)
    return found[0] if len(unique) == 1 else None


def _match_id(a: str, b: str) -> str:
    return hashlib.sha256(f"{a}|{b}|{VALIDATOR_VERSION}".encode()).hexdigest()[:24]


def match_games(
    kalshi_markets: list[NormalizedMarket],
    poly_markets: list[NormalizedMarket],
    now: datetime,
) -> list[MarketMatch]:
    """Every provable YES==YES game-winner pair. Deterministic, no model involved."""
    polys: dict[tuple[str, date], list[PolyGame]] = {}
    for pm in poly_markets:
        pg = parse_poly_game(pm)
        if pg is not None:
            polys.setdefault((pg.league, pg.game_date), []).append(pg)

    kgames = [g for g in map(parse_kalshi_game, kalshi_markets) if g is not None]
    # Two Kalshi events for the same teams on the same day (a doubleheader listed without
    # a G-suffix) make every pairing for that day ambiguous.
    events_per_slot: dict[tuple[str, date, str], set[str]] = {}
    for g in kgames:
        events_per_slot.setdefault((g.league, g.game_date, "".join(sorted(g.teams))), set()).add(g.market.event_id)

    out: list[MarketMatch] = []
    for kg in kgames:
        km = kg.market
        if len(events_per_slot[(kg.league, kg.game_date, "".join(sorted(kg.teams)))]) > 1:
            continue
        # Kalshi dates games in US Eastern; Polymarket slugs occasionally roll a late
        # game to the next UTC day. Exact date first, then +/-1 day, each requiring a
        # unique candidate.
        chosen: tuple[PolyGame, tuple[str, str]] | None = None
        for offset in (0, 1, -1):
            hits = []
            for pg in polys.get((kg.league, kg.game_date + timedelta(days=offset)), []):
                # A +/-1 day slug is only the same game when the start time, in Eastern,
                # is Kalshi's date (a late game rolled over in UTC). Without this check
                # back-to-back series paired a Kalshi game with the previous or next
                # day's game - 9 such double twins were approved (FINDINGS 57).
                if offset and pg.start_et != kg.game_date:
                    continue
                aligned = split_teams(kg.league, kg.teams, pg.codes)
                if aligned is not None:
                    hits.append((pg, aligned))
            if len(hits) == 1:
                chosen = hits[0]
                break
            if len(hits) > 1:
                break  # doubleheader or duplicate listing: ambiguous, do not guess
        if chosen is None:
            continue
        pg, aligned = chosen
        # Only the Kalshi market whose YES team is Polymarket's outcome[0] (the YES
        # token whose book we read) - so the pair needs no inversion anywhere.
        if kg.yes_code != aligned[0]:
            continue
        time_diff = ""
        if km.close_time and pg.market.close_time:
            time_diff = f"{abs((km.close_time - pg.market.close_time).total_seconds()):.0f}"
        out.append(
            MarketMatch(
                match_id=_match_id(km.canonical_id, pg.market.canonical_id),
                canonical_id_a=km.canonical_id,
                canonical_id_b=pg.market.canonical_id,
                match_confidence=MATCH_CONFIDENCE,
                same_outcome_boolean=True,
                rule_diff=(
                    f"{kg.league} moneyline {kg.game_date}: Kalshi YES '{km.title}' == "
                    f"Polymarket '{pg.outcomes[0]}' ({pg.codes[0]} vs {pg.codes[1]}); both settle "
                    "on the official game result (close times differ by design)"
                ),
                time_diff=time_diff,
                resolution_source_diff="official league result on both venues",
                human_review_required=False,
                created_at=now,
                validator_version=VALIDATOR_VERSION,
            )
        )
    return out


__all__ = ["VALIDATOR_VERSION", "codes_match", "match_games", "parse_kalshi_game", "parse_poly_game", "split_teams"]
