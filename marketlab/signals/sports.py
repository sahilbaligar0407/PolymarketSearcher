"""Sports-specific features: consensus pricing, a simple Elo baseline, and the pregame/live gate.

American-odds conversion and vig removal are **not reimplemented here** - they live in
:mod:`marketlab.core.probability` and are imported, because two independently maintained
implementations of the same de-vig math would eventually disagree by a rounding rule and
nobody would notice until a backtest and live diverged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal

from marketlab.core.instruments import PROB_QUANTUM
from marketlab.core.probability import american_to_probability, remove_vig

__all__ = [
    "american_to_probability",
    "remove_vig",
    "consensus_probability",
    "market_vs_consensus",
    "elo_rating",
    "elo_win_probability",
    "line_movement",
    "steam_move_detected",
    "GameState",
    "is_live",
    "should_use_pregame_signal",
    "time_decay_weight",
    "score_differential_probability",
]


def consensus_probability(book_probabilities: dict[str, list[Decimal]]) -> Decimal:
    """Vig-free consensus probability for one outcome across multiple bookmakers.

    Each value in ``book_probabilities`` is one bookmaker's full set of vig-inclusive
    implied probabilities for the market (e.g. ``[home_prob, away_prob]``), with the
    outcome of interest always at index 0. The vig is removed **per bookmaker first**
    (``remove_vig`` normalizes each book's own set to sum to 1), and only then are the
    de-vigged outcome-0 probabilities averaged across books.

    This order matters: averaging vig-inclusive numbers first and normalizing the average
    at the end silently assumes every book carries the same margin. A book with a fat 8%
    margin and a book with a razor 2% margin do not distort the average by the same
    amount once vig is present, so removing it per book before averaging is the only way
    to get a consensus that isn't secretly a vig-weighted blend.
    """
    if not book_probabilities:
        raise ValueError("book_probabilities must not be empty")
    de_vigged_outcome: list[Decimal] = []
    for probs in book_probabilities.values():
        if not probs:
            continue
        vig_free = remove_vig(probs)
        de_vigged_outcome.append(vig_free[0])
    if not de_vigged_outcome:
        raise ValueError("no bookmaker supplied any probabilities")
    avg = sum(de_vigged_outcome, Decimal(0)) / len(de_vigged_outcome)
    return avg.quantize(PROB_QUANTUM, rounding=ROUND_HALF_EVEN)


def market_vs_consensus(contract_price: Decimal, consensus: Decimal) -> Decimal:
    """Disagreement between the tradeable contract price and the sportsbook consensus."""
    return contract_price - consensus


# ---------------------------------------------------------------------------
# Elo baseline
# ---------------------------------------------------------------------------

DEFAULT_ELO_RATING = 1500.0


def elo_win_probability(r_home: float, r_away: float, home_advantage: float) -> float:
    """Standard logistic Elo win probability for the home team, with a rating-points bonus."""
    return 1.0 / (1.0 + 10.0 ** (-((r_home + home_advantage - r_away) / 400.0)))


def elo_rating(
    ratings: dict[str, float],
    home: str,
    away: str,
    k: float,
    home_advantage: float,
    outcome: float | None = None,
) -> tuple[float, float]:
    """Look up (and, given a result, update) Elo ratings for ``home``/``away``.

    ``ratings`` is read (defaulting unseen teams to ``DEFAULT_ELO_RATING``) and, when
    ``outcome`` is supplied, updated in place. ``outcome`` is the actual result from the
    home team's perspective: ``1.0`` home win, ``0.0`` away win, ``0.5`` draw.

    With ``outcome=None`` (the pregame case) this simply returns each team's expected
    score - i.e. ``(elo_win_probability(...), 1 - that)`` - without touching ``ratings``,
    since there is nothing to update against yet. A simple, honest baseline: no momentum
    terms, no margin-of-victory adjustment, no per-league K calibration.
    """
    r_home = ratings.get(home, DEFAULT_ELO_RATING)
    r_away = ratings.get(away, DEFAULT_ELO_RATING)
    expected_home = elo_win_probability(r_home, r_away, home_advantage)
    expected_away = 1.0 - expected_home
    if outcome is None:
        return expected_home, expected_away
    new_home = r_home + k * (outcome - expected_home)
    new_away = r_away + k * ((1.0 - outcome) - expected_away)
    ratings[home] = new_home
    ratings[away] = new_away
    return new_home, new_away


# ---------------------------------------------------------------------------
# Line movement
# ---------------------------------------------------------------------------


def line_movement(open_line: Decimal, current_line: Decimal) -> Decimal:
    return current_line - open_line


def steam_move_detected(
    line_history: list[tuple[datetime, Decimal]],
    threshold: Decimal,
    window_seconds: float,
    now: datetime,
) -> bool:
    """Whether the line has moved by at least ``threshold`` within the trailing ``window_seconds``.

    A "steam move" is a sudden, sharp line move (sharp money hitting many books at once)
    rather than the slow pregame drift ``time_decay_weight`` accounts for - the signal is
    the size of the move *within a short window*, not the size of the move overall.
    """
    eligible = [(ts, line) for ts, line in line_history if ts <= now]
    if len(eligible) < 2:
        return False
    cutoff = now.timestamp() - window_seconds
    windowed = [(ts, line) for ts, line in eligible if ts.timestamp() >= cutoff]
    if len(windowed) < 2:
        return False
    lines = [line for _, line in windowed]
    movement = max(lines) - min(lines)
    return movement >= threshold


# ---------------------------------------------------------------------------
# Pregame / live gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameState:
    """Minimal snapshot of a game's status, enough to gate pregame vs. live signal use."""

    game_id: str
    league: str = ""
    started: bool = False
    final: bool = False
    seconds_to_start: float | None = None
    seconds_remaining: float | None = None
    home_score: int | None = None
    away_score: int | None = None


def is_live(state: GameState) -> bool:
    return state.started and not state.final


def should_use_pregame_signal(state: GameState) -> bool:
    """The enforcement point: pregame signal use must stop the instant the game goes live.

    Returns ``False`` as soon as ``state.started`` is ``True``, regardless of any other
    field (score, clock, even a game that is somehow already ``final``). A pregame
    consensus/Elo edge says nothing once real information (game events) starts arriving -
    continuing to trade on it is a point-in-time discipline violation, not a stale-but-ok
    feature.
    """
    return not state.started


def time_decay_weight(seconds_to_start: float, halflife_seconds: float = 6.0 * 3600.0) -> float:
    """Pregame signal weight that rises toward 1.0 as kickoff approaches.

    Exponential decay by time-to-start with a configurable half-life (default 6 hours):
    a pregame consensus edge computed a week out is far less trustworthy (more can change)
    than the same edge computed five minutes before the opening whistle. Already-started
    games (``seconds_to_start <= 0``) get full weight, though
    :func:`should_use_pregame_signal` is what actually gates whether pregame signal is
    used at all once the game goes live.
    """
    if seconds_to_start <= 0:
        return 1.0
    return math.exp(-math.log(2.0) * seconds_to_start / halflife_seconds)


# ---------------------------------------------------------------------------
# In-game baseline (crude by design)
# ---------------------------------------------------------------------------

#: Rough "how fast this league's score differential compresses time" scale factors.
#: Larger scale = a given point differential is more decisive sooner (e.g. hockey, where
#: goals are rare and precious) - purely a hand-tuned baseline, not fit to any data.
_LEAGUE_SCALE: dict[str, float] = {
    "NBA": 0.20,
    "NCAAB": 0.22,
    "NFL": 0.45,
    "NCAAF": 0.45,
    "MLB": 0.65,
    "NHL": 0.85,
    "SOCCER": 1.10,
}
_DEFAULT_LEAGUE_SCALE = 0.5


def score_differential_probability(diff: int, seconds_remaining: float, league: str = "") -> float:
    """A crude in-game win-probability baseline from score differential and time left.

    **This is a baseline to beat, not a model to trust.** It ignores possession, timeouts,
    foul trouble, pull-the-goalie situations, run environment, and every other bit of
    context a real in-game model would use - it exists so a strategy has *something*
    principled to compare its actual model against, and so "my model beats score-diff/
    time-left" is a checkable claim.

    Uses a logistic function of the differential scaled by the inverse square root of time
    remaining (in minutes): the same lead becomes more decisive as time runs out, and the
    league-specific scale factor accounts for how "expensive" one point of differential is
    in that sport.
    """
    scale = _LEAGUE_SCALE.get(league.upper(), _DEFAULT_LEAGUE_SCALE)
    minutes_remaining = max(seconds_remaining, 1.0) / 60.0
    z = scale * diff / math.sqrt(minutes_remaining)
    return 1.0 / (1.0 + math.exp(-z))
