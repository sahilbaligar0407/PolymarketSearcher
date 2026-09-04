"""Unit tests for marketlab/signals/sports.py."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from marketlab.core.probability import american_to_probability, remove_vig
from marketlab.signals.sports import (
    GameState,
    consensus_probability,
    elo_rating,
    elo_win_probability,
    is_live,
    line_movement,
    should_use_pregame_signal,
    steam_move_detected,
    time_decay_weight,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _t(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


# ---------------------------------------------------------------------------
# -110/-110 vig removal
# ---------------------------------------------------------------------------


def test_minus_110_both_sides_raw_and_vig_free() -> None:
    home_raw = american_to_probability(-110)
    away_raw = american_to_probability(-110)
    assert home_raw == Decimal("0.5238")
    assert away_raw == Decimal("0.5238")

    home_fair, away_fair = remove_vig([home_raw, away_raw])
    assert home_fair == Decimal("0.5000")
    assert away_fair == Decimal("0.5000")


# ---------------------------------------------------------------------------
# Consensus: per-book de-vig then average, NOT average-then-normalize
# ---------------------------------------------------------------------------


def test_consensus_probability_per_book_devig_differs_from_average_then_normalize() -> None:
    # Two books quoting the same favorite but with very different margins (5% vs 10%).
    book_probs = {
        "book_a": [Decimal("0.55"), Decimal("0.50")],  # sums to 1.05
        "book_b": [Decimal("0.70"), Decimal("0.40")],  # sums to 1.10
    }

    consensus = consensus_probability(book_probs)

    # The naive (wrong) approach: average the raw vig-inclusive numbers first, then
    # normalize the pair at the end.
    raw_home_avg = (Decimal("0.55") + Decimal("0.70")) / 2
    raw_away_avg = (Decimal("0.50") + Decimal("0.40")) / 2
    naive_consensus = raw_home_avg / (raw_home_avg + raw_away_avg)

    # Per-book-first: 0.55/1.05 = 0.52380..., 0.70/1.10 = 0.63636..., avg = 0.58008...
    expected_per_book = (Decimal("0.55") / Decimal("1.05") + Decimal("0.70") / Decimal("1.10")) / 2

    assert float(consensus) == pytest.approx(float(expected_per_book), abs=1e-3)
    assert consensus != naive_consensus.quantize(Decimal("0.0001"))
    assert abs(float(consensus) - float(naive_consensus)) > 1e-4


def test_consensus_probability_single_book_matches_its_own_devig() -> None:
    consensus = consensus_probability({"only_book": [Decimal("0.55"), Decimal("0.50")]})
    expected = Decimal("0.55") / Decimal("1.05")
    assert float(consensus) == pytest.approx(float(expected), abs=1e-4)


def test_consensus_probability_empty_raises() -> None:
    with pytest.raises(ValueError):
        consensus_probability({})


# ---------------------------------------------------------------------------
# Elo baseline
# ---------------------------------------------------------------------------


def test_elo_win_probability_equal_ratings_plus_home_advantage_favors_home() -> None:
    p_home = elo_win_probability(1500.0, 1500.0, home_advantage=65.0)
    assert p_home > 0.5


def test_elo_win_probability_equal_ratings_no_advantage_is_half() -> None:
    assert elo_win_probability(1500.0, 1500.0, home_advantage=0.0) == pytest.approx(0.5)


def test_elo_rating_pregame_returns_expected_score_without_mutating() -> None:
    ratings = {"HOME": 1600.0, "AWAY": 1500.0}
    expected_home, expected_away = elo_rating(ratings, "HOME", "AWAY", k=20.0, home_advantage=0.0)
    assert expected_home > 0.5
    assert expected_home + expected_away == pytest.approx(1.0)
    # No outcome supplied -> ratings dict must be untouched.
    assert ratings == {"HOME": 1600.0, "AWAY": 1500.0}


def test_elo_rating_updates_after_a_result() -> None:
    ratings = {"HOME": 1500.0, "AWAY": 1500.0}
    new_home, new_away = elo_rating(
        ratings, "HOME", "AWAY", k=20.0, home_advantage=0.0, outcome=1.0
    )
    # Home won as an even-money favorite -> rating should rise.
    assert new_home > 1500.0
    assert new_away < 1500.0
    assert ratings["HOME"] == new_home


# ---------------------------------------------------------------------------
# Line movement / steam moves
# ---------------------------------------------------------------------------


def test_line_movement_simple_delta() -> None:
    assert line_movement(Decimal("-3.0"), Decimal("-4.5")) == Decimal("-1.5")


def test_steam_move_detected_on_a_sharp_move_within_window() -> None:
    history = [
        (_t(0), Decimal("-3.0")),
        (_t(60), Decimal("-3.0")),
        (_t(120), Decimal("-3.5")),
        (_t(125), Decimal("-4.5")),  # sharp 1-point move in 5 seconds
    ]
    now = _t(125)
    assert steam_move_detected(history, threshold=Decimal("1.0"), window_seconds=30, now=now) is True
    # Evaluated at t=120 (before the sharp move happens) there's only one point in a 10s
    # window, so there's nothing to compare against yet.
    assert (
        steam_move_detected(history, threshold=Decimal("1.0"), window_seconds=10, now=_t(120))
        is False
    )


def test_steam_move_not_detected_on_slow_drift() -> None:
    history = [
        (_t(0), Decimal("-3.0")),
        (_t(3600), Decimal("-3.5")),
        (_t(7200), Decimal("-4.0")),
    ]
    assert (
        steam_move_detected(history, threshold=Decimal("1.0"), window_seconds=60, now=_t(7200))
        is False
    )


# ---------------------------------------------------------------------------
# Pregame / live gate - the headline test
# ---------------------------------------------------------------------------


def test_should_use_pregame_signal_false_the_instant_game_starts() -> None:
    pregame = GameState(game_id="g1", started=False)
    assert should_use_pregame_signal(pregame) is True

    live = GameState(game_id="g1", started=True)
    assert should_use_pregame_signal(live) is False
    assert is_live(live) is True


def test_should_use_pregame_signal_false_even_with_other_pregame_looking_fields() -> None:
    # started=True must dominate regardless of what else the state carries.
    state = GameState(
        game_id="g1",
        started=True,
        final=False,
        seconds_to_start=-1.0,
        home_score=0,
        away_score=0,
    )
    assert should_use_pregame_signal(state) is False


def test_is_live_false_once_final() -> None:
    finished = GameState(game_id="g1", started=True, final=True)
    assert is_live(finished) is False
    assert should_use_pregame_signal(finished) is False


def test_time_decay_weight_increases_as_start_approaches() -> None:
    far = time_decay_weight(seconds_to_start=24 * 3600.0)
    near = time_decay_weight(seconds_to_start=60.0)
    assert 0.0 < far < near <= 1.0
    assert time_decay_weight(seconds_to_start=0.0) == 1.0
