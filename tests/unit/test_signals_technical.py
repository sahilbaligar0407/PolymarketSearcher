"""Unit tests for marketlab/signals/technical.py (and the rolling primitives it builds on)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from marketlab.signals.rolling import EWMA, RollingWindow, RollingZScore, WelfordVariance
from marketlab.signals.technical import (
    bounded_momentum,
    momentum,
    realized_volatility,
    simple_return,
    zscore,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _t(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


# ---------------------------------------------------------------------------
# Time-based momentum on an irregularly-spaced series
# ---------------------------------------------------------------------------


def test_momentum_time_based_on_irregular_spacing() -> None:
    # Deliberately uneven gaps: 3s, 7s, 7s.
    series = [
        (_t(0), Decimal("100")),
        (_t(3), Decimal("102")),
        (_t(10), Decimal("105")),
        (_t(17), Decimal("120")),
    ]
    now = _t(17)
    # lookback=10s -> anchor cutoff = 17-10 = 7 -> most recent point with ts<=7 is (3, 102).
    result = momentum(series, lookback_seconds=10, now=now)
    assert result is not None
    assert result == pytest.approx((120 - 102) / 102)


def test_momentum_none_when_no_anchor_old_enough() -> None:
    series = [(_t(0), Decimal("100")), (_t(1), Decimal("101"))]
    # Lookback far exceeds the whole series - no observation is old enough to anchor on.
    assert momentum(series, lookback_seconds=1000, now=_t(1)) is None


def test_momentum_empty_and_single_point_return_none() -> None:
    assert momentum([], lookback_seconds=10, now=_t(0)) is None
    assert momentum([(_t(0), Decimal("100"))], lookback_seconds=10, now=_t(0)) is None


# ---------------------------------------------------------------------------
# Logit-space (bounded) momentum vs. naive percentage return
# ---------------------------------------------------------------------------


def test_logit_momentum_small_near_boundary_large_near_middle() -> None:
    small_move_logit = bounded_momentum(Decimal("0.02"), Decimal("0.01"))
    large_move_logit = bounded_momentum(Decimal("0.60"), Decimal("0.50"))
    # The economically-tiny 0.01->0.02 move should register as smaller in logit space than
    # the substantive 0.50->0.60 move.
    assert abs(small_move_logit) < abs(large_move_logit)


def test_naive_percentage_return_gets_it_backwards() -> None:
    naive_small = simple_return(Decimal("0.01"), Decimal("0.02"))
    naive_large = simple_return(Decimal("0.50"), Decimal("0.60"))
    assert naive_small is not None and naive_large is not None
    # A raw percentage return says the tiny move (+100%) is bigger than the substantive one
    # (+20%) - exactly backwards, which is why bounded_momentum exists.
    assert naive_small > naive_large


# ---------------------------------------------------------------------------
# Realized volatility, hand-computed
# ---------------------------------------------------------------------------


def test_realized_volatility_matches_hand_computation() -> None:
    prices = [Decimal("100"), Decimal("101"), Decimal("99"), Decimal("103")]
    timestamps = [_t(0), _t(1), _t(2), _t(3)]

    log_rets = [
        math.log(101 / 100),
        math.log(99 / 101),
        math.log(103 / 99),
    ]
    mean = sum(log_rets) / 3
    var = sum((r - mean) ** 2 for r in log_rets) / 2
    expected_sd = math.sqrt(var)

    result = realized_volatility(prices, timestamps, window_seconds=3, annualize=False)
    assert result == pytest.approx(expected_sd)

    seconds_per_year = 365.0 * 24.0 * 3600.0
    expected_annualized = expected_sd * math.sqrt(seconds_per_year / 1.0)  # avg dt = 1s
    annualized = realized_volatility(prices, timestamps, window_seconds=3, annualize=True)
    assert annualized == pytest.approx(expected_annualized)


def test_realized_volatility_none_for_too_few_points() -> None:
    assert realized_volatility([Decimal("1")], [_t(0)], window_seconds=10) is None
    assert (
        realized_volatility([Decimal("1"), Decimal("2")], [_t(0), _t(1)], window_seconds=10)
        is None
    )


# ---------------------------------------------------------------------------
# zscore: zero variance must return None, never raise/divide-by-zero
# ---------------------------------------------------------------------------


def test_zscore_zero_variance_returns_none() -> None:
    assert zscore(5.0, mean=5.0, std=0.0) is None
    assert zscore(5.0, mean=5.0, std=None) is None


def test_zscore_normal_case() -> None:
    assert zscore(10.0, mean=5.0, std=2.5) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# EWMA: irregular spacing must decay by elapsed time, not tick count
# ---------------------------------------------------------------------------


def test_ewma_irregular_spacing_decays_by_elapsed_time() -> None:
    close = EWMA(halflife_seconds=10)
    close.update(100.0, _t(0))
    v_close = close.update(101.0, _t(1))  # 1 second later

    far = EWMA(halflife_seconds=10)
    far.update(100.0, _t(0))
    v_far = far.update(101.0, _t(100))  # 100 seconds later

    assert v_close is not None and v_far is not None
    # Two updates 100s apart (ten half-lives) should move almost all the way to the new
    # value; two updates 1s apart should barely move - decay is by time, not tick count.
    assert v_close < v_far
    assert v_far == pytest.approx(101.0, abs=1e-2)
    assert v_close == pytest.approx(100.0, abs=0.2)


def test_ewma_ignores_none_and_nan() -> None:
    e = EWMA(halflife_seconds=5)
    e.update(10.0, _t(0))
    assert e.update(None, _t(1)) == pytest.approx(10.0)
    assert e.update(float("nan"), _t(2)) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Supporting rolling primitives
# ---------------------------------------------------------------------------


def test_rolling_window_basic_stats_and_bounds() -> None:
    w = RollingWindow(maxlen=3)
    assert w.mean() is None
    assert w.std() is None
    w.push(1)
    w.push(2)
    w.push(3)
    assert w.full
    assert w.mean() == pytest.approx(2.0)
    assert w.min() == 1
    assert w.max() == 3
    assert w.last() == 3
    assert w.first() == 1
    w.push(4)  # evicts the 1
    assert w.full
    assert w.first() == 2
    assert w.sum() == pytest.approx(9.0)


def test_rolling_window_ignores_none_and_nan() -> None:
    w = RollingWindow(maxlen=5)
    w.push(None)
    w.push(float("nan"))
    w.push(1.0)
    assert w.len() == 1


def test_rolling_window_zero_variance_std_is_none() -> None:
    w = RollingWindow(maxlen=3)
    w.push(5)
    w.push(5)
    w.push(5)
    assert w.std() is None


def test_rolling_zscore_none_until_enough_points_then_zero_variance_is_none() -> None:
    z = RollingZScore(window=3)
    assert z.update(1.0) is None
    result = z.update(2.0)
    assert result is not None
    z2 = RollingZScore(window=2)
    z2.update(5.0)
    assert z2.update(5.0) is None  # zero variance, not division-by-zero


def test_welford_variance_matches_naive_computation() -> None:
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    wv = WelfordVariance()
    for v in values:
        wv.update(v)
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    assert wv.mean() == pytest.approx(mean)
    assert wv.variance() == pytest.approx(var)
    assert wv.std() == pytest.approx(math.sqrt(var))


def test_welford_variance_single_point_and_none_inputs() -> None:
    wv = WelfordVariance()
    wv.update(None)
    wv.update(float("nan"))
    wv.update(3.0)
    assert wv.mean() == pytest.approx(3.0)
    assert wv.variance() is None  # fewer than 2 points
