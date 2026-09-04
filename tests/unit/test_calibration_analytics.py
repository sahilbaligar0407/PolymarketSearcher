"""Tests for marketlab.analytics.calibration."""

from __future__ import annotations

import numpy as np

from marketlab.analytics.calibration import (
    brier_decomposition,
    expected_calibration_error,
    improvement_vs_market,
)


def _perfectly_calibrated_forecasts(rng: np.random.Generator, n: int) -> tuple[list[float], list[int]]:
    """Forecasts whose bin means exactly match observed frequencies by construction."""
    forecasts: list[float] = []
    outcomes: list[int] = []
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        # For probability p, generate many trials whose empirical win rate is ~p.
        for _ in range(n):
            forecasts.append(p)
            outcomes.append(1 if rng.random() < p else 0)
    return forecasts, outcomes


def test_perfectly_calibrated_forecaster_has_low_ece():
    rng = np.random.default_rng(0)
    forecasts, outcomes = _perfectly_calibrated_forecasts(rng, n=5000)
    ece = expected_calibration_error(forecasts, outcomes, bins=10)
    assert ece is not None
    assert ece < 0.02  # large n per bucket -> observed frequency converges to p


def test_overconfident_forecaster_has_high_reliability_term():
    rng = np.random.default_rng(1)
    n = 2000
    # The forecaster always says 0.95 or 0.05 (extreme confidence), but the true rate
    # backing those calls is only 70/30 -- systematically overconfident.
    forecasts = [0.95] * n + [0.05] * n
    outcomes = [1 if rng.random() < 0.70 else 0 for _ in range(n)] + [
        1 if rng.random() < 0.30 else 0 for _ in range(n)
    ]
    decomp = brier_decomposition(forecasts, outcomes, bins=10)
    assert decomp.reliability is not None
    # A well-calibrated forecaster in this same setup would have reliability near 0;
    # overconfidence should push it up substantially.
    assert decomp.reliability > 0.05


def test_well_calibrated_forecaster_has_low_reliability_for_comparison():
    rng = np.random.default_rng(2)
    n = 2000
    forecasts = [0.70] * n + [0.30] * n
    outcomes = [1 if rng.random() < 0.70 else 0 for _ in range(n)] + [
        1 if rng.random() < 0.30 else 0 for _ in range(n)
    ]
    decomp = brier_decomposition(forecasts, outcomes, bins=10)
    assert decomp.reliability is not None
    assert decomp.reliability < 0.01


def test_improvement_vs_market_negative_when_model_is_noisier_than_market():
    rng = np.random.default_rng(3)
    n = 3000
    market = rng.uniform(0.05, 0.95, size=n)
    outcomes = [1 if rng.random() < p else 0 for p in market]
    # Model = market price plus noise: strictly worse than just reading the price.
    noise = rng.normal(0, 0.15, size=n)
    model = np.clip(market + noise, 0.001, 0.999)

    bss = improvement_vs_market(model.tolist(), market.tolist(), outcomes)
    assert bss is not None
    assert bss < 0.0


def test_improvement_vs_market_positive_when_model_beats_market():
    rng = np.random.default_rng(4)
    n = 3000
    true_p = rng.uniform(0.05, 0.95, size=n)
    outcomes = [1 if rng.random() < p else 0 for p in true_p]
    # Market is a noisy version of the truth; model IS the truth.
    market_noise = rng.normal(0, 0.20, size=n)
    market = np.clip(true_p + market_noise, 0.001, 0.999)

    bss = improvement_vs_market(true_p.tolist(), market.tolist(), outcomes)
    assert bss is not None
    assert bss > 0.0


def test_murphy_decomposition_reconstructs_brier_score():
    # The Murphy identity reconstructs the Brier score exactly when every forecast in a
    # bin shares the identical value (the classical discrete-forecast-category case).
    # Binning a continuum of forecast values into ranges leaves a small residual (within-
    # bin forecast variance) that the three named terms don't capture -- so we test with
    # forecasts confined to five levels, each centered in its own bin, rather than a
    # continuum, to isolate the identity itself.
    rng = np.random.default_rng(5)
    n = 400
    levels = (0.1, 0.3, 0.5, 0.7, 0.9)
    forecasts: list[float] = []
    outcomes: list[int] = []
    for p in levels:
        for _ in range(n):
            forecasts.append(p)
            outcomes.append(1 if rng.random() < p else 0)
    decomp = brier_decomposition(forecasts, outcomes, bins=5)
    assert decomp.brier_score is not None
    reconstructed = decomp.uncertainty - decomp.resolution + decomp.reliability
    assert abs(reconstructed - decomp.brier_score) < 1e-9


def test_empty_inputs_return_none():
    assert expected_calibration_error([], []) is None
    decomp = brier_decomposition([], [])
    assert decomp.brier_score is None
    assert decomp.reliability is None
    assert decomp.resolution is None
    assert improvement_vs_market([], [], []) is None
