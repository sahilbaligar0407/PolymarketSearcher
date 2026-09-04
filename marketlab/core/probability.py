"""Probability utilities shared by strategies, analytics and the AI layer.

Kept deliberately small and dependency-free so that it is trivially testable and safe to
import from anywhere.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_EVEN, Decimal

from marketlab.core.instruments import PROB_QUANTUM, Side

EPS = Decimal("0.000001")


def clamp(p: Decimal, lo: Decimal = EPS, hi: Decimal = Decimal(1) - EPS) -> Decimal:
    return max(lo, min(hi, p))


def brier_score(forecast: Decimal, outcome: int) -> Decimal:
    """Squared error of a probabilistic forecast. Lower is better; 0.25 is a coin flip."""
    return (forecast - Decimal(outcome)) ** 2


def log_loss(forecast: Decimal, outcome: int) -> Decimal:
    p = clamp(forecast)
    return Decimal(str(-math.log(float(p if outcome == 1 else Decimal(1) - p))))


def american_to_probability(odds: int) -> Decimal:
    """Convert American sportsbook odds to an implied (vig-inclusive) probability."""
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    if odds > 0:
        p = Decimal(100) / Decimal(odds + 100)
    else:
        p = Decimal(-odds) / Decimal(-odds + 100)
    return p.quantize(PROB_QUANTUM, rounding=ROUND_HALF_EVEN)


def decimal_odds_to_probability(odds: Decimal | float) -> Decimal:
    d = Decimal(str(odds))
    if d <= 1:
        raise ValueError("decimal odds must exceed 1.0")
    return (Decimal(1) / d).quantize(PROB_QUANTUM, rounding=ROUND_HALF_EVEN)


def remove_vig(probabilities: list[Decimal]) -> list[Decimal]:
    """Normalize a set of vig-inclusive book probabilities so they sum to 1.

    Never compare a raw American-odds probability to a prediction-market price; the book's
    margin makes the sum exceed 1.
    """
    total = sum(probabilities, Decimal(0))
    if total <= 0:
        raise ValueError("probabilities must sum to a positive number")
    return [(p / total).quantize(PROB_QUANTUM, rounding=ROUND_HALF_EVEN) for p in probabilities]


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def gbm_touch_probability(
    spot: float, strike: float, sigma_annual: float, seconds_remaining: float
) -> float:
    """P(spot > strike at expiry) under driftless geometric Brownian motion.

    The workhorse for BTC threshold contracts: 'will BTC be above X at time T'.  Note this
    is the *terminal* probability, not the probability of touching X at any point, which
    is roughly double for an at-the-money barrier.
    """
    if seconds_remaining <= 0:
        return 1.0 if spot > strike else 0.0
    if spot <= 0 or strike <= 0 or sigma_annual <= 0:
        raise ValueError("spot, strike and sigma must be positive")
    t = seconds_remaining / (365.0 * 24.0 * 3600.0)
    d2 = (math.log(spot / strike) - 0.5 * sigma_annual**2 * t) / (sigma_annual * math.sqrt(t))
    return normal_cdf(d2)


def gbm_barrier_touch_probability(
    spot: float, strike: float, sigma_annual: float, seconds_remaining: float
) -> float:
    """P(spot touches strike at any time before expiry), reflection-principle approximation."""
    if seconds_remaining <= 0:
        return 1.0 if (spot >= strike) else 0.0
    terminal = gbm_touch_probability(spot, strike, sigma_annual, seconds_remaining)
    if spot < strike:
        return min(1.0, 2.0 * terminal)
    return min(1.0, 2.0 * (1.0 - gbm_touch_probability(spot, strike, sigma_annual, seconds_remaining)) + (2 * terminal - 1.0))


def kelly_fraction(p: Decimal, price: Decimal, side: Side = Side.YES) -> Decimal:
    """Full-Kelly stake fraction for a binary contract.

    Reported as a research metric only.  Real sizing stays hard-capped until calibration
    is demonstrated, because overconfident probabilities make Kelly dangerously aggressive.
    """
    price = clamp(price)
    if side is Side.NO:
        p = Decimal(1) - p
        price = Decimal(1) - price
    b = (Decimal(1) - price) / price  # net odds received on a win
    q = Decimal(1) - p
    f = (b * p - q) / b
    return max(Decimal(0), f)


def expected_edge(
    model_probability: Decimal,
    executable_price: Decimal,
    fee: Decimal,
    slippage_buffer: Decimal,
    uncertainty_buffer: Decimal,
    side: Side = Side.YES,
) -> Decimal:
    """Edge a strategy may actually claim, after every cost the PRD requires subtracting."""
    p = model_probability if side is Side.YES else Decimal(1) - model_probability
    return p - executable_price - fee - slippage_buffer - uncertainty_buffer
