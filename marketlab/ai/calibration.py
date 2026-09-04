"""Scoring the local model's forecasts against reality -- never against P&L.

A model can be well calibrated and traded badly, or poorly calibrated and still traded
profitably by luck; :class:`~marketlab.core.strategy.ProbabilityForecast` is recorded
whether or not it produces a trade for exactly this reason. Everything in this module
operates on (forecast, realized outcome) pairs, sourced from
:class:`~marketlab.core.events.SettlementEvent` -- never from a model's own confidence,
and never from unrealized mark-to-market.

The one number every other metric here is subordinate to is
:meth:`CalibrationTracker.improvement_vs_market`: whether the model's Brier score beats
the Brier score of simply reading the market price. A negative number means the model
adds value; zero or positive means it does not, and that must be reported exactly as
measured, with no rounding-in-its-favor -- "the model is worse than the market" is a
legitimate, expected finding for an early or poorly-prompted model, not a bug to hide.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from marketlab.core.probability import brier_score
from marketlab.core.probability import log_loss as _log_loss_one
from marketlab.core.strategy import ProbabilityForecast


@dataclass(frozen=True)
class ScoredForecast:
    """One forecast paired with its realized, venue-settled outcome."""

    forecast: ProbabilityForecast
    #: 1 = resolved YES, 0 = resolved NO. Never inferred -- comes from a SettlementEvent.
    outcome: int


@dataclass(frozen=True)
class CalibrationBin:
    """One bucket of a reliability diagram."""

    lo: Decimal
    hi: Decimal
    count: int
    mean_predicted: Decimal | None
    actual_frequency: Decimal | None


@dataclass(frozen=True)
class CalibrationReport:
    count: int
    brier_score: Decimal | None
    log_loss: Decimal | None
    mean_predicted_probability: Decimal | None
    actual_frequency: Decimal | None
    calibration_curve: list[CalibrationBin] = field(default_factory=list)
    #: model Brier - market Brier. Negative = model beats the market baseline.
    improvement_vs_market: Decimal | None = None
    market_brier_score: Decimal | None = None
    #: Expected Calibration Error, bin-count-weighted.
    ece: Decimal | None = None


def _bin_edges(bins: int) -> list[Decimal]:
    return [Decimal(i) / Decimal(bins) for i in range(bins + 1)]


def calibration_error(pairs: Sequence[tuple[Decimal, int]], bins: int = 10) -> Decimal:
    """Expected Calibration Error over raw ``(p_yes, outcome)`` pairs.

    Bin-count-weighted mean absolute gap between each bin's mean predicted probability
    and its actual realized frequency. Returns ``Decimal(0)`` for an empty input rather
    than raising, since "no data yet" is a normal state early in a strategy's life.
    """
    if not pairs:
        return Decimal(0)
    edges = _bin_edges(bins)
    n = len(pairs)
    total = Decimal(0)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = [(p, o) for p, o in pairs if (lo <= p < hi) or (i == bins - 1 and p == hi)]
        if not in_bin:
            continue
        mean_pred = sum((p for p, _ in in_bin), Decimal(0)) / Decimal(len(in_bin))
        freq = Decimal(sum(o for _, o in in_bin)) / Decimal(len(in_bin))
        total += Decimal(len(in_bin)) * abs(mean_pred - freq)
    return total / Decimal(n)


def shrink_toward_market(p_model: Decimal, p_market: Decimal, weight: Decimal) -> Decimal:
    """Blend the model's estimate toward the market price.

    ``weight`` in ``[0, 1]`` is how much of the model's view to keep; ``1 - weight`` is
    how much of the market's to keep. **``weight`` must be derived from measured
    historical improvement** (e.g. :meth:`CalibrationTracker.improvement_vs_market` on
    held-out, out-of-sample forecasts) -- never assumed or hand-tuned. Every call site in
    this codebase should default to ``weight=Decimal(0)`` (trust the market entirely)
    until a strategy has actually earned a higher weight empirically.
    """
    if not (Decimal(0) <= weight <= Decimal(1)):
        raise ValueError(f"weight must be in [0, 1]: {weight}")
    return weight * p_model + (Decimal(1) - weight) * p_market


class CalibrationTracker:
    """Accumulates (forecast, outcome) pairs and reports calibration metrics.

    Abstained forecasts (``forecast.abstain is True``) are excluded from every metric:
    an abstain is a deliberate non-answer, and scoring it as if it were a 0.5 (or
    whatever placeholder value it might carry) would misrepresent the model's actual
    forecasting performance in either direction.
    """

    def __init__(self) -> None:
        self._scored: list[ScoredForecast] = []

    def record(self, forecast: ProbabilityForecast, outcome: int) -> None:
        if outcome not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1, got {outcome!r}")
        self._scored.append(ScoredForecast(forecast=forecast, outcome=outcome))

    @property
    def count(self) -> int:
        return len(self._scored)

    def _scoreable(self) -> list[ScoredForecast]:
        return [s for s in self._scored if not s.forecast.abstain]

    def brier(self) -> Decimal | None:
        scored = self._scoreable()
        if not scored:
            return None
        total = sum((brier_score(s.forecast.p_yes, s.outcome) for s in scored), Decimal(0))
        return total / Decimal(len(scored))

    def market_brier(self) -> Decimal | None:
        scored = [s for s in self._scoreable() if s.forecast.market_probability is not None]
        if not scored:
            return None
        total = sum(
            (brier_score(s.forecast.market_probability, s.outcome) for s in scored),  # type: ignore[arg-type]
            Decimal(0),
        )
        return total / Decimal(len(scored))

    def log_loss(self) -> Decimal | None:
        scored = self._scoreable()
        if not scored:
            return None
        total = sum((_log_loss_one(s.forecast.p_yes, s.outcome) for s in scored), Decimal(0))
        return total / Decimal(len(scored))

    def mean_predicted_probability(self) -> Decimal | None:
        scored = self._scoreable()
        if not scored:
            return None
        return sum((s.forecast.p_yes for s in scored), Decimal(0)) / Decimal(len(scored))

    def actual_frequency(self) -> Decimal | None:
        scored = self._scoreable()
        if not scored:
            return None
        return Decimal(sum(s.outcome for s in scored)) / Decimal(len(scored))

    def improvement_vs_market(self) -> Decimal | None:
        """Model Brier minus market Brier, over forecasts where a market price was
        recorded. Negative = model beats the market; positive = model is worse than
        just reading the price. Reported as computed -- see module docstring.
        """
        scored = [s for s in self._scoreable() if s.forecast.market_probability is not None]
        if not scored:
            return None
        n = Decimal(len(scored))
        model_total = sum((brier_score(s.forecast.p_yes, s.outcome) for s in scored), Decimal(0))
        market_total = sum(
            (brier_score(s.forecast.market_probability, s.outcome) for s in scored),  # type: ignore[arg-type]
            Decimal(0),
        )
        return model_total / n - market_total / n

    def calibration_curve(self, bins: int = 10) -> list[CalibrationBin]:
        scored = self._scoreable()
        edges = _bin_edges(bins)
        out: list[CalibrationBin] = []
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            in_bin = [
                s for s in scored if (lo <= s.forecast.p_yes < hi) or (i == bins - 1 and s.forecast.p_yes == hi)
            ]
            if not in_bin:
                out.append(CalibrationBin(lo=lo, hi=hi, count=0, mean_predicted=None, actual_frequency=None))
                continue
            mean_pred = sum((s.forecast.p_yes for s in in_bin), Decimal(0)) / Decimal(len(in_bin))
            freq = Decimal(sum(s.outcome for s in in_bin)) / Decimal(len(in_bin))
            out.append(CalibrationBin(lo=lo, hi=hi, count=len(in_bin), mean_predicted=mean_pred, actual_frequency=freq))
        return out

    def calibration_error(self, bins: int = 10) -> Decimal | None:
        scored = self._scoreable()
        if not scored:
            return None
        return calibration_error([(s.forecast.p_yes, s.outcome) for s in scored], bins)

    def report(self, bins: int = 10) -> CalibrationReport:
        return CalibrationReport(
            count=self.count,
            brier_score=self.brier(),
            log_loss=self.log_loss(),
            mean_predicted_probability=self.mean_predicted_probability(),
            actual_frequency=self.actual_frequency(),
            calibration_curve=self.calibration_curve(bins),
            improvement_vs_market=self.improvement_vs_market(),
            market_brier_score=self.market_brier(),
            ece=self.calibration_error(bins),
        )
