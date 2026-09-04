"""Portfolio-wide probability quality, scored separately from P&L.

This is the analytics-layer sibling of :mod:`marketlab.ai.calibration` (which scores one
LLM forecaster's raw output). This module scores *any* stream of
:class:`~marketlab.core.strategy.ProbabilityForecast`-shaped ``(forecast, outcome)`` pairs
— across strategies, models, or the market itself treated as a forecaster — using plain
floats and numpy, and adds the Murphy decomposition, AUC discrimination and reliability
rows that the daily report needs. Import from ``marketlab.ai.calibration`` rather than
duplicating its per-forecaster tracker; this module is for portfolio-level rollups.

A model can be perfectly calibrated (reliability == 0) and still useless (resolution == 0,
meaning its forecasts never differ from the base rate) — both terms are always reported
together, never just one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

import numpy as np

Number = Decimal | float


def _floats(xs: Sequence[Number]) -> np.ndarray:
    return np.asarray([float(x) for x in xs], dtype=float)


def _validate(forecasts: Sequence[Number], outcomes: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    if len(forecasts) != len(outcomes):
        raise ValueError("forecasts and outcomes must be the same length")
    f = _floats(forecasts)
    o = np.asarray([int(x) for x in outcomes], dtype=float)
    if len(o) and not np.all((o == 0.0) | (o == 1.0)):
        raise ValueError("outcomes must be 0 or 1")
    return f, o


# ---------------------------------------------------------------------------
# Murphy (Brier) decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrierDecomposition:
    n: int
    brier_score: float | None
    #: Calibration term: how far each bin's mean forecast is from its observed frequency.
    #: Lower is better; 0 is perfect calibration.
    reliability: float | None
    #: Discrimination term: how far each bin's observed frequency is from the base rate.
    #: Higher is better; 0 means the forecaster never says anything but the base rate.
    resolution: float | None
    #: Irreducible variance of the outcome itself: base_rate * (1 - base_rate).
    uncertainty: float | None


def _bin_edges(bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, bins + 1)


def _assign_bins(forecasts: np.ndarray, bins: int) -> np.ndarray:
    edges = _bin_edges(bins)
    idx = np.digitize(forecasts, edges[1:-1], right=False)
    return np.clip(idx, 0, bins - 1)


def brier_decomposition(
    forecasts: Sequence[Number], outcomes: Sequence[int], bins: int = 10
) -> BrierDecomposition:
    """Murphy (1973) decomposition: ``Brier = uncertainty - resolution + reliability``.

    Reported together always — see module docstring for why either term alone is
    misleading.
    """
    f, o = _validate(forecasts, outcomes)
    n = f.size
    if n == 0:
        return BrierDecomposition(n=0, brier_score=None, reliability=None, resolution=None, uncertainty=None)

    brier = float(np.mean((f - o) ** 2))
    base_rate = float(np.mean(o))
    uncertainty = base_rate * (1.0 - base_rate)

    bin_idx = _assign_bins(f, bins)
    reliability = 0.0
    resolution = 0.0
    for k in range(bins):
        mask = bin_idx == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        f_bar_k = float(np.mean(f[mask]))
        o_bar_k = float(np.mean(o[mask]))
        weight = n_k / n
        reliability += weight * (f_bar_k - o_bar_k) ** 2
        resolution += weight * (o_bar_k - base_rate) ** 2

    return BrierDecomposition(
        n=n,
        brier_score=brier,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
    )


# ---------------------------------------------------------------------------
# Calibration curve / ECE / MCE
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationBinRow:
    lo: float
    hi: float
    predicted_mean: float | None
    observed_frequency: float | None
    count: int


def calibration_curve(
    forecasts: Sequence[Number], outcomes: Sequence[int], bins: int = 10
) -> list[CalibrationBinRow]:
    f, o = _validate(forecasts, outcomes)
    edges = _bin_edges(bins)
    rows: list[CalibrationBinRow] = []
    if f.size == 0:
        return [
            CalibrationBinRow(float(edges[i]), float(edges[i + 1]), None, None, 0)
            for i in range(bins)
        ]
    bin_idx = _assign_bins(f, bins)
    for k in range(bins):
        mask = bin_idx == k
        n_k = int(mask.sum())
        if n_k == 0:
            rows.append(CalibrationBinRow(float(edges[k]), float(edges[k + 1]), None, None, 0))
            continue
        rows.append(
            CalibrationBinRow(
                lo=float(edges[k]),
                hi=float(edges[k + 1]),
                predicted_mean=float(np.mean(f[mask])),
                observed_frequency=float(np.mean(o[mask])),
                count=n_k,
            )
        )
    return rows


def expected_calibration_error(
    forecasts: Sequence[Number], outcomes: Sequence[int], bins: int = 10
) -> float | None:
    """Count-weighted mean absolute gap between predicted and observed, per bin."""
    if not forecasts:
        return None
    rows = calibration_curve(forecasts, outcomes, bins)
    n = len(forecasts)
    total = 0.0
    for r in rows:
        if r.count == 0 or r.predicted_mean is None or r.observed_frequency is None:
            continue
        total += (r.count / n) * abs(r.predicted_mean - r.observed_frequency)
    return total


def maximum_calibration_error(
    forecasts: Sequence[Number], outcomes: Sequence[int], bins: int = 10
) -> float | None:
    """Worst single-bin gap between predicted and observed. Unweighted, on purpose — a
    small but badly miscalibrated bin (e.g. a rarely-hit extreme-confidence bucket) is
    exactly the kind of thing an average can hide."""
    if not forecasts:
        return None
    rows = calibration_curve(forecasts, outcomes, bins)
    gaps = [
        abs(r.predicted_mean - r.observed_frequency)
        for r in rows
        if r.count > 0 and r.predicted_mean is not None and r.observed_frequency is not None
    ]
    return max(gaps) if gaps else None


# ---------------------------------------------------------------------------
# Improvement vs market (Brier skill score)
# ---------------------------------------------------------------------------


def improvement_vs_market(
    model_forecasts: Sequence[Number],
    market_probabilities: Sequence[Number],
    outcomes: Sequence[int],
) -> float | None:
    """Brier skill score of the model against the market-price baseline.

    ``BSS = 1 - BS_model / BS_market``. Positive means the model beats simply reading the
    price; 0 means it adds nothing; **negative means the model is worse than the market**
    and the report must say so plainly. ``None`` only when there is no data or the market
    baseline itself is a perfect forecaster (``BS_market == 0``, division undefined).
    """
    if not model_forecasts or not market_probabilities or not outcomes:
        return None
    mf, o = _validate(model_forecasts, outcomes)
    mkt, _ = _validate(market_probabilities, outcomes)
    if mf.size != mkt.size:
        raise ValueError("model_forecasts and market_probabilities must be the same length")
    bs_model = float(np.mean((mf - o) ** 2))
    bs_market = float(np.mean((mkt - o) ** 2))
    if bs_market == 0.0:
        return None
    return 1.0 - (bs_model / bs_market)


# ---------------------------------------------------------------------------
# Log loss / AUC
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogLossSummary:
    n: int
    mean_log_loss: float | None
    #: Log loss of always predicting the base rate — a naive baseline for comparison.
    base_rate_log_loss: float | None


def log_loss_summary(
    forecasts: Sequence[Number], outcomes: Sequence[int], eps: float = 1e-6
) -> LogLossSummary:
    f, o = _validate(forecasts, outcomes)
    if f.size == 0:
        return LogLossSummary(n=0, mean_log_loss=None, base_rate_log_loss=None)
    p = np.clip(f, eps, 1.0 - eps)
    losses = -(o * np.log(p) + (1.0 - o) * np.log(1.0 - p))
    base_rate = float(np.mean(o))
    br = np.clip(base_rate, eps, 1.0 - eps)
    base_losses = -(o * np.log(br) + (1.0 - o) * np.log(1.0 - br))
    return LogLossSummary(
        n=f.size,
        mean_log_loss=float(np.mean(losses)),
        base_rate_log_loss=float(np.mean(base_losses)),
    )


def discrimination_auc(forecasts: Sequence[Number], outcomes: Sequence[int]) -> float | None:
    """Area under the ROC curve: can the forecaster rank eventual winners above losers?

    Computed via the Mann-Whitney U statistic (rank-sum), so no extra dependency beyond
    numpy is needed. ``None`` when there are no observations of one of the two outcome
    classes — AUC is undefined without both a positive and a negative example.
    """
    f, o = _validate(forecasts, outcomes)
    n_pos = int(np.sum(o == 1.0))
    n_neg = int(np.sum(o == 0.0))
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rankdata(f)
    sum_ranks_pos = float(np.sum(ranks[o == 1.0]))
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties get the mean rank -- avoids a scipy.stats dependency
    beyond what's already used elsewhere in this package."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# Reliability diagram rows (terminal rendering)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReliabilityRow:
    bin_label: str
    predicted_mean: float | None
    observed_frequency: float | None
    count: int
    #: ASCII bar for terminal rendering, length proportional to `count`.
    bar: str


def reliability_diagram_rows(
    forecasts: Sequence[Number],
    outcomes: Sequence[int],
    bins: int = 10,
    bar_width: int = 20,
) -> list[ReliabilityRow]:
    rows = calibration_curve(forecasts, outcomes, bins)
    max_count = max((r.count for r in rows), default=0)
    out: list[ReliabilityRow] = []
    for r in rows:
        bar_len = int(bar_width * r.count / max_count) if max_count else 0
        out.append(
            ReliabilityRow(
                bin_label=f"{r.lo:.1f}-{r.hi:.1f}",
                predicted_mean=r.predicted_mean,
                observed_frequency=r.observed_frequency,
                count=r.count,
                bar="#" * bar_len,
            )
        )
    return out
