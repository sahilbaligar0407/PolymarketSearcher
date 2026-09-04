"""Uncertainty quantification, so three lucky trades never become a promotion.

Every resampling routine here takes an explicit ``seed`` and is fully deterministic for a
fixed seed — the promotion gate must produce the same verdict on rerun.

**Confidence interval method.** All four bootstrap functions below report a CI as
``point_estimate +/- t_crit(df) * bootstrap_standard_error`` (a "bootstrap-t" / normal
approximation using the resampled standard error) rather than the raw 2.5/97.5 percentiles
of the resampled distribution. This is a deliberate choice: a plain percentile interval
built from resampling *only the observed values* can never straddle zero when every
observed value shares the same sign — e.g. three winning trades can only ever resample to
other winning-trade combinations, so a percentile CI on "3 trades, 3 wins" would report
100% confidence the true edge is positive, which is precisely the false confidence this
module exists to prevent. Scaling the bootstrap SE by a Student-t critical value (with
degrees of freedom equal to the number of independent units minus one) inflates the
interval correctly as the number of independent observations shrinks, and converges to the
same answer as the percentile method for well-behaved, larger samples.

When there are too few independent units to estimate a standard error at all (n<2, or a
single cluster), the honest answer is "we cannot bound this" — reported as an infinite
interval rather than a falsely narrow one.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

Statistic = Callable[[np.ndarray], float]


def _default_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr))


@dataclass(frozen=True)
class BootstrapResult:
    #: Statistic computed on the original (full, unresampled) sample.
    point_estimate: float | None
    ci_low: float
    ci_high: float
    confidence: float
    #: Number of raw observations (not clusters/blocks) that went in.
    n: int
    n_resamples: int
    seed: int | None
    #: Standard deviation of the resampled statistic. None when it could not be estimated.
    standard_error: float | None = None
    note: str = ""


def _empty_result(n_resamples: int, confidence: float, seed: int | None, note: str) -> BootstrapResult:
    return BootstrapResult(
        point_estimate=None,
        ci_low=float("nan"),
        ci_high=float("nan"),
        confidence=confidence,
        n=0,
        n_resamples=n_resamples,
        seed=seed,
        standard_error=None,
        note=note,
    )


def _t_interval(point: float, se: float, df: int, confidence: float) -> tuple[float, float]:
    if df < 1:
        return (float("-inf"), float("inf"))
    if se == 0.0:
        return (point, point)
    alpha = 1.0 - confidence
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    margin = t_crit * se
    return (point - margin, point + margin)


def bootstrap_statistic(
    samples: Sequence[float],
    statistic_fn: Statistic = _default_mean,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Generic percentile-resample / t-scaled-SE bootstrap for an arbitrary statistic.

    Deterministic for a fixed ``seed`` (uses ``numpy.random.default_rng(seed)``).
    """
    arr = np.asarray(list(samples), dtype=float)
    n = arr.size
    if n == 0:
        return _empty_result(n_resamples, confidence, seed, "empty sample")

    point = float(statistic_fn(arr))
    if n == 1:
        return BootstrapResult(
            point_estimate=point,
            ci_low=float("-inf"),
            ci_high=float("inf"),
            confidence=confidence,
            n=1,
            n_resamples=n_resamples,
            seed=seed,
            standard_error=None,
            note="a single observation carries no estimate of variance",
        )

    rng = np.random.default_rng(seed)
    resampled = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled[i] = statistic_fn(arr[idx])
    se = float(np.std(resampled, ddof=1))
    ci_low, ci_high = _t_interval(point, se, df=n - 1, confidence=confidence)
    return BootstrapResult(
        point_estimate=point,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        n=n,
        n_resamples=n_resamples,
        seed=seed,
        standard_error=se,
    )


def bootstrap_mean(
    samples: Sequence[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    return bootstrap_statistic(samples, _default_mean, n_resamples, confidence, seed)


def block_bootstrap(
    samples: Sequence[float],
    block_size: int,
    statistic_fn: Statistic = _default_mean,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Moving-block bootstrap for an autocorrelated per-trade series.

    Resamples contiguous blocks of ``block_size`` consecutive observations (with
    replacement over all valid start positions) and concatenates them back up to the
    original length, preserving local autocorrelation that an i.i.d. resample would
    destroy.
    """
    arr = np.asarray(list(samples), dtype=float)
    n = arr.size
    if n == 0:
        return _empty_result(n_resamples, confidence, seed, "empty sample")
    if n == 1:
        return bootstrap_statistic(samples, statistic_fn, n_resamples, confidence, seed)

    block_size = max(1, min(block_size, n))
    point = float(statistic_fn(arr))
    n_blocks_needed = -(-n // block_size)  # ceil
    max_start = n - block_size + 1
    rng = np.random.default_rng(seed)
    resampled = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        starts = rng.integers(0, max_start, size=n_blocks_needed)
        pieces = [arr[s : s + block_size] for s in starts]
        resample = np.concatenate(pieces)[:n]
        resampled[i] = statistic_fn(resample)
    se = float(np.std(resampled, ddof=1))
    ci_low, ci_high = _t_interval(point, se, df=n - 1, confidence=confidence)
    return BootstrapResult(
        point_estimate=point,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        n=n,
        n_resamples=n_resamples,
        seed=seed,
        standard_error=se,
    )


def cluster_bootstrap(
    samples: Sequence[float],
    cluster_ids: Sequence[object],
    statistic_fn: Statistic = _default_mean,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Resample whole clusters, not individual trades.

    500 trades on one election event are close to *one* observation, not five hundred:
    cluster by ``event_id`` / ``resolution_date`` / ``category`` per
    ``configs/strategies.yaml: promotion.independence``, and pass the per-trade cluster
    label here. With a single cluster there is no way to estimate between-cluster
    variance at all, so this returns an explicit, unbounded interval rather than the
    zero-width interval a naive resample would produce (resampling one cluster with
    replacement can only ever reproduce that same cluster).
    """
    arr = np.asarray(list(samples), dtype=float)
    ids = list(cluster_ids)
    n = arr.size
    if n == 0:
        return _empty_result(n_resamples, confidence, seed, "empty sample")
    if len(ids) != n:
        raise ValueError("samples and cluster_ids must be the same length")

    point = float(statistic_fn(arr))
    unique = sorted(set(ids), key=lambda x: str(x))
    g = len(unique)
    if g < 2:
        return BootstrapResult(
            point_estimate=point,
            ci_low=float("-inf"),
            ci_high=float("inf"),
            confidence=confidence,
            n=n,
            n_resamples=n_resamples,
            seed=seed,
            standard_error=None,
            note=(
                f"only {g} independent cluster(s) present; a cluster-robust interval "
                "requires at least 2 and cannot be estimated from this data"
            ),
        )

    groups: dict[object, np.ndarray] = {u: arr[[i for i, c in enumerate(ids) if c == u]] for u in unique}
    rng = np.random.default_rng(seed)
    resampled = np.empty(n_resamples, dtype=float)
    unique_arr = np.array(unique, dtype=object)
    for i in range(n_resamples):
        chosen = rng.choice(unique_arr, size=g, replace=True)
        resample = np.concatenate([groups[c] for c in chosen])
        resampled[i] = statistic_fn(resample)
    se = float(np.std(resampled, ddof=1))
    ci_low, ci_high = _t_interval(point, se, df=g - 1, confidence=confidence)
    return BootstrapResult(
        point_estimate=point,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        n=n,
        n_resamples=n_resamples,
        seed=seed,
        standard_error=se,
        note=f"{g} independent clusters",
    )


def effective_sample_size(cluster_ids: Sequence[object]) -> float:
    """How many independent observations a clustered sample is really worth.

    Uses the Kish design-effect formula ``(sum n_i)^2 / sum(n_i^2)``, which is exact under
    the conservative assumption that observations within a cluster are fully correlated
    (worst case). It degenerates to 1.0 when everything is one cluster, and to the number
    of clusters when clusters are equal-sized — matching the intuition in
    ``configs/strategies.yaml``'s ``promotion.independence`` block.
    """
    ids = list(cluster_ids)
    if not ids:
        return 0.0
    counts = Counter(ids)
    total = sum(counts.values())
    sum_sq = sum(c * c for c in counts.values())
    if sum_sq == 0:
        return 0.0
    return float(total * total) / float(sum_sq)


def ci_excludes_zero(ci: tuple[float, float]) -> bool:
    """True only when the whole interval is strictly on one side of zero.

    Used directly by the promotion gate (``configs/strategies.yaml: promotion.bootstrap.
    require_ci_excludes_zero``): an interval that straddles zero, or that is unbounded
    because too few independent clusters exist, must not promote anything.
    """
    lo, hi = ci
    if lo != lo or hi != hi:  # NaN check without importing math for one line
        return False
    return lo > 0.0 or hi < 0.0


def minimum_detectable_effect(
    n: int,
    std: float,
    confidence: float = 0.95,
    power: float = 0.80,
) -> float | None:
    """Smallest true mean effect this sample size could reliably detect.

    Standard two-sample-free power formula for a one-sample mean test:
    ``MDE = (z_(alpha/2) + z_power) * std / sqrt(n)``. Returns ``None`` for ``n <= 0``.
    Useful for telling a researcher "your 40-trade backtest could not have detected
    anything smaller than an 8% edge" before they draw any conclusion from it.
    """
    if n <= 0 or std < 0:
        return None
    if n == 0:
        return None
    alpha = 1.0 - confidence
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))
    return (z_alpha + z_power) * std / (n**0.5)
