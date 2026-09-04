"""Champion/challenger: the gate between "looks good" and "gets more capital."

Reads the ``promotion:`` and ``champion:`` blocks of ``configs/strategies.yaml``.
Nothing here ever promotes to a ``LIVE_*`` status - that is always an explicit human
action taken through :meth:`marketlab.experiments.registry.ExperimentRegistry.transition`
with ``allow_live=True``, never something this engine decides on its own.

**Independence.** 500 trades on one election is close to *one* observation, not five
hundred.  Every gate below clusters observations by ``event_id`` / ``resolution_date`` /
``category`` (per ``promotion.independence.cluster_by``), caps how much of the sample one
cluster may dominate, and runs the confidence interval through a cluster-robust bootstrap
rather than treating every trade as independent.  ``marketlab.analytics.bootstrap``
(owned by another team) is imported defensively: if it is not there yet, a small local
fallback with a ``TODO`` keeps this module importable and testable in isolation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from marketlab.core.strategy import ProbabilityForecast
from marketlab.experiments.identity import ExperimentIdentity
from marketlab.logging import get_logger
from marketlab.storage.state import Experiment, ExperimentStatus

log = get_logger(__name__)

try:
    from marketlab.analytics.bootstrap import (
        BootstrapResult,
        ci_excludes_zero,
        cluster_bootstrap,
        effective_sample_size,
    )
except ImportError:  # pragma: no cover - defensive; the real module exists as of writing.
    # TODO(experiments-team): remove this fallback once marketlab.analytics.bootstrap is
    # guaranteed present. This is a deliberately minimal stand-in, not a replacement: it
    # does not implement the bootstrap-t interval the real module uses, only an honest
    # "cannot estimate with fewer than 2 clusters" degenerate case plus a normal-
    # approximation interval otherwise, so promotion still fails closed rather than open.
    import math
    from dataclasses import dataclass as _dataclass

    @_dataclass(frozen=True)
    class BootstrapResult:  # type: ignore[no-redef]
        point_estimate: float | None
        ci_low: float
        ci_high: float
        confidence: float
        n: int
        n_resamples: int
        seed: int | None
        standard_error: float | None = None
        note: str = ""

    def effective_sample_size(cluster_ids: Sequence[object]) -> float:  # type: ignore[no-redef]
        ids = list(cluster_ids)
        if not ids:
            return 0.0
        counts = Counter(ids)
        total = sum(counts.values())
        sum_sq = sum(c * c for c in counts.values())
        return float(total * total) / float(sum_sq) if sum_sq else 0.0

    def cluster_bootstrap(  # type: ignore[no-redef,misc]
        samples: Sequence[float],
        cluster_ids: Sequence[object],
        statistic_fn: Any = None,
        n_resamples: int = 2000,
        confidence: float = 0.95,
        seed: int | None = None,
    ) -> BootstrapResult:
        arr = list(samples)
        n = len(arr)
        if n == 0:
            return BootstrapResult(None, float("nan"), float("nan"), confidence, 0, n_resamples, seed)
        point = sum(arr) / n
        unique = sorted(set(cluster_ids), key=str)
        if len(unique) < 2:
            return BootstrapResult(
                point, float("-inf"), float("inf"), confidence, n, n_resamples, seed,
                note=f"only {len(unique)} independent cluster(s); cannot estimate a CI",
            )
        # Crude cluster-mean normal approximation - not a real bootstrap, fail-closed only.
        means = []
        for u in unique:
            vals = [v for v, c in zip(arr, cluster_ids, strict=True) if c == u]
            means.append(sum(vals) / len(vals))
        g = len(means)
        mean_of_means = sum(means) / g
        var = sum((m - mean_of_means) ** 2 for m in means) / (g - 1) if g > 1 else 0.0
        se = math.sqrt(var / g) if g > 0 else None
        margin = 1.96 * se if se else 0.0
        return BootstrapResult(
            point, point - margin, point + margin, confidence, n, n_resamples, seed, standard_error=se,
        )

    def ci_excludes_zero(ci: tuple[float, float]) -> bool:  # type: ignore[no-redef]
        lo, hi = ci
        if lo != lo or hi != hi:
            return False
        return lo > 0.0 or hi < 0.0


# ---------------------------------------------------------------------------
# Duck-typed inputs (this module's own contract - no shared frozen model exists yet)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeObservation:
    """One resolved, cost-adjusted trade outcome feeding the promotion gate.

    ``pnl_after_costs`` MUST already reflect a conservative fill model (the
    ``TRADE_THROUGH`` or ``QUEUE`` result from ``ExecutionConfig.limit_fill_model``) -
    never the optimistic ``TOUCH`` model. The cluster fields mirror
    ``promotion.independence.cluster_by`` in ``configs/strategies.yaml``.
    """

    pnl_after_costs: Decimal
    event_id: str = ""
    resolution_date: str = ""
    category: str = ""


@dataclass(frozen=True)
class PromotionMetrics:
    """Everything :meth:`PromotionEngine.evaluate` needs about one PAPER experiment."""

    frequency_class: str  # "fast" | "medium" | "slow_event"
    orders_submitted: int = 0
    resolved_trades: Sequence[TradeObservation] = field(default_factory=tuple)
    forward_days: float = 0.0
    #: Only meaningful for frequency_class == "slow_event".
    resolved_events: int = 0
    #: Must not be "TOUCH" - see docstring on TradeObservation.
    fill_model: str = "TRADE_THROUGH"


@dataclass(frozen=True)
class DegradationMetrics:
    drawdown_pct: Decimal = Decimal(0)
    consecutive_losing_days: int = 0
    calibration_degradation: Decimal = Decimal(0)


@dataclass(frozen=True)
class PromotionDecision:
    #: None means "no status change recommended" (either blocked, or nothing triggered).
    new_status: ExperimentStatus | None
    reasons: list[str]
    blocking: list[str]


@dataclass(frozen=True)
class FrozenChampion:
    experiment_id: str
    identity: ExperimentIdentity
    frozen_at: datetime


@dataclass(frozen=True)
class ChallengeResult:
    replaced: bool
    reasons: list[str]


class FrozenChampionError(Exception):
    """Raised whenever anything tries to retune a frozen champion in place."""


def _cluster_id(t: TradeObservation) -> str:
    return f"{t.event_id}|{t.resolution_date}|{t.category}"


def _max_cluster_weight(cluster_ids: Sequence[object]) -> float:
    if not cluster_ids:
        return 0.0
    counts = Counter(cluster_ids)
    return max(counts.values()) / len(cluster_ids)


class PromotionEngine:
    """Champion/challenger logic driven entirely by ``configs/strategies.yaml``."""

    def __init__(self, strategies_config: dict[str, Any]) -> None:
        self._promotion_cfg: dict[str, Any] = strategies_config.get("promotion", {}) or {}
        self._champion_cfg: dict[str, Any] = strategies_config.get("champion", {}) or {}
        self._independence_cfg: dict[str, Any] = self._promotion_cfg.get("independence", {}) or {}
        self._bootstrap_cfg: dict[str, Any] = self._promotion_cfg.get("bootstrap", {}) or {}
        self._frozen: dict[str, FrozenChampion] = {}

    # ------------------------------------------------------------------
    # PAPER -> QUALIFIED
    # ------------------------------------------------------------------

    def evaluate(
        self,
        experiment: Experiment,
        metrics: PromotionMetrics,
        forecasts: Sequence[ProbabilityForecast] = (),
    ) -> PromotionDecision:
        del forecasts  # reserved for a future calibration gate; not required to qualify yet
        reasons: list[str] = []
        blocking: list[str] = []

        if experiment.status != ExperimentStatus.PAPER:
            blocking.append(
                f"experiment is {experiment.status}, not PAPER; this gate only evaluates "
                f"PAPER -> QUALIFIED"
            )
            return PromotionDecision(None, reasons, blocking)

        if metrics.fill_model.upper() == "TOUCH":
            blocking.append(
                "fill_model is TOUCH; promotion requires a conservative result "
                "(TRADE_THROUGH or QUEUE), never the optimistic touch-fill model"
            )

        freq = metrics.frequency_class
        if freq == "fast":
            cfg = self._promotion_cfg.get("fast", {})
            min_orders = cfg.get("min_orders", 0)
            min_resolved = cfg.get("min_resolved_trades", 0)
            min_days = cfg.get("min_forward_days", 0)
            if metrics.orders_submitted < min_orders:
                blocking.append(f"orders_submitted={metrics.orders_submitted} < required {min_orders}")
            if len(metrics.resolved_trades) < min_resolved:
                blocking.append(
                    f"resolved_trades={len(metrics.resolved_trades)} < required {min_resolved}"
                )
            if metrics.forward_days < min_days:
                blocking.append(f"forward_days={metrics.forward_days} < required {min_days}")
            require_positive = cfg.get("require_positive_after_conservative_costs", True)
        elif freq == "medium":
            cfg = self._promotion_cfg.get("medium", {})
            min_independent = cfg.get("min_independent_trades", 0)
            min_days = cfg.get("min_forward_days", 0)
            ess = effective_sample_size([_cluster_id(t) for t in metrics.resolved_trades])
            if ess < min_independent:
                blocking.append(
                    f"effective independent trades={ess:.1f} < required {min_independent}"
                )
            if metrics.forward_days < min_days:
                blocking.append(f"forward_days={metrics.forward_days} < required {min_days}")
            require_positive = True
        elif freq == "slow_event":
            cfg = self._promotion_cfg.get("slow_event", {})
            min_events = cfg.get("min_resolved_events", 0)
            min_days = cfg.get("min_forward_days", 0)
            if metrics.resolved_events < min_events:
                blocking.append(
                    f"resolved_events={metrics.resolved_events} < required {min_events}"
                )
            if metrics.forward_days < min_days:
                blocking.append(f"forward_days={metrics.forward_days} < required {min_days}")
            require_positive = True
        else:
            blocking.append(f"unknown frequency_class {freq!r}; must be fast/medium/slow_event")
            require_positive = True

        if not metrics.resolved_trades:
            blocking.append("no resolved trades to evaluate independence or expectancy from")
        else:
            pnls = [float(t.pnl_after_costs) for t in metrics.resolved_trades]
            cluster_ids = [_cluster_id(t) for t in metrics.resolved_trades]

            max_weight_cfg = float(self._independence_cfg.get("max_weight_per_cluster", 0.25))
            dominant = _max_cluster_weight(cluster_ids)
            if dominant > max_weight_cfg:
                blocking.append(
                    f"one cluster carries {dominant:.0%} of observations, exceeding "
                    f"promotion.independence.max_weight_per_cluster={max_weight_cfg:.0%} "
                    f"(500 trades on one event is not 500 observations)"
                )

            boot = cluster_bootstrap(
                pnls,
                cluster_ids,
                n_resamples=int(self._bootstrap_cfg.get("resamples", 2000)),
                confidence=float(self._bootstrap_cfg.get("confidence", 0.95)),
                seed=abs(hash(experiment.experiment_id)) % (2**31),
            )
            excludes_zero = ci_excludes_zero((boot.ci_low, boot.ci_high))
            if self._bootstrap_cfg.get("require_ci_excludes_zero", True) and not excludes_zero:
                blocking.append(
                    f"bootstrap CI on expectancy [{boot.ci_low:.4f}, {boot.ci_high:.4f}] "
                    f"does not exclude zero ({boot.note or 'insufficient independent evidence'})"
                )

            expectancy = sum(pnls) / len(pnls)
            if require_positive and expectancy <= 0:
                blocking.append(
                    f"expectancy after conservative costs is not positive ({expectancy:.4f})"
                )

        if blocking:
            return PromotionDecision(None, reasons, blocking)
        reasons.append(f"met every promotion threshold for frequency_class={freq}")
        return PromotionDecision(ExperimentStatus.QUALIFIED, reasons, blocking)

    # ------------------------------------------------------------------
    # champion freeze / anti-retune
    # ------------------------------------------------------------------

    def freeze(self, experiment: Experiment, identity: ExperimentIdentity, at: datetime) -> FrozenChampion:
        """Freeze a champion's version/parameters/feature/data pipeline identity.

        "If it isn't broke, don't fix it": once frozen, this experiment's configuration
        is locked. Any improvement must run as a *challenger* with its own new
        ``experiment_id`` - never a retune of this one.
        """
        if experiment.status != ExperimentStatus.CHAMPION:
            raise ValueError(f"only a CHAMPION may be frozen; {experiment.experiment_id} is {experiment.status}")
        frozen = FrozenChampion(experiment_id=experiment.experiment_id, identity=identity, frozen_at=at)
        self._frozen[experiment.experiment_id] = frozen
        log.info("promotion.champion_frozen", experiment_id=experiment.experiment_id, frozen_at=at.isoformat())
        return frozen

    def is_frozen(self, experiment_id: str) -> bool:
        return experiment_id in self._frozen

    def frozen_identity(self, experiment_id: str) -> FrozenChampion | None:
        return self._frozen.get(experiment_id)

    def attempt_retune(self, experiment_id: str, new_params: dict[str, Any]) -> None:
        """Always refused for a frozen champion - see :meth:`freeze`."""
        del new_params
        if experiment_id in self._frozen:
            raise FrozenChampionError(
                f"{experiment_id} is a frozen champion; in-place retuning is forbidden. "
                f"Launch a challenger with its own experiment id instead."
            )

    # ------------------------------------------------------------------
    # challenger vs champion (forward performance only)
    # ------------------------------------------------------------------

    def challenge(
        self, champion_metrics: PromotionMetrics, challenger_metrics: PromotionMetrics
    ) -> ChallengeResult:
        """A challenger replaces the champion only on forward performance, never backtest.

        Both ``*_metrics`` arguments must already be forward (PAPER/LIVE), out-of-sample
        observations - this function has no way to detect a backtest snuck in here, so
        that discipline is the caller's responsibility (see ``docs/CONTRACTS.md``).
        """
        if not self._champion_cfg.get("challenger_must_win_forward", True):
            return ChallengeResult(False, ["challenger_must_win_forward is disabled; champion retained"])

        champ = self._bootstrap_expectancy(champion_metrics)
        chal = self._bootstrap_expectancy(challenger_metrics)
        if champ is None or chal is None or champ.point_estimate is None or chal.point_estimate is None:
            return ChallengeResult(False, ["insufficient forward data for champion and/or challenger"])
        if not ci_excludes_zero((chal.ci_low, chal.ci_high)):
            return ChallengeResult(False, ["challenger's forward bootstrap CI does not exclude zero"])
        if chal.point_estimate <= champ.point_estimate:
            return ChallengeResult(
                False,
                [
                    f"challenger forward expectancy {chal.point_estimate:.4f} does not exceed "
                    f"champion's {champ.point_estimate:.4f}"
                ],
            )
        return ChallengeResult(
            True,
            [
                f"challenger forward expectancy {chal.point_estimate:.4f} beat champion's "
                f"{champ.point_estimate:.4f} with a CI excluding zero"
            ],
        )

    def _bootstrap_expectancy(self, metrics: PromotionMetrics) -> BootstrapResult | None:
        if not metrics.resolved_trades:
            return None
        pnls = [float(t.pnl_after_costs) for t in metrics.resolved_trades]
        cluster_ids = [_cluster_id(t) for t in metrics.resolved_trades]
        return cluster_bootstrap(
            pnls,
            cluster_ids,
            n_resamples=int(self._bootstrap_cfg.get("resamples", 2000)),
            confidence=float(self._bootstrap_cfg.get("confidence", 0.95)),
            seed=0,
        )

    # ------------------------------------------------------------------
    # degradation
    # ------------------------------------------------------------------

    def check_degradation(
        self, champion: Experiment, recent_metrics: DegradationMetrics
    ) -> PromotionDecision:
        """Demote a champion on drawdown / losing-streak / calibration triggers.

        Past success grants no immunity: every check here looks only at
        ``recent_metrics``, never at the champion's historical track record.
        """
        cfg = self._champion_cfg.get("demote_on", {}) or {}
        triggered: list[str] = []

        dd_trigger = Decimal(str(cfg.get("drawdown_pct", 1)))
        if recent_metrics.drawdown_pct >= dd_trigger:
            triggered.append(f"drawdown {recent_metrics.drawdown_pct} >= trigger {dd_trigger}")

        losing_days_trigger = cfg.get("consecutive_losing_days")
        if losing_days_trigger is not None and recent_metrics.consecutive_losing_days >= losing_days_trigger:
            triggered.append(
                f"consecutive_losing_days={recent_metrics.consecutive_losing_days} >= "
                f"trigger {losing_days_trigger}"
            )

        calib_trigger = Decimal(str(cfg.get("calibration_degradation", 1)))
        if recent_metrics.calibration_degradation >= calib_trigger:
            triggered.append(
                f"calibration_degradation {recent_metrics.calibration_degradation} >= "
                f"trigger {calib_trigger}"
            )

        if not triggered:
            return PromotionDecision(None, ["no degradation trigger hit"], [])
        if champion.status != ExperimentStatus.CHAMPION:
            triggered.append(f"note: experiment is {champion.status}, not CHAMPION")
        return PromotionDecision(ExperimentStatus.DEGRADED, triggered, [])

    # ------------------------------------------------------------------
    # live promotion: always refused here
    # ------------------------------------------------------------------

    def promote_to_live(self, *_args: Any, **_kwargs: Any) -> PromotionDecision:
        """Always refuses. Live promotion is exclusively a human action.

        See :meth:`marketlab.experiments.registry.ExperimentRegistry.transition`, which
        requires ``allow_live=True`` explicitly passed by a human-triggered call path -
        never by this engine or any automated scheduler.
        """
        return PromotionDecision(
            None,
            [],
            [
                "live promotion is never automatic; a human must explicitly call "
                "ExperimentRegistry.transition(..., allow_live=True)"
            ],
        )
