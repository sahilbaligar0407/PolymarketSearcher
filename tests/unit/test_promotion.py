"""Tests for marketlab.experiments.promotion."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from marketlab.experiments.identity import ExperimentIdentity
from marketlab.experiments.promotion import (
    DegradationMetrics,
    FrozenChampionError,
    PromotionEngine,
    PromotionMetrics,
    TradeObservation,
)
from marketlab.storage.state import Experiment, ExperimentStatus

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _promotion_config() -> dict:
    return {
        "promotion": {
            "fast": {
                "min_orders": 200,
                "min_resolved_trades": 100,
                "min_forward_days": 14,
                "require_positive_after_conservative_costs": True,
            },
            "medium": {"min_independent_trades": 75, "min_forward_days": 30},
            "slow_event": {"min_resolved_events": 30, "min_forward_days": 30},
            "independence": {
                "cluster_by": ["event_id", "resolution_date", "category"],
                "max_weight_per_cluster": 0.25,
            },
            "bootstrap": {"resamples": 300, "confidence": 0.95, "require_ci_excludes_zero": True},
        },
        "champion": {
            "freeze_on_promotion": True,
            "challenger_must_win_forward": True,
            "demote_on": {
                "drawdown_pct": Decimal("0.25"),
                "consecutive_losing_days": 10,
                "calibration_degradation": Decimal("0.05"),
            },
        },
    }


def _experiment(status: ExperimentStatus, experiment_id: str = "exp_1") -> Experiment:
    return Experiment(
        experiment_id=experiment_id,
        strategy_name="momentum",
        strategy_version="1.0.0",
        market_universe="btc_1h",
        venue="kalshi",
        status=status,
        created_at=T0,
    )


def _homogeneous_trades(n: int, n_clusters: int, pnl: Decimal = Decimal("0.05")) -> list[TradeObservation]:
    """``n`` trades split across ``n_clusters`` distinct events, identical pnl.

    Identical pnl makes the bootstrap standard error exactly zero, so the resulting
    confidence interval collapses to a single point at the true mean - deterministic and
    exactly on one side of zero, with no dependence on a random seed.
    """
    return [
        TradeObservation(pnl_after_costs=pnl, event_id=f"evt_{i % n_clusters}", resolution_date="2026-01-01", category="crypto")
        for i in range(n)
    ]


def test_three_trades_three_wins_is_not_promoted() -> None:
    engine = PromotionEngine(_promotion_config())
    experiment = _experiment(ExperimentStatus.PAPER)
    metrics = PromotionMetrics(
        frequency_class="fast",
        orders_submitted=3,
        resolved_trades=[
            TradeObservation(Decimal("0.10"), event_id="evt_1"),
            TradeObservation(Decimal("0.20"), event_id="evt_2"),
            TradeObservation(Decimal("0.15"), event_id="evt_3"),
        ],
        forward_days=1,
        fill_model="TRADE_THROUGH",
    )
    decision = engine.evaluate(experiment, metrics)
    assert decision.new_status is None
    assert any("min_orders" in b or "orders_submitted" in b for b in decision.blocking)
    assert any("resolved_trades" in b for b in decision.blocking)
    assert any("forward_days" in b for b in decision.blocking)


def test_meeting_every_threshold_is_promoted() -> None:
    engine = PromotionEngine(_promotion_config())
    experiment = _experiment(ExperimentStatus.PAPER)
    metrics = PromotionMetrics(
        frequency_class="fast",
        orders_submitted=200,
        resolved_trades=_homogeneous_trades(n=100, n_clusters=100),
        forward_days=14,
        fill_model="TRADE_THROUGH",
    )
    decision = engine.evaluate(experiment, metrics)
    assert decision.blocking == []
    assert decision.new_status is ExperimentStatus.QUALIFIED


def test_touch_fill_model_blocks_promotion() -> None:
    engine = PromotionEngine(_promotion_config())
    experiment = _experiment(ExperimentStatus.PAPER)
    metrics = PromotionMetrics(
        frequency_class="fast",
        orders_submitted=200,
        resolved_trades=_homogeneous_trades(n=100, n_clusters=100),
        forward_days=14,
        fill_model="TOUCH",
    )
    decision = engine.evaluate(experiment, metrics)
    assert decision.new_status is None
    assert any("TOUCH" in b for b in decision.blocking)


def test_500_trades_one_cluster_fails_independence() -> None:
    engine = PromotionEngine(_promotion_config())
    experiment = _experiment(ExperimentStatus.PAPER)
    metrics = PromotionMetrics(
        frequency_class="fast",
        orders_submitted=500,
        resolved_trades=_homogeneous_trades(n=500, n_clusters=1),
        forward_days=30,
        fill_model="TRADE_THROUGH",
    )
    decision = engine.evaluate(experiment, metrics)
    assert decision.new_status is None
    assert any("cluster" in b.lower() for b in decision.blocking)


def test_500_trades_across_100_clusters_passes() -> None:
    engine = PromotionEngine(_promotion_config())
    experiment = _experiment(ExperimentStatus.PAPER)
    metrics = PromotionMetrics(
        frequency_class="fast",
        orders_submitted=500,
        resolved_trades=_homogeneous_trades(n=500, n_clusters=100),
        forward_days=30,
        fill_model="TRADE_THROUGH",
    )
    decision = engine.evaluate(experiment, metrics)
    assert decision.blocking == []
    assert decision.new_status is ExperimentStatus.QUALIFIED


def test_evaluate_only_applies_to_paper_experiments() -> None:
    engine = PromotionEngine(_promotion_config())
    experiment = _experiment(ExperimentStatus.CHAMPION)
    metrics = PromotionMetrics(frequency_class="fast", resolved_trades=_homogeneous_trades(100, 100))
    decision = engine.evaluate(experiment, metrics)
    assert decision.new_status is None
    assert any("not PAPER" in b for b in decision.blocking)


def test_degradation_triggers_demote_a_champion() -> None:
    engine = PromotionEngine(_promotion_config())
    champion = _experiment(ExperimentStatus.CHAMPION)
    triggered = engine.check_degradation(
        champion, DegradationMetrics(drawdown_pct=Decimal("0.30"), consecutive_losing_days=2, calibration_degradation=Decimal("0.0"))
    )
    assert triggered.new_status is ExperimentStatus.DEGRADED
    assert any("drawdown" in r for r in triggered.reasons)

    not_triggered = engine.check_degradation(
        champion, DegradationMetrics(drawdown_pct=Decimal("0.01"), consecutive_losing_days=1, calibration_degradation=Decimal("0.0"))
    )
    assert not_triggered.new_status is None


def test_champion_parameters_are_frozen_and_retune_rejected() -> None:
    engine = PromotionEngine(_promotion_config())
    champion = _experiment(ExperimentStatus.CHAMPION)
    identity = ExperimentIdentity(
        strategy_name="momentum",
        strategy_version="1.0.0",
        git_commit="abc123",
        parameter_hash="deadbeef0000",
        market_universe="btc_1h",
        venue="kalshi",
        data_version="data.2026.09.04",
        execution_model_version="exec.v1",
        feature_version="feat.v1",
        llm_model_id=None,
        prompt_hash=None,
        start_timestamp=T0,
        starting_bankroll=Decimal("50.00"),
    )
    frozen = engine.freeze(champion, identity, at=T0)
    assert frozen.experiment_id == champion.experiment_id
    assert engine.is_frozen(champion.experiment_id)

    with pytest.raises(FrozenChampionError):
        engine.attempt_retune(champion.experiment_id, {"threshold": Decimal("0.05")})


def test_freeze_requires_champion_status() -> None:
    engine = PromotionEngine(_promotion_config())
    not_champion = _experiment(ExperimentStatus.PAPER)
    identity = ExperimentIdentity(
        strategy_name="momentum", strategy_version="1.0.0", git_commit="abc", parameter_hash="x",
        market_universe="btc_1h", venue="kalshi", data_version="d", execution_model_version="e",
        feature_version="f", llm_model_id=None, prompt_hash=None, start_timestamp=T0,
        starting_bankroll=Decimal("50.00"),
    )
    with pytest.raises(ValueError):
        engine.freeze(not_champion, identity, at=T0)


def test_live_promotion_is_always_refused() -> None:
    engine = PromotionEngine(_promotion_config())
    decision = engine.promote_to_live()
    assert decision.new_status is None
    assert decision.blocking
    assert any("human" in b.lower() for b in decision.blocking)
