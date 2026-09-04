"""Combines forecasts from QUALIFIED strategies only. Disabled by default.

``configs/strategies.yaml`` keeps ``ensemble.enabled: false`` and ``universes: []`` until
QUALIFIED components exist - enabling it early would launder unproven per-strategy signals
into a single number that looks more authoritative than any of its inputs. This module
enforces the same discipline in code: :class:`EnsembleStrategy` **refuses to construct
itself from non-QUALIFIED inputs**, raising ``ValueError`` at ``__init__`` time rather than
silently degrading.

**Weighting.** Each component is described by a :class:`ComponentSpec` carrying pre-scored
metadata (forward expected value, sample reliability, calibration, consistency, drawdown,
correlation to other components, execution sensitivity, recent degradation, liquidity) -
this module does not itself compute promotion statistics (that lives in the promotion/
champion-challenger pipeline described in ``configs/strategies.yaml``); it only turns
already-scored components into weights. Weights are non-negative, capped per strategy
(``max_weight_per_strategy``), rise with forward EV / sample reliability / calibration /
consistency, and fall with drawdown / correlation / execution sensitivity / recent
degradation / low liquidity.

**Decision rule.** A position is taken only when (a) at least two *independent* signal
families agree on direction, (b) the resulting ensemble expected edge clears fees plus
slippage plus the uncertainty buffer, and (c) liquidity is sufficient. "Independent"
means distinct ``family`` labels (e.g. evidence class, or a coarser hand-assigned grouping)
- five correlated momentum variants agreeing is one opinion, not five.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from marketlab.core.instruments import ONE, Side
from marketlab.core.orders import Action, OrderType
from marketlab.core.strategy import ProbabilityForecast
from marketlab.strategies.base import BaseStrategy, clamp_probability

DEFAULT_MAX_WEIGHT_PER_STRATEGY = Decimal("0.40")
DEFAULT_MIN_AGREEING_FAMILIES = 2
DEFAULT_MIN_OPEN_INTEREST = Decimal("1")

# Tunable, honestly-unfitted coefficients for the raw weight score. Positive terms pull
# weight up, negative terms pull it down; the whole expression is clamped at 0 before
# per-strategy capping and normalization.
_W_FORWARD_EV = Decimal("1.0")
_W_SAMPLE_RELIABILITY = Decimal("0.5")
_W_CALIBRATION = Decimal("0.5")
_W_CONSISTENCY = Decimal("0.5")
_P_DRAWDOWN = Decimal("1.0")
_P_CORRELATION = Decimal("0.5")
_P_EXECUTION_SENSITIVITY = Decimal("0.5")
_P_DEGRADATION = Decimal("0.5")
_P_ILLIQUIDITY = Decimal("0.3")


@dataclass(frozen=True)
class ComponentSpec:
    """Pre-scored metadata for one QUALIFIED strategy component, for one market."""

    strategy_id: str
    family: str
    qualified: bool
    forecast: ProbabilityForecast | None
    forward_ev: Decimal = Decimal(0)
    sample_reliability: Decimal = Decimal(0)
    calibration_score: Decimal = Decimal(0)
    consistency_score: Decimal = Decimal(0)
    drawdown_pct: Decimal = Decimal(0)
    correlation_to_others: Decimal = Decimal(0)
    execution_sensitivity: Decimal = Decimal(0)
    recent_degradation: Decimal = Decimal(0)
    liquidity_score: Decimal = Decimal(1)


def _raw_score(c: ComponentSpec) -> Decimal:
    positive = (
        c.forward_ev * _W_FORWARD_EV
        + c.sample_reliability * _W_SAMPLE_RELIABILITY
        + c.calibration_score * _W_CALIBRATION
        + c.consistency_score * _W_CONSISTENCY
    )
    negative = (
        c.drawdown_pct * _P_DRAWDOWN
        + c.correlation_to_others * _P_CORRELATION
        + c.execution_sensitivity * _P_EXECUTION_SENSITIVITY
        + c.recent_degradation * _P_DEGRADATION
        + (Decimal(1) - c.liquidity_score) * _P_ILLIQUIDITY
    )
    return max(Decimal(0), positive - negative)


class EnsembleStrategy(BaseStrategy):
    """Combines QUALIFIED component forecasts into one Kalshi trade, or none."""

    name = "ensemble"
    version = "1.0.0"
    evidence_class = "B"

    def __init__(self, strategy_id: str, experiment_id: str, ctx: Any, params: dict | None = None) -> None:
        super().__init__(strategy_id, experiment_id, ctx, params)
        components: list[ComponentSpec] = list(self.param("components", []) or [])
        unqualified = [c.strategy_id for c in components if not c.qualified]
        if unqualified:
            raise ValueError(
                f"EnsembleStrategy refuses non-QUALIFIED components: {unqualified}. "
                "Only strategies that have already cleared promotion may feed the ensemble."
            )
        self._components = components
        self._max_weight_per_strategy = Decimal(
            str(self.param("max_weight_per_strategy", DEFAULT_MAX_WEIGHT_PER_STRATEGY))
        )

    def set_components(self, components: list[ComponentSpec]) -> None:
        """Replace the component set with a fresh (still QUALIFIED-only) snapshot."""
        unqualified = [c.strategy_id for c in components if not c.qualified]
        if unqualified:
            raise ValueError(f"EnsembleStrategy refuses non-QUALIFIED components: {unqualified}")
        self._components = components

    def weights(self) -> dict[str, Decimal]:
        """Normalized, per-strategy-capped weights across the current component set."""
        raw = {c.strategy_id: _raw_score(c) for c in self._components}
        capped = {sid: min(score, self._max_weight_per_strategy) for sid, score in raw.items()}
        total = sum(capped.values(), Decimal(0))
        if total <= 0:
            return dict.fromkeys(capped, Decimal(0))
        return {sid: score / total for sid, score in capped.items()}

    def combine(self, canonical_id: str) -> None:
        """Evaluate the decision rule for one market and emit at most one intent."""
        relevant = [
            c
            for c in self._components
            if c.forecast is not None and c.forecast.canonical_id == canonical_id and not c.forecast.abstain
        ]
        if not relevant:
            return

        families_yes = {c.family for c in relevant if c.forecast.p_yes > Decimal("0.5")}  # type: ignore[union-attr]
        families_no = {c.family for c in relevant if c.forecast.p_yes <= Decimal("0.5")}  # type: ignore[union-attr]
        majority_families = families_yes if len(families_yes) >= len(families_no) else families_no
        min_agreeing = int(self.param("min_agreeing_families", DEFAULT_MIN_AGREEING_FAMILIES))
        if len(majority_families) < min_agreeing:
            return

        majority_yes = families_yes >= families_no
        contributing = [c for c in relevant if c.family in majority_families]
        weights = self.weights()
        weighted_sum = sum((weights.get(c.strategy_id, Decimal(0)) * c.forecast.p_yes for c in contributing), Decimal(0))  # type: ignore[union-attr]
        weight_total = sum((weights.get(c.strategy_id, Decimal(0)) for c in contributing), Decimal(0))
        if weight_total <= 0:
            return
        ensemble_p = clamp_probability(weighted_sum / weight_total)

        if self.should_skip(canonical_id) is not None:
            return
        market = self.ctx.market(canonical_id)
        book = self.ctx.book(canonical_id)
        assert market is not None and book is not None
        if market.open_interest < Decimal(str(self.param("min_open_interest", DEFAULT_MIN_OPEN_INTEREST))):
            return

        side = Side.YES if majority_yes else Side.NO
        model_p = ensemble_p if side is Side.YES else clamp_probability(ONE - ensemble_p)
        price = self.executable_price(book, side, Action.BUY)
        if price is None:
            return
        edge = self.edge_after_costs(model_p, price, market, side)
        min_edge = Decimal(str(self.param("min_edge", "0.02")))
        if edge < min_edge:
            return
        cooldown = float(self.param("cooldown_seconds", 60.0))
        if self.on_cooldown(canonical_id, cooldown):
            return

        intent = self.make_intent(
            canonical_id=canonical_id,
            side=side,
            action=Action.BUY,
            quantity=self.sensible_quantity(price),
            order_type=OrderType.LIMIT,
            limit_price=price,
            rationale=(
                f"Ensemble: {len(majority_families)} independent families "
                f"({sorted(majority_families)}) agree on {side.value}; ensemble_p="
                f"{ensemble_p}, edge={edge} clears min_edge={min_edge}."
            ),
            features={
                "families": sorted(majority_families),
                "ensemble_probability": float(ensemble_p),
                "component_strategy_ids": [c.strategy_id for c in contributing],
                "weights": {sid: float(w) for sid, w in weights.items()},
                "side": side.value,
            },
            model_probability=model_p,
            expected_edge=edge,
        )
        if self.emit_if_profitable(intent):
            self.mark_fired(canonical_id)


__all__ = ["ComponentSpec", "EnsembleStrategy"]
