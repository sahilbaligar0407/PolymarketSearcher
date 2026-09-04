"""AI-analyst probability strategy. Evidence class F - requires a local LLM.

Pipeline, strictly, per market: build a point-in-time evidence bundle
(:mod:`marketlab.ai.retrieval`) -> call the injected
:class:`~marketlab.ai.provider.LLMProvider` with the schema from
:mod:`marketlab.ai.schemas` -> **validate with :mod:`marketlab.ai.validator`** -> only a
valid, non-abstaining assessment ever becomes a forecast, and only a forecast that clears
both ``entry_edge`` and ``min_confidence`` ever becomes a trade.

**Invalid or malformed model output means abstain - never repair it.** This module never
touches a field of a failed :class:`~marketlab.ai.validator.ValidationResult`; a failure
short-circuits straight to an abstaining :class:`~marketlab.core.strategy.ProbabilityForecast`
and zero intents, exactly mirroring :mod:`marketlab.ai.validator`'s own stated contract.

**Injection, not construction-time wiring.** Every strategy is instantiated uniformly by
the runner as ``cls(strategy_id, experiment_id, ctx, params)`` (see
``configs/strategies.yaml``'s ``class: module:ClassName`` dispatch), so there is no room
for an extra required constructor argument. The :class:`~marketlab.ai.provider.LLMProvider`
and the (optional) retrieval store are therefore supplied through ``params["llm_provider"]``
/ ``params["retrieval_store"]`` - the same "narrow context, collaborators via params"
pattern used by :mod:`marketlab.strategies.cross_venue` for match records. When
``llm_provider`` is absent, this module defaults to
:class:`~marketlab.ai.provider.DisabledProvider`, which always abstains - "the provider is
disabled -> the strategy runs and emits nothing, cleanly" falls out of that default for
free rather than needing a separate code path.

**Sync handlers, async inference.** :class:`~marketlab.core.strategy.Strategy`'s
``on_*`` handlers are synchronous, but retrieval and LLM calls are ``async``. Handlers
therefore only ever *queue* a canonical_id for inference (event-triggered: high-impact
news, a filing, a large market move, a tracked trader action - mirroring
``configs/default.yaml: ai.triggers`` - plus a scheduled refresh on a timer, per
``ai.scheduled_refresh_seconds``); the actual awaited work happens in
:meth:`process_pending`, which the runner calls from its own event loop and which honours
``max_concurrent_inferences`` via a semaphore. This is the one clean way to reconcile a
synchronous strategy-handler contract with an inherently asynchronous local-model call.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from marketlab.ai.provider import DisabledProvider, LLMProvider
from marketlab.ai.retrieval import RetrievalService
from marketlab.ai.schemas import MarketAssessment, json_schema_for
from marketlab.ai.validator import assessment_to_forecast, parse_llm_output, validate_assessment
from marketlab.core.events import (
    BookUpdateEvent,
    FilingEvent,
    NewsEvent,
    SocialEvent,
    TimerEvent,
    TraderActionEvent,
)
from marketlab.core.instruments import ONE, Side
from marketlab.core.orders import Action, OrderType
from marketlab.core.strategy import ProbabilityForecast
from marketlab.strategies.base import BaseStrategy, clamp_probability

DEFAULT_LARGE_MOVE_THRESHOLD = Decimal("0.05")
DEFAULT_SCHEDULED_REFRESH_SECONDS = 1800.0
DEFAULT_MAX_CONCURRENT_INFERENCES = 2


class NewsProbabilityStrategy(BaseStrategy):
    """Trades a Kalshi contract against a validated local-LLM probability assessment."""

    name = "news_probability"
    version = "1.0.0"
    evidence_class = "F"

    def __init__(self, strategy_id: str, experiment_id: str, ctx: Any, params: dict | None = None) -> None:
        super().__init__(strategy_id, experiment_id, ctx, params)
        self._provider: LLMProvider = self.param("llm_provider", None) or DisabledProvider()
        self._retrieval = RetrievalService(self.param("retrieval_store", None), self.ctx.clock)
        self._pending_inferences: set[str] = set()
        self._last_mid: dict[str, Decimal] = {}
        self._semaphore = asyncio.Semaphore(
            int(self.param("max_concurrent_inferences", DEFAULT_MAX_CONCURRENT_INFERENCES))
        )

    # ------------------------------------------------------------------ event triggers

    def on_news(self, event: NewsEvent) -> None:
        if self.is_high_impact_news(event):
            self._queue_universe("high_impact_news")

    def on_filing(self, event: FilingEvent) -> None:
        del event
        self._queue_universe("sec_filing")

    def on_trader_action(self, event: TraderActionEvent) -> None:
        del event
        self._queue_universe("tracked_trader_action")

    def on_social(self, event: SocialEvent) -> None:
        del event
        # A public statement alone is not "high impact news" by this module's own
        # heuristic (that's news-source-class specific); it may still be worth a refresh
        # if it moves the tracked market, so leave triggering to on_book_update's
        # large-move check rather than queuing on every social post unconditionally.

    def on_book_update(self, event: BookUpdateEvent) -> None:
        mid = event.book.mid
        if mid is None:
            return
        canonical_id = event.canonical_id
        previous = self._last_mid.get(canonical_id)
        self._last_mid[canonical_id] = mid
        if previous is None:
            return
        threshold = Decimal(str(self.param("large_move_threshold", DEFAULT_LARGE_MOVE_THRESHOLD)))
        if abs(mid - previous) >= threshold:
            self._queue_if_eligible(canonical_id)

    def on_timer(self, event: TimerEvent) -> None:
        del event
        refresh_seconds = float(self.param("scheduled_refresh_seconds", DEFAULT_SCHEDULED_REFRESH_SECONDS))
        for market in self.ctx.markets():
            cid = market.canonical_id
            if not self.on_cooldown(f"ai_refresh:{cid}", refresh_seconds):
                self._queue_if_eligible(cid)

    # ------------------------------------------------------------------ queueing

    def _queue_universe(self, trigger: str) -> None:
        del trigger
        for market in self.ctx.markets():
            self._queue_if_eligible(market.canonical_id)

    def _queue_if_eligible(self, canonical_id: str) -> None:
        if self.should_skip(canonical_id) is not None:
            return
        min_spacing = float(self.param("min_inference_spacing_seconds", 60.0))
        if self.on_cooldown(f"ai_infer:{canonical_id}", min_spacing):
            return
        self._pending_inferences.add(canonical_id)

    # ------------------------------------------------------------------ async inference

    async def process_pending(self) -> int:
        """Drain the trigger queue, running each inference under the concurrency cap.

        Called by the runner's own event loop (never by a synchronous Strategy handler).
        Returns the number of markets processed.
        """
        pending, self._pending_inferences = self._pending_inferences, set()
        if not pending:
            return 0
        await asyncio.gather(*(self._run_with_semaphore(cid) for cid in pending))
        return len(pending)

    async def _run_with_semaphore(self, canonical_id: str) -> None:
        async with self._semaphore:
            await self.evaluate_market(canonical_id)

    async def evaluate_market(self, canonical_id: str) -> None:
        market = self.ctx.market(canonical_id)
        if market is None or self.should_skip(canonical_id) is not None:
            return
        book = self.ctx.book(canonical_id)
        assert book is not None
        self.mark_fired(f"ai_refresh:{canonical_id}")
        self.mark_fired(f"ai_infer:{canonical_id}")

        decision_time = self.now()
        bundle = await self._retrieval.build_bundle(
            canonical_id, decision_time, market=market, market_price=book.mid
        )
        prompt = (
            f"Assess the probability that market {market.canonical_id!r} "
            f"({market.title!r}) resolves YES, using only the evidence below.\n\n"
            f"Resolution rules: {market.resolution_rules or market.description}\n\n"
            f"{bundle.to_prompt_context()}"
        )
        try:
            response = await self._provider.generate(
                prompt,
                schema=json_schema_for(MarketAssessment),
                system=(
                    "You are a careful probability forecaster. Cite only the given "
                    "evidence ids. Abstain if the evidence is insufficient or the "
                    "resolution rules are ambiguous."
                ),
                timeout=float(self.param("timeout_seconds", 120.0)),
            )
        except Exception as exc:  # noqa: BLE001 - a provider failure must never raise here
            self._record_abstain(canonical_id, book.mid, [f"provider_error: {exc}"])
            return

        assessment = parse_llm_output(response.parsed if response.parsed is not None else response.text or None)
        result = validate_assessment(assessment, market, bundle, decision_time)
        if not result.valid or result.assessment_or_none is None:
            self._record_abstain(canonical_id, book.mid, result.failures, response=response)
            return

        validated = result.assessment_or_none
        forecast = assessment_to_forecast(validated, self.strategy_id, self.experiment_id, book.mid)
        forecast.features["llm_model_id"] = response.model
        forecast.features["prompt_hash"] = response.prompt_hash
        self.forecast(forecast)

        if validated.abstain:
            return
        min_confidence = Decimal(str(self.param("min_confidence", "0.5")))
        if validated.confidence < min_confidence:
            return

        self._maybe_trade(canonical_id, market, book, validated.p_yes, response)

    def _record_abstain(
        self, canonical_id: str, market_probability: Decimal | None, failures: list[str], response: Any = None
    ) -> None:
        features: dict[str, Any] = {"validation_failures": failures}
        if response is not None:
            features["llm_model_id"] = response.model
            features["prompt_hash"] = response.prompt_hash
        self.forecast(
            ProbabilityForecast(
                strategy_id=self.strategy_id,
                experiment_id=self.experiment_id,
                canonical_id=canonical_id,
                as_of=self.now(),
                p_yes=Decimal("0.5"),
                abstain=True,
                market_probability=market_probability,
                features=features,
                rationale="Abstaining: " + "; ".join(failures) if failures else "Abstaining: no usable assessment.",
            )
        )

    def _maybe_trade(
        self, canonical_id: str, market: Any, book: Any, model_probability: Decimal, response: Any
    ) -> None:
        price_yes = self.executable_price(book, Side.YES, Action.BUY)
        price_no = self.executable_price(book, Side.NO, Action.BUY)
        candidates: list[tuple[Side, Decimal, Decimal]] = []
        if price_yes is not None:
            candidates.append((Side.YES, price_yes, self.edge_after_costs(model_probability, price_yes, market, Side.YES)))
        if price_no is not None:
            no_p = clamp_probability(ONE - model_probability)
            candidates.append((Side.NO, price_no, self.edge_after_costs(no_p, price_no, market, Side.NO)))
        if not candidates:
            return
        side, price, edge = max(candidates, key=lambda c: c[2])

        entry_edge = Decimal(str(self.param("entry_edge", "0.05")))
        if edge <= entry_edge:
            return

        intent = self.make_intent(
            canonical_id=canonical_id,
            side=side,
            action=Action.BUY,
            quantity=self.sensible_quantity(price),
            order_type=OrderType.LIMIT,
            limit_price=price,
            rationale=(
                f"AI analyst: model_p={model_probability} (model={response.model}) vs "
                f"executable {side.value} price {price}; edge {edge} clears entry_edge "
                f"{entry_edge}."
            ),
            features={
                "llm_model_id": response.model,
                "prompt_hash": response.prompt_hash,
                "model_probability": float(model_probability),
                "executable_price": float(price),
                "side": side.value,
            },
            model_probability=model_probability,
            expected_edge=edge,
        )
        if self.emit_if_profitable(intent):
            self.mark_fired(canonical_id)


__all__ = ["NewsProbabilityStrategy"]
