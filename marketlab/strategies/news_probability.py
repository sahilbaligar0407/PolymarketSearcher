"""AI-analyst probability strategy. Evidence class F - requires an LLM tier.

Pipeline, strictly, per market: build a point-in-time evidence bundle
(:mod:`marketlab.ai.retrieval`) -> call the injected
:class:`~marketlab.ai.provider.LLMProvider` with the schema from
:mod:`marketlab.ai.schemas` -> **validate with :mod:`marketlab.ai.validator`** -> only a
valid, non-abstaining assessment ever becomes a forecast, and only a forecast that clears
both ``entry_edge`` and ``min_confidence`` ever becomes a trade.

**Invalid or malformed model output means abstain - never repair it.** A validation
failure short-circuits straight to an abstaining
:class:`~marketlab.core.strategy.ProbabilityForecast` and zero intents.

**Code owns identity, the model owns judgement.** The model is asked only for the
judgement fields of :class:`~marketlab.ai.schemas.MarketAssessment`; ``market_id``,
``as_of`` and ``information_cutoff`` are stamped by code when the model leaves them out. A
model that *names a different market* still fails validation.

**AI stack arms.** The runner injects ``llm_provider`` (the primary tier) and, for an
escalating arm, ``escalation_provider`` (the budget-capped OpenAI second opinion; see
:mod:`marketlab.ai.stack`). Escalation happens only when the primary assessment would
actually trade, and the trade goes ahead only if the second opinion independently clears
the same edge on the same side. Every variant of an arm shares one inference per market
through the injected ``assessment_cache``. With no ``llm_provider`` the strategy defaults
to :class:`~marketlab.ai.provider.DisabledProvider`, which always abstains.

**Sync handlers, async inference.** Handlers only *queue* a canonical_id (high-impact
news, a filing, a large market move, a tracked trader action, or the scheduled refresh).
The awaited work happens in :meth:`process_pending`, which the runner starts in the
background via :meth:`start_background_inference` so a slow model never blocks dispatch.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from marketlab.ai.provider import DisabledProvider, LLMProvider, LLMResponse
from marketlab.ai.retrieval import EvidenceBundle, RetrievalService
from marketlab.ai.schemas import MarketAssessment, json_schema_for
from marketlab.ai.validator import assessment_to_forecast, parse_llm_output, validate_assessment
from marketlab.core.events import (
    BookUpdateEvent,
    FilingEvent,
    MarketUpdateEvent,
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

#: Fields code stamps itself; the model is never asked for them.
_CODE_OWNED_FIELDS = ("market_id", "as_of", "information_cutoff")

_SYSTEM_PROMPT = (
    "You are a careful, calibrated probability forecaster for prediction markets. "
    "Estimate the probability the market resolves YES from the resolution rules, the "
    "dates given, the evidence listed, and well-established background knowledge. "
    "Cite evidence only by the ids shown. In question_interpretation restate the exact "
    "question including its key entity, threshold and date. Set abstain=true when you "
    "genuinely cannot estimate, and set resolution_rule_warning=true only if the rules "
    "themselves are ambiguous. Confidence is how sure you are of your probability "
    "estimate, not of the outcome."
)


def judgement_schema() -> dict[str, Any]:
    """MarketAssessment's JSON schema minus the code-owned identity fields."""
    schema = json_schema_for(MarketAssessment)
    props = dict(schema.get("properties", {}))
    for name in _CODE_OWNED_FIELDS:
        props.pop(name, None)
    schema["properties"] = props
    schema["required"] = [r for r in schema.get("required", []) if r not in _CODE_OWNED_FIELDS]
    return schema


@dataclass(frozen=True)
class _Assessed:
    """One inference outcome, shareable across variants through the assessment cache."""

    decision_time: datetime
    bundle: EvidenceBundle | None
    response: LLMResponse | None
    error: str = ""


def _evidence_count(bundle: EvidenceBundle) -> int:
    return len(bundle.news) + len(bundle.social) + len(bundle.filings) + len(bundle.trader_activity)


class NewsProbabilityStrategy(BaseStrategy):
    """Trades a Kalshi contract against a validated LLM probability assessment."""

    name = "news_probability"
    version = "2.1.0"
    evidence_class = "F"

    def __init__(self, strategy_id: str, experiment_id: str, ctx: Any, params: dict | None = None) -> None:
        super().__init__(strategy_id, experiment_id, ctx, params)
        self._provider: LLMProvider = self.param("llm_provider", None) or DisabledProvider()
        self._escalation: LLMProvider | None = self.param("escalation_provider", None)
        self._cache = self.param("assessment_cache", None)
        self._retrieval = RetrievalService(self.param("retrieval_store", None), self.ctx.clock)
        self._pending_inferences: set[str] = set()
        self._last_mid: dict[str, Decimal] = {}
        #: Markets routed to this sleeve's universe. TimerEvents are broadcast and
        #: ctx.markets() is every market the engine knows, so the universe has to be
        #: learned from the universe-targeted book/market events.
        self._universe_ids: set[str] = set()
        self._last_assessed: dict[str, datetime] = {}
        self._semaphore = asyncio.Semaphore(
            int(self.param("max_concurrent_inferences", DEFAULT_MAX_CONCURRENT_INFERENCES))
        )
        self._task: asyncio.Future[int] | None = None

    # ------------------------------------------------------------------ event triggers

    def on_news(self, event: NewsEvent) -> None:
        if self.is_high_impact_news(event):
            self._queue_universe("high_impact_news")

    def on_filing(self, event: FilingEvent) -> None:
        del event
        self._queue_universe("sec_filing")

    def on_trader_action(self, event: TraderActionEvent) -> None:
        if event.canonical_id and event.canonical_id in self._universe_ids:
            self._queue_if_eligible(event.canonical_id)

    def on_social(self, event: SocialEvent) -> None:
        del event
        # A public statement alone is not "high impact news"; leave triggering to the
        # large-move check in on_book_update rather than queuing on every post.

    def on_market_update(self, event: MarketUpdateEvent) -> None:
        self._universe_ids.add(event.market.canonical_id)

    def on_book_update(self, event: BookUpdateEvent) -> None:
        canonical_id = event.canonical_id
        self._universe_ids.add(canonical_id)
        mid = event.book.mid
        if mid is None:
            return
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
        for cid in self._universe_ids:
            if not self.on_cooldown(f"ai_refresh:{cid}", refresh_seconds):
                self._queue_if_eligible(cid)

    # ------------------------------------------------------------------ queueing

    def _queue_universe(self, trigger: str) -> None:
        del trigger
        for cid in self._universe_ids:
            self._queue_if_eligible(cid)

    def _worth_assessing(self, canonical_id: str) -> bool:
        """Cheap deterministic triage before any model time is spent."""
        market = self.ctx.market(canonical_id)
        book = self.ctx.book(canonical_id)
        if market is None or book is None or book.mid is None:
            return False
        lo = Decimal(str(self.param("min_mid", "0.04")))
        if not (lo <= book.mid <= ONE - lo):
            return False  # priced at a near-certainty; nothing left to find
        if market.close_time is not None:
            hours_left = (market.close_time - self.now()).total_seconds() / 3600
            if hours_left < float(self.param("min_hours_to_close", 2.0)):
                return False  # too close to resolution for a slow model to add anything
        return True

    def _queue_if_eligible(self, canonical_id: str) -> None:
        if self.should_skip(canonical_id) is not None:
            return
        min_spacing = float(self.param("min_inference_spacing_seconds", 60.0))
        if self.on_cooldown(f"ai_infer:{canonical_id}", min_spacing):
            return
        if not self._worth_assessing(canonical_id):
            return
        self._pending_inferences.add(canonical_id)

    # ------------------------------------------------------------------ async inference

    def start_background_inference(self) -> bool:
        """Kick off :meth:`process_pending` without blocking the caller.

        Called by the runner on every tick. A model call takes seconds to minutes, so it
        must never run inline on the event-dispatch path. At most one drain is in flight
        per sleeve; intents it produces are picked up by the runner's next dispatch.
        """
        if self._task is not None and not self._task.done():
            return False
        if not self._pending_inferences:
            return False
        self._task = asyncio.ensure_future(self.process_pending())
        return True

    async def process_pending(self) -> int:
        """Drain the trigger queue (stalest first, capped per pass) under the semaphore."""
        pending, self._pending_inferences = self._pending_inferences, set()
        if not pending:
            return 0
        cap = int(self.param("max_markets_per_pass", 4))
        epoch = datetime.min.replace(tzinfo=self.now().tzinfo)
        ordered = sorted(pending, key=lambda cid: self._last_assessed.get(cid, epoch))
        batch, overflow = ordered[:cap], ordered[cap:]
        # Overflow is not lost: it waits for the next pass.
        self._pending_inferences.update(overflow)
        await asyncio.gather(*(self._run_with_semaphore(cid) for cid in batch), return_exceptions=True)
        return len(batch)

    async def _run_with_semaphore(self, canonical_id: str) -> None:
        async with self._semaphore:
            await self.evaluate_market(canonical_id)

    def _prompt(self, market: Any, bundle: EvidenceBundle, decision_time: datetime) -> str:
        close = market.close_time.isoformat() if market.close_time else "unknown"
        return (
            f"Today is {decision_time.date().isoformat()} (UTC {decision_time.strftime('%H:%M')}).\n"
            f"Market: {market.title!r}\n"
            f"Market closes: {close}\n"
            f"Resolution rules: {market.resolution_rules or market.description or '(not provided)'}\n\n"
            f"{bundle.to_prompt_context()}\n\n"
            "Estimate the probability this market resolves YES."
        )

    async def _assess(self, provider: LLMProvider, market: Any, canonical_id: str, ttl: float) -> _Assessed:
        async def compute() -> _Assessed:
            decision_time = self.now()
            try:
                bundle = await self._retrieval.build_bundle(
                    canonical_id, decision_time, market=market, market_price=None
                )
            except Exception as exc:  # noqa: BLE001
                return _Assessed(decision_time, None, None, f"retrieval_error: {exc}")
            try:
                response = await provider.generate(
                    self._prompt(market, bundle, decision_time),
                    schema=judgement_schema(),
                    system=_SYSTEM_PROMPT,
                    timeout=float(self.param("timeout_seconds", 120.0)),
                )
            except Exception as exc:  # noqa: BLE001 - a provider failure must never raise here
                return _Assessed(decision_time, bundle, None, f"provider_error: {exc}")
            return _Assessed(decision_time, bundle, response)

        if self._cache is None:
            return await compute()
        key = (f"{provider.name}:{provider.model}", canonical_id)
        result: _Assessed = await self._cache.get_or_compute(key, ttl, compute)
        return result

    def _validated(self, assessed: _Assessed, market: Any, canonical_id: str) -> tuple[Any, list[str]]:
        """(validated assessment or None, failures)."""
        if assessed.response is None or assessed.bundle is None:
            return None, [assessed.error or "no_response"]
        # Gated after inference, not before: the inference is shared across variants,
        # and only this variant's decision depends on how much evidence it demands.
        min_items = int(self.param("min_evidence_items", 0))
        if _evidence_count(assessed.bundle) < min_items:
            return None, [f"insufficient_evidence(<{min_items})"]
        raw = assessed.response.parsed
        if raw is None:
            return None, [str(assessed.response.raw.get("reason", "unparseable_output"))]
        payload = dict(raw)
        payload.setdefault("market_id", canonical_id)
        payload.setdefault("as_of", assessed.decision_time.isoformat())
        payload.setdefault("information_cutoff", assessed.decision_time.isoformat())
        result = validate_assessment(parse_llm_output(payload), market, assessed.bundle, assessed.decision_time)
        if not result.valid or result.assessment_or_none is None:
            return None, result.failures
        validated = result.assessment_or_none
        # A bare 0.50 is a small model's "I don't know", not a forecast. Measured on the
        # first live run: qwen2.5:3b answered exactly 0.50 at confidence 0.9 on a Nasdaq
        # range bucket priced 0.075, which reads as a 42-point edge and got traded.
        band = Decimal(str(self.param("uninformative_band", "0.015")))
        if not validated.abstain and abs(validated.p_yes - Decimal("0.5")) < band:
            return None, ["uninformative_p_0.5"]
        return validated, []

    async def evaluate_market(self, canonical_id: str) -> None:
        market = self.ctx.market(canonical_id)
        if market is None or self.should_skip(canonical_id) is not None:
            return
        book = self.ctx.book(canonical_id)
        assert book is not None
        self.mark_fired(f"ai_refresh:{canonical_id}")
        self.mark_fired(f"ai_infer:{canonical_id}")
        self._last_assessed[canonical_id] = self.now()
        ttl = float(self.param("scheduled_refresh_seconds", DEFAULT_SCHEDULED_REFRESH_SECONDS)) * 0.9

        assessed = await self._assess(self._provider, market, canonical_id, ttl)
        validated, failures = self._validated(assessed, market, canonical_id)
        # The book may have moved while the model was thinking; trade on the live one.
        book = self.ctx.book(canonical_id) or book
        if validated is None:
            self._record_abstain(canonical_id, book.mid, failures, response=assessed.response)
            return
        response = assessed.response
        assert response is not None

        forecast = assessment_to_forecast(validated, self.strategy_id, self.experiment_id, book.mid)
        forecast.features["llm_model_id"] = response.model
        forecast.features["prompt_hash"] = response.prompt_hash
        forecast.features["ai_stack"] = self.param("ai_stack", "local")
        self.forecast(forecast)

        if validated.abstain:
            return
        min_confidence = Decimal(str(self.param("min_confidence", "0.5")))
        if validated.confidence < min_confidence:
            return

        candidate = self._best_candidate(market, book, validated.p_yes)
        if candidate is None:
            return
        side, price, edge = candidate

        second: Any = None
        second_response: LLMResponse | None = None
        if self._escalation is not None:
            # Only a would-be trade earns a paid second opinion.
            escalated = await self._assess(self._escalation, market, canonical_id, ttl)
            second, failures = self._validated(escalated, market, canonical_id)
            second_response = escalated.response
            if second is None or second.abstain or second.confidence < min_confidence:
                self._record_abstain(
                    canonical_id, book.mid, ["escalation_declined", *failures], response=second_response
                )
                return
            confirm = self._best_candidate(market, book, second.p_yes)
            if confirm is None or confirm[0] is not side:
                self._record_abstain(
                    canonical_id, book.mid,
                    [f"escalation_disagrees: primary={validated.p_yes} second={second.p_yes}"],
                    response=second_response,
                )
                return
            # Trade at the more conservative of the two edges.
            if confirm[2] < edge:
                side, price, edge = confirm

        self._emit(canonical_id, side, price, edge, validated, response, second, second_response, assessed)

    def _record_abstain(
        self, canonical_id: str, market_probability: Decimal | None, failures: list[str], response: Any = None
    ) -> None:
        features: dict[str, Any] = {"validation_failures": failures, "ai_stack": self.param("ai_stack", "local")}
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

    def _best_candidate(
        self, market: Any, book: Any, model_probability: Decimal
    ) -> tuple[Side, Decimal, Decimal] | None:
        price_yes = self.executable_price(book, Side.YES, Action.BUY)
        price_no = self.executable_price(book, Side.NO, Action.BUY)
        candidates: list[tuple[Side, Decimal, Decimal]] = []
        if price_yes is not None:
            candidates.append((Side.YES, price_yes, self.edge_after_costs(model_probability, price_yes, market, Side.YES)))
        if price_no is not None:
            no_p = clamp_probability(ONE - model_probability)
            candidates.append((Side.NO, price_no, self.edge_after_costs(no_p, price_no, market, Side.NO)))
        if not candidates:
            return None
        best = max(candidates, key=lambda c: c[2])
        entry_edge = Decimal(str(self.param("entry_edge", "0.05")))
        return best if best[2] > entry_edge else None

    def _emit(
        self,
        canonical_id: str,
        side: Side,
        price: Decimal,
        edge: Decimal,
        validated: Any,
        response: LLMResponse,
        second: Any,
        second_response: LLMResponse | None,
        assessed: _Assessed,
    ) -> None:
        evidence: list[str] = []
        if assessed.bundle is not None:
            evidence = [f"{it.evidence_id}: {it.headline[:90]}" for it in assessed.bundle.all_items()[:6]]
        why = (
            f"AI analyst [{self.param('ai_stack', 'local')}]: {response.model} says "
            f"p_yes={validated.p_yes} (conf {validated.confidence})"
        )
        if second is not None and second_response is not None:
            why += f"; confirmed by {second_response.model} p_yes={second.p_yes} (conf {second.confidence})"
        why += (
            f". Executable {side.value} at {price}; edge {edge} after fees clears "
            f"entry_edge {self.param('entry_edge', '0.05')}."
        )
        if validated.supporting_facts:
            why += " Supporting: " + "; ".join(validated.supporting_facts[:3])
        intent = self.make_intent(
            canonical_id=canonical_id,
            side=side,
            action=Action.BUY,
            quantity=self.sensible_quantity(price),
            order_type=OrderType.LIMIT,
            limit_price=price,
            rationale=why[:1500],
            features={
                "llm_model_id": response.model,
                "prompt_hash": response.prompt_hash,
                "ai_stack": self.param("ai_stack", "local"),
                "model_probability": float(validated.p_yes),
                "model_confidence": float(validated.confidence),
                "second_opinion_model": second_response.model if second_response else None,
                "second_opinion_p_yes": float(second.p_yes) if second is not None else None,
                "executable_price": float(price),
                "side": side.value,
                "evidence": evidence,
                "contradicting": list(validated.contradicting_facts[:3]),
            },
            model_probability=second.p_yes if second is not None else validated.p_yes,
            expected_edge=edge,
        )
        if self.emit_if_profitable(intent):
            self.mark_fired(canonical_id)


__all__ = ["NewsProbabilityStrategy", "judgement_schema"]
