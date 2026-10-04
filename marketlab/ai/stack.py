"""The runtime AI stack shared by every AI sleeve.

Three tiers, cheapest first:

* ``local`` -- the Ollama model on this machine. Does all routine assessment.
* ``jev`` -- an optional free remote model (see :mod:`marketlab.ai.cloud`).
* ``openai`` -- a budget-capped second opinion, consulted only by *escalating* arms and
  only when the primary assessment would actually trade.

A sleeve names its arm with the ``ai_stack`` parameter:

====================  =================  ===================
``ai_stack``          primary            escalation
====================  =================  ===================
``local``             local              --
``hybrid``            local              openai
``jev``               jev                --
``jev_hybrid``        jev                openai
====================  =================  ===================

Arms whose providers are not configured are simply not created, so the tournament
compares exactly the stacks that can run. The two caches here are what make the stack
affordable: :class:`AssessmentCache` lets every parameter variant of one arm share a
single inference per market, and :class:`EvidenceCache` gives the retrieval layer the
live news/social/filing/trader stream that the SQLite store does not persist.
"""

from __future__ import annotations

import asyncio
import re
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from marketlab.ai.provider import LLMProvider
from marketlab.clock import Clock
from marketlab.core.events import (
    BaseEvent,
    ExternalPriceEvent,
    FilingEvent,
    NewsEvent,
    SocialEvent,
    TraderActionEvent,
)
from marketlab.core.instruments import NormalizedMarket
from marketlab.logging import get_logger

log = get_logger(__name__)

#: ai_stack name -> (primary provider key, escalation provider key or None)
STACKS: dict[str, tuple[str, str | None]] = {
    "local": ("local", None),
    "hybrid": ("local", "openai"),
    "jev": ("jev", None),
    "jev_hybrid": ("jev", "openai"),
}

_STOPWORDS = frozenset(
    ["will", "what", "when", "where", "which", "who", "whom", "this", "that", "these", "those", "there", "their", "them", "they", "with", "from", "into", "onto", "than", "then", "have", "has", "had", "been", "being", "does", "did", "done", "before", "after", "above", "below", "over", "under", "between", "during", "more", "most", "less", "least", "than", "such", "only", "other", "some", "any", "each", "every", "market", "markets", "price", "prices", "close", "closes", "above", "below", "least", "between", "yes", "no", "the", "and", "for", "2024", "2025", "2026", "2027", "2028", "end", "of", "by", "on", "in", "at", "to", "a", "an", "be", "is", "are", "or", "as", "it", "its"]
)


def market_keywords(market: NormalizedMarket) -> list[str]:
    """Distinctive words from a market's title, used to decide which news is relevant."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9.'-]{2,}", f"{market.title} {market.subcategory}")
    out: list[str] = []
    for w in words:
        lw = w.lower().strip(".'-")
        if len(lw) >= 3 and lw not in _STOPWORDS and lw not in out:
            out.append(lw)
    return out[:12]


def _relevant(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return False
    lowered = text.lower()
    hits = sum(1 for k in keywords if k in lowered)
    # One hit on a short title is enough; longer titles need corroboration so that a
    # single generic word ("trump", "fed") does not drag in the whole news firehose.
    return hits >= (1 if len(keywords) <= 2 else 2)


class EvidenceCache:
    """Bounded in-memory history of the live evidence stream, queryable per market.

    Duck-types the store interface :class:`~marketlab.ai.retrieval.RetrievalService`
    expects. Point-in-time safety is still enforced by the retrieval layer's own
    ``visible_at`` filter; this class only decides *relevance*.
    """

    def __init__(self, market_lookup: Callable[[str], NormalizedMarket | None], maxlen: int = 4000) -> None:
        self._market_lookup = market_lookup
        self._news: deque[NewsEvent] = deque(maxlen=maxlen)
        self._social: deque[SocialEvent] = deque(maxlen=maxlen)
        self._filings: deque[FilingEvent] = deque(maxlen=maxlen // 4)
        self._trader: deque[TraderActionEvent] = deque(maxlen=maxlen)
        self._prices: dict[str, ExternalPriceEvent] = {}

    def record(self, event: BaseEvent) -> None:
        if isinstance(event, NewsEvent):
            if event.duplicate_of is None:
                self._news.append(event)
        elif isinstance(event, SocialEvent):
            self._social.append(event)
        elif isinstance(event, FilingEvent):
            self._filings.append(event)
        elif isinstance(event, TraderActionEvent):
            self._trader.append(event)
        elif isinstance(event, ExternalPriceEvent):
            self._prices[event.symbol] = event

    def _keywords(self, canonical_id: str) -> list[str]:
        market = self._market_lookup(canonical_id)
        return market_keywords(market) if market is not None else []

    def get_news(self, canonical_id: str) -> list[NewsEvent]:
        kw = self._keywords(canonical_id)
        return [e for e in self._news if _relevant(f"{e.title} {e.body[:600]} {' '.join(e.entities)}", kw)]

    def get_social(self, canonical_id: str) -> list[SocialEvent]:
        kw = self._keywords(canonical_id)
        return [e for e in self._social if _relevant(f"{e.text} {' '.join(e.mentioned_entities)}", kw)]

    def get_filings(self, canonical_id: str) -> list[FilingEvent]:
        kw = self._keywords(canonical_id)
        return [e for e in self._filings if _relevant(f"{e.company} {e.ticker} {e.form_type}", kw)]

    def get_trader_activity(self, canonical_id: str) -> list[TraderActionEvent]:
        kw = self._keywords(canonical_id)
        return [
            e for e in self._trader
            if e.canonical_id == canonical_id or (e.title and _relevant(e.title, kw))
        ]

    def get_external_prices(self, canonical_id: str) -> list[ExternalPriceEvent]:
        kw = self._keywords(canonical_id)
        return [e for e in self._prices.values() if _relevant(e.symbol.replace("-", " "), kw)]

    def get_market(self, canonical_id: str) -> NormalizedMarket | None:
        return self._market_lookup(canonical_id)

    def sizes(self) -> dict[str, int]:
        return {
            "news": len(self._news), "social": len(self._social), "filings": len(self._filings),
            "trader": len(self._trader), "prices": len(self._prices),
        }


@dataclass
class _Entry:
    at: datetime
    value: Any


class AssessmentCache:
    """Share one inference per (provider, market) across every variant of an arm.

    Six ``news_probability`` variants that differ only in entry threshold would otherwise
    ask the same model the same question six times. Concurrent requests for the same key
    await the same in-flight call instead of starting a second one.
    """

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._inflight: dict[tuple[str, str], asyncio.Future[Any]] = {}

    def fresh(self, key: tuple[str, str], ttl_seconds: float) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if (self._clock.now() - entry.at).total_seconds() > ttl_seconds:
            return None
        return entry.value

    async def get_or_compute(
        self, key: tuple[str, str], ttl_seconds: float, compute: Callable[[], Awaitable[Any]]
    ) -> Any:
        cached = self.fresh(key, ttl_seconds)
        if cached is not None:
            return cached
        pending = self._inflight.get(key)
        if pending is not None:
            return await asyncio.shield(pending)
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            value = await compute()
            self._entries[key] = _Entry(self._clock.now(), value)
            future.set_result(value)
            return value
        except BaseException as exc:
            future.set_exception(exc)
            # Retrieve so an un-awaited future does not log "exception never retrieved".
            future.exception()
            raise
        finally:
            self._inflight.pop(key, None)
            if len(self._entries) > 5000:
                for old in sorted(self._entries, key=lambda k: self._entries[k].at)[:1000]:
                    del self._entries[old]


@dataclass
class AIStack:
    providers: dict[str, LLMProvider]
    evidence: EvidenceCache
    assessments: AssessmentCache
    #: Per-provider default inference timeout, seconds.
    timeouts: dict[str, float] = field(default_factory=dict)

    def resolve(self, stack: str) -> tuple[LLMProvider, LLMProvider | None] | None:
        """(primary, escalation) for an arm, or None if any required tier is missing."""
        spec = STACKS.get(stack)
        if spec is None:
            return None
        primary_key, escalation_key = spec
        primary = self.providers.get(primary_key)
        if primary is None:
            return None
        if escalation_key is None:
            return primary, None
        escalation = self.providers.get(escalation_key)
        if escalation is None:
            return None
        return primary, escalation

    def available_stacks(self) -> list[str]:
        return [name for name in STACKS if self.resolve(name) is not None]


async def build_ai_stack(
    settings: Any,
    detected: LLMProvider,
    clock: Clock,
    market_lookup: Callable[[str], NormalizedMarket | None],
) -> AIStack:
    """Assemble the tiers that are actually available on this machine right now.

    The local tier reuses the autodetected Ollama server but swaps in
    ``ai.runtime_model`` when that model is installed: the autodetector prefers the
    strongest model (gpt-oss:20b, ~2 min per assessment on this CPU) while the
    tournament needs throughput. Anything unreachable is simply absent.
    """
    from marketlab.ai.cloud import build_jev_provider, build_openai_provider
    from marketlab.ai.ollama import OllamaProvider
    from marketlab.ai.provider import _probe_ollama

    ai_cfg = dict(getattr(settings, "ai", {}) or {})
    providers: dict[str, LLMProvider] = {}

    runtime_model = str(ai_cfg.get("runtime_model") or "")
    if detected.name == "disabled" and runtime_model:
        # Ollama was not answering at boot (still starting, or briefly busy). The stack
        # is built once per process, so dropping the tier here would silently remove the
        # local arms for the whole run. Register it anyway: calls abstain (= no trade)
        # until the server answers, then work with no restart.
        base = str(getattr(getattr(settings, "sources", None), "ollama_base", "") or "http://localhost:11434")
        log.warning("ai.stack.local_unreachable_at_boot", base_url=base, model=runtime_model)
        providers["local"] = OllamaProvider(
            base_url=base, model=runtime_model, concurrency=int(ai_cfg.get("runtime_concurrency", 1))
        )
    elif detected.name != "disabled":
        local: LLMProvider = detected
        base_url = getattr(detected, "base_url", None)
        if runtime_model and base_url and getattr(detected, "api_style", "ollama") == "ollama":
            installed = await _probe_ollama(base_url) or []
            if runtime_model in installed:
                local = OllamaProvider(
                    base_url=base_url,
                    model=runtime_model,
                    concurrency=int(ai_cfg.get("runtime_concurrency", 1)),
                )
            else:
                log.warning("ai.stack.runtime_model_missing", model=runtime_model, using=detected.model)
        providers["local"] = local

    jev = build_jev_provider(settings)
    if jev is not None:
        providers["jev"] = jev
    openai = build_openai_provider(settings)
    if openai is not None:
        providers["openai"] = openai

    stack = AIStack(providers=providers, evidence=EvidenceCache(market_lookup), assessments=AssessmentCache(clock))
    log.info(
        "ai.stack.ready",
        tiers={k: f"{v.name}:{v.model}" for k, v in providers.items()},
        arms=stack.available_stacks(),
    )
    return stack


__all__ = [
    "STACKS",
    "build_ai_stack",
    "AIStack",
    "AssessmentCache",
    "EvidenceCache",
    "market_keywords",
]
