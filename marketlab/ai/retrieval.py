"""Evidence bundles: the only thing the local model is ever allowed to look at.

The single mandatory rule in this file is the point-in-time gate: a bundle built "as of"
time ``T`` must never contain a record whose ``first_seen_time`` is later than ``T``.
Every event type in :mod:`marketlab.core.events` carries ``first_seen_time`` for exactly
this reason, and :meth:`~marketlab.core.events.BaseEvent.visible_at` is the sanctioned
check. This module has a dedicated test (``tests/unit/test_ai_retrieval.py``) proving the
gate holds to the second.

``RetrievalService`` takes a duck-typed store and an injected :class:`~marketlab.clock.
Clock` rather than importing a concrete ``StateStore`` -- storage is owned by another
team. We call ``getattr(store, "get_news", None)`` etc. defensively so that a store
missing a method (or not yet built) degrades to an empty category instead of raising.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from marketlab.clock import Clock
from marketlab.core.events import (
    ExternalPriceEvent,
    FilingEvent,
    NewsEvent,
    SocialEvent,
    TraderActionEvent,
)
from marketlab.core.instruments import NormalizedMarket
from marketlab.logging import get_logger

log = get_logger(__name__)

# `marketlab.signals.news` is owned by another team. Keep the seam clean: prefer their
# dedup logic when available, fall back to a local one otherwise, and never let an
# import error (or a bug in their function) break retrieval. Their `dedupe` keeps every
# record for audit and only *marks* `duplicate_of`; we additionally drop the marked
# repeats before handing evidence to the model, since ten wire copies of one story
# should read as one piece of evidence, not ten.
_external_dedupe: Callable[[list[NewsEvent]], list[NewsEvent]] | None
try:  # pragma: no cover - exercised implicitly by whichever branch is live
    from marketlab.signals.news import dedupe as _external_dedupe  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _external_dedupe = None


@dataclass(frozen=True)
class EvidenceItem:
    """One citable, id-tagged fact handed to the model.

    ``timestamp`` is always the record's ``first_seen_time`` (or, for the synthetic
    market/prior items, the bundle's ``as_of``) -- the one field point-in-time
    discipline and the validator's look-ahead check are allowed to use.
    """

    evidence_id: str
    kind: str
    timestamp: datetime
    headline: str
    text: str
    source: str = ""
    #: Sentiment/direction in roughly [-1, 1] when known; used to pick counterevidence.
    tone: float | None = None
    #: Recency/relevance multiplier applied on top of the built-in recency decay.
    weight: float = 1.0


@dataclass(frozen=True)
class EvidenceBundle:
    """Everything the model is allowed to see for one market at one point in time."""

    market: NormalizedMarket
    rules: str
    news: tuple[EvidenceItem, ...]
    social: tuple[EvidenceItem, ...]
    filings: tuple[EvidenceItem, ...]
    external_prices: tuple[EvidenceItem, ...]
    trader_activity: tuple[EvidenceItem, ...]
    prior_assessment: EvidenceItem | None
    counterevidence: tuple[EvidenceItem, ...]
    as_of: datetime
    #: Market midpoint (YES probability) at build time, if known.
    market_price: Decimal | None = None

    def market_evidence_id(self) -> str:
        return f"market:{self.market.canonical_id}"

    def all_items(self) -> list[EvidenceItem]:
        """Every citable item *except* the synthetic market entry (see :meth:`index`)."""
        items: list[EvidenceItem] = [
            *self.news,
            *self.social,
            *self.filings,
            *self.external_prices,
            *self.trader_activity,
            *self.counterevidence,
        ]
        if self.prior_assessment is not None:
            items.append(self.prior_assessment)
        return items

    def index(self) -> dict[str, EvidenceItem]:
        """``evidence_id -> EvidenceItem`` for every item, including the synthetic
        ``market:<canonical_id>`` entry that stands in for "the current market price is
        visible as of this bundle's ``as_of``" -- there is always exactly one of these,
        it is never look-ahead by construction, and the validator relies on it existing.
        """
        idx = {item.evidence_id: item for item in self.all_items()}
        idx[self.market_evidence_id()] = EvidenceItem(
            evidence_id=self.market_evidence_id(),
            kind="market",
            timestamp=self.as_of,
            headline=self.market.title,
            text=(
                f"current market price (YES probability) = {self.market_price}"
                if self.market_price is not None
                else "current market price unavailable"
            ),
            source=str(self.market.venue),
        )
        return idx

    def to_prompt_context(self, max_chars: int = 6000) -> str:
        """Compact, id-tagged text for the prompt. Never dumps a whole archive -- the
        caps applied in :class:`RetrievalService` bound the item counts, and this
        additionally hard-truncates the rendered text to ``max_chars``.
        """
        sections: list[tuple[str, list[EvidenceItem]]] = [
            ("NEWS", list(self.news)),
            ("COUNTEREVIDENCE", list(self.counterevidence)),
            ("SOCIAL", list(self.social)),
            ("FILINGS", list(self.filings)),
            ("EXTERNAL_PRICES", list(self.external_prices)),
            ("TRADER_ACTIVITY", list(self.trader_activity)),
        ]
        if self.prior_assessment is not None:
            sections.append(("PRIOR_ASSESSMENT", [self.prior_assessment]))

        lines: list[str] = []
        for label, items in sections:
            if not items:
                continue
            lines.append(f"-- {label} --")
            for item in items:
                body = item.text or item.headline
                lines.append(f"[{item.evidence_id}] ({item.timestamp.isoformat()}) {item.headline}: {body}")
        text = "\n".join(lines) if lines else "(no evidence available)"
        if len(text) > max_chars:
            text = text[: max(0, max_chars - 20)].rstrip() + "\n...[truncated]"
        return text


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _news_to_item(e: NewsEvent) -> EvidenceItem:
    key = e.body_hash or e.canonical_url or e.url or e.title
    return EvidenceItem(
        evidence_id=f"news:{_stable_hash(key)}",
        kind="news",
        timestamp=e.first_seen_time,
        headline=e.title,
        text=e.body[:1000],
        source=e.source,
        tone=e.tone,
    )


def _social_to_item(e: SocialEvent) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"social:{e.post_id}",
        kind="social",
        timestamp=e.first_seen_time,
        headline=f"@{e.author} on {e.platform}",
        text=e.text[:500],
        source=e.platform,
    )


def _filing_to_item(e: FilingEvent) -> EvidenceItem:
    headline = f"{e.form_type or 'filing'} - {e.company or e.ticker}".strip(" -")
    text = (
        f"insider={e.insider_name or 'n/a'} role={e.insider_role or 'n/a'} "
        f"code={e.transaction_code or 'n/a'} value={e.transaction_value}"
    )
    return EvidenceItem(
        evidence_id=f"sec:{e.accession}",
        kind="filing",
        timestamp=e.first_seen_time,
        headline=headline,
        text=text,
        source="sec",
    )


def _price_to_item(e: ExternalPriceEvent) -> EvidenceItem:
    eid = f"price:{e.symbol}:{_stable_hash(e.symbol, str(e.price), e.event_time.isoformat())}"
    return EvidenceItem(
        evidence_id=eid,
        kind="external_price",
        timestamp=e.first_seen_time,
        headline=f"{e.symbol} external price",
        text=f"price={e.price} implied_probability={e.implied_probability}",
        source=e.venue,
    )


def _trader_to_item(e: TraderActionEvent) -> EvidenceItem:
    tx = e.transaction_hash or _stable_hash(e.wallet, e.canonical_id, e.event_time.isoformat())
    return EvidenceItem(
        evidence_id=f"trader:{e.wallet}:{tx}",
        kind="trader_activity",
        timestamp=e.first_seen_time,
        headline=f"{e.username or e.wallet} {e.action or 'traded'} {e.side or ''}".strip(),
        text=f"price={e.price} size={e.size} usd_size={e.usd_size} title={e.title}",
        source="polymarket",
    )


def _prior_to_item(prior: Any, canonical_id: str) -> EvidenceItem | None:
    """Duck-types a prior forecast/assessment object (ProbabilityForecast or
    MarketAssessment-shaped) into an EvidenceItem. Returns None if it doesn't quack.
    """
    as_of = getattr(prior, "as_of", None)
    if not isinstance(as_of, datetime):
        return None
    p_yes = getattr(prior, "p_yes", None)
    eid = f"prior:{canonical_id}:{_stable_hash(canonical_id, as_of.isoformat())}"
    return EvidenceItem(
        evidence_id=eid,
        kind="prior_assessment",
        timestamp=as_of,
        headline="Prior model assessment",
        text=f"p_yes={p_yes}",
        source="ai",
    )


def _local_dedup_news(events: list[NewsEvent]) -> list[NewsEvent]:
    """Fallback dedup used only when marketlab.signals.news isn't importable."""
    seen: set[str] = set()
    out: list[NewsEvent] = []
    for e in events:
        key = e.body_hash or e.canonical_url or e.url or e.title.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _dedup_news(events: list[NewsEvent]) -> list[NewsEvent]:
    if _external_dedupe is not None:
        try:
            deduped = _external_dedupe(events)
            return [e for e in deduped if e.duplicate_of is None]
        except Exception:
            log.warning("ai.retrieval.external_dedup_failed", exc_info=True)
    return _local_dedup_news(events)


async def _fetch(store: Any, method: str, canonical_id: str) -> list[Any]:
    """Call an optional duck-typed store method; missing or failing => empty list."""
    fn = getattr(store, method, None)
    if fn is None:
        return []
    try:
        result = fn(canonical_id)
        if inspect.isawaitable(result):
            result = await result
        return list(result or [])
    except Exception:
        log.warning("ai.retrieval.store_method_failed", method=method, exc_info=True)
        return []


async def _fetch_one(store: Any, method: str, canonical_id: str) -> Any | None:
    fn = getattr(store, method, None)
    if fn is None:
        return None
    try:
        result = fn(canonical_id)
        if inspect.isawaitable(result):
            result = await result
        return result
    except Exception:
        log.warning("ai.retrieval.store_method_failed", method=method, exc_info=True)
        return None


class RetrievalService:
    """Builds point-in-time-safe :class:`EvidenceBundle`\\ s for the analyst layer.

    ``store`` is duck-typed: any object exposing some subset of ``get_news``,
    ``get_social``, ``get_filings``, ``get_external_prices``, ``get_trader_activity``,
    ``get_market`` and ``get_prior_assessment`` (sync or async, each taking
    ``canonical_id``) works. Missing methods simply yield an empty category rather than
    raising, since the real ``StateStore`` is owned by another team and its exact
    surface may still change.
    """

    def __init__(
        self,
        store: Any,
        clock: Clock,
        *,
        max_items_per_category: int = 8,
        default_max_chars: int = 6000,
        counterevidence_limit: int = 3,
    ) -> None:
        self._store = store
        self._clock = clock
        self._max_items = max_items_per_category
        self._default_max_chars = default_max_chars
        self._counter_limit = counterevidence_limit

    async def build_bundle(
        self,
        canonical_id: str,
        as_of: datetime,
        *,
        market: NormalizedMarket | None = None,
        market_price: Decimal | None = None,
        keywords: Sequence[str] = (),
    ) -> EvidenceBundle:
        if as_of > self._clock.now():
            log.warning("ai.retrieval.future_as_of", canonical_id=canonical_id, as_of=as_of.isoformat())

        if market is None:
            market = await _fetch_one(self._store, "get_market", canonical_id)
        if market is None:
            raise ValueError(f"no market available for {canonical_id}; caller must supply one")

        rules = market.resolution_rules

        raw_news = [e for e in await _fetch(self._store, "get_news", canonical_id) if e.visible_at(as_of)]
        raw_news = _dedup_news(raw_news)
        news_items = self._filter_and_rank([_news_to_item(e) for e in raw_news], as_of, keywords)

        raw_social = [e for e in await _fetch(self._store, "get_social", canonical_id) if e.visible_at(as_of)]
        social_items = self._filter_and_rank([_social_to_item(e) for e in raw_social], as_of, keywords)

        raw_filings = [e for e in await _fetch(self._store, "get_filings", canonical_id) if e.visible_at(as_of)]
        filing_items = self._filter_and_rank([_filing_to_item(e) for e in raw_filings], as_of, keywords)

        raw_prices = [
            e for e in await _fetch(self._store, "get_external_prices", canonical_id) if e.visible_at(as_of)
        ]
        price_items = self._filter_and_rank([_price_to_item(e) for e in raw_prices], as_of, ())

        raw_trader = [
            e for e in await _fetch(self._store, "get_trader_activity", canonical_id) if e.visible_at(as_of)
        ]
        trader_items = self._filter_and_rank([_trader_to_item(e) for e in raw_trader], as_of, ())

        prior = await _fetch_one(self._store, "get_prior_assessment", canonical_id)
        prior_item = _prior_to_item(prior, canonical_id) if prior is not None else None
        if prior_item is not None and not prior_item.timestamp <= as_of:
            # A prior assessment made "in the future" relative to this bundle is
            # look-ahead by definition; drop it rather than smuggle it in.
            log.warning("ai.retrieval.dropping_future_prior", canonical_id=canonical_id)
            prior_item = None

        counterevidence = self._select_counterevidence(news_items + social_items, market_price)

        return EvidenceBundle(
            market=market,
            rules=rules,
            news=tuple(news_items),
            social=tuple(social_items),
            filings=tuple(filing_items),
            external_prices=tuple(price_items),
            trader_activity=tuple(trader_items),
            prior_assessment=prior_item,
            counterevidence=tuple(counterevidence),
            as_of=as_of,
            market_price=market_price,
        )

    def _filter_and_rank(
        self, items: list[EvidenceItem], as_of: datetime, keywords: Sequence[str]
    ) -> list[EvidenceItem]:
        if keywords:
            lowered = [k.lower() for k in keywords if k]
            filtered = [it for it in items if any(k in (it.headline + " " + it.text).lower() for k in lowered)]
            # A keyword filter that zeroes out everything is more likely a bad keyword
            # list than "no relevant evidence exists" -- fall back to unfiltered rather
            # than silently handing the model an empty bundle.
            if filtered:
                items = filtered
        ranked = sorted(items, key=lambda it: self._recency_weight(it, as_of), reverse=True)
        return ranked[: self._max_items]

    @staticmethod
    def _recency_weight(item: EvidenceItem, as_of: datetime) -> float:
        age_seconds = max(0.0, (as_of - item.timestamp).total_seconds())
        return item.weight / (1.0 + age_seconds / 3600.0)

    def _select_counterevidence(
        self, items: list[EvidenceItem], market_price: Decimal | None
    ) -> list[EvidenceItem]:
        """Items whose tone opposes the current market price -- deliberately included
        so the model isn't only shown evidence that agrees with where the price sits.
        """
        if market_price is None:
            return []
        favors_yes = market_price >= Decimal("0.5")
        candidates = [
            it
            for it in items
            if it.tone is not None and ((favors_yes and it.tone < 0) or (not favors_yes and it.tone > 0))
        ]
        candidates.sort(key=lambda it: abs(it.tone or 0.0), reverse=True)
        return candidates[: self._counter_limit]
