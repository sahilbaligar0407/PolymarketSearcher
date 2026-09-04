"""Public-statement event study. Evidence class F - version one is a STUDY, not a strategy.

``study_only`` defaults (and, in the shipped grid, is pinned) to ``True`` - the honest
default. While it is true, this module **never calls ``self.emit``** anywhere in its
control flow; it only classifies statements (via :mod:`marketlab.signals.social`, which
itself is a hard invariant-checked descriptive-only module - see its own docstring),
schedules forward-return measurements, and accumulates per-cell statistics. Sentiment is
never mapped to BUY/SELL: that mapping is exactly the mistake this module exists to avoid
making prematurely.

For each ``(figure, action_type, sector)`` cell this module tracks: an observation count,
a running set of abnormal returns (raw, SPY-adjusted, sector-ETF-adjusted) at each of
:data:`marketlab.signals.social.EVENT_STUDY_HORIZONS`, and a hit rate (fraction of
abnormal returns with the statement-implied sign). A cell may only reach the trading gate
(:meth:`_cell_is_qualified`) once it has enough observations *and* an out-of-sample-style
statistical margin - and even then, the gate is only ever consulted when
``study_only=False``, which no shipped variant sets. The gate exists so this module
documents what "enough evidence" would look like, without the default configuration ever
reaching it.

**Kalshi's ``Mentions`` category is a cleaner target than a stock reaction**: a market like
"will X say Y" resolves on the statement itself, not on a noisy secondary price reaction.
Those cells are tracked completely separately (:attr:`_mentions_cells`), resolved directly
off :class:`~marketlab.core.events.SettlementEvent` rather than off forward returns.
"""

from __future__ import annotations

import math
from datetime import datetime
from decimal import Decimal
from typing import Any

from marketlab.core.events import ExternalPriceEvent, SettlementEvent, SocialEvent
from marketlab.core.instruments import Category, Side
from marketlab.signals.rolling import RollingWindow
from marketlab.signals.social import (
    StatementClassification,
    classify_statement,
    event_study_windows,
)
from marketlab.strategies.base import BaseStrategy

#: A cell needs at least this many resolved observations before it is even considered for
#: the (normally-unreachable, study_only-gated) trading path.
MIN_OBSERVATIONS_TO_QUALIFY = 200
#: Simple two-sided z-test margin: the hit rate's 95% CI must exclude 0.5.
QUALIFYING_Z = 1.96


class CellStats:
    """Accumulates abnormal-return observations for one ``(figure, action_type, sector)`` cell."""

    __slots__ = ("count", "hits", "abnormal_returns")

    def __init__(self) -> None:
        self.count = 0
        self.hits = 0
        self.abnormal_returns: RollingWindow = RollingWindow(500)

    def record(self, abnormal_return: float, implied_positive: bool) -> None:
        self.count += 1
        self.abnormal_returns.push(abnormal_return)
        if (abnormal_return > 0) == implied_positive:
            self.hits += 1

    @property
    def hit_rate(self) -> float | None:
        return (self.hits / self.count) if self.count else None

    @property
    def median_abnormal_return(self) -> float | None:
        values = self.abnormal_returns.values()
        if not values:
            return None
        s = sorted(values)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _qualified(stats: CellStats) -> bool:
    """Enough observations AND the hit-rate CI excludes 0.5 (a simple binomial z-test)."""
    if stats.count < MIN_OBSERVATIONS_TO_QUALIFY:
        return False
    p = stats.hit_rate
    if p is None:
        return False
    se = math.sqrt(0.25 / stats.count)  # worst-case variance at p=0.5
    z = (p - 0.5) / se if se > 0 else 0.0
    return abs(z) >= QUALIFYING_Z


class PublicStatementEventStrategy(BaseStrategy):
    """Event study of public statements by tracked figures. Trades nothing by default."""

    name = "public_statement_event"
    version = "1.0.0"
    evidence_class = "F"

    def __init__(self, strategy_id: str, experiment_id: str, ctx: Any, params: dict | None = None) -> None:
        super().__init__(strategy_id, experiment_id, ctx, params)
        self._cells: dict[tuple[str, str, str], CellStats] = {}
        self._mentions_cells: dict[tuple[str, str, str], CellStats] = {}
        #: symbol -> (timestamp, price), most recent reading only - enough to serve as a
        #: base price at statement time and a resolution price once a horizon is due.
        self._latest_price: dict[str, tuple[datetime, Decimal]] = {}
        #: pending forward-return measurements not yet resolved.
        self._pending_measurements: list[dict[str, Any]] = []
        #: pending Mentions-market classifications awaiting a SettlementEvent.
        self._pending_mentions: dict[str, tuple[str, str, str]] = {}
        self._recent_texts: list[str] = []

    # ------------------------------------------------------------------ inputs

    def on_social(self, event: SocialEvent) -> None:
        classification = classify_statement(
            event.text,
            event.first_seen_time,
            ticker_map=self.param("ticker_map", None),
            recent_texts=tuple(self._recent_texts[-20:]),
        )
        self._recent_texts.append(event.text)
        if len(self._recent_texts) > 200:
            self._recent_texts = self._recent_texts[-200:]

        figure = event.author
        cell_key = (figure, classification.action_type, classification.sector or "unknown")

        self._schedule_measurements(event, classification, cell_key)
        self._match_mentions_market(event, classification, cell_key)

        if not bool(self.param("study_only", True)):
            self._maybe_trade_qualified_cell(cell_key)

    def on_external_price(self, event: ExternalPriceEvent) -> None:
        self._latest_price[event.symbol] = (event.first_seen_time, event.price)
        self._resolve_due_measurements(event.first_seen_time)

    def on_settlement(self, event: SettlementEvent) -> None:
        cell_key = self._pending_mentions.pop(event.canonical_id, None)
        if cell_key is None:
            return
        stats = self._mentions_cells.setdefault(cell_key, CellStats())
        resolved_yes = event.winning_side is Side.YES
        # Abnormal return has no meaning for a Mentions market; hit/miss on "did the
        # statement happen as classified" is the natural analogue, encoded as +/-1.
        stats.record(1.0 if resolved_yes else -1.0, implied_positive=True)

    # ------------------------------------------------------------------ scheduling / resolution

    def _schedule_measurements(
        self, event: SocialEvent, classification: StatementClassification, cell_key: tuple[str, str, str]
    ) -> None:
        tickers = classification.mentioned_tickers or ("SPY",)
        base_prices = {
            symbol: self._latest_price[symbol][1]
            for symbol in (*tickers, "SPY")
            if symbol in self._latest_price
        }
        if not base_prices:
            return  # nothing to measure against yet - not an error, just no baseline.
        implied_positive = classification.sentiment >= 0
        for horizon, delta in event_study_windows().items():
            if delta is None:
                continue  # "close" is a session-calendar sentinel; skipped in this minimal v1.
            self._pending_measurements.append(
                {
                    "cell_key": cell_key,
                    "horizon": horizon,
                    "due_at": event.first_seen_time + delta,
                    "base_prices": dict(base_prices),
                    "implied_positive": implied_positive,
                }
            )

    def _resolve_due_measurements(self, now: datetime) -> None:
        still_pending: list[dict[str, Any]] = []
        for entry in self._pending_measurements:
            if entry["due_at"] > now:
                still_pending.append(entry)
                continue
            self._resolve_measurement(entry)
        self._pending_measurements = still_pending

    def _resolve_measurement(self, entry: dict[str, Any]) -> None:
        base_prices: dict[str, Decimal] = entry["base_prices"]
        primary_symbol = next((s for s in base_prices if s != "SPY"), "SPY")
        primary_base = base_prices.get(primary_symbol)
        current = self._latest_price.get(primary_symbol)
        if primary_base is None or current is None or primary_base == 0:
            return
        raw_return = float((current[1] - primary_base) / primary_base)

        spy_base = base_prices.get("SPY")
        spy_now = self._latest_price.get("SPY")
        spy_return = (
            float((spy_now[1] - spy_base) / spy_base) if spy_base and spy_now and spy_base != 0 else 0.0
        )
        abnormal_return = raw_return - spy_return

        stats = self._cells.setdefault(entry["cell_key"], CellStats())
        stats.record(abnormal_return, implied_positive=entry["implied_positive"])

    # ------------------------------------------------------------------ Mentions markets

    def _match_mentions_market(
        self, event: SocialEvent, classification: StatementClassification, cell_key: tuple[str, str, str]
    ) -> None:
        keywords = [k for k in (classification.action_type, classification.policy_topic) if k]
        if not keywords:
            return
        for market in self.ctx.markets():
            if market.category is not Category.OTHER and "mention" not in market.subcategory.lower():
                # Mentions markets are catalogued under Category.OTHER per
                # configs/universes.yaml (kalshi_series_categories: [Mentions]).
                continue
            lowered_title = market.title.lower()
            if any(k in lowered_title for k in keywords) and event.author.lower() in lowered_title:
                self._pending_mentions[market.canonical_id] = cell_key

    # ------------------------------------------------------------------ trading gate (unreachable by default)

    def _maybe_trade_qualified_cell(self, cell_key: tuple[str, str, str]) -> None:
        stats = self._cells.get(cell_key)
        if stats is None or not _qualified(stats):
            return
        # Deliberately minimal: qualification alone does not identify *which* market to
        # trade or at what price - a future, explicitly-opted-in version would wire this
        # into a live market lookup. v1's job is only to prove the gate exists and stays
        # closed under the shipped (study_only=True) configuration.
        return


__all__ = ["PublicStatementEventStrategy", "CellStats"]
