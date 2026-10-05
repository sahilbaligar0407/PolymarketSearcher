"""Copy what the top Polymarket wallets are *holding*, on the matched Kalshi contract.

The operator's manual process: take the top 100 wallets on a leaderboard (all-time,
monthly or weekly), list every open position, find the positions many of them share, and
place the same trade. ``marketlab.signals.holdings`` builds that consensus every refresh;
this strategy decides which rows to copy. Execution is Kalshi-only: a row trades only when
its Polymarket market has an approved Kalshi twin (YES==YES), where Polymarket outcome 0 is
a Kalshi YES and outcome 1 a Kalshi NO.

The variants are the questions the manual process leaves open:

* ``board`` / ``top_n`` - whose holdings count: all-time, month, week, or all three.
* ``min_holders`` - how many must agree. ``net_only`` drops wallets holding both sides
  of a market (hedgers and market makers; common at the top of the board).
* ``entry`` - ``blind`` copies whatever the price now is. ``edge`` refuses when the
  price has run more than ``max_chase`` above the holders' average entry, or when less
  than ``min_remaining`` of upside is left ("$5 to make $0.50": a 0.91 contract pays 9%
  at best, and one miss erases ten wins).
* ``confirm`` - ``none``, ``news`` (a relevant story in the evidence cache), ``jev``
  (TypeSafe's Jev must put the outcome's probability above the price by ``jev_min_edge``)
  or ``news_jev`` (Jev, shown the headlines).

Every sleeve holds to settlement and enters a market at most once. The PRD's positive
edge rule still applies: without a model, ``copy_edge_assumption`` is the assumed worth
of a consensus, recorded with every intent so it is measured rather than trusted.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from decimal import Decimal
from typing import Any

from marketlab.ai.typesafe import noul
from marketlab.core.events import TimerEvent
from marketlab.core.instruments import ONE, Side
from marketlab.core.orders import Action, Fill, OrderType
from marketlab.logging import get_logger
from marketlab.signals.holdings import BOARDS, ConsensusRow
from marketlab.strategies.base import BaseStrategy, clamp_probability

log = get_logger(__name__)

VALID_ENTRY = ("blind", "edge")
VALID_CONFIRM = ("none", "news", "jev", "news_jev")
#: How often a sleeve re-reads the consensus (the snapshot itself refreshes ~15 min).
EVALUATE_EVERY_SECONDS = 60.0
NEWS_LOOKBACK = timedelta(hours=72)


class HoldingsConsensusStrategy(BaseStrategy):
    name = "holdings_consensus"
    version = "1.0.0"
    evidence_class = "D"

    def __init__(self, strategy_id: str, experiment_id: str, ctx: Any, params: dict | None = None) -> None:
        super().__init__(strategy_id, experiment_id, ctx, params)
        self.board = str(self.param("board", "union"))
        self.entry = str(self.param("entry", "blind"))
        self.confirm = str(self.param("confirm", "none"))
        if self.board not in BOARDS:
            raise ValueError(f"board must be one of {BOARDS}, got {self.board!r}")
        if self.entry not in VALID_ENTRY:
            raise ValueError(f"entry must be one of {VALID_ENTRY}, got {self.entry!r}")
        if self.confirm not in VALID_CONFIRM:
            raise ValueError(f"confirm must be one of {VALID_CONFIRM}, got {self.confirm!r}")
        self.top_n = int(self.param("top_n", 100))
        self.min_holders = int(self.param("min_holders", 5))
        self.net_only = bool(self.param("net_only", True))
        self._entered: set[str] = set()
        self._last_eval = None
        self._jev_tasks: dict[str, asyncio.Future[Any]] = {}
        self.refusals: dict[str, int] = {}

    # ------------------------------------------------------------------ events

    def on_timer(self, event: TimerEvent) -> None:
        now = self.now()
        if self._last_eval is not None and (now - self._last_eval).total_seconds() < EVALUATE_EVERY_SECONDS:
            return
        self._last_eval = now
        book = self.param("holdings")
        if book is None or book.snapshot_time is None:
            return
        for row in book.view(self.board, self.top_n, self.net_only):
            if row.n_holders < self.min_holders:
                break  # rows are sorted by holder count
            self._consider(row, book.snapshot_id)

    def on_fill(self, fill: Fill) -> None:
        self._entered.add(fill.canonical_id)

    # ------------------------------------------------------------------ decision

    def _refuse(self, reason: str) -> None:
        self.refusals[reason] = self.refusals.get(reason, 0) + 1

    def _kalshi_leg(self, row: ConsensusRow) -> tuple[str, Side] | None:
        matches = self.param("matches")
        match = matches.get(f"poly:{row.condition_id}") if matches is not None else None
        if match is None or getattr(match, "same_outcome_boolean", True) is False:
            return None
        if row.outcome_index not in (0, 1):
            return None
        return match.canonical_id_a, Side.YES if row.outcome_index == 0 else Side.NO

    def _consider(self, row: ConsensusRow, snapshot_id: int) -> None:
        count = row.net_holders if self.net_only else row.n_holders
        if count < self.min_holders:
            self._refuse("not_enough_net_holders")
            return
        leg = self._kalshi_leg(row)
        if leg is None:
            self._refuse("no_kalshi_twin")
            return
        canonical_id, side = leg
        if canonical_id in self._entered or self.on_cooldown(canonical_id, 3600):
            return
        if self.should_skip(canonical_id) is not None:
            self._refuse("book_unusable")
            return
        market = self.ctx.market(canonical_id)
        book = self.ctx.book(canonical_id)
        price = self.executable_price(book, side, Action.BUY) if book is not None else None
        if market is None or price is None:
            self._refuse("no_liquidity")
            return

        chase = price - row.avg_entry
        remaining = ONE - price
        if self.entry == "edge":
            if chase > Decimal(str(self.param("max_chase", "0.05"))):
                self._refuse("price_ran_past_holders_entry")
                return
            if remaining < Decimal(str(self.param("min_remaining", "0.10"))):
                self._refuse("too_little_upside_left")
                return

        headlines: list[str] = []
        if self.confirm in ("news", "news_jev"):
            headlines = self._headlines(canonical_id)
            if self.confirm == "news" and not headlines:
                self._refuse("no_supporting_news")
                return

        jev_p: float | None = None
        if self.confirm in ("jev", "news_jev"):
            jev_p = self._jev_probability(row, canonical_id, side, price, headlines, snapshot_id)
            if jev_p is None:
                return  # asked; the answer lands before the next evaluation
            if Decimal(str(jev_p)) - price < Decimal(str(self.param("jev_min_edge", "0.03"))):
                self._refuse("jev_disagrees")
                return

        if jev_p is not None:
            p_outcome = Decimal(str(jev_p))
        else:
            p_outcome = price + Decimal(str(self.param("copy_edge_assumption", "0.06")))
        p_outcome = clamp_probability(p_outcome)
        model_probability = p_outcome if side is Side.YES else ONE - p_outcome
        edge = self.edge_after_costs(model_probability, price, market, side)
        quantity = self.sensible_quantity(price, risk_fraction=Decimal(str(self.param("risk_fraction", "0.03"))))

        features: dict[str, Any] = {
            "board": self.board, "top_n": self.top_n, "net_only": self.net_only,
            "entry": self.entry, "confirm": self.confirm,
            "holders": row.n_holders, "net_holders": row.net_holders, "opposing": len(row.opposing),
            "holders_usd": float(row.usd_value), "holders_avg_entry": float(row.avg_entry),
            "poly_price": float(row.cur_price), "kalshi_price": float(price),
            "chase": float(chase), "remaining_upside": float(remaining),
            "jev_p_outcome": jev_p, "headlines": headlines[:5],
            "poly_condition_id": row.condition_id, "poly_outcome": row.outcome,
            "snapshot_id": snapshot_id,
        }
        intent = self.make_intent(
            canonical_id=canonical_id,
            side=side,
            action=Action.BUY,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=price,
            rationale=(
                f"Holdings consensus ({self.board} top {self.top_n}, {self.entry}, confirm={self.confirm}): "
                f"{row.net_holders if self.net_only else row.n_holders} wallets hold '{row.outcome}' on "
                f"'{row.title}' ({len(row.opposing)} on the other side), avg entry {row.avg_entry:.3f}; "
                f"buying Kalshi {side.value} at {price} (chase {chase:+.3f}, upside left {remaining:.2f})"
                + (f"; Jev p={jev_p:.2f}" if jev_p is not None else "")
                + f". edge={edge}."
            ),
            features=features,
            model_probability=model_probability,
            expected_edge=edge,
            evidence_ids=(f"holdings:{snapshot_id}:{row.condition_id}:{row.outcome_index}",),
        )
        if self.emit_if_profitable(intent):
            self.mark_fired(canonical_id)
        else:
            self._refuse("no_edge_after_costs")

    # ------------------------------------------------------------------ confirmation tiers

    def _headlines(self, canonical_id: str) -> list[str]:
        evidence = self.param("evidence")
        if evidence is None:
            return []
        cutoff = self.now() - NEWS_LOOKBACK
        try:
            news = evidence.get_news(canonical_id)
        except Exception:  # noqa: BLE001 - evidence is an enrichment, never a dependency
            return []
        return [n.title for n in news if n.first_seen_time >= cutoff][-8:]

    def _jev_probability(
        self, row: ConsensusRow, canonical_id: str, side: Side, price: Decimal,
        headlines: list[str], snapshot_id: int,
    ) -> float | None:
        jev = self.param("jev")
        if jev is None:
            self._refuse("jev_unavailable")
            return None
        key = f"holdings:{snapshot_id}:{row.condition_id}:{row.outcome_index}:{bool(headlines)}"
        task = self._jev_tasks.get(key)
        if task is not None and task.done():
            return noul(task.result(), "outcome")
        if task is None:
            state = {
                "market": row.title,
                "outcome_in_question": row.outcome,
                "kalshi_price_for_outcome": float(price),
                "top_wallets_holding_outcome": row.n_holders,
                "top_wallets_on_other_side": len(row.opposing),
                "holders_average_entry_price": round(float(row.avg_entry), 3),
                "holders_total_usd": round(float(row.usd_value)),
                "recent_headlines": headlines,
            }
            questions = {"outcome": {
                "type": "noul",
                "instructions": f"Will '{row.outcome}' be the winning outcome of '{row.title}'?",
                "criteria": {"true": f"'{row.outcome}' resolves as the winner", "false": "Any other result"},
            }}
            try:
                self._jev_tasks[key] = asyncio.ensure_future(jev.evaluate(key, state, questions))
            except RuntimeError:  # no running loop (offline replay): no Jev tier
                self._refuse("jev_unavailable")
            if len(self._jev_tasks) > 500:
                for k in [k for k, t in self._jev_tasks.items() if t.done()][:250]:
                    self._jev_tasks.pop(k, None)
        return None


__all__ = ["HoldingsConsensusStrategy"]
