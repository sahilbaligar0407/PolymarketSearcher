"""Leaderboard holdings consensus: what the top Polymarket wallets are sitting in *now*.

This is the operator's manual process, automated: take the top wallets on a leaderboard,
read every open position each one holds, and find the positions many of them share
("11 of the top 100 hold X"). It complements ``copy_trader``, which reacts to individual
*trades* as they happen; this reads *holdings*, a slower and steadier signal.

Two corrections the manual process cannot make, both measured on the live API
(2026-10-04, top 100 overall): the best agreement was 6 of 100, not 11, and the same top
wallets often held *both* sides of one game (6 on Panthers, 4 on Lions). Many leaderboard
whales are market makers and hedgers, so a raw holder count can be mostly noise. Each
consensus row therefore also reports the wallets on the other side, and ``net_only``
counts only wallets holding a single outcome of that market.

``HoldingsBook`` is shared: ingest replaces its snapshot every refresh, strategies read
it. Consensus for one (board, top_n, net_only) view is computed once per snapshot and
cached, since dozens of sleeves ask for the same view.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

#: Boards a view can draw wallets from. "union" is all three combined.
BOARDS = ("all", "month", "week", "union")

#: Positions priced at or beyond these are effectively decided; nobody is "betting" there.
MIN_LIVE_PRICE = Decimal("0.02")
MAX_LIVE_PRICE = Decimal("0.98")


@dataclass(frozen=True)
class Holding:
    wallet: str
    condition_id: str
    outcome_index: int
    outcome: str
    title: str
    size: Decimal
    avg_price: Decimal
    cur_price: Decimal
    usd_value: Decimal
    end_date: str = ""

    @property
    def market_key(self) -> tuple[str, int]:
        return (self.condition_id, self.outcome_index)


@dataclass
class ConsensusRow:
    condition_id: str
    outcome_index: int
    outcome: str
    title: str
    cur_price: Decimal
    holders: list[str] = field(default_factory=list)
    #: Wallets in the same view holding a *different* outcome of the same market.
    opposing: list[str] = field(default_factory=list)
    usd_value: Decimal = Decimal(0)
    #: Size-weighted average entry price across holders.
    avg_entry: Decimal = Decimal(0)

    @property
    def n_holders(self) -> int:
        return len(self.holders)

    @property
    def net_holders(self) -> int:
        return len(self.holders) - len(self.opposing)


def parse_position(wallet: str, raw: dict) -> Holding | None:
    """One ``data-api /positions`` row, or None for resolved/dead/unreadable positions."""
    try:
        if raw.get("redeemable"):
            return None
        cur = Decimal(str(raw.get("curPrice") or 0))
        size = Decimal(str(raw.get("size") or 0))
        if size <= 0 or cur <= MIN_LIVE_PRICE or cur >= MAX_LIVE_PRICE:
            return None
        return Holding(
            wallet=wallet,
            condition_id=str(raw["conditionId"]),
            outcome_index=int(raw.get("outcomeIndex", 0)),
            outcome=str(raw.get("outcome") or ""),
            title=str(raw.get("title") or ""),
            size=size,
            avg_price=Decimal(str(raw.get("avgPrice") or 0)),
            cur_price=cur,
            usd_value=Decimal(str(raw.get("currentValue") or 0)),
            end_date=str(raw.get("endDate") or ""),
        )
    except (KeyError, ValueError, ArithmeticError):
        return None


def consensus(
    holdings: list[Holding], wallets: list[str], *, net_only: bool
) -> list[ConsensusRow]:
    """Aggregate ``holdings`` over ``wallets`` into one row per (market, outcome).

    With ``net_only`` a wallet holding more than one outcome of a market is dropped from
    that market entirely: it is hedged or making a market, not expressing a view.
    """
    members = set(wallets)
    by_wallet_market: dict[tuple[str, str], set[int]] = defaultdict(set)
    for h in holdings:
        if h.wallet in members:
            by_wallet_market[(h.wallet, h.condition_id)].add(h.outcome_index)

    rows: dict[tuple[str, int], ConsensusRow] = {}
    weight: dict[tuple[str, int], Decimal] = defaultdict(Decimal)
    for h in holdings:
        if h.wallet not in members:
            continue
        two_sided = len(by_wallet_market[(h.wallet, h.condition_id)]) > 1
        if net_only and two_sided:
            continue
        row = rows.get(h.market_key)
        if row is None:
            row = ConsensusRow(h.condition_id, h.outcome_index, h.outcome, h.title, h.cur_price)
            rows[h.market_key] = row
        if h.wallet not in row.holders:
            row.holders.append(h.wallet)
        row.usd_value += h.usd_value
        row.avg_entry += h.avg_price * h.size
        weight[h.market_key] += h.size

    for key, row in rows.items():
        if weight[key] > 0:
            row.avg_entry = row.avg_entry / weight[key]
        holders = set(row.holders)
        row.opposing = sorted(
            w for (w, cid), outs in by_wallet_market.items()
            if cid == row.condition_id
            and w not in holders
            and any(o != row.outcome_index for o in outs)
            and (not net_only or len(outs) == 1)
        )
    return sorted(rows.values(), key=lambda r: (-r.n_holders, -r.usd_value))


class HoldingsBook:
    """The latest snapshot, shared between the ingest loop (writer) and sleeves (readers)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.snapshot_time: datetime | None = None
        self.snapshot_id = 0
        self._boards: dict[str, list[str]] = {}
        self._holdings: list[Holding] = []
        self._cache: dict[tuple[str, int, bool], list[ConsensusRow]] = {}

    def replace(self, boards: dict[str, list[str]], holdings: list[Holding], at: datetime) -> None:
        with self._lock:
            self._boards = {k: list(v) for k, v in boards.items()}
            self._holdings = list(holdings)
            self._cache = {}
            self.snapshot_time = at
            self.snapshot_id += 1

    def wallets(self, board: str, top_n: int) -> list[str]:
        if board == "union":
            seen: list[str] = []
            for name in ("all", "month", "week"):
                for w in self._boards.get(name, [])[:top_n]:
                    if w not in seen:
                        seen.append(w)
            return seen
        return self._boards.get(board, [])[:top_n]

    def view(self, board: str, top_n: int, net_only: bool) -> list[ConsensusRow]:
        key = (board, top_n, net_only)
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                cached = consensus(self._holdings, self.wallets(board, top_n), net_only=net_only)
                self._cache[key] = cached
            return cached

    def __len__(self) -> int:
        return len(self._holdings)


__all__ = [
    "BOARDS",
    "ConsensusRow",
    "Holding",
    "HoldingsBook",
    "consensus",
    "parse_position",
]
