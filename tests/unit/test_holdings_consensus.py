"""Leaderboard holdings consensus: the aggregation and the copy strategy built on it."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from marketlab.core.instruments import Side
from marketlab.signals.holdings import Holding, HoldingsBook, consensus, parse_position
from marketlab.strategies.holdings_consensus import HoldingsConsensusStrategy
from tests.fixtures.strategy_harness import StrategyHarness, make_book, make_market

NOW = datetime(2026, 10, 4, 22, 0, tzinfo=UTC)
GAME = "0xgame"
KALSHI = "kalshi:kxmlbgame-26oct04atllad-atl"


def _h(wallet: str, outcome_index: int, avg: str = "0.34", cur: str = "0.40", cid: str = GAME) -> Holding:
    return Holding(
        wallet=wallet, condition_id=cid, outcome_index=outcome_index,
        outcome="Braves" if outcome_index == 0 else "Dodgers", title="Braves vs. Dodgers",
        size=Decimal(100), avg_price=Decimal(avg), cur_price=Decimal(cur), usd_value=Decimal(40),
    )


# ---------------------------------------------------------------------------- aggregation


def test_parse_position_drops_resolved_and_decided_markets() -> None:
    live = {"conditionId": GAME, "outcomeIndex": 1, "outcome": "Dodgers", "title": "t",
            "size": 10, "avgPrice": 0.6, "curPrice": 0.61, "currentValue": 6.1}
    assert parse_position("w", live).outcome_index == 1
    assert parse_position("w", {**live, "redeemable": True}) is None
    assert parse_position("w", {**live, "curPrice": 0.995}) is None
    assert parse_position("w", {**live, "curPrice": 0.0}) is None


def test_consensus_counts_holders_and_flags_the_other_side() -> None:
    holdings = [_h("a", 0), _h("b", 0), _h("c", 0), _h("d", 1), _h("e", 0), _h("e", 1)]
    rows = consensus(holdings, ["a", "b", "c", "d", "e"], net_only=False)
    braves = next(r for r in rows if r.outcome_index == 0)
    assert braves.n_holders == 4 and braves.opposing == ["d"]
    # net_only drops e, who holds both sides: a hedge is not a view.
    net = next(r for r in consensus(holdings, ["a", "b", "c", "d", "e"], net_only=True) if r.outcome_index == 0)
    assert net.holders == ["a", "b", "c"] and net.opposing == ["d"]
    assert net.net_holders == 2


def test_consensus_only_counts_wallets_in_the_view() -> None:
    rows = consensus([_h("a", 0), _h("outsider", 0)], ["a"], net_only=True)
    assert rows[0].holders == ["a"]


def test_book_views_by_board_and_union() -> None:
    book = HoldingsBook()
    book.replace({"all": ["a", "b"], "month": ["b", "c"], "week": ["d"]},
                 [_h(w, 0) for w in "abcd"], NOW)
    assert book.wallets("union", 100) == ["a", "b", "c", "d"]
    assert book.view("month", 100, True)[0].holders == ["b", "c"]
    assert book.view("all", 1, True)[0].holders == ["a"]


# ---------------------------------------------------------------------------- strategy


class _Matches:
    def get(self, key: str):  # noqa: ANN201
        if key == f"poly:{GAME}":
            return SimpleNamespace(canonical_id_a=KALSHI, same_outcome_boolean=True)
        return None


def _harness(holders: int, ask: str, avg: str = "0.34", **params) -> StrategyHarness:  # noqa: ANN003
    book = HoldingsBook()
    wallets = [f"w{i}" for i in range(holders)]
    book.replace({"all": wallets, "month": [], "week": []}, [_h(w, 0, avg=avg) for w in wallets], NOW)
    h = StrategyHarness(HoldingsConsensusStrategy, params={
        "board": "all", "min_holders": 3, "holdings": book, "matches": _Matches(), **params,
    })
    h.set_market(make_market(KALSHI))
    bid = str(Decimal(ask) - Decimal("0.02"))
    h.set_book(make_book(KALSHI, [(bid, 500)], [(ask, 500)], h.now()))
    return h


def test_blind_copies_the_shared_holding_as_kalshi_yes() -> None:
    h = _harness(holders=4, ask="0.41", entry="blind")
    h.feed_timer()
    assert len(h.intents) == 1
    intent = h.intents[0]
    assert intent.canonical_id == KALSHI and intent.side is Side.YES
    assert intent.features["holders"] == 4


def test_too_few_holders_does_nothing() -> None:
    h = _harness(holders=2, ask="0.41", entry="blind")
    h.feed_timer()
    assert h.intents == []


def test_edge_refuses_a_price_that_ran_past_the_holders_entry() -> None:
    h = _harness(holders=4, ask="0.60", avg="0.34", entry="edge", max_chase="0.05")
    h.feed_timer()
    assert h.intents == []
    assert h.strategy.refusals["price_ran_past_holders_entry"] == 1


def test_edge_refuses_five_dollars_to_make_fifty_cents() -> None:
    h = _harness(holders=4, ask="0.93", avg="0.92", entry="edge", max_chase="0.05", min_remaining="0.10")
    h.feed_timer()
    assert h.intents == []
    assert h.strategy.refusals["too_little_upside_left"] == 1


def test_enters_a_market_once() -> None:
    h = _harness(holders=4, ask="0.41", entry="blind")
    h.feed_timer()
    h.advance(120)
    h.feed_timer()
    assert len(h.intents) == 1


def test_news_confirm_needs_a_story() -> None:
    evidence = SimpleNamespace(get_news=lambda cid: [])
    h = _harness(holders=4, ask="0.41", entry="blind", confirm="news", evidence=evidence)
    h.feed_timer()
    assert h.intents == [] and h.strategy.refusals["no_supporting_news"] == 1


class _Jev:
    def __init__(self, p: float) -> None:
        self.p = p
        self.calls = 0

    async def evaluate(self, key, state, questions):  # noqa: ANN001, ANN201
        self.calls += 1
        return {"outcome": {"type": "noul", "noul": self.p}}


@pytest.mark.parametrize(("p", "trades"), [(0.60, True), (0.42, False)])
async def test_jev_confirm_trades_only_when_jev_beats_the_price(p: float, trades: bool) -> None:
    jev = _Jev(p)
    h = _harness(holders=4, ask="0.41", entry="blind", confirm="jev", jev=jev, jev_min_edge="0.03")
    h.feed_timer()  # asks Jev
    assert h.intents == []
    await asyncio.sleep(0)
    h.advance(61)
    h.set_book(make_book(KALSHI, [("0.39", 500)], [("0.41", 500)], h.now()))
    h.feed_timer()  # uses the answer
    assert (len(h.intents) == 1) is trades
    assert jev.calls == 1


def test_kelly_bets_more_on_bigger_edges_and_nothing_without_one() -> None:
    h = StrategyHarness(HoldingsConsensusStrategy, params={"bankroll": "100.00"})
    s = h.strategy
    assert s.kelly_quantity(Decimal("0.50"), Decimal("0.50")) == 0
    small = s.kelly_quantity(Decimal("0.50"), Decimal("0.54"))
    big = s.kelly_quantity(Decimal("0.50"), Decimal("0.70"))
    assert 0 < small < big
    # capped at 4% of the bankroll: $4 at 50c is 8 contracts, however large the edge
    assert s.kelly_quantity(Decimal("0.50"), Decimal("0.99")) == 8


def test_fixed_sizing_scales_with_the_bankroll() -> None:
    small = StrategyHarness(HoldingsConsensusStrategy, params={"bankroll": "50.00"}).strategy
    big = StrategyHarness(HoldingsConsensusStrategy, params={"bankroll": "100.00"}).strategy
    assert big.sensible_quantity(Decimal("0.10")) == 2 * small.sensible_quantity(Decimal("0.10"))
