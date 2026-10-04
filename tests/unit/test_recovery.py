"""Boot recovery must not hang when Kalshi is unreachable (network down at boot)."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from decimal import Decimal

from marketlab.clock import SimulatedClock
from marketlab.core.instruments import Side
from marketlab.core.portfolio import Portfolio, Position
from marketlab.daemon import recovery as recovery_mod
from marketlab.daemon.recovery import RecoveryService
from marketlab.daemon.registry import BookRegistry, MarketRegistry, PortfolioRegistry

NOW = datetime(2026, 10, 4, 22, 38, tzinfo=UTC)


class _Exp:
    def __init__(self, experiment_id: str) -> None:
        self.experiment_id = experiment_id
        self.status = "PAPER"


class _Store:
    def __init__(self, n: int) -> None:
        self.n = n

    def list_experiments(self) -> list[_Exp]:
        return [_Exp(f"E{i}") for i in range(self.n)]

    def load_portfolio(self, experiment_id: str) -> Portfolio:
        cid = f"kalshi:mkt-{experiment_id}"
        pos = Position(canonical_id=cid, side=Side.YES, quantity=2, average_price=Decimal("0.40"))
        return Portfolio(experiment_id=experiment_id, strategy_id="s", cash=Decimal("49.20"),
                         positions={Portfolio.key(cid, Side.YES): pos})

    def open_orders(self) -> list:
        return []


class _HangingKalshi:
    async def get_market(self, ticker: str) -> dict:
        await asyncio.sleep(3600)
        return {}


async def test_restore_finishes_by_the_deadline_when_kalshi_hangs(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(recovery_mod, "RECONCILE_DEADLINE_SECONDS", 0.3)
    clock = SimulatedClock(NOW)
    service = RecoveryService(
        clock,
        market_registry=MarketRegistry(),
        book_registry=BookRegistry(),
        portfolio_registry=PortfolioRegistry(),
        kalshi_rest=_HangingKalshi(),
    )
    started = time.monotonic()
    summary = await service.restore(_Store(200), broker=None)
    assert time.monotonic() - started < 5
    assert summary.experiments_restored == 200
    assert any("unchecked" in note for note in summary.notes)
