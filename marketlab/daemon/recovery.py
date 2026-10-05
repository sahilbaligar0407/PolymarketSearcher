"""Crash/restart recovery: rebuild in-memory state from the durable store.

Nothing about a restart should look like a fresh start. ``RecoveryService.restore``
reloads every sleeve's real bankroll from :class:`~marketlab.storage.state.StateStore`
(never re-creates one), rebuilds the broker's resting-order book, and reconciles any
market that resolved while the process was down. It is the one place that gets to say
"nothing was lost" -- or to say precisely what could not be recovered.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from marketlab.adapters.kalshi.normalize import (
    normalize_market as normalize_kalshi_market,
)
from marketlab.adapters.kalshi.normalize import normalize_settlement
from marketlab.clock import Clock
from marketlab.core.instruments import MarketStatus, Venue
from marketlab.daemon.registry import BookRegistry, MarketRegistry, PortfolioRegistry
from marketlab.logging import get_logger

log = get_logger(__name__)

#: Settlement lookups in flight at once during boot recovery, and the total time allowed.
RECONCILE_CONCURRENCY = 8
RECONCILE_DEADLINE_SECONDS = 90.0


@dataclass
class RecoverySummary:
    experiments_restored: int = 0
    open_orders_restored: int = 0
    positions_restored: int = 0
    total_equity: Decimal = Decimal("0")
    settled_while_down: int = 0
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        return (
            f"restored {self.experiments_restored} experiment(s), "
            f"{self.open_orders_restored} open order(s), "
            f"{self.positions_restored} open position(s), "
            f"total equity ${self.total_equity}, "
            f"{self.settled_while_down} market(s) settled while down"
        )


class RecoveryService:
    """Orchestrates the crash-recovery half of the supervisor's boot sequence.

    The experiment runner (another team's module) owns reconstructing *sleeve
    identities* from ``configs/strategies.yaml`` variants; this service owns
    reconstructing *money* -- portfolios, resting orders, and settling anything that
    resolved on the venue while nobody was watching.
    """

    def __init__(
        self,
        clock: Clock,
        *,
        market_registry: MarketRegistry,
        book_registry: BookRegistry,
        portfolio_registry: PortfolioRegistry,
        kalshi_rest: Any | None = None,
    ) -> None:
        self.clock = clock
        self.markets = market_registry
        self.books = book_registry
        self.portfolios = portfolio_registry
        self.kalshi_rest = kalshi_rest

    async def restore(self, store: Any, broker: Any, runner: Any = None) -> RecoverySummary:
        summary = RecoverySummary()
        if store is None:
            summary.notes.append("no StateStore configured; starting cold (nothing to restore)")
            log.warning("recovery_no_store", detail=summary.notes[-1])
            return summary

        # 1. Resting paper orders. No `store` argument: PaperBroker's own `store`
        # (wired at construction via `daemon.supervisor._PaperBrokerStoreBridge`)
        # already satisfies its `StateStoreLike` protocol; the raw StateStore passed
        # into this method does not (see that bridge's docstring for the mismatch).
        if broker is not None and hasattr(broker, "load_open_orders"):
            try:
                broker.load_open_orders()
            except Exception as exc:  # noqa: BLE001 - recovery must never crash the boot
                summary.notes.append(f"load_open_orders failed: {exc}")
                log.warning("recovery_load_open_orders_failed", error=str(exc))

        # 2. Portfolios -- the real bankroll, never re-created.
        try:
            # Only live sleeves come back. A DEAD sleeve stays dead and a DISABLED
            # (retired) cohort must not be resurrected into the live portfolio registry -
            # otherwise every status report counts retired sleeves as alive, and the
            # operator sees 776 "alive" when 388 are actually trading.
            experiments = [
                e
                for e in store.list_experiments()
                if str(getattr(e, "status", "")) not in ("DEAD", "DISABLED")
            ]
        except Exception as exc:  # noqa: BLE001
            summary.notes.append(f"list_experiments failed: {exc}")
            log.warning("recovery_list_experiments_failed", error=str(exc))
            experiments = []

        marks = self.books.mark_prices()
        for exp in experiments:
            # A sleeve the runner already loaded keeps that one Portfolio object. Loading a
            # second copy here replaced the registry entry, so after every restart the
            # broker traded one object while the runner snapshotted and death-checked a
            # frozen other, and both were persisted (4,244 conflicting balance rows,
            # FINDINGS 58).
            existing = self.portfolios.get(exp.experiment_id)
            if existing is not None:
                summary.experiments_restored += 1
                summary.total_equity += existing.equity(marks)
                summary.positions_restored += sum(1 for p in existing.positions.values() if p.quantity > 0)
                continue
            try:
                portfolio = store.load_portfolio(exp.experiment_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("recovery_load_portfolio_failed", experiment_id=exp.experiment_id, error=str(exc))
                continue
            if portfolio is None:
                continue
            self.portfolios.register(portfolio)
            summary.experiments_restored += 1
            summary.total_equity += portfolio.equity(marks)
            summary.positions_restored += sum(1 for p in portfolio.positions.values() if p.quantity > 0)

        try:
            summary.open_orders_restored = len(store.open_orders())
        except Exception as exc:  # noqa: BLE001
            log.warning("recovery_open_orders_count_failed", error=str(exc))

        # 3. Reconcile: any market with an open position that resolved while we were down.
        canonical_ids_with_positions = {
            k.split("|", 1)[0]
            for portfolio in self.portfolios.all().values()
            for k, pos in portfolio.positions.items()
            if pos.quantity > 0
        }
        if self.kalshi_rest is not None:
            # Bounded: the per-market lookups run a few at a time under one deadline. Run
            # serially, hundreds of markets x 5 retried 10s requests blocked boot for hours
            # whenever the network was down (seen 2026-10-04). Anything not reconciled here
            # is still settled by the ingest settlements loop, which works from open
            # positions once the daemon is up.
            gate = asyncio.Semaphore(RECONCILE_CONCURRENCY)

            async def reconcile(canonical_id: str) -> None:
                ticker = canonical_id.split(":", 1)[1].upper()
                async with gate:
                    try:
                        raw = await self.kalshi_rest.get_market(ticker)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("recovery_market_reconcile_failed", canonical_id=canonical_id, error=str(exc))
                        return
                market_raw = raw.get("market", raw)
                status = normalize_kalshi_market(market_raw).status
                if status is not MarketStatus.SETTLED:
                    return
                winning_side, voided = normalize_settlement(market_raw)
                for portfolio in self.portfolios.all().values():
                    if any(k.startswith(canonical_id) for k in portfolio.positions):
                        portfolio.settle(canonical_id, winning_side)
                if hasattr(store, "save_settlement"):
                    from marketlab.storage.state import Settlement

                    store.save_settlement(
                        Settlement(
                            canonical_id=canonical_id,
                            venue=Venue.KALSHI,
                            winning_side=winning_side,
                            voided=voided,
                            settled_at=self.clock.now(),
                            first_seen_time=self.clock.now(),
                        )
                    )
                summary.settled_while_down += 1

            kalshi_ids = [c for c in canonical_ids_with_positions if c.startswith("kalshi:")]
            tasks = [asyncio.ensure_future(reconcile(c)) for c in kalshi_ids]
            if tasks:
                _done, pending = await asyncio.wait(tasks, timeout=RECONCILE_DEADLINE_SECONDS)
                for task in pending:
                    task.cancel()
                if pending:
                    summary.notes.append(
                        f"settlement reconcile: {len(pending)} of {len(tasks)} markets unchecked "
                        f"after {RECONCILE_DEADLINE_SECONDS:.0f}s; the settlements loop will finish them"
                    )
                    log.warning("recovery_reconcile_deadline", unchecked=len(pending), total=len(tasks))

        log.info(
            "recovery_complete",
            experiments=summary.experiments_restored,
            open_orders=summary.open_orders_restored,
            positions=summary.positions_restored,
            total_equity=str(summary.total_equity),
            settled_while_down=summary.settled_while_down,
        )
        return summary
