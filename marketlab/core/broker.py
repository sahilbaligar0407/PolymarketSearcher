"""Broker interface.

The single place that decides whether an :class:`OrderIntent` becomes a simulated fill or
a real venue order.  Strategies hold a reference to nothing here - the runner owns the
broker and feeds results back to the strategy through ``on_fill`` / ``on_order_update``.
"""

from __future__ import annotations

import abc
from enum import StrEnum

from marketlab.core.orders import Order, OrderIntent


class Mode(StrEnum):
    """Operating mode. LIVE must be impossible to reach accidentally."""

    DATA_ONLY = "DATA_ONLY"
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"

    @property
    def allows_orders(self) -> bool:
        return self in {Mode.BACKTEST, Mode.PAPER, Mode.LIVE}

    @property
    def is_real_money(self) -> bool:
        return self is Mode.LIVE


class Broker(abc.ABC):
    """Anything that can turn intents into orders."""

    mode: Mode

    @abc.abstractmethod
    async def submit(self, intent: OrderIntent) -> Order:
        """Accept or reject an intent and return the resulting order state."""

    @abc.abstractmethod
    async def cancel(self, order_id: str) -> Order | None:
        """Cancel a resting order. Returns the final order state, or None if unknown."""

    @abc.abstractmethod
    async def open_orders(self, strategy_id: str | None = None) -> list[Order]:
        """Currently resting orders, optionally filtered to one strategy."""

    @abc.abstractmethod
    async def get_order(self, order_id: str) -> Order | None: ...
