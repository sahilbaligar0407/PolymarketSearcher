"""Clock abstraction.

No strategy, signal, or analytics code may call ``datetime.now()`` directly.  All time
access is dependency-injected through a :class:`Clock` so that replay is deterministic
and point-in-time discipline is enforceable.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Time source injected into every component that needs 'now'."""

    def now(self) -> datetime:
        """Current time, always timezone-aware UTC."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Advance time by ``seconds``."""
        ...


class LiveClock:
    """Wall-clock time. Used in DATA_ONLY, PAPER and LIVE modes."""

    __slots__ = ()

    def now(self) -> datetime:
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class SimulatedClock:
    """Deterministic clock driven by replayed events.

    ``sleep`` advances the simulated instant without any real waiting, which makes
    backtests reproducible and fast.
    """

    __slots__ = ("_now",)

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("SimulatedClock requires a timezone-aware start time")
        self._now = start.astimezone(UTC)

    def now(self) -> datetime:
        return self._now

    def set(self, instant: datetime) -> None:
        """Jump the clock forward to ``instant``. Time never moves backwards."""
        if instant.tzinfo is None:
            raise ValueError("SimulatedClock requires timezone-aware instants")
        instant = instant.astimezone(UTC)
        if instant < self._now:
            raise ValueError(f"clock moved backwards: {self._now} -> {instant}")
        self._now = instant

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)

    async def sleep(self, seconds: float) -> None:
        self.advance(seconds)
