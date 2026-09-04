"""Bounded, O(1)-update online accumulators for streaming feature computation.

Strategies run continuously for days across hundreds of markets. They cannot keep
unbounded history of every tick, so every primitive here is memory-bounded (a
``deque(maxlen=...)`` or a handful of scalars) and updates in O(1) per observation.

**No wall-clock access anywhere in this module.** Every time-dependent method takes an
explicit ``now`` (or ``timestamp``) argument, exactly like the rest of MarketLab, so that
replay against historical data is bit-for-bit deterministic and these accumulators are
trivially testable without mocking ``datetime.now()``.

All accumulators tolerate ``None``/``NaN`` input by ignoring it rather than raising or
poisoning subsequent results, because a live feed will eventually hand you a gap.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from decimal import Decimal
from statistics import fmean

Number = float | int | Decimal


def _is_bad(x: float | None) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _to_float(x: Number) -> float:
    return float(x)


class RollingWindow:
    """Fixed-capacity FIFO window of numeric samples with O(1) push and cheap stats.

    Bad values (``None``/``NaN``) are silently dropped rather than occupying a slot -
    a strategy asking "what's my last 20 valid prints" should not get a window half full
    of holes.
    """

    def __init__(self, maxlen: int) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self._buf: deque[float] = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def push(self, x: Number | None) -> None:
        if x is None:
            return
        f = _to_float(x)
        if math.isnan(f):
            return
        self._buf.append(f)

    def __len__(self) -> int:
        return len(self._buf)

    def len(self) -> int:
        return len(self._buf)

    @property
    def full(self) -> bool:
        return len(self._buf) == self._maxlen

    def mean(self) -> float | None:
        if not self._buf:
            return None
        return fmean(self._buf)

    def std(self) -> float | None:
        """Sample standard deviation. ``None`` for fewer than 2 points or zero variance."""
        n = len(self._buf)
        if n < 2:
            return None
        m = fmean(self._buf)
        var = sum((x - m) ** 2 for x in self._buf) / (n - 1)
        if var <= 0:
            return None
        return math.sqrt(var)

    def min(self) -> float | None:
        return min(self._buf) if self._buf else None

    def max(self) -> float | None:
        return max(self._buf) if self._buf else None

    def last(self) -> float | None:
        return self._buf[-1] if self._buf else None

    def first(self) -> float | None:
        return self._buf[0] if self._buf else None

    def sum(self) -> float | None:
        if not self._buf:
            return None
        return math.fsum(self._buf)

    def values(self) -> list[float]:
        return list(self._buf)


class EWMA:
    """Exponentially weighted moving average that decays by *elapsed time*, not by tick.

    Market data does not arrive on a fixed grid, so weighting each update by a constant
    decay factor (as a naive tick-based EWMA would) implicitly treats a quiet market and a
    busy one identically. Instead the decay factor between two updates is
    ``exp(-ln(2) * dt / halflife)`` where ``dt`` is the actual elapsed wall time between
    observations, so two updates a second apart barely move the average while two updates
    a minute apart move it most of the way to the new value.
    """

    def __init__(self, halflife_seconds: float) -> None:
        if halflife_seconds <= 0:
            raise ValueError("halflife_seconds must be positive")
        self.halflife_seconds = halflife_seconds
        self._value: float | None = None
        self._last_timestamp: datetime | None = None

    def update(self, x: Number | None, timestamp: datetime) -> float | None:
        if x is None:
            return self._value
        f = _to_float(x)
        if math.isnan(f):
            return self._value
        if self._value is None or self._last_timestamp is None:
            self._value = f
            self._last_timestamp = timestamp
            return self._value
        dt = (timestamp - self._last_timestamp).total_seconds()
        if dt < 0:
            # Out-of-order data: don't let it corrupt the average or run time backwards.
            return self._value
        alpha = 1.0 - math.exp(-math.log(2.0) * dt / self.halflife_seconds)
        self._value = self._value + alpha * (f - self._value)
        self._last_timestamp = timestamp
        return self._value

    @property
    def value(self) -> float | None:
        return self._value


class RollingZScore:
    """Z-score of the latest observation against a trailing window of prior observations."""

    def __init__(self, window: int) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        self._window = RollingWindow(window)

    def update(self, x: Number | None) -> float | None:
        """Push ``x`` and return its z-score against the window *including* itself.

        Returns ``None`` until the window holds at least two valid points, or if the
        window has zero variance (all values identical) - a z-score against zero spread
        is undefined, not infinite.
        """
        if x is None:
            return None
        f = _to_float(x)
        if math.isnan(f):
            return None
        self._window.push(f)
        mean = self._window.mean()
        std = self._window.std()
        if mean is None or std is None:
            return None
        return (f - mean) / std

    def full(self) -> bool:
        return self._window.full


class TimeWindow:
    """Keeps ``(timestamp, value)`` pairs and evicts anything older than ``now - seconds``.

    Every method takes an explicit ``now`` - this class never reads the wall clock, so a
    backtest replaying historical data gets identical eviction behavior to live trading.
    """

    def __init__(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        self.seconds = seconds
        self._items: deque[tuple[datetime, object]] = deque()

    def add(self, timestamp: datetime, value: object) -> None:
        self._items.append((timestamp, value))

    def evict(self, now: datetime) -> None:
        cutoff = now.timestamp() - self.seconds
        while self._items and self._items[0][0].timestamp() < cutoff:
            self._items.popleft()

    def values(self, now: datetime) -> list[object]:
        self.evict(now)
        return [v for _, v in self._items]

    def items(self, now: datetime) -> list[tuple[datetime, object]]:
        self.evict(now)
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def len(self, now: datetime) -> int:
        self.evict(now)
        return len(self._items)


class WelfordVariance:
    """Numerically stable online mean/variance (Welford's algorithm).

    Preferred over a naive sum-of-squares accumulator, which loses precision catastrophically
    for long-running series with a large mean and small variance - exactly the shape of a
    prediction-contract price series sitting near 0.95 for days.
    """

    def __init__(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, x: Number | None) -> None:
        if x is None:
            return
        f = _to_float(x)
        if math.isnan(f):
            return
        self._n += 1
        delta = f - self._mean
        self._mean += delta / self._n
        delta2 = f - self._mean
        self._m2 += delta * delta2

    @property
    def n(self) -> int:
        return self._n

    def mean(self) -> float | None:
        return self._mean if self._n > 0 else None

    def variance(self) -> float | None:
        """Sample variance. ``None`` for fewer than 2 points or exactly-zero variance."""
        if self._n < 2:
            return None
        var = self._m2 / (self._n - 1)
        return var if var > 0 else None

    def std(self) -> float | None:
        var = self.variance()
        return math.sqrt(var) if var is not None else None
