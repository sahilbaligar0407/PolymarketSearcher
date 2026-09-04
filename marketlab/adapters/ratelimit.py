"""Async token-bucket rate limiting with per-endpoint costs.

Kalshi (and most venues) do not expose one universal request budget: different
endpoints cost different numbers of tokens, and a 429 response should shrink the
effective rate temporarily rather than being treated as a fixed, hard-coded ceiling.
This module is deliberately venue-agnostic so any adapter (not just Kalshi) can share it.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


@dataclass(slots=True)
class _BucketState:
    """Mutable bucket state, isolated so it can be swapped atomically under a lock."""

    rate: float  # tokens refilled per second
    burst: float  # max tokens the bucket can hold
    tokens: float
    last_refill: float


class RateLimiter:
    """Async token bucket supporting variable per-call costs and dynamic capacity.

    Usage::

        limiter = RateLimiter(rate=10.0, burst=20.0)
        await limiter.acquire(cost=2)   # e.g. an endpoint that costs 2 tokens

    On a 429, call :meth:`on_429` with the server's ``Retry-After`` (seconds, may be
    ``None``). The limiter halves its rate immediately and recovers linearly back to the
    configured rate over ``recovery_seconds`` once the retry-after window has elapsed.
    """

    def __init__(
        self,
        rate: float = 10.0,
        burst: float | None = None,
        *,
        recovery_seconds: float = 60.0,
        clock: _MonoClock | None = None,
    ) -> None:
        self._base_rate = rate
        self._base_burst = burst if burst is not None else rate
        self._recovery_seconds = recovery_seconds
        self._clock = clock or _MonoClock()
        now = self._clock.now()
        self._state = _BucketState(rate=rate, burst=self._base_burst, tokens=self._base_burst, last_refill=now)
        self._lock = asyncio.Lock()
        self._degraded_until: float | None = None
        self._degrade_started_at: float | None = None

    def update_capacity(self, rate: float, burst: float | None = None) -> None:
        """Reconfigure the steady-state rate/burst, e.g. from a discovered header."""
        self._base_rate = rate
        self._base_burst = burst if burst is not None else rate
        if self._degraded_until is None:
            self._state.rate = rate
            self._state.burst = self._base_burst

    def on_429(self, retry_after: float | None) -> None:
        """Halve the rate temporarily; it recovers back to normal afterward.

        ``retry_after`` (seconds) delays the next token grant outright when given.
        """
        now = self._clock.now()
        self._state.rate = max(self._state.rate / 2.0, 0.01)
        self._degrade_started_at = now
        wait = retry_after or 0.0
        self._degraded_until = now + wait
        # Draining tokens to zero forces the next acquire() to wait out the reduced rate
        # (and the retry-after floor) rather than spending a stale burst allowance.
        self._state.tokens = 0.0

    def _refill(self, now: float) -> None:
        state = self._state
        # If we are past a degrade window, ease the rate back toward baseline instead of
        # snapping instantly - a burst of 429s should produce a gradual recovery.
        if self._degrade_started_at is not None:
            elapsed = now - self._degrade_started_at
            if elapsed >= self._recovery_seconds:
                state.rate = self._base_rate
                state.burst = self._base_burst
                self._degrade_started_at = None
                self._degraded_until = None
            else:
                # Linear interpolation from half-rate back to base rate over the window.
                frac = elapsed / self._recovery_seconds
                half = self._base_rate / 2.0
                state.rate = half + (self._base_rate - half) * frac
        elapsed = max(now - state.last_refill, 0.0)
        state.tokens = min(state.burst, state.tokens + elapsed * state.rate)
        state.last_refill = now

    async def acquire(self, cost: float = 1.0) -> None:
        """Block until ``cost`` tokens are available, then spend them."""
        while True:
            async with self._lock:
                now = self._clock.now()
                if self._degraded_until is not None and now < self._degraded_until:
                    wait = self._degraded_until - now
                else:
                    self._refill(now)
                    if self._state.tokens >= cost:
                        self._state.tokens -= cost
                        return
                    deficit = cost - self._state.tokens
                    wait = deficit / self._state.rate if self._state.rate > 0 else 1.0
            await asyncio.sleep(max(wait, 0.001))

    @property
    def current_rate(self) -> float:
        return self._state.rate


class _MonoClock:
    """Wall-clock-independent monotonic time source for the limiter's internal math.

    Deliberately separate from :class:`marketlab.clock.Clock` (which deals in
    tz-aware ``datetime`` for event ordering); this is plain float seconds for bucket
    arithmetic and is fine to back with ``time.monotonic``.
    """

    def now(self) -> float:
        return time.monotonic()
