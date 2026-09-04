"""Shared adapter/HTTP contract used by every venue and data-source integration.

This is the one module every team's adapter (Kalshi, Polymarket, SEC, weather, ...)
imports.  It intentionally knows nothing about any particular venue: no Kalshi-specific
headers, no Polymarket-specific query params.  Keep it that way.

Design rules baked in here, because they're easy to get wrong under deadline pressure:

* A missing credential must never raise. ``probe()`` reports ``NO_CREDENTIALS`` and the
  rest of the engine keeps running in degraded mode.
* Every HTTP request is logged structurally (method, path, status, elapsed_ms) and never
  logs headers, since auth material lives there.
* Retries are policy, not ad-hoc ``except`` blocks: 429/5xx/network errors retry with
  exponential backoff and jitter, honoring ``Retry-After`` when present.
"""

from __future__ import annotations

import abc
from datetime import datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from marketlab.adapters.ratelimit import RateLimiter
from marketlab.clock import Clock, LiveClock
from marketlab.logging import get_logger

log = get_logger(__name__)


class SourceStatus(StrEnum):
    """Health state of a data source or venue connection, surfaced by `marketlab doctor`."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STALE = "stale"
    DOWN = "down"
    NO_CREDENTIALS = "no_credentials"
    DISABLED = "disabled"


class SourceHealth(BaseModel):
    """A point-in-time health snapshot for one adapter."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: SourceStatus
    last_message_at: datetime | None = None
    latency_ms: float | None = None
    error_count: int = 0
    reconnect_count: int = 0
    detail: str = ""


class RetryableHttpError(Exception):
    """Wraps a retryable HTTP failure so tenacity's predicate can be simple and typed."""

    def __init__(self, status_code: int, retry_after: float | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, RetryableHttpError | httpx.TransportError | httpx.TimeoutException)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    delta = (dt - datetime.now(dt.tzinfo)).total_seconds()
    return max(delta, 0.0)


class HttpAdapter:
    """A retrying, rate-limited, health-tracked wrapper around ``httpx.AsyncClient``.

    Subclasses (or composing adapters) call :meth:`get_json` / :meth:`post_json` and get,
    for free: exponential backoff on 429/5xx/network errors that respects
    ``Retry-After``, a shared token-bucket rate limiter with per-call ``cost``, structured
    per-request logging that never includes headers, and a rolling health snapshot.
    """

    def __init__(
        self,
        base_url: str,
        *,
        name: str,
        default_headers: dict[str, str] | None = None,
        timeout: float = 10.0,
        rate_limiter: RateLimiter | None = None,
        clock: Clock | None = None,
        max_attempts: int = 5,
    ) -> None:
        self.name = name
        self._clock = clock or LiveClock()
        self._rate_limiter = rate_limiter or RateLimiter()
        self._max_attempts = max_attempts
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=default_headers or {},
            timeout=timeout,
        )
        self._status: SourceStatus = SourceStatus.HEALTHY
        self._last_message_at: datetime | None = None
        self._last_latency_ms: float | None = None
        self._error_count = 0
        self._reconnect_count = 0
        self._detail = ""

    @property
    def rate_limiter(self) -> RateLimiter:
        return self._rate_limiter

    async def close(self) -> None:
        await self._client.aclose()

    def health(self) -> SourceHealth:
        return SourceHealth(
            name=self.name,
            status=self._status,
            last_message_at=self._last_message_at,
            latency_ms=self._last_latency_ms,
            error_count=self._error_count,
            reconnect_count=self._reconnect_count,
            detail=self._detail,
        )

    def _mark_success(self) -> None:
        self._status = SourceStatus.HEALTHY
        self._last_message_at = self._clock.now()
        self._detail = ""

    def _mark_error(self, detail: str) -> None:
        self._error_count += 1
        self._status = SourceStatus.DEGRADED
        self._detail = detail

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        cost: float = 1.0,
    ) -> dict[str, Any]:
        await self._rate_limiter.acquire(cost=cost)

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=0.5, max=20.0),
            retry=retry_if_exception(_is_retryable),
        )
        async def _do() -> httpx.Response:
            started = self._clock.now()
            try:
                resp = await self._client.request(
                    method, path, params=params, headers=headers, json=json_body
                )
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                elapsed_ms = (self._clock.now() - started).total_seconds() * 1000
                log.warning(
                    "http_request_failed",
                    source=self.name,
                    method=method,
                    path=path,
                    elapsed_ms=round(elapsed_ms, 1),
                    error=str(exc),
                )
                self._mark_error(str(exc))
                raise
            elapsed_ms = (self._clock.now() - started).total_seconds() * 1000
            log.info(
                "http_request",
                source=self.name,
                method=method,
                path=path,
                status=resp.status_code,
                elapsed_ms=round(elapsed_ms, 1),
            )
            self._last_latency_ms = elapsed_ms
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = _parse_retry_after(resp.headers.get("retry-after"))
                if resp.status_code == 429:
                    self._rate_limiter.on_429(retry_after)
                self._mark_error(f"http {resp.status_code}")
                raise RetryableHttpError(resp.status_code, retry_after, f"http {resp.status_code}")
            return resp

        resp = await _do()
        resp.raise_for_status()
        self._mark_success()
        if not resp.content:
            return {}
        return resp.json()

    async def get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        cost: float = 1.0,
    ) -> dict[str, Any]:
        return await self._request("GET", path, params=params, headers=headers, cost=cost)

    async def post_json(
        self,
        path: str,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        cost: float = 1.0,
    ) -> dict[str, Any]:
        return await self._request("POST", path, headers=headers, json_body=json_body, cost=cost)


class Adapter(abc.ABC):
    """Base for every venue/data-source adapter driven by the engine."""

    name: str

    @abc.abstractmethod
    async def probe(self) -> SourceHealth:
        """Cheap connectivity/auth check used by `marketlab doctor`.

        Must never raise: a missing credential or an unreachable host is a
        ``SourceHealth`` with status ``NO_CREDENTIALS`` / ``DOWN``, not an exception.
        """
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        """Release any held connections (HTTP clients, websockets, ...)."""
        ...
