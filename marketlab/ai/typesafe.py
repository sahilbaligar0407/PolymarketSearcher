"""TypeSafe's Jev: a "System One" model for typed, calibrated judgements.

Jev is not a chat model. ``POST https://api.typesafe.ai/v1/systemone`` takes a ``state``
(any JSON) and a map of typed ``questions`` - ``noul`` (yes/no, answered as a probability),
``choice`` and ``score`` - and returns one answer per question in ~0.5 s (measured
2026-10-04: 469 ms for two questions). That suits a per-candidate trade check far better
than a multi-second generative call.

The client is shared by every sleeve. Identical (key) requests share one in-flight call
and are cached, because a dozen variants evaluating the same consensus row in the same
snapshot would otherwise each pay for the same answer. Failures never raise into a
strategy: ``evaluate`` returns ``None`` and the caller treats that as "no confirmation".
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from marketlab.logging import get_logger

log = get_logger(__name__)

API_URL = "https://api.typesafe.ai/v1/systemone"


class JevClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "jev-latest",
        timeout: float = 10.0,
        max_concurrency: int = 8,
        cache_seconds: float = 900.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._key = api_key
        self.model = model
        self._http = http or httpx.AsyncClient(timeout=timeout)
        self._gate = asyncio.Semaphore(max_concurrency)
        self._cache_seconds = cache_seconds
        self._cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
        self._inflight: dict[str, asyncio.Future[dict[str, Any] | None]] = {}
        self.calls = 0
        self.failures = 0
        self.input_tokens = 0

    async def evaluate(
        self, key: str, state: Any, questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any] | None:
        """``answers`` for ``questions`` against ``state``, or None on any failure."""
        hit = self._cache.get(key)
        if hit is not None and time.monotonic() - hit[0] < self._cache_seconds:
            return hit[1]
        pending = self._inflight.get(key)
        if pending is not None:
            return await pending
        future: asyncio.Future[dict[str, Any] | None] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            answers = await self._post(state, questions)
            self._cache[key] = (time.monotonic(), answers)
            future.set_result(answers)
            return answers
        finally:
            self._inflight.pop(key, None)
            if not future.done():
                future.set_result(None)

    async def _post(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        body = {"model": self.model, "state": state, "questions": questions}
        async with self._gate:
            for attempt in range(3):
                try:
                    self.calls += 1
                    resp = await self._http.post(
                        API_URL, json=body, headers={"Authorization": f"Bearer {self._key}"}
                    )
                    if resp.status_code == 429 and attempt < 2:
                        await asyncio.sleep(float(resp.headers.get("retry-after", 1.0)))
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    self.input_tokens += int((data.get("usage") or {}).get("input_tokens") or 0)
                    answers = data.get("answers")
                    return answers if isinstance(answers, dict) else None
                except Exception as exc:  # noqa: BLE001 - a confirmation tier must never raise
                    if attempt == 2:
                        self.failures += 1
                        log.warning("jev_request_failed", error=str(exc)[:200])
                        return None
                    await asyncio.sleep(0.5 * (attempt + 1))
        return None

    async def close(self) -> None:
        await self._http.aclose()


def noul(answers: dict[str, Any] | None, question_id: str) -> float | None:
    """The yes-probability of one ``noul`` answer, if present."""
    if not answers:
        return None
    answer = answers.get(question_id) or {}
    value = answer.get("noul")
    return float(value) if isinstance(value, int | float) else None


def build_jev_client(settings: Any) -> JevClient | None:
    key = str(getattr(settings.secrets, "typesafe_api_key", "") or "")
    if not key:
        return None
    return JevClient(key, model=str(getattr(settings.secrets, "typesafe_model", "") or "jev-latest"))


__all__ = ["API_URL", "JevClient", "build_jev_client", "noul"]
