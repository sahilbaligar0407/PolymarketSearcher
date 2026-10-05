"""TypeSafe Jev client: shared calls, retry on 429, never raises."""

from __future__ import annotations

import asyncio

import httpx

from marketlab.ai.typesafe import JevClient, noul

Q = {"outcome": {"type": "noul", "instructions": "?"}}


def _client(handler) -> JevClient:  # noqa: ANN001
    return JevClient("k", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_identical_requests_share_one_call_and_cache() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json={"answers": {"outcome": {"type": "noul", "noul": 0.7}}})

    jev = _client(handler)
    a, b = await asyncio.gather(jev.evaluate("x", {}, Q), jev.evaluate("x", {}, Q))
    c = await jev.evaluate("x", {}, Q)
    assert noul(a, "outcome") == noul(b, "outcome") == noul(c, "outcome") == 0.7
    assert len(calls) == 1


async def test_rate_limit_is_retried() -> None:
    responses = [httpx.Response(429, headers={"retry-after": "0"}),
                 httpx.Response(200, json={"answers": {"outcome": {"type": "noul", "noul": 0.2}}})]
    jev = _client(lambda request: responses.pop(0))
    assert noul(await jev.evaluate("y", {}, Q), "outcome") == 0.2


async def test_failure_returns_none() -> None:
    jev = _client(lambda request: httpx.Response(500))
    assert await jev.evaluate("z", {}, Q) is None
    assert jev.failures == 1
