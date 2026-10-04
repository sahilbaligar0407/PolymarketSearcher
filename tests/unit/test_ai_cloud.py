"""Remote LLM tiers: the OpenAI spend ledger and the generic chat-completions client."""

from __future__ import annotations

import json
from decimal import Decimal

import httpx

from marketlab.ai.cloud import ChatCompletionsProvider, SpendLedger, _extract_json, price_for

SCHEMA = {"type": "object", "properties": {"p_yes": {"type": "number"}}, "required": ["p_yes"]}


def _ok(content: str, tokens_in: int = 100, tokens_out: int = 20) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
        },
    )


def _provider(handler, ledger: SpendLedger | None = None, name: str = "openai") -> ChatCompletionsProvider:
    return ChatCompletionsProvider(
        name=name, base_url="https://example.test", api_key="k", model="gpt-4.1-nano",
        ledger=ledger, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def test_ledger_persists_and_caps(tmp_path):
    path = tmp_path / "spend.json"
    ledger = SpendLedger(path, Decimal("0.01"))
    assert await ledger.reserve(Decimal("0.006"))
    assert not await ledger.reserve(Decimal("0.006"))  # would exceed the cap
    await ledger.settle(Decimal("0.006"), Decimal("0.001"))
    assert ledger.spent_today() == Decimal("0.001")
    # A restart must not reset the day's spend.
    assert SpendLedger(path, Decimal("0.01")).spent_today() == Decimal("0.001")


async def test_successful_call_is_metered(tmp_path):
    ledger = SpendLedger(tmp_path / "s.json", Decimal("0.02"))
    provider = _provider(lambda req: _ok('{"p_yes": 0.7}', 1000, 100), ledger)
    resp = await provider.generate("prompt", schema=SCHEMA)
    assert resp.parsed == {"p_yes": 0.7}
    assert resp.model == "openai:gpt-4.1-nano"
    price_in, price_out = price_for("gpt-4.1-nano")
    expected = (1000 * price_in + 100 * price_out) / Decimal(1_000_000)
    assert ledger.spent_today() == expected


async def test_exhausted_budget_abstains_without_calling(tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        return _ok('{"p_yes": 0.7}')

    provider = _provider(handler, SpendLedger(tmp_path / "s.json", Decimal("0.0000001")))
    resp = await provider.generate("prompt", schema=SCHEMA)
    assert resp.parsed is None
    assert resp.raw["reason"] == "budget_exhausted"
    assert calls == []


async def test_http_error_abstains_and_keeps_worst_case_booked(tmp_path):
    ledger = SpendLedger(tmp_path / "s.json", Decimal("0.02"))
    provider = _provider(lambda req: httpx.Response(500, json={"error": "boom"}), ledger)
    resp = await provider.generate("prompt", schema=SCHEMA)
    assert resp.parsed is None
    assert resp.raw["reason"].startswith("error")
    # Without usage numbers the worst case stays booked: never under-count spend.
    assert ledger.spent_today() > 0


async def test_falls_back_when_response_format_unsupported():
    bodies = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": "response_format not supported"})
        return _ok('Sure! ```json\n{"p_yes": 0.4}\n```')

    provider = _provider(handler, name="jev")
    resp = await provider.generate("prompt", schema=SCHEMA)
    assert resp.parsed == {"p_yes": 0.4}
    assert "JSON schema" in bodies[-1]["messages"][-1]["content"]
    # The next call skips the doomed response_format attempt.
    await provider.generate("prompt", schema=SCHEMA)
    assert "response_format" not in bodies[-1]


def test_extract_json_tolerates_wrappers():
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('text before {"a": 2} after') == {"a": 2}
    assert _extract_json("no json here") is None
    assert _extract_json("[1, 2]") is None


def test_unknown_model_priced_conservatively():
    assert price_for("some-new-model") >= price_for("gpt-4.1-nano")
