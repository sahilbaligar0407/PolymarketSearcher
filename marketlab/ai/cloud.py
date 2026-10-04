"""Remote chat-completions providers: OpenAI (budget-capped) and Jev.

The local Ollama model does the bulk of the work. These are the two remote tiers:

* **OpenAI** -- a *second opinion* consulted only when a local assessment would actually
  trade. Every call is metered against a persisted daily spend ledger, and a call whose
  worst-case cost would exceed what is left of the day's budget is never sent. Running
  out of budget degrades to an abstain, which the strategy treats as NO TRADE.
* **Jev** -- an OpenAI-compatible endpoint configured entirely by ``JEV_BASE_URL`` /
  ``JEV_API_KEY`` / ``JEV_MODEL``. Free to use, so no ledger by default. When the URL is
  unset, no Jev provider exists and Jev-arm sleeves are simply not created.

Both obey the same contract as every :class:`~marketlab.ai.provider.LLMProvider`: never
raise into the engine, return ``parsed=None`` for anything unusable, and never receive a
tool, a credential other than their own bearer token, or anything that can act on a venue.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from marketlab.ai.provider import LLMProvider, LLMResponse, ProviderHealth
from marketlab.logging import get_logger
from marketlab.settings import DATA_DIR, Settings

log = get_logger(__name__)

#: USD per 1M tokens, (input, output). Unknown models are priced at the most expensive
#: listed rate so an unrecognised model can only ever under-spend the budget.
MODEL_PRICES_PER_M: dict[str, tuple[Decimal, Decimal]] = {
    "gpt-4.1-nano": (Decimal("0.10"), Decimal("0.40")),
    "gpt-4o-mini": (Decimal("0.15"), Decimal("0.60")),
    "gpt-4.1-mini": (Decimal("0.40"), Decimal("1.60")),
    "gpt-5-nano": (Decimal("0.05"), Decimal("0.40")),
    "gpt-5-mini": (Decimal("0.25"), Decimal("2.00")),
}
_FALLBACK_PRICE = (Decimal("0.40"), Decimal("2.00"))

#: Conservative chars-per-token for the pre-call worst-case estimate. Real English text
#: runs ~4; JSON and tickers run lower, so 3 over-estimates rather than under-estimates.
_CHARS_PER_TOKEN = 3


def price_for(model: str) -> tuple[Decimal, Decimal]:
    for prefix, price in MODEL_PRICES_PER_M.items():
        if model == prefix or model.startswith(prefix + "-"):
            return price
    return _FALLBACK_PRICE


class SpendLedger:
    """A persisted per-UTC-day spend counter with a hard cap.

    Persisted so a restart cannot reset the day's spend and quietly double the budget.
    ``reserve`` is the only way to get permission to spend: it books the worst-case cost
    up front, and ``settle`` later swaps that reservation for the metered actual.
    """

    def __init__(self, path: Path, daily_cap_usd: Decimal) -> None:
        self.path = path
        self.daily_cap = daily_cap_usd
        self._lock = asyncio.Lock()
        self._spent: dict[str, str] = {}
        try:
            self._spent = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._spent = {}

    @staticmethod
    def _today() -> str:
        return datetime.now(UTC).date().isoformat()

    def spent_today(self) -> Decimal:
        return Decimal(self._spent.get(self._today(), "0"))

    def remaining(self) -> Decimal:
        return max(Decimal("0"), self.daily_cap - self.spent_today())

    def _add(self, amount: Decimal) -> None:
        day = self._today()
        self._spent[day] = str(Decimal(self._spent.get(day, "0")) + amount)
        # Keep a short history only; this is a budget, not an accounting system.
        for old in sorted(self._spent)[:-30]:
            del self._spent[old]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._spent, indent=2), encoding="utf-8")
        except OSError:
            log.error("ai.ledger.persist_failed", path=str(self.path), exc_info=True)

    async def reserve(self, worst_case: Decimal) -> bool:
        async with self._lock:
            if worst_case > self.remaining():
                return False
            self._add(worst_case)
            return True

    async def settle(self, reserved: Decimal, actual: Decimal) -> None:
        async with self._lock:
            self._add(actual - reserved)


@dataclass(frozen=True)
class _Usage:
    tokens_in: int | None
    tokens_out: int | None


def _extract_json(text: str) -> dict[str, Any] | None:
    """Parse a JSON object, tolerating a model that wraps it in prose or a code fence."""
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
        return value if isinstance(value, dict) else None
    except ValueError:
        return None


class ChatCompletionsProvider(LLMProvider):
    """Any OpenAI-compatible ``/v1/chat/completions`` endpoint.

    ``ledger`` is optional: OpenAI gets one, a free endpoint such as Jev does not. When
    the endpoint rejects ``response_format`` (many compatible servers do), the call is
    retried once with the schema stated in the prompt instead.
    """

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        ledger: SpendLedger | None = None,
        max_output_tokens: int = 600,
        concurrency: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self._api_key = api_key
        self._ledger = ledger
        self._max_out = max_output_tokens
        self._sem = asyncio.Semaphore(concurrency)
        self._client = client or httpx.AsyncClient(timeout=None)
        self._owns_client = client is None
        self._schema_unsupported = False

    @property
    def ledger(self) -> SpendLedger | None:
        return self._ledger

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _worst_case_cost(self, chars: int) -> Decimal:
        price_in, price_out = price_for(self.model)
        tokens_in = chars // _CHARS_PER_TOKEN + 50
        return (Decimal(tokens_in) * price_in + Decimal(self._max_out) * price_out) / Decimal(1_000_000)

    def _actual_cost(self, usage: _Usage, fallback: Decimal) -> Decimal:
        if usage.tokens_in is None or usage.tokens_out is None:
            return fallback
        price_in, price_out = price_for(self.model)
        return (Decimal(usage.tokens_in) * price_in + Decimal(usage.tokens_out) * price_out) / Decimal(1_000_000)

    def _abstain(self, prompt_hash: str, started: float, reason: str) -> LLMResponse:
        log.warning("ai.cloud.abstain", provider=self.name, model=self.model, reason=reason)
        return LLMResponse(
            text="",
            parsed=None,
            model=f"{self.name}:{self.model}",
            prompt_hash=prompt_hash,
            latency_ms=(time.monotonic() - started) * 1000,
            raw={"reason": reason},
        )

    async def _post(self, body: dict[str, Any]) -> httpx.Response:
        return await self._client.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=body)

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        timeout: float = 90.0,  # noqa: ASYNC109
    ) -> LLMResponse:
        started = time.monotonic()
        prompt_hash = hashlib.sha256(
            json.dumps([system, prompt, schema, self.model], sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        reserved = Decimal("0")
        if self._ledger is not None:
            reserved = self._worst_case_cost(len(prompt) + len(system or "") + len(json.dumps(schema or {})))
            if not await self._ledger.reserve(reserved):
                return self._abstain(prompt_hash, started, "budget_exhausted")

        usage = _Usage(None, None)
        try:
            async with self._sem:
                text, usage = await asyncio.wait_for(
                    self._call(prompt, schema, system, temperature), timeout=timeout
                )
        except TimeoutError:
            return self._abstain(prompt_hash, started, "timeout")
        except Exception as exc:  # noqa: BLE001 - a remote failure must never reach the engine
            return self._abstain(prompt_hash, started, f"error: {type(exc).__name__}: {exc}")
        finally:
            if self._ledger is not None:
                await self._ledger.settle(reserved, self._actual_cost(usage, reserved))

        parsed = _extract_json(text) if schema is not None else None
        if self._ledger is not None:
            log.info(
                "ai.cloud.call",
                provider=self.name,
                model=self.model,
                tokens_in=usage.tokens_in,
                tokens_out=usage.tokens_out,
                spent_today=str(self._ledger.spent_today()),
            )
        return LLMResponse(
            text=text,
            parsed=parsed,
            model=f"{self.name}:{self.model}",
            prompt_hash=prompt_hash,
            latency_ms=(time.monotonic() - started) * 1000,
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
        )

    async def _call(
        self, prompt: str, schema: dict[str, Any] | None, system: str | None, temperature: float
    ) -> tuple[str, _Usage]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": self._max_out,
        }
        if schema is not None and not self._schema_unsupported:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "assessment", "schema": schema, "strict": False},
            }
            body["messages"] = [*messages, {"role": "user", "content": prompt}]
            resp = await self._post(body)
            if resp.status_code == 400:
                # Many OpenAI-compatible servers reject response_format; fall back to
                # stating the schema in the prompt and remember not to try again.
                log.info("ai.cloud.response_format_unsupported", provider=self.name, detail=resp.text[:200])
                self._schema_unsupported = True
            else:
                resp.raise_for_status()
                return self._unpack(resp.json())

        body.pop("response_format", None)
        content = prompt
        if schema is not None:
            content += (
                "\n\nRespond with ONLY a JSON object matching this JSON schema, no prose:\n"
                + json.dumps(schema)
            )
        body["messages"] = [*messages, {"role": "user", "content": content}]
        resp = await self._post(body)
        resp.raise_for_status()
        return self._unpack(resp.json())

    @staticmethod
    def _unpack(data: dict[str, Any]) -> tuple[str, _Usage]:
        choice = (data.get("choices") or [{}])[0]
        text = str((choice.get("message") or {}).get("content") or "")
        usage = data.get("usage") or {}
        return text, _Usage(usage.get("prompt_tokens"), usage.get("completion_tokens"))

    async def probe(self) -> ProviderHealth:
        started = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                self._client.get(f"{self.base_url}/models", headers=self._headers()), timeout=10
            )
            ok = resp.status_code == 200
            detail = "" if ok else f"HTTP {resp.status_code}"
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        if self._ledger is not None:
            detail = (detail + " " if detail else "") + (
                f"budget ${self._ledger.spent_today():.4f} / ${self._ledger.daily_cap} today"
            )
        return ProviderHealth(
            ok=ok, provider=self.name, model=self.model, detail=detail,
            latency_ms=(time.monotonic() - started) * 1000,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def build_openai_provider(settings: Settings, ledger_path: Path | None = None) -> ChatCompletionsProvider | None:
    secrets = settings.secrets
    if not secrets.openai_api_key:
        return None
    cap = Decimal(str(secrets.openai_daily_budget_usd or "0"))
    if cap <= 0:
        return None
    ledger = SpendLedger(ledger_path or DATA_DIR / "openai_spend.json", cap)
    return ChatCompletionsProvider(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key=secrets.openai_api_key,
        model=secrets.openai_model or "gpt-4.1-nano",
        ledger=ledger,
        max_output_tokens=500,
        concurrency=1,
    )


def build_jev_provider(settings: Settings) -> ChatCompletionsProvider | None:
    secrets = settings.secrets
    if not secrets.jev_base_url:
        return None
    return ChatCompletionsProvider(
        name="jev",
        base_url=secrets.jev_base_url,
        api_key=secrets.jev_api_key,
        model=secrets.jev_model or "jev",
        ledger=None,
        concurrency=2,
    )


__all__ = [
    "ChatCompletionsProvider",
    "SpendLedger",
    "build_jev_provider",
    "build_openai_provider",
    "price_for",
]
