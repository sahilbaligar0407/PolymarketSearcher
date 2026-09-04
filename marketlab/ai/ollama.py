"""``OllamaProvider`` -- the local-model backend used in this deployment.

Honesty note baked into this module's design, not just its docstring: **LLM output is
not bit-reproducible across model versions or Ollama/runtime versions**, even with a
fixed seed and temperature 0. Quantization kernels, batching, and llama.cpp/Ollama
internals all change generation in ways a seed does not pin down. That is exactly why
``llm_model_id`` (the ``model`` string recorded on every :class:`~marketlab.ai.provider.
LLMResponse`) and ``prompt_hash`` are part of the immutable experiment identity: replay
is "as reproducible as a local model allows," not bit-exact, and the identity fields are
what let a later analysis tell "same model+prompt, different run" apart from "different
model or prompt entirely."
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Literal

import httpx

from marketlab.ai.provider import LLMProvider, LLMResponse, ProviderHealth
from marketlab.logging import get_logger

log = get_logger(__name__)

#: Fixed so that, model/runtime version held constant, replay is as reproducible as a
#: local model can be made. See the module docstring for the honest limits of that.
DEFAULT_SEED = 7
DEFAULT_NUM_CTX = 8192
DEFAULT_CONCURRENCY = 2


def _hash_prompt(system: str, prompt: str, schema: dict[str, Any] | None, model: str, options: dict[str, Any]) -> str:
    """sha256 of (system + prompt + schema + model + options), hex, first 16 chars.

    Any change to the prompt text, the schema, the model, or the sampling options
    produces a different hash -- which is the point: it lets replay distinguish "the
    exact same call" from "something, however small, changed."
    """
    payload = json.dumps(
        {
            "system": system,
            "prompt": prompt,
            "schema": schema,
            "model": model,
            "options": options,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _abstain(model: str, prompt_hash: str, latency_ms: float, reason: str) -> LLMResponse:
    """Build the response returned whenever we cannot get a usable answer.

    Never raises into the engine. ``parsed=None`` is the universal "treat this as an
    abstain" signal callers must honor -- never attempt to salvage a partial payload.
    """
    log.warning("ai.ollama.abstain", model=model, reason=reason)
    return LLMResponse(
        text="",
        parsed=None,
        model=model,
        prompt_hash=prompt_hash,
        latency_ms=latency_ms,
        tokens_in=None,
        tokens_out=None,
        raw={"reason": reason},
    )


class OllamaProvider(LLMProvider):
    """Talks to a local Ollama (or, in a degraded mode, an OpenAI-compatible /
    llama.cpp) server over HTTP. Never touches credentials, never runs a shell command,
    never constructs anything that could act on a venue.

    ``api_style`` lets :func:`marketlab.ai.provider.detect_provider` reuse this class
    for the fallback tiers it may detect (an OpenAI-compatible localhost server, or a
    llama.cpp server) without a second HTTP client implementation. In this deployment
    ``api_style="ollama"`` (the default) is what is actually exercised -- Ollama is
    confirmed running locally with ``format``-constrained structured output support.
    """

    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_style: Literal["ollama", "openai", "llamacpp"] = "ollama",
        seed: int = DEFAULT_SEED,
        num_ctx: int = DEFAULT_NUM_CTX,
        concurrency: int = DEFAULT_CONCURRENCY,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_style = api_style
        self._seed = seed
        self._num_ctx = num_ctx
        # httpx's own default client timeout is 5s -- far too short for a local 20B
        # model. `generate`'s `timeout` parameter (enforced via asyncio.wait_for) is the
        # single source of truth for how long a call may run, so the client itself is
        # given no timeout of its own.
        self._client = client or httpx.AsyncClient(timeout=None)
        self._sem = asyncio.Semaphore(concurrency)
        self._owns_client = client is None

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        timeout: float = 90.0,  # noqa: ASYNC109
    ) -> LLMResponse:
        options = {"temperature": float(temperature), "num_ctx": self._num_ctx, "seed": self._seed}
        prompt_hash = _hash_prompt(system or "", prompt, schema, self.model, options)

        start = time.monotonic()
        try:
            async with self._sem:
                if self.api_style == "ollama":
                    resp_data = await asyncio.wait_for(
                        self._call_ollama_chat(prompt, schema, system, options), timeout=timeout
                    )
                elif self.api_style == "openai":
                    resp_data = await asyncio.wait_for(
                        self._call_openai_chat(prompt, schema, system, options), timeout=timeout
                    )
                else:
                    resp_data = await asyncio.wait_for(
                        self._call_llamacpp(prompt, system, options), timeout=timeout
                    )
        except TimeoutError:
            return _abstain(self.model, prompt_hash, (time.monotonic() - start) * 1000, "timeout")
        except httpx.HTTPError as exc:
            return _abstain(
                self.model, prompt_hash, (time.monotonic() - start) * 1000, f"http_error:{type(exc).__name__}:{exc}"
            )

        latency_ms = (time.monotonic() - start) * 1000
        if resp_data is None:
            return _abstain(self.model, prompt_hash, latency_ms, "non_200_response")

        content, raw, tokens_in, tokens_out = resp_data
        parsed: dict[str, Any] | None = None
        if schema is not None:
            try:
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    parsed = None
            except json.JSONDecodeError:
                log.warning("ai.ollama.unparseable_json", model=self.model, prompt_hash=prompt_hash)
                parsed = None

        return LLMResponse(
            text=content,
            parsed=parsed,
            model=self.model,
            prompt_hash=prompt_hash,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            raw=raw,
        )

    async def _call_ollama_chat(
        self, prompt: str, schema: dict[str, Any] | None, system: str | None, options: dict[str, Any]
    ) -> tuple[str, dict[str, Any], int | None, int | None] | None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if schema is not None:
            payload["format"] = schema
        resp = await self._client.post(f"{self.base_url}/api/chat", json=payload)
        if resp.status_code != 200:
            log.warning("ai.ollama.bad_status", status=resp.status_code, body=resp.text[:500])
            return None
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        return content, data, data.get("prompt_eval_count"), data.get("eval_count")

    async def _call_openai_chat(
        self, prompt: str, schema: dict[str, Any] | None, system: str | None, options: dict[str, Any]
    ) -> tuple[str, dict[str, Any], int | None, int | None] | None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": options["temperature"],
            "seed": options["seed"],
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "assessment", "schema": schema},
            }
        resp = await self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        if resp.status_code != 200:
            log.warning("ai.openai_compat.bad_status", status=resp.status_code, body=resp.text[:500])
            return None
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return content, data, usage.get("prompt_tokens"), usage.get("completion_tokens")

    async def _call_llamacpp(
        self, prompt: str, system: str | None, options: dict[str, Any]
    ) -> tuple[str, dict[str, Any], int | None, int | None] | None:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        payload = {
            "prompt": full_prompt,
            "temperature": options["temperature"],
            "seed": options["seed"],
            "n_predict": 1024,
        }
        resp = await self._client.post(f"{self.base_url}/completion", json=payload)
        if resp.status_code != 200:
            log.warning("ai.llamacpp.bad_status", status=resp.status_code, body=resp.text[:500])
            return None
        data = resp.json()
        content = data.get("content", "")
        return content, data, data.get("tokens_evaluated"), data.get("tokens_predicted")

    async def probe(self) -> ProviderHealth:
        start = time.monotonic()
        try:
            if self.api_style == "ollama":
                resp = await self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
                ok = resp.status_code == 200 and any(
                    m.get("name") == self.model for m in resp.json().get("models", [])
                )
            elif self.api_style == "openai":
                resp = await self._client.get(f"{self.base_url}/v1/models", timeout=5.0)
                ok = resp.status_code == 200
            else:
                resp = await self._client.get(f"{self.base_url}/health", timeout=5.0)
                ok = resp.status_code == 200
        except httpx.HTTPError as exc:
            return ProviderHealth(ok=False, provider=self.name, model=self.model, detail=str(exc))
        latency_ms = (time.monotonic() - start) * 1000
        return ProviderHealth(ok=ok, provider=self.name, model=self.model, latency_ms=latency_ms)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
