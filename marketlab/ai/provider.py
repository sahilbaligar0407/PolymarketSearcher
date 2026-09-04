"""Local-model provider autodetection and the provider interface.

**No provider in this module ever downloads or pulls a model.** Detection only ever
probes an already-running local server; if nothing answers, :func:`detect_provider`
returns :class:`DisabledProvider` and the caller logs one WARN and moves on. The engine
must run perfectly well with AI disabled -- the analyst layer is an optional enrichment,
never a dependency of the trading loop.

The model itself never receives credentials, never gets a tool that can act, and never
sees anything resembling a shell or a venue SDK. It only ever emits text/JSON that
:mod:`marketlab.ai.validator` inspects before anything downstream may use it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import httpx

from marketlab.logging import get_logger
from marketlab.settings import Settings

log = get_logger(__name__)

#: Preference order when LOCAL_LLM_MODEL is unset and we must pick from what's installed.
PREFERRED_MODELS: tuple[str, ...] = (
    "gpt-oss:20b",
    "glm-4.7-flash:latest",
    "qwen2.5vl:7b",
    "qwen3:0.6b",
)

#: Common ports an OpenAI-compatible localhost server (vLLM, LM Studio, text-generation-
#: webui, etc.) might be listening on.
_OPENAI_COMPAT_PORTS: tuple[int, ...] = (8000, 1234, 5000, 8080)

_DEFAULT_PROBE_TIMEOUT = 2.0


@dataclass(frozen=True)
class LLMResponse:
    """The result of one ``generate`` call, provider-agnostic."""

    text: str
    #: Parsed JSON when a schema was supplied and parsing/validation succeeded.
    #: ``None`` means "no usable structured output" -- callers must treat that exactly
    #: like an abstain, never attempt to coerce a partial payload out of ``text``.
    parsed: dict[str, Any] | None
    model: str
    prompt_hash: str
    latency_ms: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderHealth:
    ok: bool
    provider: str
    model: str
    detail: str = ""
    latency_ms: float | None = None


class LLMProvider(abc.ABC):
    """The one interface every backend (Ollama, OpenAI-compatible, llama.cpp, disabled)
    implements. Nothing outside this module and :mod:`marketlab.ai.ollama` should ever
    branch on provider type -- callers program to this ABC only.
    """

    name: str
    model: str

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        timeout: float = 90.0,  # noqa: ASYNC109
    ) -> LLMResponse: ...

    @abc.abstractmethod
    async def probe(self) -> ProviderHealth: ...

    @abc.abstractmethod
    async def close(self) -> None: ...


class DisabledProvider(LLMProvider):
    """Null object used whenever no local model is reachable.

    ``generate`` always returns an abstain-shaped response (``parsed=None``,
    ``text=""``) and never raises, never makes a network call, and never blocks. The
    rest of the pipeline (retrieval, validator, strategies) must treat a
    ``DisabledProvider`` exactly like a model that always abstains.
    """

    name = "disabled"
    model = "none"

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        timeout: float = 90.0,  # noqa: ASYNC109
    ) -> LLMResponse:
        return LLMResponse(
            text="",
            parsed=None,
            model=self.model,
            prompt_hash="",
            latency_ms=0.0,
            tokens_in=0,
            tokens_out=0,
            raw={"reason": "ai_disabled"},
        )

    async def probe(self) -> ProviderHealth:
        return ProviderHealth(ok=False, provider=self.name, model=self.model, detail="AI disabled: no local LLM reachable")

    async def close(self) -> None:
        return None


async def _probe_ollama(base_url: str, timeout: float = _DEFAULT_PROBE_TIMEOUT) -> list[str] | None:  # noqa: ASYNC109
    """Return the list of installed model names, or None if unreachable."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
        if resp.status_code != 200:
            return None
        data = resp.json()
        return [m.get("name", "") for m in data.get("models", [])]
    except (httpx.HTTPError, ValueError):
        return None


async def _probe_openai_compat(base_url: str, timeout: float = _DEFAULT_PROBE_TIMEOUT) -> list[str] | None:  # noqa: ASYNC109
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/v1/models")
        if resp.status_code != 200:
            return None
        data = resp.json()
        return [m.get("id", "") for m in data.get("data", [])]
    except (httpx.HTTPError, ValueError):
        return None


async def _probe_llama_cpp(base_url: str, timeout: float = _DEFAULT_PROBE_TIMEOUT) -> bool:  # noqa: ASYNC109
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/health")
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def choose_model(installed: list[str], forced: str | None) -> str | None:
    """Pick a model id from what's installed, honoring an explicit override.

    Never triggers a download -- this only selects among ``installed`` names. Returns
    None if neither the forced choice nor any preferred model is present.
    """
    if forced:
        return forced
    for preferred in PREFERRED_MODELS:
        if preferred in installed:
            return preferred
    return installed[0] if installed else None


async def detect_provider(settings: Settings) -> LLMProvider:
    """Autodetect a locally-running LLM backend, in this exact priority order.

    1. An existing Ollama server at ``settings.sources.ollama_base``.
    2. An existing OpenAI-compatible server on a common localhost port.
    3. An existing llama.cpp server (``/health``).
    4. An explicit ``LOCAL_LLM_BASE_URL`` the user configured, tried against every
       known protocol shape.
    5. :class:`DisabledProvider`.

    This function NEVER pulls, downloads, or otherwise installs a model. It only ever
    asks an already-running server what it has.
    """
    from marketlab.ai.ollama import OllamaProvider  # local import: avoid import cycle

    forced_model = settings.secrets.local_llm_model or None
    explicit_base = settings.secrets.local_llm_base_url or None

    # 1. Ollama at the configured base URL.
    ollama_base = settings.sources.ollama_base
    installed = await _probe_ollama(ollama_base)
    if installed is not None:
        model = choose_model(installed, forced_model)
        if model:
            log.info("ai.provider.detected", provider="ollama", base_url=ollama_base, model=model)
            return OllamaProvider(base_url=ollama_base, model=model)
        log.warning("ai.provider.ollama_no_usable_model", base_url=ollama_base, installed=installed)

    # 2. OpenAI-compatible localhost server.
    for port in _OPENAI_COMPAT_PORTS:
        base = f"http://localhost:{port}"
        models = await _probe_openai_compat(base)
        if models is not None and models:
            model = forced_model or models[0]
            log.info("ai.provider.detected", provider="openai-compat", base_url=base, model=model)
            return OllamaProvider(base_url=base, model=model, api_style="openai")

    # 3. llama.cpp server.
    for port in (8080, 8000):
        base = f"http://localhost:{port}"
        if await _probe_llama_cpp(base):
            model = forced_model or "llama.cpp"
            log.info("ai.provider.detected", provider="llama.cpp", base_url=base, model=model)
            return OllamaProvider(base_url=base, model=model, api_style="llamacpp")

    # 4. Explicit LOCAL_LLM_BASE_URL, tried against every protocol shape we know.
    if explicit_base:
        installed = await _probe_ollama(explicit_base)
        if installed is not None:
            model = choose_model(installed, forced_model)
            if model:
                log.info("ai.provider.detected", provider="ollama", base_url=explicit_base, model=model)
                return OllamaProvider(base_url=explicit_base, model=model)
        models = await _probe_openai_compat(explicit_base)
        if models is not None:
            model = forced_model or (models[0] if models else "unknown")
            log.info("ai.provider.detected", provider="openai-compat", base_url=explicit_base, model=model)
            return OllamaProvider(base_url=explicit_base, model=model, api_style="openai")
        if await _probe_llama_cpp(explicit_base):
            model = forced_model or "llama.cpp"
            log.info("ai.provider.detected", provider="llama.cpp", base_url=explicit_base, model=model)
            return OllamaProvider(base_url=explicit_base, model=model, api_style="llamacpp")

    # 5. Nothing reachable: degrade, don't crash.
    log.warning("ai.provider.disabled", reason="no local LLM reachable; AI layer degraded to abstain-only")
    return DisabledProvider()
