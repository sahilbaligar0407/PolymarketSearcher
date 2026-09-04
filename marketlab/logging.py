"""Structured logging.

Two sinks: readable console output for a human watching the daemon, and newline-delimited
JSON on disk so every trade decision can be reconstructed later.

Secrets never reach either sink - :func:`_redact` strips anything that looks like key
material before rendering.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import structlog

#: Field names whose values are replaced wholesale.
_SECRET_KEYS = {
    "api_key",
    "api_key_id",
    "private_key",
    "private_key_path",
    "secret",
    "secret_key",
    "bearer",
    "bearer_token",
    "password",
    "token",
    "authorization",
    "kalshi_access_key",
    "kalshi_signature",
}

#: Anything that looks like PEM material, even if it lands in a free-text field.
_PEM_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        if _PEM_RE.search(value):
            return "<REDACTED PRIVATE KEY>"
        return value
    if isinstance(value, dict):
        return {k: ("<REDACTED>" if k.lower() in _SECRET_KEYS else _redact_value(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_redact_value(v) for v in value)
    return value


def _redact(_logger: Any, _method: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    return {
        k: ("<REDACTED>" if k.lower() in _SECRET_KEYS else _redact_value(v))
        for k, v in event_dict.items()
    }


_CONFIGURED = False
_CONFIGURED_LOG_DIR: Path | None = None


def configure_logging(
    log_dir: Path | None = None,
    level: str = "INFO",
    json_console: bool = False,
) -> None:
    """Idempotently configure structlog + stdlib logging for the whole process.

    Idempotent, with one deliberate exception: a later call that supplies a ``log_dir``
    when the current configuration has none DOES reconfigure, so the on-disk JSONL audit
    trail gets attached. Without that, a CLI entry point that configures logging early
    (console only) silently prevented the daemon from ever opening its log file - the
    audit trail every trade decision is supposed to be reconstructable from.
    """
    global _CONFIGURED, _CONFIGURED_LOG_DIR
    if _CONFIGURED and not (log_dir is not None and _CONFIGURED_LOG_DIR is None):
        return

    handlers: list[logging.Handler] = []
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(message)s"))
    handlers.append(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "marketlab.jsonl", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(fh)

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )
    # Third-party chatter is not useful at INFO.
    for noisy in ("httpx", "httpcore", "websockets", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_console or log_dir is not None
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True
    _CONFIGURED_LOG_DIR = log_dir


def get_logger(name: str) -> Any:
    """Get a bound structlog logger. Safe to call before :func:`configure_logging`."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)
