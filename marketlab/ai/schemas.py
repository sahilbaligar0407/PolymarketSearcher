"""The strict output contract every local-model call must produce.

Every model in this module mirrors a JSON shape the local LLM is constrained to emit
(via Ollama's ``format`` structured-output parameter, see :func:`json_schema_for`).
Nothing here enforces range/consistency semantics -- that is deliberately left to
:mod:`marketlab.ai.validator`, the deterministic gate, so that a malformed or
out-of-range value can be *observed and rejected* rather than silently clamped away by
a validator that runs before we ever get to look at it.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MarketAssessment(BaseModel):
    """One local-model probability assessment of one market, at one point in time.

    ``p_yes``/``confidence`` are intentionally *not* range-constrained at the pydantic
    level (no ``ge``/``le``) -- an out-of-range value must survive parsing so the
    deterministic validator can see it, log it, and abstain, rather than have pydantic
    silently reject or clamp it before the validator ever runs.
    """

    model_config = ConfigDict(frozen=True)

    market_id: str
    as_of: datetime
    question_interpretation: str
    #: The model's estimate that the market resolves YES. Expected in [0, 1]; the
    #: validator, not this schema, is what rejects an out-of-range value.
    p_yes: Decimal
    #: The model's own confidence in its estimate, in [0, 1] (validator-checked).
    confidence: Decimal
    #: True if the model believes the evidence is insufficient or resolution rules are
    #: ambiguous. The model must never be pressured into a number it doesn't believe.
    abstain: bool = False
    #: Every evidence_id the model actually relied on. Must all resolve inside the
    #: EvidenceBundle that was handed to it -- the validator checks this.
    evidence_ids: list[str] = Field(default_factory=list)
    supporting_facts: list[str] = Field(default_factory=list)
    contradicting_facts: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    #: True if the model thinks the resolution rules themselves are ambiguous or
    #: contestable. The validator forces abstain=True whenever this is set.
    resolution_rule_warning: bool = False
    #: The latest evidence timestamp the model was told about. Must be <= decision time.
    information_cutoff: datetime


class TraderStyle(StrEnum):
    """Observable trading-behavior buckets. Never an identity or intent claim."""

    SPORTS_SPECIALIST = "sports_specialist"
    LATE_FAVORITE_BUYER = "late_favorite_buyer"
    NEWS_TRADER = "news_trader"
    MARKET_MAKER = "market_maker"
    HIGH_TURNOVER = "high_turnover"
    LONG_TAIL_BETTOR = "long_tail_bettor"
    CRYPTO_MICROSTRUCTURE = "crypto_microstructure"
    EVENT_ARBITRAGE = "event_arbitrage"


class TraderStyleClassification(BaseModel):
    """A wallet's trading style, derived strictly from its observable trade history.

    This label describes *behavior visible on-chain/on-venue* (timing relative to news,
    market categories traded, position sizing, holding period, trade cadence) -- never
    identity or intent. In particular, a wallet must never be labeled an "AI bot" or
    "automated" merely because it trades frequently or with low latency; plenty of
    attentive humans and manually-run specialist strategies look exactly like that from
    the outside. ``high_turnover`` describes the observation, not a claim about who or
    what is placing the orders.
    """

    model_config = ConfigDict(frozen=True)

    wallet: str
    as_of: datetime
    styles: list[TraderStyle] = Field(default_factory=list)
    primary_style: TraderStyle
    confidence: Decimal
    #: Trade ids the model actually looked at to reach this classification.
    evidence_trade_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class NewsRelevance(BaseModel):
    """Does one article bear on one market, and which resolution criterion does it touch."""

    model_config = ConfigDict(frozen=True)

    market_id: str
    news_id: str
    as_of: datetime
    relevant: bool
    reason: str
    #: Which specific clause/threshold/date in the resolution rules this touches, if any.
    resolution_criterion: str = ""
    confidence: Decimal


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Render a pydantic model as a plain JSON Schema for Ollama's ``format`` parameter.

    Ollama (and most local structured-output backends) work best with a flat, ref-free
    schema. The three schemas above have no nested BaseModel fields, so pydantic's
    default ``model_json_schema()`` is already ref-free; if that ever changes, inline
    any ``$defs``/``$ref`` here rather than at every call site.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", None)
    if defs:
        schema = _inline_refs(schema, defs)
    _force_numeric_decimals(model, schema)
    return schema


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            return _inline_refs(defs[ref.removeprefix("#/$defs/")], defs)
        return {k: _inline_refs(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def _force_numeric_decimals(model: type[BaseModel], schema: dict[str, Any]) -> None:
    """Collapse pydantic's default ``anyOf: [number, pattern-constrained string]`` for
    ``Decimal`` fields down to a bare ``{"type": "number"}``.

    Observed live against gpt-oss:20b: Ollama's structured-output grammar enforces JSON
    *type* but does not reliably enforce a string's regex ``pattern``, so the model used
    the string branch to write a qualitative word ("moderate") into ``confidence``
    instead of a number -- valid JSON, useless as a probability. Every Decimal field in
    this module is a probability/confidence value, so dropping the string branch removes
    that escape hatch entirely; pydantic still parses a bare JSON number into ``Decimal``
    losslessly (via ``Decimal(str(value))``), so nothing is lost on the parsing side.
    """
    properties = schema.get("properties", {})
    for name, field_info in model.model_fields.items():
        if field_info.annotation is Decimal and name in properties:
            properties[name] = {"type": "number", "title": properties[name].get("title", name)}
