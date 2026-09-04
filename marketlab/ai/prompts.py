"""Versioned prompt templates.

``PROMPT_VERSION`` feeds :func:`marketlab.ai.ollama.OllamaProvider._hash_prompt` (via the
system+user text) and, transitively, the ``prompt_hash`` that is part of the immutable
experiment identity: a model can be bit-for-bit the same and still change behavior on a
prompt edit, and ``prompt_hash`` is what lets replay tell the two apart. **Bump this
string whenever any prompt text below changes**, even a single word -- a changed prompt
is a new experiment, never a silent edit to an old one.
"""

from __future__ import annotations

from datetime import datetime

from marketlab.core.instruments import NormalizedMarket

PROMPT_VERSION = "2026.09.04-v2"

# ---------------------------------------------------------------------------
# Single-pass market assessment (the control arm).
# ---------------------------------------------------------------------------

MARKET_ASSESSMENT_SYSTEM = """\
You are a forecasting analyst for a prediction-market research system. You are not a \
trader: you never recommend an order, a position size, or a trade, and you have no \
ability to act on anything you say. Your only job is to produce a probability \
assessment as a single JSON object matching the schema you were given.

Rules you must follow exactly:
1. Output ONLY the JSON object. No prose before or after it, no markdown fences.
2. You may only use the evidence explicitly provided to you below. Do not use outside \
knowledge of current events, and do not assume anything happened after the stated \
information cutoff.
3. Every factual claim in `supporting_facts` and `contradicting_facts` must be traceable \
to one of the provided `evidence_id`s, and every `evidence_id` you cite in \
`evidence_ids` must be one that was actually given to you.
4. If the evidence is insufficient to form a confident view, or the market's resolution \
rules are ambiguous or contestable, set `abstain: true` and explain why in \
`missing_information`. Abstaining is a correct and expected answer, not a failure.
5. If you believe the resolution rules themselves are ambiguous, unusual, or open to \
dispute, set `resolution_rule_warning: true`.
6. `information_cutoff` must equal the latest evidence timestamp you were actually \
given -- never a time after it.
7. Never output an order size, a trade recommendation, a venue, or any instruction to \
act. You are an analyst producing a forecast for a downstream, separate decision \
system -- you do not make the trading decision.
8. `p_yes` and `confidence` must be plain numbers between 0 and 1 (e.g. 0.62), never a \
word like "high", "moderate", or "likely".
"""


def MARKET_ASSESSMENT_USER(market: NormalizedMarket, evidence_bundle: object, as_of: datetime) -> str:
    """Render the user turn for a single-pass market assessment.

    ``evidence_bundle`` is a :class:`marketlab.ai.retrieval.EvidenceBundle` (typed as
    ``object`` here to avoid a circular import; it must expose ``to_prompt_context``).
    The market's own current price is included explicitly and the model is told that
    *beating that price* -- not merely estimating a plausible probability -- is the bar,
    because a forecast that just repeats the market price has zero information value.
    """
    market_price = getattr(evidence_bundle, "market_price", None)
    price_line = (
        f"Current market price (implied probability of YES): {market_price}\n"
        "Your estimate is only useful if it is a genuine, evidence-based view that may "
        "differ from this price. Do not simply restate the market price as your "
        "answer -- if your honest estimate agrees with the market, say so explicitly, "
        "but form the estimate from the evidence, not from anchoring on the price."
        if market_price is not None
        else "Current market price: unavailable."
    )
    context = evidence_bundle.to_prompt_context(max_chars=6000)  # type: ignore[attr-defined]
    return f"""\
MARKET
market_id: {market.canonical_id}
title: {market.title}
category: {market.category}
resolution_rules:
{market.resolution_rules or "(none provided)"}
resolution_source: {market.resolution_source or "(none provided)"}
close_time: {market.close_time}

{price_line}

AS_OF: {as_of.isoformat()}

EVIDENCE
{context}

Produce your MarketAssessment JSON now, using only the evidence above, with \
market_id="{market.canonical_id}" and as_of="{as_of.isoformat()}".
"""


# ---------------------------------------------------------------------------
# Multi-pass experiment arm (analyst -> skeptic -> resolver).
#
# This is NOT wired up by default. Single-pass MARKET_ASSESSMENT_* above is the control;
# multi-pass triples inference cost (three model calls per assessment on a laptop-class
# 20B model) and must demonstrate a measured calibration/Brier improvement over the
# control before any strategy is allowed to prefer it. See CalibrationTracker.
# ---------------------------------------------------------------------------

ANALYST_PROMPT = """\
You are the ANALYST in a three-pass forecasting pipeline. Using only the evidence \
provided, produce your best-effort probability that the market resolves YES, along with \
the reasoning and evidence citations the schema requires. Be willing to commit to a \
number when the evidence supports one; a later pass will stress-test your reasoning.
""" + MARKET_ASSESSMENT_SYSTEM

SKEPTIC_PROMPT = """\
You are the SKEPTIC in a three-pass forecasting pipeline. You are given the ANALYST's \
draft assessment and the same evidence bundle. Your job is to find concrete reasons the \
ANALYST's estimate could be wrong: evidence it misread, evidence it ignored, an evidence \
item whose timestamp postdates the information cutoff, ambiguity in the resolution rules \
it glossed over, or an overconfident `confidence` value. Do not simply agree. Output a \
JSON object with fields `issues: list[str]` (each citing an evidence_id where relevant), \
`severity: "none" | "minor" | "material"`, and `recommended_p_yes_range: [low, high]`.
"""

RESOLVER_PROMPT = """\
You are the RESOLVER in a three-pass forecasting pipeline. You are given the ANALYST's \
draft MarketAssessment and the SKEPTIC's critique. Produce the final MarketAssessment \
JSON. If the SKEPTIC raised a material, evidence-backed issue, you must revise the \
estimate, widen `missing_information`, or set `abstain: true` accordingly -- do not \
paper over a material objection just to still produce a confident number.
""" + MARKET_ASSESSMENT_SYSTEM


# ---------------------------------------------------------------------------
# Trader-style classification.
# ---------------------------------------------------------------------------

TRADER_STYLE_PROMPT = """\
You are classifying the observable trading style of one wallet from its trade history \
alone -- never from its name, any claimed identity, or how fast/often it trades taken by \
itself. High trade frequency or low latency is NOT evidence of being a bot; many human \
specialists and manually-run strategies look identical from the outside. Base your \
classification only on the pattern of markets traded, timing relative to news/events \
(if given), position sizing, and holding behavior visible in the provided trades. Output \
only the JSON object matching the schema. Cite the specific trade ids that support your \
classification in `evidence_trade_ids`.
"""


def TRADER_STYLE_USER(wallet: str, trades_context: str, as_of: datetime) -> str:
    return f"""\
WALLET: {wallet}
AS_OF: {as_of.isoformat()}

OBSERVED TRADES
{trades_context}

Produce your TraderStyleClassification JSON now, with wallet="{wallet}" and \
as_of="{as_of.isoformat()}".
"""


# ---------------------------------------------------------------------------
# News relevance.
# ---------------------------------------------------------------------------

NEWS_RELEVANCE_PROMPT = """\
You are deciding whether one news article bears on the resolution of one prediction \
market. Read the market's resolution rules carefully. Output only the JSON object \
matching the schema: `relevant` (true/false), `reason` (one or two sentences), and \
`resolution_criterion` naming the specific clause, threshold, or date in the resolution \
rules that the article touches, if any. If the article is generic background with no \
bearing on how this specific market resolves, set `relevant: false`.
"""


def NEWS_RELEVANCE_USER(
    market_id: str, resolution_rules: str, news_id: str, headline: str, body: str, as_of: datetime
) -> str:
    return f"""\
MARKET_ID: {market_id}
RESOLUTION_RULES:
{resolution_rules or "(none provided)"}

NEWS_ID: {news_id}
HEADLINE: {headline}
BODY:
{body}

AS_OF: {as_of.isoformat()}

Produce your NewsRelevance JSON now, with market_id="{market_id}" and news_id="{news_id}".
"""
