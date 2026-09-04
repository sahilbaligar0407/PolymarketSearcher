"""The local LLM analyst layer.

Architecture (never deviate from this order)::

    DATA -> RETRIEVAL -> LOCAL MODEL -> STRICT JSON ASSESSMENT -> DETERMINISTIC VALIDATOR
         -> STRATEGY -> RISK GATEWAY -> BROKER

The model in this package is an *analyst*, never the broker.  It never receives
credentials, never invokes a shell, never constructs an order, and never changes a risk
limit or activates live mode.  Every piece of output it produces is strict JSON that
passes through :mod:`marketlab.ai.validator` before anything downstream may act on it.
If validation fails for any reason, the deterministic gate abstains -- it never repairs
a malformed recommendation.
"""

from __future__ import annotations
