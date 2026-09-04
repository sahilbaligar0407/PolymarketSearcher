"""Derived signals computed from raw information-layer events.

These modules never fetch data themselves; they take already-ingested events
(``NewsEvent``, ``FilingEvent``, ``SocialEvent``) and turn them into features a
strategy or the AI analyst can consume.
"""

from __future__ import annotations
