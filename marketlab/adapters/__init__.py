"""Venue and data-source adapters.

``base.py`` defines the shared ``Adapter``/``HttpAdapter`` contract every venue adapter
builds on. ``ratelimit.py`` provides the token-bucket limiter. Venue-specific adapters
live in their own subpackages (``kalshi/``, ``polymarket_global/``, ...).
"""

from __future__ import annotations
