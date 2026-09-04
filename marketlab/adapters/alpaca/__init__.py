"""Alpaca Market Data adapter (optional): equity bars/quotes and news.

Requires ``ALPACA_API_KEY``/``ALPACA_SECRET_KEY``; without them ``probe()`` reports
``NO_CREDENTIALS`` and every method returns an empty result.
"""

from __future__ import annotations
