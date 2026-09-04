"""Polymarket global adapters: READ-ONLY intelligence source.

Global Polymarket is close-only from the United States. Nothing in this package (or
its submodules) ever POSTs an order to Polymarket - it supplies (a) market/book prices
for cross-venue comparison against Kalshi, and (b) public top-trader activity for the
copy-trading strategy family, which trades the *equivalent Kalshi contract*, never a
Polymarket position. See ``geoblock.py`` for the enforcement of that boundary and
``clob.py`` for why order-submission code is deliberately absent.

Submodules:

``gamma``      Market/event discovery (``gamma-api.polymarket.com``).
``clob``       Live order-book state, read-only (``clob.polymarket.com``).
``data_api``   Public wallet positions/activity/trades (``data-api.polymarket.com``).
``leaderboard`` Dynamic trader discovery across both leaderboard surfaces.
``geoblock``   The safety gate: confirms blocked, locks execution off.
``normalize``  Gamma/CLOB/Data-API payloads -> ``marketlab.core`` universal types.
"""

from __future__ import annotations
