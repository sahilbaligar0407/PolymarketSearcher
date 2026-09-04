"""Kalshi adapter: the only execution venue in this deployment.

Submodules:

``auth``      RSA-PSS request signing (``auth.py``).
``rest``      REST client for public + authenticated (read-only) endpoints.
``ws``        Websocket client with local order-book reconstruction.
``normalize`` Kalshi payloads -> ``marketlab.core.instruments`` universal types.
``fees``      Kalshi's quadratic taker-fee formula.

Order placement (``create_order`` / ``cancel_order``) is deliberately **not** here - that
belongs to ``marketlab/execution/kalshi_live.py``, owned by the execution team.
"""

from __future__ import annotations
