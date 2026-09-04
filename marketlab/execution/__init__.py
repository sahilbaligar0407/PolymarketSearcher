"""Execution layer: the PaperBroker, its fill/latency models, the risk gateway, and the
(disabled-by-default) live Kalshi broker.

This package is the only place an :class:`~marketlab.core.orders.OrderIntent` turns into a
:class:`~marketlab.core.orders.Order`.  Everything here is built against the frozen
contracts in ``marketlab/core`` and ``marketlab/settings.py`` - see ``docs/CONTRACTS.md``.
"""

from __future__ import annotations
