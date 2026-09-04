"""Polymarket US adapter: public/unauthenticated calls only.

Kalshi is the only execution venue in this deployment. Polymarket US is technically an
execution-eligible venue (see ``Venue.POLY_US`` in ``core/instruments.py``), but no
order code is written here - see ``public.py`` for why (Ed25519 request signing and
live order placement are deliberately out of scope; if this venue is ever activated for
execution, that is a separate, explicitly-scoped piece of work, not a byproduct of this
adapter).
"""

from __future__ import annotations
