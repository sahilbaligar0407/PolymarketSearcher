"""BTC/ETH reference spot prices: the low-latency feed BTC threshold-contract
strategies (Kalshi ``KXBTC*`` and friends) key off of.

Coinbase Exchange's public API is primary (no key required); Binance is a fallback that
is expected to be unreachable from a US IP (DNS-level block or an HTTP 451) and is
disabled cleanly the first time that happens rather than retried forever.
"""

from __future__ import annotations
