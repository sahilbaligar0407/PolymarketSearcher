"""FRED / ALFRED adapter: economic time series with point-in-time (vintage) discipline.

Requires ``FRED_API_KEY``; without it every method degrades to an empty result and
``probe()`` reports ``NO_CREDENTIALS`` rather than raising.
"""

from __future__ import annotations
