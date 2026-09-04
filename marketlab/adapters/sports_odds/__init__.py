"""The Odds API adapter (optional): sportsbook lines converted to vig-free probabilities.

Requires ``THE_ODDS_API_KEY``; without it ``probe()`` reports ``NO_CREDENTIALS`` and
``get_odds`` returns an empty list rather than making a guaranteed-401 request.
"""

from __future__ import annotations
