"""Official X (Twitter) API v2 adapter. No scraping fallback - ``X_BEARER_TOKEN`` or
the source degrades to ``NO_CREDENTIALS`` and stays silent. Usage-priced, so a monthly
request budget is enforced client-side in addition to whatever X itself enforces.
"""

from __future__ import annotations
