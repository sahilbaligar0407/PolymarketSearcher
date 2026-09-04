"""SEC EDGAR adapter: corporate disclosures (filings, company facts, Form 4 ownership).

Submodules:

``client``  ``SecAdapter`` - REST client over ``data.sec.gov`` / ``www.sec.gov`` / EDGAR
            full-text search, rate-limited to 10 req/sec and under a descriptive
            User-Agent (SEC blocks requests without one).
``forms``   Form 4 (insider ownership) XML parsing and the ``is_routine`` classifier
            that keeps scheduled 10b5-1/award transactions from being read as
            discretionary buy/sell signals.
"""

from __future__ import annotations
