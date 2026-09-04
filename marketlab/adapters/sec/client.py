"""SEC EDGAR adapter: submissions, XBRL company facts/concepts, ticker map, filings feed.

Endpoints probed live against production on 2026-09-04:

* ``GET data.sec.gov/submissions/CIK##########.json``                    200, JSON - works
* ``GET data.sec.gov/api/xbrl/companyfacts/CIK##########.json``          (same host/pattern, not
  separately probed but documented identically by SEC - same auth/rate rules apply)
* ``GET www.sec.gov/files/company_tickers.json``                         200, JSON - works, ~10k rows
* ``GET efts.sec.gov/LATEST/search-index?q=...&forms=...``               200, JSON - **this is the
  endpoint EDGAR's own full-text-search UI calls**; confirmed reliable and used for
  ``iter_recent_filings``/``full_text_search``.
* ``GET www.sec.gov/cgi-bin/browse-edgar?...&output=atom``                200, but the *pseudo-XML*
  SEC serves here was observed with a live bug: filing ``<entry title="ARRAY(0x...)">`` and
  ``<company-info name="ARRAY(0x...)">`` - a PHP/Perl array-to-string artifact leaking into the
  XML on production. Do not build anything load-bearing on this feed; kept unused here in favor
  of the EFTS JSON endpoint above.

Every request carries ``settings.secrets.sec_user_agent`` as the ``User-Agent`` header - SEC
returns 403 without one. Rate limited to <=10 req/sec (default 8, politely under the ceiling
SEC documents), shared across all three hosts this adapter talks to since they're all "SEC".
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from marketlab.adapters.base import Adapter, HttpAdapter, SourceHealth, SourceStatus
from marketlab.adapters.ratelimit import RateLimiter
from marketlab.clock import Clock, LiveClock
from marketlab.core.events import FilingEvent, SourceClass
from marketlab.logging import get_logger

log = get_logger(__name__)

_EFTS_BASE = "https://efts.sec.gov"
_DEFAULT_UA_PLACEHOLDER = "set SEC_USER_AGENT"


class SecAdapter(Adapter):
    """EDGAR client. No API key required, but SEC blocks anonymous-looking User-Agents."""

    name = "sec"

    def __init__(
        self,
        *,
        sec_base: str,
        edgar_base: str,
        user_agent: str,
        clock: Clock | None = None,
        requests_per_second: float = 8.0,
    ) -> None:
        self._clock = clock or LiveClock()
        self._user_agent = user_agent.strip() or "MarketLab research (contact: set SEC_USER_AGENT)"
        if _DEFAULT_UA_PLACEHOLDER in self._user_agent:
            log.warning(
                "sec_user_agent_placeholder",
                detail="SEC_USER_AGENT is unset; using a generic placeholder. "
                "Set a real contact email in .env before sustained polling.",
            )
        headers = {"User-Agent": self._user_agent, "Accept-Encoding": "gzip, deflate"}
        # One limiter shared across every host this adapter talks to: SEC's 10 req/sec
        # guidance is a courtesy budget for "this client", not "this specific endpoint".
        rate = min(requests_per_second, 10.0)
        self._limiter = RateLimiter(rate=rate, burst=max(rate * 2, 2.0))

        self._data = HttpAdapter(
            sec_base, name="sec_data", default_headers=headers, rate_limiter=self._limiter, clock=self._clock
        )
        self._efts = HttpAdapter(
            _EFTS_BASE, name="sec_efts", default_headers=headers, rate_limiter=self._limiter, clock=self._clock
        )
        self._www = HttpAdapter(
            edgar_base, name="sec_www", default_headers=headers, rate_limiter=self._limiter, clock=self._clock
        )
        # `get_json` on HttpAdapter always parses the response as JSON; Form 4 XML
        # payloads need raw text, so a small dedicated httpx client fetches those,
        # sharing the same rate limiter/User-Agent as everything else.
        self._archive_client = httpx.AsyncClient(base_url=edgar_base, headers=headers, timeout=15.0)

        self._ticker_to_cik: dict[str, str] | None = None
        self._company_tickers_raw: dict[str, Any] | None = None

    @property
    def has_usable_user_agent(self) -> bool:
        """SEC's Akamai bot filter 403s any ``User-Agent`` without an ``@`` in it.

        Verified live: the shipped placeholder contains no address and is rejected every
        time. Checking here turns a mysterious blanket 403 into an actionable
        ``NO_CREDENTIALS`` telling the operator exactly which variable to set.
        """
        return "@" in self._user_agent

    async def probe(self) -> SourceHealth:
        if not self.has_usable_user_agent:
            return SourceHealth(
                name=self.name,
                status=SourceStatus.NO_CREDENTIALS,
                detail=(
                    "SEC_USER_AGENT must contain a contact email address - SEC rejects "
                    "any User-Agent without one. Set e.g. "
                    'SEC_USER_AGENT="MarketLab research (you@example.com)" in .env'
                ),
            )
        try:
            await self.get_ticker_map()
            return SourceHealth(name=self.name, status=SourceStatus.HEALTHY, last_message_at=self._clock.now())
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            log.warning("sec_probe_failed", error=str(exc))
            return SourceHealth(name=self.name, status=SourceStatus.DOWN, detail=str(exc))

    async def close(self) -> None:
        await asyncio.gather(self._data.close(), self._efts.close(), self._www.close(), self._archive_client.aclose())

    def health(self) -> SourceHealth:
        return self._data.health()

    @staticmethod
    def _pad_cik(cik: str | int) -> str:
        return str(cik).strip().lstrip("0").zfill(10) if str(cik).strip() else "0000000000"

    # -- reference data -----------------------------------------------------------

    async def get_ticker_map(self) -> dict[str, str]:
        """``ticker -> zero-padded 10-digit CIK``, cached after the first fetch."""
        if self._ticker_to_cik is not None:
            return self._ticker_to_cik
        payload = await self._www.get_json("/files/company_tickers.json")
        self._company_tickers_raw = payload
        out: dict[str, str] = {}
        for row in payload.values():
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker", "")).upper().strip()
            cik = row.get("cik_str")
            if ticker and cik is not None:
                out[ticker] = self._pad_cik(cik)
        self._ticker_to_cik = out
        return out

    async def get_company_tickers_raw(self) -> dict[str, Any]:
        """Raw ``{index: {cik_str, ticker, title}}`` payload.

        Feeds ``marketlab.signals.news.build_ticker_map`` directly (ticker->title),
        which is what entity/ticker extraction needs rather than the cik lookup.
        """
        if self._company_tickers_raw is None:
            await self.get_ticker_map()
        return self._company_tickers_raw or {}

    # -- submissions / XBRL --------------------------------------------------------

    async def get_submissions(self, cik: str | int) -> dict[str, Any]:
        return await self._data.get_json(f"/submissions/CIK{self._pad_cik(cik)}.json")

    async def get_company_facts(self, cik: str | int) -> dict[str, Any]:
        return await self._data.get_json(f"/api/xbrl/companyfacts/CIK{self._pad_cik(cik)}.json")

    async def get_company_concept(self, cik: str | int, taxonomy: str, tag: str) -> dict[str, Any]:
        return await self._data.get_json(f"/api/xbrl/companyconcept/CIK{self._pad_cik(cik)}/{taxonomy}/{tag}.json")

    # -- filings feed (EDGAR full-text search; the endpoint that actually works) --

    async def full_text_search(
        self,
        query: str = "*",
        *,
        forms: list[str] | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        start: int = 0,
    ) -> dict[str, Any]:
        """Raw EDGAR full-text search response (``efts.sec.gov/LATEST/search-index``)."""
        params: dict[str, Any] = {"q": query, "from": start}
        if forms:
            params["forms"] = ",".join(forms)
        if date_from:
            params["startdt"] = date_from.isoformat()
        if date_to:
            params["enddt"] = date_to.isoformat()
        return await self._efts.get_json("/LATEST/search-index", params=params)

    async def iter_recent_filings(
        self,
        forms: list[str] | None = None,
        since: datetime | None = None,
        query: str = "*",
    ) -> list[dict[str, Any]]:
        """Raw hit rows from full-text search, newest EDGAR index first."""
        forms = forms or ["8-K", "10-Q", "10-K", "4", "13D", "13G"]
        date_from = since.date() if since else None
        payload = await self.full_text_search(query, forms=forms, date_from=date_from)
        return payload.get("hits", {}).get("hits", [])

    async def iter_filing_events(
        self,
        forms: list[str] | None = None,
        since: datetime | None = None,
        query: str = "*",
    ) -> list[FilingEvent]:
        """Full-text search results as ``FilingEvent``s.

        ``first_seen_time`` is always the injected Clock's ``now()`` - never derived
        from ``file_date`` - since that's the point at which *this process* observed
        the filing, however long ago it was actually filed.
        """
        hits = await self.iter_recent_filings(forms=forms, since=since, query=query)
        now = self._clock.now()
        events: list[FilingEvent] = []
        for hit in hits:
            src = hit.get("_source", {})
            accession = str(src.get("adsh", ""))
            ciks = src.get("ciks") or []
            cik = str(ciks[0]) if ciks else ""
            display_names = src.get("display_names") or []
            company = display_names[0] if display_names else ""
            form_type = str(src.get("form") or (src.get("root_forms") or [""])[0])
            file_date_str = src.get("file_date", "")
            published: datetime | None = None
            if file_date_str:
                try:
                    published = datetime.strptime(file_date_str, "%Y-%m-%d").replace(tzinfo=UTC)
                except ValueError:
                    published = None
            file_id = str(hit.get("_id", ""))
            primary_doc = file_id.split(":")[-1] if ":" in file_id else ""
            url = ""
            if cik and accession:
                cik_int = cik.lstrip("0") or "0"
                url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                    f"{accession.replace('-', '')}/{primary_doc}"
                )
            events.append(
                FilingEvent(
                    event_time=published or now,
                    published_time=published,
                    first_seen_time=now,
                    ingested_time=now,
                    source="sec_edgar_fts",
                    source_class=SourceClass.OFFICIAL_PRIMARY,
                    accession=accession,
                    cik=cik,
                    form_type=form_type,
                    company=company,
                    url=url,
                )
            )
        return events

    # -- Form 4 raw document fetch --------------------------------------------------

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(initial=0.5, max=10.0),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
    )
    async def _get_text(self, path: str) -> str:
        await self._limiter.acquire(cost=1)
        resp = await self._archive_client.get(path)
        resp.raise_for_status()
        return resp.text

    async def get_form4_xml(self, cik: str | int, accession: str, primary_doc: str) -> str:
        """Fetch one Form 4's raw XML from the EDGAR archive.

        ``accession`` may be dashed (``0001140361-26-035636``) or not; the archive path
        requires it undashed. ``primary_doc`` is the filename (e.g. ``form4.xml``) found
        via the filing index or a full-text-search hit's ``_id``.
        """
        acc_nodash = accession.replace("-", "")
        cik_int = str(int(str(cik).lstrip("0") or "0"))
        path = f"/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary_doc}"
        return await self._get_text(path)
