"""The safety gate: confirms global Polymarket is geoblocked and locks execution off.

This module is the one place in the Polymarket-global adapter family that is allowed
to touch ``Settings.polymarket_global_execution`` - and it may only ever set it to
``False``. There is no code path anywhere in this deployment, including here, that sets
it to ``True``. ``assert_no_global_execution`` exists purely so that if some future
change ever tries, it fails loudly instead of silently arming a US-illegal order path.

If the geoblock endpoint is unreachable or returns something we can't parse, we fail
closed: ``blocked=True``. An unreachable safety check is not evidence of safety.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

from marketlab.adapters.base import HttpAdapter
from marketlab.logging import get_logger

log = get_logger(__name__)

#: NON-NEGOTIABLE: this constant documents (it does not enforce by itself -
#: `assert_no_global_execution` does that) that global Polymarket execution is never
#: turned on in this deployment. Global Polymarket is close-only from the US; Kalshi is
#: the only execution venue. Do not add a code path that sets
#: `Settings.polymarket_global_execution = True` anywhere, ever.
GLOBAL_EXECUTION_PERMANENTLY_DISABLED = True


class GeoblockResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    blocked: bool
    country: str = ""
    #: Polymarket's public geoblock endpoint does not itself report a "close-only" flag
    #: (only `blocked`/`ip`/`country`/`region`, confirmed by live probe 2026-09-04); per
    #: publicly documented Polymarket policy, a blocked US IP is close-only (existing
    #: positions may be closed, no new positions may be opened), so we treat
    #: `close_only == blocked` as the conservative, documented business rule.
    close_only: bool
    raw: dict[str, Any] = {}


async def check_geoblock(http: HttpAdapter, path: str = "") -> GeoblockResult:
    """Query the geoblock endpoint. Fails closed (``blocked=True``) on any error."""
    try:
        data = await http.get_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"unexpected geoblock payload type: {type(data)!r}")
        blocked = bool(data.get("blocked", True))
        country = str(data.get("country", ""))
        return GeoblockResult(blocked=blocked, country=country, close_only=blocked, raw=data)
    except Exception as exc:
        log.warning("poly_geoblock_check_failed_failing_closed", error=str(exc))
        return GeoblockResult(blocked=True, country="", close_only=True, raw={"error": str(exc)})


async def enforce(settings: Any, *, http: HttpAdapter | None = None) -> GeoblockResult:
    """Run the geoblock check and lock ``settings.polymarket_global_execution`` off.

    Regardless of what the probe reports - blocked, unblocked, or unreachable - this
    function always ends by setting the flag to ``False``. If the probe were ever to
    report ``blocked=False`` (which would be surprising: this deployment runs from a US
    IP), we log a warning but still do not enable anything; enabling execution is not a
    decision this function is allowed to make.
    """
    split = urlsplit(str(settings.sources.poly_geoblock))
    origin = f"{split.scheme}://{split.netloc}"
    path = split.path or "/"

    owns_http = http is None
    client = http or HttpAdapter(origin, name="poly_geoblock")
    try:
        result = await check_geoblock(client, path)
    finally:
        if owns_http:
            await client.close()

    settings.polymarket_global_execution = False
    if not result.blocked:
        log.warning("poly_geoblock_unexpected_unblocked", country=result.country)
    else:
        log.info("poly_geoblock_confirmed_blocked", country=result.country)
    return result


def assert_no_global_execution(settings: Any) -> None:
    """Raise if anything ever managed to flip global Polymarket execution on.

    Call this from startup/health checks as a tripwire, independent of ``enforce``.
    """
    if getattr(settings, "polymarket_global_execution", False):
        raise RuntimeError(
            "polymarket_global_execution is True - this must never happen. Global "
            "Polymarket is a read-only intelligence source in this deployment; Kalshi "
            "is the only execution venue."
        )
