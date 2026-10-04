"""The ``marketlab`` command-line interface.

Every command in the PRD lives here. Commands that need live network access build
their own short-lived adapters directly (so ``marketlab markets`` shows real rows even
if no daemon is running); commands that report on a running daemon read the PID file,
the daemon's own heartbeat status file, and :class:`~marketlab.storage.state.StateStore`
-- there is no other IPC channel between a `paper status`/`paper stop` invocation and a
`paper start` process running in another terminal.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from marketlab.adapters.base import SourceHealth, SourceStatus
from marketlab.ai.provider import detect_provider
from marketlab.clock import LiveClock
from marketlab.core.broker import Mode
from marketlab.core.instruments import Venue
from marketlab.daemon.ingest import IngestService, select_tracked_markets
from marketlab.daemon.supervisor import Supervisor
from marketlab.execution.kalshi_live import KalshiLiveBroker
from marketlab.logging import get_logger
from marketlab.settings import load_settings
from marketlab.storage.parquet import ParquetWriter
from marketlab.storage.state import StateStore

log = get_logger(__name__)
console = Console()

app = typer.Typer(add_completion=False, no_args_is_help=True, help="MarketLab: prediction-market research engine.")
paper_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Paper-trading daemon controls.")
experiments_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Strategy tournament experiments.")
report_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Reports.")
ai_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Local AI analyst.")
live_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Real-money live trading (hard-gated).")
app.add_typer(paper_app, name="paper")
app.add_typer(experiments_app, name="experiments")
app.add_typer(report_app, name="report")
app.add_typer(ai_app, name="ai")
app.add_typer(live_app, name="live")


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# PID file / process management
# ---------------------------------------------------------------------------


def _pid_file(settings: Any) -> Path:
    return Path(settings.data_dir) / "marketlab.pid"


def _status_file(settings: Any) -> Path:
    return Path(settings.data_dir) / "daemon_status.json"


def _write_pid_file(settings: Any, pid: int | None = None) -> None:
    settings.ensure_dirs()
    _pid_file(settings).write_text(str(pid or os.getpid()), encoding="utf-8")


def _read_pid_file(settings: Any) -> int | None:
    p = _pid_file(settings)
    if not p.exists():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _remove_pid_file(settings: Any) -> None:
    with contextlib.suppress(FileNotFoundError, OSError):
        _pid_file(settings).unlink()


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        try:
            out = subprocess.run(  # noqa: S603, S607
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, timeout=5
            )
            return str(pid) in out.stdout
        except Exception:  # noqa: BLE001
            return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _signal_stop(pid: int) -> None:
    if os.name == "nt":
        with contextlib.suppress(OSError):
            os.kill(pid, signal.CTRL_BREAK_EVENT)
            return
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)
        return
    os.kill(pid, signal.SIGTERM)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _fmt_health(health: SourceHealth) -> str:
    if health.status == SourceStatus.HEALTHY:
        return "OK"
    if health.status == SourceStatus.NO_CREDENTIALS:
        return f"NO CREDENTIALS{f' ({health.detail})' if health.detail else ''}"
    if health.status == SourceStatus.DISABLED:
        return f"DISABLED{f' ({health.detail})' if health.detail else ''}"
    return f"{health.status.value.upper()}{f' ({health.detail})' if health.detail else ''}"


async def _probe(ingest: IngestService, name: str, timeout_seconds: float = 8.0) -> SourceHealth:
    adapters = ingest.all_adapters()
    adapter = adapters.get(name)
    if adapter is None:
        # Distinguish "never wired" from "failed to construct". The latter is a bug and
        # must say so - reporting a TypeError as "not implemented" hid a real wiring
        # break in the GDELT adapter for an entire build.
        build_error = ingest.adapter_build_errors().get(name)
        if build_error:
            return SourceHealth(
                name=name,
                status=SourceStatus.DOWN,
                detail=f"adapter failed to construct - {build_error}",
            )
        return SourceHealth(name=name, status=SourceStatus.DISABLED, detail="adapter not configured")
    try:
        return await asyncio.wait_for(adapter.probe(), timeout=timeout_seconds)
    except Exception as exc:  # noqa: BLE001
        return SourceHealth(name=name, status=SourceStatus.DOWN, detail=str(exc))


@dataclass
class _DoctorRow:
    label: str
    value: str
    required: bool = False
    ok: bool = True


async def _run_doctor(settings: Any) -> tuple[list[_DoctorRow], int]:
    clock = LiveClock()
    rows: list[_DoctorRow] = []
    exit_ok = True

    try:
        store = StateStore.open(settings.db_path)
        store.close()
        rows.append(_DoctorRow("Database", "OK"))
    except Exception as exc:  # noqa: BLE001
        rows.append(_DoctorRow("Database", f"FAIL ({exc})", required=True, ok=False))
        exit_ok = False

    try:
        settings.parquet_dir.mkdir(parents=True, exist_ok=True)
        probe_file = settings.parquet_dir / ".doctor_probe"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink()
        rows.append(_DoctorRow("Parquet storage", "OK"))
    except Exception as exc:  # noqa: BLE001
        rows.append(_DoctorRow("Parquet storage", f"FAIL ({exc})", required=True, ok=False))
        exit_ok = False

    try:
        _ = clock.now()
        rows.append(_DoctorRow("Clock", "OK"))
    except Exception as exc:  # noqa: BLE001
        rows.append(_DoctorRow("Clock", f"FAIL ({exc})", required=True, ok=False))
        exit_ok = False

    ingest = IngestService(settings, clock)
    try:
        (
            kalshi_health,
            poly_gamma_health,
            poly_us_health,
            sec_health,
            gdelt_health,
            x_health,
            bluesky_health,
            weather_health,
        ) = await asyncio.gather(
            _probe(ingest, "kalshi_rest"),
            _probe(ingest, "poly_gamma"),
            _probe(ingest, "poly_us_rest"),
            _probe(ingest, "sec"),
            _probe(ingest, "gdelt"),
            _probe(ingest, "x"),
            _probe(ingest, "bluesky"),
            _probe(ingest, "weather_nws"),
        )

        rows.append(_DoctorRow("Kalshi REST", _fmt_health(kalshi_health), required=True))
        if kalshi_health.status is SourceStatus.DOWN:
            exit_ok = False

        ws = ingest.kalshi_ws_adapter([])
        try:
            ws_health = await asyncio.wait_for(ws.probe(), timeout=8.0)
        except Exception as exc:  # noqa: BLE001
            ws_health = SourceHealth(name="kalshi_ws", status=SourceStatus.DOWN, detail=str(exc))
        ws_label = {
            SourceStatus.HEALTHY: "AUTHENTICATED" if ingest.kalshi_rest._auth.is_configured else "REACHABLE",  # noqa: SLF001
            SourceStatus.NO_CREDENTIALS: "NO CREDENTIALS",
        }.get(ws_health.status, _fmt_health(ws_health))
        rows.append(_DoctorRow("Kalshi WebSocket", ws_label))

        unmet_live = KalshiLiveBroker.preflight(settings, acknowledge_real_money_risk=False, rest_adapter=object())
        rows.append(_DoctorRow("Kalshi Live Orders", "DISABLED" if unmet_live else "ENABLED"))

        rows.append(_DoctorRow("Polymarket Global Data", _fmt_health(poly_gamma_health)))

        geo = await ingest.geoblock_check()
        geo_label = "BLOCKED FOR NEW US ORDERS" if geo.blocked else "NOT BLOCKED (unexpected -- execution stays disabled regardless)"
        rows.append(_DoctorRow("Polymarket Global Geo", geo_label))
        rows.append(_DoctorRow("Polymarket Global Trade", "DISABLED"))
        rows.append(_DoctorRow("Polymarket US Public", _fmt_health(poly_us_health)))

        rows.append(_DoctorRow("GDELT", _fmt_health(gdelt_health)))
        rows.append(_DoctorRow("SEC EDGAR", _fmt_health(sec_health)))
        rows.append(_DoctorRow("X", _fmt_health(x_health)))
        rows.append(_DoctorRow("Bluesky", _fmt_health(bluesky_health)))
        rows.append(_DoctorRow("NWS", _fmt_health(weather_health)))

        provider = await detect_provider(settings)
        provider_health = await provider.probe()
        rows.append(_DoctorRow("Ollama", "OK" if provider_health.ok else "UNAVAILABLE"))
        rows.append(_DoctorRow("Model", provider.model if provider_health.ok else "none"))
        await provider.close()

        rows.append(_DoctorRow("Mode", settings.mode.value))
        rows.append(_DoctorRow("Real-money trading", "HARD DISABLED" if not settings.live_allowed else "ARMED"))
    finally:
        for adapter in ingest.all_adapters().values():
            with contextlib.suppress(Exception):
                await adapter.close()
        if ingest._kalshi_ws is not None:  # noqa: SLF001
            with contextlib.suppress(Exception):
                await ingest._kalshi_ws.close()  # noqa: SLF001

    return rows, (0 if exit_ok else 1)


@app.command("doctor")
def doctor() -> None:
    """Probe every source/venue concurrently and report health."""
    settings = load_settings()
    rows, exit_code = _run(_run_doctor(settings))
    console.print("[bold]MarketLab Doctor[/bold]")
    console.print("=" * 40)
    width = max(len(r.label) for r in rows) + 2
    for r in rows:
        style = "red" if not r.ok else ""
        console.print(f"{r.label:<{width}} {r.value}", style=style or None)
    raise typer.Exit(code=exit_code)


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


@app.command("sources")
def sources_cmd() -> None:
    settings = load_settings()
    toggles = (settings.source_toggles or {}).get("sources", {})
    table = Table(title="Configured Sources")
    for col in ("Source", "Enabled", "Required", "Poll Seconds"):
        table.add_column(col)
    for name, cfg in toggles.items():
        cfg = cfg or {}
        table.add_row(name, str(cfg.get("enabled", True)), str(cfg.get("required", False)), str(cfg.get("poll_seconds", "-")))
    console.print(table)


# ---------------------------------------------------------------------------
# markets
# ---------------------------------------------------------------------------


async def _fetch_kalshi_markets_live(settings: Any, limit: int, category: str | None) -> list[Any]:
    from marketlab.adapters.kalshi.normalize import normalize_market
    from marketlab.adapters.kalshi.rest import KalshiRestAdapter

    clock = LiveClock()
    adapter = KalshiRestAdapter(settings, clock=clock)
    try:
        raw_rows: list[dict[str, Any]] = []
        async for page in adapter.iter_all_markets(status="open", limit=1000, max_pages=25):
            rows = page.get("markets") or []
            if not rows:
                break
            raw_rows.extend(rows)
        normalized = []
        for raw in raw_rows:
            try:
                normalized.append(normalize_market(raw))
            except Exception:  # noqa: BLE001
                continue
        selected = select_tracked_markets(normalized, settings.universes, max_tracked_markets=0)
        if category:
            selected = [m for m in selected if m.category.value == category]
        return selected[:limit]
    finally:
        await adapter.close()


async def _fetch_poly_markets_live(settings: Any, limit: int, category: str | None) -> list[Any]:
    from marketlab.adapters.polymarket_global.gamma import GammaAdapter
    from marketlab.adapters.polymarket_global.normalize import normalize_market as normalize_poly

    clock = LiveClock()
    adapter = GammaAdapter(settings.sources.poly_gamma, clock)
    try:
        raw_list = await adapter.get_markets(limit=max(limit * 3, 100), closed=False, order="volume24hr", ascending=False)
        normalized = []
        for raw in raw_list:
            try:
                normalized.append(normalize_poly(raw))
            except Exception:  # noqa: BLE001
                continue
        if category:
            normalized = [m for m in normalized if m.category.value == category]
        return normalized[:limit]
    finally:
        await adapter.close()


@app.command("markets")
def markets_cmd(
    venue: str = typer.Option("kalshi", "--venue", help="kalshi | poly-global | poly-us"),
    limit: int = typer.Option(25, "--limit"),
    category: str | None = typer.Option(None, "--category"),
) -> None:
    settings = load_settings()
    if venue == Venue.POLY_US.value:
        from marketlab.adapters.polymarket_us.public import PolymarketUsAdapter

        clock = LiveClock()

        async def _probe_us() -> SourceHealth:
            adapter = PolymarketUsAdapter(settings.sources.poly_us_rest, clock)
            try:
                return await adapter.probe()
            finally:
                await adapter.close()

        health = _run(_probe_us())
        console.print(f"Polymarket US: {_fmt_health(health)}")
        console.print("[dim]No public data tier is available on this host; see docs/FINDINGS.md #15.[/dim]")
        return

    if venue not in (Venue.POLY_GLOBAL.value, Venue.KALSHI.value):
        console.print(f"[red]unknown venue: {venue!r}[/red]")
        raise typer.Exit(code=1)

    # Prefer what we have actually ingested and are tracking. A live fetch returns
    # whatever the venue happens to hand back first - on Kalshi that is dominated by
    # zero-volume KXMVE parlays (docs/FINDINGS.md #8) and tells the operator nothing
    # about what the tournament is really trading.
    rows: list[Any] = []
    source_label = "database"
    try:
        store = StateStore.open(settings.db_path)
        try:
            rows = store.list_markets(venue=venue, category=category, limit=limit)
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001 - a fresh checkout has no database yet
        log.warning("markets_db_read_failed", error=str(exc))

    if not rows:
        source_label = "live venue (database empty - run 'marketlab ingest --once')"
        if venue == Venue.POLY_GLOBAL.value:
            rows = _run(_fetch_poly_markets_live(settings, limit, category))
        else:
            rows = _run(_fetch_kalshi_markets_live(settings, limit, category))

    table = Table(title=f"Markets ({venue})")
    for col in ("Canonical ID", "Title", "Category", "Status", "Volume", "Close Time"):
        table.add_column(col)
    for m in rows:
        table.add_row(
            m.canonical_id,
            (m.title[:50] + "...") if len(m.title) > 53 else m.title,
            m.category.value,
            m.status.value,
            str(m.volume),
            m.close_time.isoformat() if m.close_time else "-",
        )
    console.print(table)
    console.print(f"[dim]{len(rows)} row(s) shown (source: {source_label})[/dim]")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@app.command("ingest")
def ingest_cmd(
    once: bool = typer.Option(False, "--once"),
    daemon: bool = typer.Option(False, "--daemon"),
) -> None:
    settings = load_settings()
    clock = LiveClock()

    if once:
        # A store (and a parquet writer) are not optional here: without them run_once
        # fetches several hundred markets, reports them, and drops every one on the
        # floor. `marketlab markets` would then show an empty database right after an
        # apparently successful ingest.
        store = StateStore.open(settings.db_path)
        parquet = ParquetWriter(settings.parquet_dir)
        service = IngestService(settings, clock, store=store, parquet=parquet)

        async def _once() -> Any:
            try:
                return await service.run_once()
            finally:
                await service.stop()
                with contextlib.suppress(Exception):
                    parquet.flush_all()
                    parquet.close()
                with contextlib.suppress(Exception):
                    store.close()

        result = _run(_once())
        table = Table(title="Ingest -- single pass")
        for col in ("Metric", "Count"):
            table.add_column(col)
        table.add_row("Kalshi markets", str(result.kalshi_markets))
        table.add_row("Kalshi books", str(result.kalshi_books))
        table.add_row("Kalshi trades", str(result.kalshi_trades))
        table.add_row("Polymarket markets", str(result.poly_markets))
        table.add_row("Polymarket books", str(result.poly_books))
        table.add_row("Polymarket activity", str(result.poly_activity))
        table.add_row("Leaderboard rows", str(result.leaderboard_rows))
        table.add_row("Geoblock confirmed", str(result.geoblock_confirmed))
        console.print(table)
        if result.errors:
            console.print("[yellow]Errors during this pass:[/yellow]")
            for e in result.errors:
                console.print(f"  - {e}")
        if not result.ok:
            console.print(
                "[bold red]FAILURE: zero markets ingested across every venue. "
                "This is not a quiet no-op -- something is broken.[/bold red]"
            )
            raise typer.Exit(code=1)
        raise typer.Exit(code=0)

    if daemon:
        store = StateStore.open(settings.db_path)
        parquet = ParquetWriter(settings.parquet_dir)
        service = IngestService(settings, clock, store=store, parquet=parquet)

        async def _daemon() -> None:
            await service.start()
            console.print("[green]Ingest daemon running. Press Ctrl+C to stop.[/green]")
            try:
                while True:
                    await asyncio.sleep(30)
                    console.print(
                        f"tracked markets={len(service.markets)} tracked books={len(service.books)} "
                        f"queue_size={service.queue.qsize()}"
                    )
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            finally:
                await service.stop()

        with contextlib.suppress(KeyboardInterrupt):
            _run(_daemon())
        return

    console.print("[yellow]specify --once or --daemon[/yellow]")
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# trader-discover / trader-show
# ---------------------------------------------------------------------------


@app.command("trader-discover")
def trader_discover(limit: int = typer.Option(20, "--limit")) -> None:
    from marketlab.adapters.polymarket_global.leaderboard import LeaderboardAdapter

    settings = load_settings()
    clock = LiveClock()

    async def _run_discover() -> list[dict[str, Any]]:
        adapter = LeaderboardAdapter(settings.sources.poly_leaderboard, settings.sources.poly_lb_legacy, clock)
        try:
            return await adapter.get_official(category="overall", period="week", metric="pnl", limit=limit)
        finally:
            await adapter.close()

    rows = _run(_run_discover())
    table = Table(title="Discovered Traders (Polymarket, overall/week/pnl)")
    for col in ("Rank", "Wallet", "Username", "PnL", "Volume"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            str(r.get("rank", "-")),
            str(r.get("proxyWallet", ""))[:14] + "...",
            str(r.get("userName") or "-"),
            str(r.get("pnl", "-")),
            str(r.get("vol", "-")),
        )
    console.print(table)


@app.command("trader-show")
def trader_show(wallet: str) -> None:
    from marketlab.adapters.polymarket_global.data_api import DataApiAdapter

    settings = load_settings()
    clock = LiveClock()

    async def _run_show() -> tuple[list[dict[str, Any]], Decimal | None]:
        adapter = DataApiAdapter(settings.sources.poly_data, clock)
        try:
            positions = await adapter.get_positions(wallet, limit=25)
            value = await adapter.get_value(wallet)
            return positions, value
        finally:
            await adapter.close()

    positions, value = _run(_run_show())
    console.print(f"[bold]Wallet:[/bold] {wallet}")
    console.print(f"[bold]Portfolio value:[/bold] {value if value is not None else 'unknown'}")
    table = Table(title="Open Positions")
    for col in ("Market", "Outcome", "Size", "Avg Price"):
        table.add_column(col)
    for p in positions[:25]:
        table.add_row(
            str(p.get("title") or p.get("conditionId", "")),
            str(p.get("outcome", "")),
            str(p.get("size", "")),
            str(p.get("avgPrice", "")),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# ai doctor / ai assess
# ---------------------------------------------------------------------------


@ai_app.command("doctor")
def ai_doctor() -> None:
    settings = load_settings()

    async def _run_ai_doctor() -> tuple[str, str, bool, str]:
        provider = await detect_provider(settings)
        health = await provider.probe()
        await provider.close()
        return provider.name, provider.model, health.ok, health.detail

    name, model, ok, detail = _run(_run_ai_doctor())
    table = Table(title="AI Doctor")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Provider", name)
    table.add_row("Model", model)
    table.add_row("Status", "OK" if ok else "UNAVAILABLE")
    table.add_row("Detail", detail or "-")
    console.print(table)


@ai_app.command("stack")
def ai_stack_cmd() -> None:
    """Show which AI tiers (local / Jev / OpenAI) are live and which arms will run."""
    from marketlab.ai.stack import STACKS, build_ai_stack

    settings = load_settings()

    async def _probe() -> list[tuple[str, str, bool, str]]:
        detected = await detect_provider(settings)
        stack = await build_ai_stack(settings, detected, LiveClock(), lambda _cid: None)
        rows = []
        for tier in ("local", "jev", "openai"):
            provider = stack.providers.get(tier)
            if provider is None:
                rows.append((tier, "-", False, "not configured"))
                continue
            health = await provider.probe()
            rows.append((tier, provider.model, health.ok, health.detail))
            await provider.close()
        rows.append(("arms", ", ".join(stack.available_stacks()) or "none", True,
                     "of " + ", ".join(STACKS)))
        return rows

    table = Table(title="AI Stack")
    for col in ("Tier", "Model", "Status", "Detail"):
        table.add_column(col)
    for tier, model, ok, detail in _run(_probe()):
        table.add_row(tier, model, "OK" if ok else "UNAVAILABLE", detail or "-")
    console.print(table)


@ai_app.command("assess")
def ai_assess(market_id: str) -> None:
    settings = load_settings()

    async def _run_assess() -> tuple[str, str]:
        provider = await detect_provider(settings)
        try:
            prompt = (
                f"Assess the prediction market {market_id!r}. In two or three sentences, give a "
                "probability estimate for YES and your rationale. If you have no real information "
                "about this market, say so plainly rather than inventing facts."
            )
            response = await provider.generate(prompt)
            return provider.model, response.text
        finally:
            await provider.close()

    model, text = _run(_run_assess())
    console.print(f"[bold]Model:[/bold] {model}")
    console.print(text or "[dim](no output -- AI disabled or the model abstained)[/dim]")


# ---------------------------------------------------------------------------
# paper start / status / leaderboard / stop
# ---------------------------------------------------------------------------


def _print_paper_banner(settings: Any) -> None:
    console.print(
        Panel.fit(
            f"MODE: {settings.mode.value}\n"
            "Kalshi Market Data:       LIVE\n"
            "Polymarket Global Data:   LIVE\n"
            "Polymarket Global Orders: DISABLED\n"
            "Polymarket US Data:       UNAVAILABLE (401)\n"
            "Real Orders Submitted:    0\n"
            "Paper Strategies:         running\n"
            f"Virtual Bankroll:         ${settings.paper.bankroll_per_strategy} per experiment",
            title="MarketLab Paper Trading",
        )
    )


@paper_app.command("start")
def paper_start(
    bankroll_per_strategy: float = typer.Option(50.0, "--bankroll-per-strategy"),
    foreground: bool = typer.Option(True, "--foreground/--background"),
) -> None:
    settings = load_settings()
    if bankroll_per_strategy != 50.0:
        settings.paper = settings.paper.model_copy(update={"bankroll_per_strategy": Decimal(str(bankroll_per_strategy))})

    existing = _read_pid_file(settings)
    if existing is not None and _pid_alive(existing):
        console.print(f"[yellow]a paper daemon is already running (pid {existing})[/yellow]")
        raise typer.Exit(code=1)

    _print_paper_banner(settings)

    if not foreground:
        settings.ensure_dirs()
        log_path = Path(settings.log_dir) / "daemon_stdout.log"
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0)
        with open(log_path, "ab") as fh:
            proc = subprocess.Popen(  # noqa: S603
                [sys.executable, "-m", "marketlab.cli", "run"],
                stdout=fh,
                stderr=fh,
                creationflags=creationflags,
            )
        _write_pid_file(settings, proc.pid)
        console.print(f"Started paper daemon in the background (pid {proc.pid}). Logs: {log_path}")
        return

    _write_pid_file(settings)
    try:
        _run(Supervisor(settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
    finally:
        _remove_pid_file(settings)


@paper_app.command("status")
def paper_status() -> None:
    settings = load_settings()
    pid = _read_pid_file(settings)
    running = pid is not None and _pid_alive(pid)

    table = Table(title="Paper Daemon Status")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Running", "yes" if running else "no")
    table.add_row("PID", str(pid) if pid else "-")

    status_path = _status_file(settings)
    if status_path.exists():
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
            for key in (
                "mode", "started_at", "uptime_seconds", "events_processed", "markets_tracked",
                "sleeves_alive", "sleeves_dead", "orders_submitted", "orders_blocked",
                "trading_allowed", "trading_detail", "last_heartbeat",
            ):
                table.add_row(key, str(data.get(key)))
        except Exception as exc:  # noqa: BLE001
            table.add_row("status_file_error", str(exc))
    else:
        table.add_row("status_file", "not found yet (no heartbeat since last start)")

    try:
        store = StateStore.open(settings.db_path)
        try:
            table.add_row("experiments_in_store", str(len(store.list_experiments())))
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001
        table.add_row("store_error", str(exc))

    console.print(table)
    if not running:
        raise typer.Exit(code=1)


@paper_app.command("leaderboard")
def paper_leaderboard(
    only_traded: bool = typer.Option(
        False, "--traded", help="Show only sleeves that have actually placed a trade"
    ),
) -> None:
    from marketlab.analytics.reports import StrategyResult, strategy_league

    settings = load_settings()
    store = StateStore.open(settings.db_path)
    try:
        results = []
        traded = 0
        total_equity = Decimal(0)
        for exp in store.list_experiments():
            portfolio = store.load_portfolio(exp.experiment_id)
            if portfolio is None:
                continue
            equity = portfolio.equity()
            total_equity += equity
            if portfolio.trade_count:
                traded += 1
            # Include the universe and a short id discriminator: ~400 sleeves share only
            # 16 strategy names, so a bare name produces pages of identical-looking rows
            # and the operator cannot tell which variant is which.
            label = f"{exp.strategy_name}/{exp.market_universe}"
            suffix = exp.experiment_id.rsplit("__", 1)[-1][:6]
            results.append(
                StrategyResult(
                    strategy_id=f"{label}#{suffix}",
                    category=exp.market_universe or "-",
                    equity=equity,
                    net_pnl=equity - portfolio.initial_capital,
                    max_drawdown=portfolio.max_drawdown,
                    n_trades=portfolio.trade_count,
                    status="dead" if portfolio.is_dead(settings.risk.death_floor) else "active",
                )
            )
    finally:
        store.close()

    if only_traded:
        results = [r for r in results if r.n_trades]

    console.print(strategy_league(results))
    # A sleeve that has not traded sits at exactly its starting $50 and therefore sorts
    # above every sleeve that has actually risked anything. Saying so plainly stops that
    # from being read as "the untraded strategies are winning".
    console.print(
        f"[dim]{len(results)} sleeve(s) shown | {traded} have traded | "
        f"total virtual equity ${total_equity:,.2f} | "
        f"untraded sleeves sit at their full $50 starting bankroll[/dim]"
    )


@paper_app.command("stop")
def paper_stop() -> None:
    settings = load_settings()
    pid = _read_pid_file(settings)
    if pid is None or not _pid_alive(pid):
        console.print("[yellow]no running paper daemon found[/yellow]")
        raise typer.Exit(code=1)
    _signal_stop(pid)
    console.print(f"Sent stop signal to pid {pid}. Graceful shutdown flushes storage before exit.")


# ---------------------------------------------------------------------------
# experiments list / show / compare
# ---------------------------------------------------------------------------


@experiments_app.command("list")
def experiments_list(status: str | None = typer.Option(None, "--status")) -> None:
    settings = load_settings()
    store = StateStore.open(settings.db_path)
    try:
        exps = store.list_experiments(status=status)
        table = Table(title="Experiments")
        for col in ("ID", "Strategy", "Version", "Universe", "Status", "Created"):
            table.add_column(col)
        for e in exps:
            table.add_row(
                e.experiment_id, e.strategy_name, e.strategy_version, e.market_universe,
                str(e.status), e.created_at.isoformat() if e.created_at else "-",
            )
        console.print(table)
    finally:
        store.close()


@experiments_app.command("show")
def experiments_show(experiment_id: str) -> None:
    settings = load_settings()
    store = StateStore.open(settings.db_path)
    try:
        exp = store.get_experiment(experiment_id)
        if exp is None:
            console.print(f"[red]no such experiment: {experiment_id}[/red]")
            raise typer.Exit(code=1)
        portfolio = store.load_portfolio(experiment_id)
        table = Table(title=f"Experiment {experiment_id}")
        table.add_column("Field")
        table.add_column("Value")
        for field_name in (
            "strategy_name", "strategy_version", "market_universe", "venue", "status", "cohort",
            "parameter_hash", "starting_bankroll", "created_at",
        ):
            table.add_row(field_name, str(getattr(exp, field_name)))
        if portfolio is not None:
            table.add_row("equity", str(portfolio.equity()))
            table.add_row("cash", str(portfolio.cash))
            table.add_row("realized_pnl", str(portfolio.realized_pnl))
            table.add_row("trade_count", str(portfolio.trade_count))
            table.add_row("max_drawdown", str(portfolio.max_drawdown))
        console.print(table)
    finally:
        store.close()


@experiments_app.command("compare")
def experiments_compare(id_a: str, id_b: str) -> None:
    settings = load_settings()
    store = StateStore.open(settings.db_path)
    try:
        pa = store.load_portfolio(id_a)
        pb = store.load_portfolio(id_b)
        table = Table(title=f"Compare {id_a} vs {id_b}")
        table.add_column("Metric")
        table.add_column(id_a)
        table.add_column(id_b)

        def row(label: str, a: Any, b: Any) -> None:
            table.add_row(label, str(a), str(b))

        row("equity", pa.equity() if pa else "-", pb.equity() if pb else "-")
        row("cash", pa.cash if pa else "-", pb.cash if pb else "-")
        row("realized_pnl", pa.realized_pnl if pa else "-", pb.realized_pnl if pb else "-")
        row("trades", pa.trade_count if pa else "-", pb.trade_count if pb else "-")
        row("max_drawdown", pa.max_drawdown if pa else "-", pb.max_drawdown if pb else "-")
        console.print(table)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# report daily / strategies / categories / traders / risk
# ---------------------------------------------------------------------------


def _load_strategy_results(settings: Any) -> list[Any]:
    from marketlab.analytics.reports import StrategyResult

    store = StateStore.open(settings.db_path)
    try:
        results = []
        for exp in store.list_experiments():
            portfolio = store.load_portfolio(exp.experiment_id)
            if portfolio is None:
                continue
            equity = portfolio.equity()
            results.append(
                StrategyResult(
                    strategy_id=exp.strategy_name or exp.experiment_id,
                    category=exp.market_universe or "-",
                    equity=equity,
                    net_pnl=equity - portfolio.initial_capital,
                    max_drawdown=portfolio.max_drawdown,
                    n_trades=portfolio.trade_count,
                    status="dead" if portfolio.is_dead(settings.risk.death_floor) else "active",
                )
            )
        return results
    finally:
        store.close()


@report_app.command("daily")
def report_daily() -> None:
    settings = load_settings()
    path = Path(settings.reports_dir) / f"daily_{date.today().isoformat()}.json"
    if path.exists():
        console.print_json(path.read_text(encoding="utf-8"))
    else:
        console.print(f"[yellow]no daily report yet at {path} (the daemon writes one once per day it runs)[/yellow]")


@report_app.command("strategies")
def report_strategies() -> None:
    from marketlab.analytics.reports import best_by_category, strategy_league

    settings = load_settings()
    results = _load_strategy_results(settings)
    console.print(strategy_league(results))
    console.print(best_by_category(results))


@report_app.command("categories")
def report_categories() -> None:
    from marketlab.analytics.reports import category_report

    settings = load_settings()
    console.print(category_report(_load_strategy_results(settings)))


@report_app.command("traders")
def report_traders() -> None:
    settings = load_settings()
    store = StateStore.open(settings.db_path)
    try:
        traders = store.list_traders()
        table = Table(title="Tracked Traders")
        for col in ("Wallet", "Username", "Status", "Rank@Discovery", "All-Time PnL"):
            table.add_column(col)
        for t in traders:
            table.add_row(t.wallet[:12] + "...", t.username or "-", str(t.status), str(t.rank_at_discovery), str(t.all_time_pnl))
        console.print(table)
    finally:
        store.close()


@report_app.command("risk")
def report_risk() -> None:
    from marketlab.analytics.reports import risk_report

    settings = load_settings()
    console.print(risk_report(_load_strategy_results(settings)))


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


@app.command("replay")
def replay(
    day: str = typer.Argument(None, help="UTC date to replay, e.g. 2026-09-12"),
    list_dates: bool = typer.Option(False, "--list", help="List the recorded dates and exit."),
    strategies: str = typer.Option("", "--strategies", help="Comma-separated strategy names (default: all non-AI)."),
    tick_seconds: float = typer.Option(10.0, "--tick-seconds", help="Simulated seconds between timer ticks."),
) -> None:
    """Replay a recorded day's real Kalshi books through the strategies (no look-ahead)."""
    from marketlab.experiments.replay import available_dates, run_replay

    settings = load_settings()
    dates = available_dates(Path(settings.parquet_dir))
    if list_dates or not day:
        console.print("Recorded dates: " + (", ".join(dates) if dates else "none yet"))
        return
    if day not in dates:
        console.print(f"[yellow]no recorded books for {day}[/yellow]; recorded: {', '.join(dates) or 'none'}")
        raise typer.Exit(code=1)
    chosen = {s.strip() for s in strategies.split(",") if s.strip()} or None
    console.print(f"Replaying {day} ... (real recorded books, simulated clock, fresh sleeves)")
    result = _run(run_replay(settings, day, strategies=chosen, tick_seconds=tick_seconds))

    table = Table(title=f"Replay {day}: top sleeves")
    for col in ("Strategy", "Category", "Equity", "P&L", "Trades"):
        table.add_column(col)
    traded = [r for r in result.leaderboard if r["trades"]]
    for r in sorted(traded, key=lambda r: r["pnl"], reverse=True)[:25]:
        table.add_row(r["strategy"], r["universe"], f"${r['equity']:.2f}", f"{r['pnl']:+.2f}", str(r["trades"]))
    console.print(table)
    families: dict[str, list[float]] = {}
    for r in traded:
        families.setdefault(r["strategy"], []).append(r["pnl"])
    console.print("Family totals: " + ", ".join(
        f"{k} {sum(v):+.2f} ({len(v)} sleeves)" for k, v in sorted(families.items(), key=lambda kv: -sum(kv[1]))
    ))
    out = Path(settings.data_dir) / "reports" / f"replay_{day}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "day": day, "events": result.events, "book_snapshots": result.book_snapshots,
        "settlements": result.settlements, "sleeves": result.sleeves, "leaderboard": result.leaderboard,
    }, indent=2), encoding="utf-8")
    console.print(
        f"[dim]{result.book_snapshots:,} book snapshots, {result.settlements} settlements, "
        f"{result.sleeves} sleeves. Database: {result.db_path}. Summary: {out}[/dim]"
    )


# ---------------------------------------------------------------------------
# live doctor / live start
# ---------------------------------------------------------------------------


@live_app.command("doctor")
def live_doctor() -> None:
    settings = load_settings()
    unmet = KalshiLiveBroker.preflight(settings, acknowledge_real_money_risk=False, rest_adapter=object())

    static_gates = [
        ("MARKETLAB_MODE=LIVE", settings.mode is Mode.LIVE),
        ("LIVE_TRADING_ENABLED=YES_I_ACCEPT_REAL_LOSS", settings.secrets.live_armed),
        ("Kalshi credentials configured", bool(settings.secrets.kalshi_api_key_id and settings.secrets.kalshi_private_key_path)),
        ("Kalshi environment is production", (settings.secrets.kalshi_environment or "production").strip().lower() == "production"),
    ]
    table = Table(title="Live Trading Gate Checklist")
    table.add_column("Gate")
    table.add_column("Status")
    all_pass = True
    for label, ok in static_gates:
        table.add_row(label, "PASS" if ok else "FAIL")
        all_pass = all_pass and ok
    table.add_row("acknowledge_real_money_risk (pass --acknowledge-real-money-risk to `live start`)", "N/A (per-invocation)")
    for u in unmet:
        if u not in {g[0] for g in static_gates}:
            table.add_row(f"unmet: {u}", "FAIL")
            all_pass = False
    console.print(table)
    if not all_pass:
        console.print("[red]Live trading is NOT eligible to start.[/red]")
        raise typer.Exit(code=1)
    console.print(
        "[yellow]Static gates pass. `live start --acknowledge-real-money-risk` still re-checks every "
        "gate, plus per-order risk/health checks, before any real order can be placed.[/yellow]"
    )


@live_app.command("start")
def live_start(
    acknowledge_real_money_risk: bool = typer.Option(False, "--acknowledge-real-money-risk"),
) -> None:
    settings = load_settings()
    unmet = KalshiLiveBroker.preflight(settings, acknowledge_real_money_risk=acknowledge_real_money_risk, rest_adapter=object())
    if unmet:
        console.print("[bold red]Live trading refused. Unmet gates:[/bold red]")
        for u in unmet:
            console.print(f"  - {u}")
        raise typer.Exit(code=1)

    console.print("[bold red]LIVE MODE ARMED. Real money is at risk.[/bold red]")
    _write_pid_file(settings)
    try:
        _run(Supervisor(settings).run(acknowledge_real_money_risk=True))
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
    finally:
        _remove_pid_file(settings)


# ---------------------------------------------------------------------------
# kill switch / dashboard
# ---------------------------------------------------------------------------


@app.command("kill")
def kill_cmd(
    stop_daemon: bool = typer.Option(True, "--stop-daemon/--keep-running",
                                     help="Also stop the daemon after engaging the switch."),
    reason: str = typer.Option("manual emergency stop", "--reason"),
) -> None:
    """EMERGENCY STOP: engage the global kill switch (refuses all new risk) and stop the daemon."""
    from marketlab.execution.risk_gateway import KILL_SWITCH_PATH

    KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    KILL_SWITCH_PATH.write_text(f"{datetime.now(UTC).isoformat()} {reason}\n", encoding="utf-8")
    console.print(f"[bold red]KILL SWITCH ENGAGED[/bold red] ({KILL_SWITCH_PATH}). No new positions will open.")
    if stop_daemon:
        settings = load_settings()
        pid = _read_pid_file(settings)
        if pid is not None and _pid_alive(pid):
            _signal_stop(pid)
            console.print(f"Sent stop signal to daemon pid {pid}.")
    console.print("Run `marketlab unkill` to clear it.")


@app.command("unkill")
def unkill_cmd() -> None:
    """Clear the global kill switch."""
    from marketlab.execution.risk_gateway import KILL_SWITCH_PATH

    if KILL_SWITCH_PATH.exists():
        KILL_SWITCH_PATH.unlink()
        console.print("[green]kill switch cleared[/green]")
    else:
        console.print("kill switch was not engaged")


@app.command("dashboard")
def dashboard_cmd(
    port: int = typer.Option(8765, "--port"),
    open_browser: bool = typer.Option(False, "--open/--no-open"),
) -> None:
    """Serve the local read-only web dashboard (http://127.0.0.1:PORT)."""
    from marketlab.dashboard.server import serve

    settings = load_settings()
    serve(settings, port=port, open_browser=open_browser)


# ---------------------------------------------------------------------------
# run -- the foreground daemon used by service scripts
# ---------------------------------------------------------------------------


def _keep_system_awake() -> None:
    """Ask Windows not to idle-sleep while the daemon runs (released when it exits).

    A sleeping laptop silently stops the tournament. This does not keep the display on
    and cannot override closing the lid if the power plan says "sleep on lid close".
    """
    if os.name != "nt":
        return
    with contextlib.suppress(Exception):
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


@app.command("run")
def run_cmd() -> None:
    """Run the full daemon in the foreground. Used by service-manager scripts."""
    settings = load_settings()
    _keep_system_awake()
    _write_pid_file(settings)
    try:
        _run(Supervisor(settings).run())
    except KeyboardInterrupt:
        pass
    finally:
        _remove_pid_file(settings)


if __name__ == "__main__":
    app()
