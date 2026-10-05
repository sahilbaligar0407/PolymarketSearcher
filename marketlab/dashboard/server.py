"""Local read-only web dashboard.

A stdlib HTTP server bound to 127.0.0.1 that reads the daemon's SQLite database through a
``mode=ro`` connection and its heartbeat file. It never writes to the database, never
talks to a venue, and never imports the broker, so it cannot place or alter an order. It
can run while the daemon is up or down; with the daemon down it simply shows the last
recorded state.

Every query here is chosen to hit an index: the database is several GB after a week, and
a dashboard left open in a browser must not compete with the daemon for the disk.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import webbrowser
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import median
from typing import Any

from marketlab.execution.risk_gateway import KILL_SWITCH_PATH
from marketlab.logging import get_logger

log = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
CACHE_SECONDS = 20.0

#: Strategy families that exist to be beaten. A real strategy's P&L means little until
#: it is compared against these on the same markets and the same fill model.
CONTROL_FAMILIES = ("random_control", "fade_control")

_INACTIVE = ("DEAD", "DISABLED", "RETIRED", "dead", "disabled", "retired")


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    cur = conn.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def _short_params(params_json: str) -> str:
    try:
        params = json.loads(params_json or "{}")
    except ValueError:
        return ""
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))[:160]


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class DashboardData:
    """Builds the dashboard's JSON snapshot, cached for :data:`CACHE_SECONDS`."""

    def __init__(self, db_path: Path, data_dir: Path) -> None:
        self.db_path = db_path
        self.data_dir = data_dir
        self._lock = threading.Lock()
        self._cached: dict[str, Any] | None = None
        self._cached_at = 0.0

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True, timeout=15)
        conn.execute("PRAGMA query_only = 1")
        return conn

    def summary(self) -> dict[str, Any]:
        with self._lock:
            if self._cached is not None and time.monotonic() - self._cached_at < CACHE_SECONDS:
                return self._cached
            started = time.monotonic()
            try:
                data = self._build()
            except sqlite3.Error as exc:
                log.error("dashboard.query_failed", error=str(exc), exc_info=True)
                data = {"error": f"database unavailable: {exc}", "status": self._status()}
            data["generated_at"] = datetime.now(UTC).isoformat()
            data["build_ms"] = round((time.monotonic() - started) * 1000)
            self._cached, self._cached_at = data, time.monotonic()
            return data

    # ------------------------------------------------------------------ sections

    def _status(self) -> dict[str, Any]:
        status: dict[str, Any] = {}
        path = self.data_dir / "daemon_status.json"
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = {}
        pid = None
        try:
            pid = int((self.data_dir / "marketlab.pid").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = None
        running = pid is not None and _pid_alive(pid)
        heartbeat_age = None
        if status.get("last_heartbeat"):
            try:
                hb = datetime.fromisoformat(status["last_heartbeat"])
                heartbeat_age = (datetime.now(UTC) - hb).total_seconds()
            except ValueError:
                heartbeat_age = None
        feeds = status.get("feed_health", {}) or {}
        return {
            "running": running and (heartbeat_age is None or heartbeat_age < 180),
            "pid": pid,
            "mode": status.get("mode", "PAPER"),
            "started_at": status.get("started_at"),
            "uptime_seconds": status.get("uptime_seconds"),
            "heartbeat_age_seconds": heartbeat_age,
            "events_processed": status.get("events_processed"),
            "markets_tracked": status.get("markets_tracked"),
            "orders_submitted": status.get("orders_submitted"),
            "trading_allowed": status.get("trading_allowed"),
            "trading_detail": status.get("trading_detail", ""),
            "kill_switch": KILL_SWITCH_PATH.exists(),
            "ai": status.get("ai", {}),
            "risk_rejects": status.get("risk_rejects", {}),
            "feeds": {
                name: {"status": f.get("status"), "required": f.get("required"), "detail": f.get("detail", "")}
                for name, f in feeds.items()
            },
        }

    def _sleeves(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        latest = _rows(
            conn,
            """
            SELECT e.experiment_id, e.strategy_name, e.market_universe, e.parameters_json,
                   e.status AS exp_status, e.llm_model_id, b.equity, b.cash, b.realized_pnl,
                   b.unrealized_pnl, b.exposure, b.max_drawdown, b.trade_count,
                   b.resolved_trade_count, b.fees_paid, b.initial_capital, b.timestamp
            FROM (SELECT x.experiment_id,
                         (SELECT MAX(id) FROM balances WHERE experiment_id = x.experiment_id) AS mid
                  FROM experiments x) m
            JOIN balances b ON b.id = m.mid
            JOIN experiments e ON e.experiment_id = m.experiment_id
            """,
        )
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        day_open = {
            r["experiment_id"]: _f(r["equity"])
            for r in _rows(
                conn,
                """
                SELECT x.experiment_id,
                       (SELECT equity FROM balances
                        WHERE experiment_id = x.experiment_id AND timestamp >= ?
                        ORDER BY timestamp, id LIMIT 1) AS equity
                FROM experiments x
                """,
                (today,),
            )
        }
        out = []
        for r in latest:
            equity = _f(r["equity"]) or 0.0
            initial = _f(r["initial_capital"]) or 50.0
            opened = day_open.get(r["experiment_id"])
            out.append({
                "experiment_id": r["experiment_id"],
                "strategy": r["strategy_name"],
                "universe": r["market_universe"],
                "params": _short_params(r["parameters_json"]),
                "status": r["exp_status"],
                "model": r["llm_model_id"] or "",
                "equity": round(equity, 2),
                "pnl": round(equity - initial, 2),
                "roi": round((equity - initial) / initial, 4) if initial else 0.0,
                "today_pnl": round(equity - opened, 2) if opened is not None else 0.0,
                "max_drawdown": _f(r["max_drawdown"]),
                "trades": int(r["trade_count"] or 0),
                "resolved": int(r["resolved_trade_count"] or 0),
                "fees": round(_f(r["fees_paid"]) or 0.0, 2),
                "exposure": round(_f(r["exposure"]) or 0.0, 2),
                "updated": r["timestamp"],
            })
        return out

    @staticmethod
    def _families(sleeves: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by: dict[str, list[dict[str, Any]]] = {}
        for s in sleeves:
            by.setdefault(s["strategy"], []).append(s)
        out = []
        for name, items in by.items():
            traded = [s for s in items if s["trades"] > 0]
            pnls = [s["pnl"] for s in traded]
            out.append({
                "strategy": name,
                "sleeves": len(items),
                "traded": len(traded),
                "total_pnl": round(sum(pnls), 2),
                "median_pnl": round(median(pnls), 2) if pnls else 0.0,
                "best_pnl": round(max(pnls), 2) if pnls else 0.0,
                "worst_pnl": round(min(pnls), 2) if pnls else 0.0,
                "share_profitable": round(sum(1 for p in pnls if p > 0) / len(pnls), 3) if pnls else None,
                "resolved_trades": sum(s["resolved"] for s in items),
                "fees": round(sum(s["fees"] for s in items), 2),
                "today_pnl": round(sum(s["today_pnl"] for s in items), 2),
                "is_control": name in CONTROL_FAMILIES,
            })
        control = [f["median_pnl"] for f in out if f["is_control"] and f["traded"]]
        baseline = max(control) if control else None
        for f in out:
            f["beats_control"] = None if baseline is None or not f["traded"] else f["median_pnl"] > baseline
            # Below ~30 resolved trades a P&L is an anecdote, not evidence.
            f["credible"] = f["resolved_trades"] >= 30
        out.sort(key=lambda f: f["total_pnl"], reverse=True)
        return out

    def _recent_trades(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = _rows(
            conn,
            """
            SELECT f.fill_id, f.order_id, f.experiment_id, f.canonical_id, f.side, f.action,
                   f.price, f.quantity, f.fee, f.timestamp, e.strategy_name, e.market_universe,
                   m.title, s.winning_side, s.voided, d.rationale, d.features_json,
                   d.model_probability, d.expected_edge
            FROM (SELECT * FROM fills ORDER BY timestamp DESC LIMIT 60) f
            JOIN experiments e ON e.experiment_id = f.experiment_id
            LEFT JOIN market_registry m ON m.canonical_id = f.canonical_id
            LEFT JOIN settlements s ON s.canonical_id = f.canonical_id
            LEFT JOIN orders o ON o.order_id = f.order_id
            LEFT JOIN decisions d ON d.intent_id = o.intent_id
            ORDER BY f.timestamp DESC
            """,
        )
        out = []
        for r in rows:
            price = _f(r["price"]) or 0.0
            qty = int(r["quantity"] or 0)
            fee = _f(r["fee"]) or 0.0
            outcome, realized = "open", None
            if r["winning_side"] and not r["voided"] and r["action"] == "buy":
                won = r["winning_side"] == r["side"]
                outcome = "won" if won else "lost"
                realized = round(((1.0 - price) if won else -price) * qty - fee, 2)
            elif r["voided"]:
                outcome = "voided"
            features: dict[str, Any] = {}
            if r["features_json"]:
                try:
                    features = json.loads(r["features_json"])
                except ValueError:
                    features = {}
            out.append({
                "time": r["timestamp"],
                "strategy": r["strategy_name"],
                "universe": r["market_universe"],
                "market": r["title"] or r["canonical_id"],
                "canonical_id": r["canonical_id"],
                "side": r["side"],
                "action": r["action"],
                "price": price,
                "quantity": qty,
                "fee": round(fee, 2),
                "outcome": outcome,
                "realized": realized,
                "rationale": r["rationale"] or "(placed before decision logging existed)",
                "model_probability": _f(r["model_probability"]),
                "expected_edge": _f(r["expected_edge"]),
                "evidence": features.get("evidence") or [],
                "second_opinion": features.get("second_opinion_model"),
                "risk": "passed all risk-gateway checks (order was accepted and filled)",
            })
        return out

    def _open_positions(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = _rows(
            conn,
            """
            SELECT p.experiment_id, p.canonical_id, p.side, p.quantity, p.average_price,
                   p.last_update, e.strategy_name, m.title, m.close_time
            FROM positions p
            JOIN experiments e ON e.experiment_id = p.experiment_id
            LEFT JOIN market_registry m ON m.canonical_id = p.canonical_id
            LEFT JOIN settlements s ON s.canonical_id = p.canonical_id
            WHERE p.quantity > 0 AND s.canonical_id IS NULL
            ORDER BY p.last_update DESC LIMIT 60
            """,
        )
        return [{
            "strategy": r["strategy_name"],
            "market": r["title"] or r["canonical_id"],
            "side": r["side"],
            "quantity": r["quantity"],
            "avg_price": _f(r["average_price"]),
            "cost": round((_f(r["average_price"]) or 0) * (r["quantity"] or 0), 2),
            "closes": r["close_time"],
            "updated": r["last_update"],
        } for r in rows]

    def _ai_decisions(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = _rows(
            conn,
            """
            SELECT f.as_of, f.canonical_id, f.p_yes, f.confidence, f.market_probability,
                   f.abstain, f.rationale, f.features_json, e.parameters_json, m.title
            FROM forecasts f
            JOIN experiments e ON e.experiment_id = f.experiment_id
            LEFT JOIN market_registry m ON m.canonical_id = f.canonical_id
            WHERE f.experiment_id IN (
                SELECT experiment_id FROM experiments WHERE strategy_name = 'news_probability'
            )
            ORDER BY f.id DESC LIMIT 40
            """,
        )
        out = []
        for r in rows:
            try:
                features = json.loads(r["features_json"] or "{}")
            except ValueError:
                features = {}
            out.append({
                "time": r["as_of"],
                "market": r["title"] or r["canonical_id"],
                "stack": features.get("ai_stack", ""),
                "model": features.get("llm_model_id", ""),
                "p_yes": _f(r["p_yes"]),
                "confidence": _f(r["confidence"]),
                "market_p": _f(r["market_probability"]),
                "abstain": bool(r["abstain"]),
                "why": (r["rationale"] or "")[:300],
                "failures": features.get("validation_failures") or [],
            })
        return out

    def _traders(self, conn: sqlite3.Connection) -> dict[str, Any]:
        latest = conn.execute("SELECT MAX(snapshot_time) FROM trader_leaderboard_snapshots").fetchone()[0]
        top: list[dict[str, Any]] = []
        if latest:
            top = _rows(
                conn,
                """
                SELECT wallet, MAX(username) AS username, MAX(CAST(pnl AS REAL)) AS pnl,
                       MAX(CAST(volume AS REAL)) AS volume, GROUP_CONCAT(DISTINCT category) AS categories
                FROM trader_leaderboard_snapshots
                WHERE snapshot_time >= datetime(?, '-2 hours')
                GROUP BY wallet ORDER BY pnl DESC LIMIT 25
                """,
                (latest,),
            )
        scored: list[dict[str, Any]] = []
        has_scores = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trader_scores'"
        ).fetchone()
        if has_scores:
            scored = _rows(
                conn,
                """
                SELECT wallet, username, score, status, resolved_positions, realized_pnl, roi, win_rate,
                       profit_factor, max_drawdown, sharpe_like, favorite_share, largest_win_share,
                       recent_roi, top_category, reasons, computed_at
                FROM trader_scores WHERE resolved_positions > 0 ORDER BY score DESC LIMIT 30
                """,
            )
        counts = dict(conn.execute("SELECT status, COUNT(*) FROM trader_scores GROUP BY status").fetchall()) if has_scores else {}
        since = datetime.now(UTC).timestamp() - 86400
        since_iso = datetime.fromtimestamp(since, UTC).isoformat()
        consensus = _rows(
            conn,
            """
            SELECT poly_condition_id, MAX(title) AS title, outcome, MAX(canonical_id) AS kalshi_match,
                   COUNT(DISTINCT wallet) AS wallets, COUNT(*) AS trades,
                   ROUND(SUM(CAST(usd_size AS REAL)), 0) AS usd, MAX(first_seen_time) AS last_seen
            FROM trader_actions
            WHERE first_seen_time >= ? AND action = 'buy'
            GROUP BY poly_condition_id, outcome
            HAVING COUNT(DISTINCT wallet) >= 2
            ORDER BY wallets DESC, usd DESC LIMIT 20
            """,
            (since_iso,),
        )
        return {"snapshot_time": latest, "top": top, "scored": scored, "status_counts": counts, "consensus": consensus}

    def _divergences(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        return _rows(
            conn,
            """
            SELECT mm.canonical_id_a AS kalshi, ka.title AS kalshi_title,
                   mm.canonical_id_b AS polymarket, pb.title AS poly_title,
                   CAST(mm.match_confidence AS REAL) AS confidence, mm.same_outcome_boolean AS same_outcome,
                   mm.human_review_required AS needs_review, mm.created_at
            FROM market_matches mm
            LEFT JOIN market_registry ka ON ka.canonical_id = mm.canonical_id_a
            LEFT JOIN market_registry pb ON pb.canonical_id = mm.canonical_id_b
            ORDER BY mm.created_at DESC LIMIT 25
            """,
        )

    def _rejections(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        return _rows(
            conn,
            """
            SELECT reject_reason AS reason, SUBSTR(reject_detail, 1, 90) AS detail, COUNT(*) AS n
            FROM (SELECT reject_reason, reject_detail FROM orders ORDER BY rowid DESC LIMIT 20000)
            WHERE reject_reason IS NOT NULL
            GROUP BY reject_reason, SUBSTR(reject_detail, 1, 40)
            ORDER BY n DESC LIMIT 15
            """,
        )

    def _alerts(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        return _rows(
            conn,
            "SELECT timestamp, severity, component, message FROM alerts ORDER BY id DESC LIMIT 15",
        )

    def _build(self) -> dict[str, Any]:
        status = self._status()
        conn = self._connect()
        try:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            sleeves = self._sleeves(conn)
            active = [s for s in sleeves if s["status"] not in _INACTIVE]
            # Retired cohorts (superseded or invalidated) must not count toward a family.
            families = self._families(active)
            traded = [s for s in active if s["trades"] > 0]
            portfolio = {
                "sleeves": len(active),
                "traded": len(traded),
                "equity": round(sum(s["equity"] for s in active), 2),
                "pnl": round(sum(s["pnl"] for s in active), 2),
                "today_pnl": round(sum(s["today_pnl"] for s in active), 2),
                "fees": round(sum(s["fees"] for s in active), 2),
                "profitable": sum(1 for s in traded if s["pnl"] > 0),
            }
            leaders = sorted(traded, key=lambda s: s["pnl"], reverse=True)
            return {
                "status": status,
                "portfolio": portfolio,
                "families": families,
                "top_sleeves": leaders[:25],
                "bottom_sleeves": leaders[-10:][::-1] if len(leaders) > 25 else [],
                "recent_trades": self._recent_trades(conn) if "decisions" in tables else [],
                "open_positions": self._open_positions(conn),
                "ai_decisions": self._ai_decisions(conn),
                "traders": self._traders(conn),
                "divergences": self._divergences(conn),
                "rejections": self._rejections(conn),
                "alerts": self._alerts(conn),
            }
        finally:
            conn.close()


def _handler(data: DashboardData) -> type[BaseHTTPRequestHandler]:
    index_html = (STATIC_DIR / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                pass  # the browser closed the tab or gave up mid-response; nothing to do

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, index_html, "text/html; charset=utf-8")
            elif path == "/api/summary":
                body = json.dumps(data.summary(), default=str).encode("utf-8")
                self._send(200, body, "application/json")
            elif path == "/healthz":
                self._send(200, b"ok", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return  # keep the console quiet; this is polled every 30s

    return Handler


def serve(settings: Any, *, port: int = 8765, open_browser: bool = False) -> None:
    data = DashboardData(Path(settings.db_path), Path(settings.data_dir))
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(data))
    url = f"http://127.0.0.1:{port}/"
    print(f"MarketLab dashboard: {url}  (read-only; Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["DashboardData", "serve"]
