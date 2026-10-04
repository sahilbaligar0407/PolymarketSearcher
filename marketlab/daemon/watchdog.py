"""Hang watchdog: exits the process when the daemon stops making progress.

The supervisor's own error handling covers anything that *raises*. It cannot cover an
``await`` that never returns, or an event loop starved by CPU or memory pressure: on
2026-10-04 a daemon booted while the Wi-Fi was down sat idle in boot for 25+ minutes,
and earlier one stalled under memory pressure, both with the process alive and no
heartbeat. START_TRADING.bat only restarts a process that *exits*, so this watchdog makes
a hang into an exit.

It is a plain OS thread rather than an asyncio task, so a stuck or starved event loop
cannot stop it. Progress is counted in the watchdog's own check intervals, not wall time,
so a laptop that slept for eight hours wakes to one missed check rather than a kill.
"""

from __future__ import annotations

import os
import threading
import time

from marketlab.logging import get_logger

log = get_logger(__name__)

#: Process exit code for a watchdog kill (distinct from crashes in the restart log).
HANG_EXIT_CODE = 75


class HangWatchdog(threading.Thread):
    def __init__(
        self,
        limit_seconds: float,
        *,
        check_seconds: float = 30.0,
        exit_fn=os._exit,  # noqa: ANN001 - injectable for tests
    ) -> None:
        super().__init__(name="marketlab-hang-watchdog", daemon=True)
        self._check = check_seconds
        self._lock = threading.Lock()
        self._missed = 0
        self._allowed = self._checks_for(limit_seconds)
        self._stage = "boot"
        self._exit = exit_fn
        self._stopped = threading.Event()

    def _checks_for(self, seconds: float) -> int:
        return max(1, int(seconds // self._check))

    def beat(self, stage: str | None = None) -> None:
        """Record progress. Call from the heartbeat loop and at boot milestones."""
        with self._lock:
            self._missed = 0
            if stage is not None:
                self._stage = stage

    def set_limit(self, seconds: float, stage: str) -> None:
        with self._lock:
            self._allowed = self._checks_for(seconds)
            self._missed = 0
            self._stage = stage

    def stop(self) -> None:
        self._stopped.set()

    def run(self) -> None:
        while not self._stopped.wait(self._check):
            with self._lock:
                self._missed += 1
                missed, allowed, stage = self._missed, self._allowed, self._stage
            if missed >= allowed:
                log.critical(
                    "watchdog_hang_detected",
                    stage=stage,
                    seconds_without_progress=missed * self._check,
                    action="exit for the START_TRADING watchdog to restart",
                )
                time.sleep(0.5)  # let the log line reach disk
                self._exit(HANG_EXIT_CODE)
                return
