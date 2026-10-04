"""The hang watchdog exits on missed checks and survives a long suspend."""

from __future__ import annotations

import threading

from marketlab.daemon.watchdog import HANG_EXIT_CODE, HangWatchdog


def _run(wd: HangWatchdog, exited: threading.Event) -> None:
    wd.start()
    exited.wait(2.0)
    wd.stop()


def test_exits_after_the_limit_without_beats() -> None:
    exited = threading.Event()
    codes: list[int] = []
    wd = HangWatchdog(0.2, check_seconds=0.05, exit_fn=lambda c: (codes.append(c), exited.set()))
    _run(wd, exited)
    assert codes == [HANG_EXIT_CODE]


def test_beats_keep_it_alive() -> None:
    exited = threading.Event()
    wd = HangWatchdog(0.2, check_seconds=0.05, exit_fn=lambda c: exited.set())
    wd.start()
    for _ in range(20):
        wd.beat()
        exited.wait(0.03)
    wd.stop()
    assert not exited.is_set()


def test_limit_is_counted_in_checks_not_wall_time() -> None:
    # A suspended laptop makes one long wait() return once: that is one missed check.
    wd = HangWatchdog(600.0, check_seconds=30.0, exit_fn=lambda c: None)
    assert wd._allowed == 20
