"""One-off cleanup after the 2026-10-04 redeploy. Safe to run with the daemon up.

1. Retires PAPER market_maker sleeves from strategy version 1.0.0 that hold no open
   position: the base_spread 0.02 cohort (which can never clear the fee gate) and the
   0.05 cohort that quoted off the tick grid for two hours. Neither runs any more (1.1.0
   replaced them), but the dashboard still counts them as active. Sleeves still holding
   a position are left alone so those positions settle into their P&L.
2. Deletes the matcher's validator 1.0.0 rows. None was ever approved, so none could
   trade; 63 were plainly wrong (e.g. Texas Senate paired with Michigan Senate). The
   1.1.0 matcher rebuilds candidates on its next pass.

Run:  uv run python scripts/cleanup_2026_10_04.py
"""

from __future__ import annotations

from marketlab.clock import LiveClock
from marketlab.experiments.registry import ExperimentRegistry
from marketlab.settings import load_settings
from marketlab.storage.state import ExperimentStatus, StateStore


def main() -> None:
    store = StateStore(load_settings().db_path)
    registry = ExperimentRegistry(store, LiveClock())
    retired = kept = 0
    for exp in store.list_experiments():
        if exp.strategy_name != "market_maker":
            continue
        if str(getattr(exp.status, "value", exp.status)) != ExperimentStatus.PAPER.value:
            continue
        if exp.strategy_version != "1.0.0":
            continue
        portfolio = store.load_portfolio(exp.experiment_id)
        if portfolio is not None and any(p.quantity > 0 for p in portfolio.positions.values()):
            kept += 1
            continue
        registry.transition(
            exp.experiment_id,
            ExperimentStatus.DISABLED,
            "market_maker 1.0.0 replaced by 1.1.0 (tick-grid quotes; base_spread 0.05)",
        )
        retired += 1
    print(f"retired market_maker 1.0.0 sleeves: {retired} (kept {kept} with open positions)")

    with store._lock:
        cur = store._conn.execute("DELETE FROM market_matches WHERE validator_version = '1.0.0'")
        store._conn.commit()
    print(f"deleted validator 1.0.0 match rows: {cur.rowcount}")
    store.close()


if __name__ == "__main__":
    main()
