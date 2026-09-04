"""The experiment lifecycle: registration, legal status transitions, and cohorts.

``ExperimentRegistry`` is a thin, opinionated layer over :class:`~marketlab.storage.
state.StateStore` - it owns exactly one thing the store itself does not: which status
transitions are *legal*.  Nothing here ever deletes or edits a historical record; a dead
sleeve's row stays dead forever, and a fresh attempt at the same idea is a brand new
``experiment_id`` (:meth:`new_cohort`), never a resurrection of the old one.
"""

from __future__ import annotations

from typing import Any

from marketlab.clock import Clock
from marketlab.core.events import Alert
from marketlab.experiments.identity import ExperimentIdentity
from marketlab.logging import get_logger
from marketlab.storage.state import Experiment, ExperimentStatus, StateStore

log = get_logger(__name__)


class IllegalTransitionError(Exception):
    """Raised when a status transition is not on the legal-transition table."""


class LiveNotAllowedError(IllegalTransitionError):
    """Raised when a LIVE_* transition is attempted without explicit human sign-off."""


#: Statuses that require an explicit human ``allow_live=True`` to reach. Promotion code
#: must never set these on its own - see ``docs/CONTRACTS.md`` / ``promotion.py``.
_LIVE_STATUSES: frozenset[ExperimentStatus] = frozenset(
    {ExperimentStatus.LIVE_SMALL, ExperimentStatus.LIVE_PROVEN}
)

#: The legal-transition table. DEAD is reachable from every non-terminal status (a sleeve
#: can die at any point in its life) and is handled as a blanket rule in `transition`
#: rather than being repeated in every entry below.
_LEGAL_TRANSITIONS: dict[ExperimentStatus, frozenset[ExperimentStatus]] = {
    ExperimentStatus.IDEA: frozenset({ExperimentStatus.BACKTESTING, ExperimentStatus.DISABLED}),
    ExperimentStatus.BACKTESTING: frozenset({ExperimentStatus.PAPER, ExperimentStatus.DISABLED}),
    ExperimentStatus.PAPER: frozenset({ExperimentStatus.QUALIFIED, ExperimentStatus.DISABLED}),
    ExperimentStatus.QUALIFIED: frozenset({ExperimentStatus.CHAMPION, ExperimentStatus.DISABLED}),
    ExperimentStatus.CHAMPION: frozenset(
        {ExperimentStatus.DEGRADED, ExperimentStatus.DISABLED, ExperimentStatus.LIVE_SMALL}
    ),
    ExperimentStatus.DEGRADED: frozenset({ExperimentStatus.DISABLED}),
    ExperimentStatus.DISABLED: frozenset(),
    ExperimentStatus.LIVE_SMALL: frozenset(
        {ExperimentStatus.LIVE_PROVEN, ExperimentStatus.DEGRADED, ExperimentStatus.DISABLED}
    ),
    ExperimentStatus.LIVE_PROVEN: frozenset({ExperimentStatus.DEGRADED, ExperimentStatus.DISABLED}),
    ExperimentStatus.DEAD: frozenset(),
}


class ExperimentRegistry:
    """Lifecycle operations over the injected :class:`StateStore`.

    ``clock`` is required (never ``datetime.now()``) for ``created_at`` /
    ``start_timestamp`` on newly registered experiments and cohorts.
    """

    def __init__(self, store: StateStore, clock: Clock) -> None:
        self._store = store
        self._clock = clock

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def register(
        self,
        identity: ExperimentIdentity,
        params: dict[str, Any],
        universe: str,
        notes: str = "",
        cohort: str | None = None,
    ) -> Experiment:
        """Idempotent: registering the same identity twice returns the existing row."""
        existing = self._store.get_experiment(identity.experiment_id)
        if existing is not None:
            return existing
        record = Experiment(
            experiment_id=identity.experiment_id,
            strategy_name=identity.strategy_name,
            strategy_version=identity.strategy_version,
            git_commit=identity.git_commit,
            parameter_hash=identity.parameter_hash,
            parameters=dict(params),
            market_universe=universe,
            venue=identity.venue,
            data_version=identity.data_version,
            execution_model_version=identity.execution_model_version,
            feature_version=identity.feature_version,
            llm_model_id=identity.llm_model_id or "",
            prompt_hash=identity.prompt_hash or "",
            start_timestamp=identity.start_timestamp,
            starting_bankroll=identity.starting_bankroll,
            status=ExperimentStatus.IDEA,
            # Stable across restarts (identity minus start_timestamp), so a resumed run
            # finds its own sleeves. Falls back to the experiment id when a caller does
            # not supply one, which keeps every row's cohort non-empty.
            cohort=cohort or identity.cohort_key,
            created_at=self._clock.now(),
            notes=notes,
        )
        return self._store.create_experiment(record)

    def get(self, experiment_id: str) -> Experiment | None:
        return self._store.get_experiment(experiment_id)

    def list(
        self, status: ExperimentStatus | str | None = None, strategy: str | None = None
    ) -> list[Experiment]:
        status_str = str(status) if status is not None else None
        return self._store.list_experiments(status=status_str, strategy=strategy)

    # ------------------------------------------------------------------
    # transitions
    # ------------------------------------------------------------------

    def transition(
        self,
        experiment_id: str,
        new_status: ExperimentStatus,
        reason: str,
        *,
        allow_live: bool = False,
    ) -> Experiment:
        """Move an experiment to ``new_status``, or raise if that move is not legal.

        ``DEAD`` is always legal from any non-terminal status (a sleeve can die at any
        point). Any ``LIVE_*`` target additionally requires ``allow_live=True`` - the
        promotion engine never sets this itself; only an explicit human action may.
        """
        experiment = self._store.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(f"no experiment registered with id {experiment_id!r}")
        current = experiment.status

        if current in (ExperimentStatus.DEAD,):
            raise IllegalTransitionError(
                f"{experiment_id} is DEAD; that status is terminal and no further "
                f"transitions are possible"
            )

        if new_status == ExperimentStatus.DEAD:
            pass  # always legal from any non-terminal status
        elif new_status in _LIVE_STATUSES:
            if not allow_live:
                raise LiveNotAllowedError(
                    f"promotion of {experiment_id} to {new_status} requires an explicit "
                    f"human action (allow_live=True); no automated code path may do this"
                )
            if new_status not in _LEGAL_TRANSITIONS.get(current, frozenset()):
                raise IllegalTransitionError(f"illegal transition {current} -> {new_status}")
        else:
            if new_status not in _LEGAL_TRANSITIONS.get(current, frozenset()):
                raise IllegalTransitionError(f"illegal transition {current} -> {new_status}")

        self._store.update_experiment_status(experiment_id, str(new_status))
        self._store.save_alert(
            Alert(
                timestamp=self._clock.now(),
                severity="info",
                component="experiment_registry",
                message=f"{experiment_id}: {current} -> {new_status}",
                detail={"experiment_id": experiment_id, "from": str(current), "to": str(new_status), "reason": reason},
            )
        )
        log.info(
            "registry.transition",
            experiment_id=experiment_id,
            from_status=str(current),
            to_status=str(new_status),
            reason=reason,
        )
        updated = self._store.get_experiment(experiment_id)
        assert updated is not None
        return updated

    # ------------------------------------------------------------------
    # cohorts
    # ------------------------------------------------------------------

    def new_cohort(self, dead_experiment_id: str) -> ExperimentIdentity:
        """A fresh identity for a new attempt at a dead sleeve's idea.

        The dead record is never touched: this only *computes* a new
        :class:`ExperimentIdentity` (new ``start_timestamp``, freshly re-read
        ``git_commit()`` since the code may have changed) for the caller to
        :meth:`register` separately. Every other field (strategy name/version, universe,
        venue, parameter hash, data/execution/feature versions, AI fields, bankroll) is
        carried forward unchanged, since this is the *same idea* getting a new attempt,
        not a different one.
        """
        from marketlab.experiments.identity import (
            git_commit as _git_commit,  # local: avoid cycle at import time
        )

        dead = self._store.get_experiment(dead_experiment_id)
        if dead is None:
            raise KeyError(f"no experiment registered with id {dead_experiment_id!r}")
        if dead.status != ExperimentStatus.DEAD:
            raise ValueError(
                f"new_cohort requires a DEAD experiment; {dead_experiment_id} is {dead.status}"
            )
        return ExperimentIdentity(
            strategy_name=dead.strategy_name,
            strategy_version=dead.strategy_version,
            git_commit=_git_commit(),
            parameter_hash=dead.parameter_hash,
            market_universe=dead.market_universe,
            venue=dead.venue,
            data_version=dead.data_version,
            execution_model_version=dead.execution_model_version,
            feature_version=dead.feature_version,
            llm_model_id=dead.llm_model_id or None,
            prompt_hash=dead.prompt_hash or None,
            start_timestamp=self._clock.now(),
            starting_bankroll=dead.starting_bankroll,
        )

    # ------------------------------------------------------------------
    # variant-count bookkeeping (how much searching produced a winner)
    # ------------------------------------------------------------------

    def record_variant_count(self, strategy_name: str, n: int) -> None:
        self._store.set_checkpoint("experiment_registry", f"variant_count::{strategy_name}", str(n))

    def variant_count(self, strategy_name: str) -> int | None:
        value = self._store.get_checkpoint("experiment_registry", f"variant_count::{strategy_name}")
        return None if value is None else int(value)
