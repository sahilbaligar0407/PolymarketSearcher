"""Immutable experiment identity.

An :class:`ExperimentIdentity` is the full provenance of one strategy/parameter/data
vintage run.  Two identities are the same experiment if and only if every field below is
equal; changing *any* field - a parameter, a strategy version bump, a data pipeline
change, a different prompt - must produce a different, brand new ``experiment_id``.  This
is what lets champion/challenger comparisons and crash recovery both trust the id: the
same id always means "the same thing happened," and a different thing always gets a new
id rather than silently mutating history.

``data_version`` / ``execution_model_version`` / ``feature_version`` are module constants
a developer bumps by hand whenever the corresponding subsystem's *semantics* change (not
on every commit - only when the meaning of the numbers it produces would change):

* ``DATA_VERSION`` - the market/event ingestion and normalization pipeline: adapter
  parsing rules, the Kalshi dollar-string vs cents handling, universe expansion, dedup
  logic.  Bump when a change would make an old backtest's inputs not reproducible from
  today's ingestion code.
* ``EXECUTION_MODEL_VERSION`` - the fill/latency simulation: :mod:`marketlab.execution`
  (latency model, fill models, fee calculation).  Bump when a change would make an old
  experiment's fills not reproducible from today's simulator.
* ``FEATURE_VERSION`` - feature computation consumed by strategies/AI (rolling stats,
  order-book features, retrieval).  Bump when a change would make an old experiment's
  feature snapshots not reproducible from today's feature code.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

#: Bump when marketlab's data ingestion/normalization semantics change. See module
#: docstring for exactly what this covers.
DATA_VERSION = "data.2026.09.04"

#: Bump when the fill/latency/fee simulation semantics change.
EXECUTION_MODEL_VERSION = "exec.v1"

#: Bump when feature computation semantics change.
FEATURE_VERSION = "feat.v1"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        # Rendered exactly as given - "0.02" and "0.020" are deliberately different
        # strings, because they came from different config text even if numerically equal.
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"object of type {type(obj)!r} is not canonically serializable")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, exact ``Decimal`` rendering.

    The single serialization every hash in this module is computed over.  Sorting keys
    means a dict built in a different order (e.g. after a round trip through a different
    YAML loader) hashes identically; nothing here depends on insertion order.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def parameter_hash(params: dict[str, Any]) -> str:
    """Sha256 of the canonical JSON of ``params``, first 12 hex characters.

    Stable under dict key reordering (canonical_json sorts keys) and exact for
    ``Decimal`` values (rendered via ``str``, not normalized).
    """
    digest = hashlib.sha256(canonical_json(params).encode("utf-8")).hexdigest()
    return digest[:12]


def git_commit(repo_root: Path | str | None = None) -> str:
    """The current commit hash, or ``"nogit"``.  Never raises.

    Covers: git not installed, not a repo, a repo with zero commits (``rev-parse HEAD``
    fails), permission errors, and any other subprocess failure.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root is not None else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return "nogit"
    if result.returncode != 0:
        return "nogit"
    commit = result.stdout.strip()
    return commit if commit else "nogit"


def _slug(value: str) -> str:
    """Uppercase, alnum-and-underscore-only slug for the readable id prefix."""
    out = []
    prev_underscore = False
    for ch in value.upper():
        if ch.isalnum():
            out.append(ch)
            prev_underscore = False
        elif not prev_underscore:
            out.append("_")
            prev_underscore = True
    slug = "".join(out).strip("_")
    return slug or "X"


@dataclass(frozen=True)
class ExperimentIdentity:
    """Everything that makes one experiment run a distinct, reproducible thing.

    Note this dataclass deliberately does NOT carry the raw ``params`` dict - only their
    ``parameter_hash``.  The full parameter dict lives on the storage-layer
    :class:`marketlab.storage.state.Experiment` record keyed by ``experiment_id``; keeping
    it out of the identity itself means the identity's hash input is always the same
    shape regardless of how many parameters a strategy happens to have.
    """

    strategy_name: str
    strategy_version: str
    git_commit: str
    parameter_hash: str
    market_universe: str
    venue: str
    data_version: str
    execution_model_version: str
    feature_version: str
    llm_model_id: str | None
    prompt_hash: str | None
    start_timestamp: datetime
    starting_bankroll: Decimal

    def _fields_for_hash(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "git_commit": self.git_commit,
            "parameter_hash": self.parameter_hash,
            "market_universe": self.market_universe,
            "venue": self.venue,
            "data_version": self.data_version,
            "execution_model_version": self.execution_model_version,
            "feature_version": self.feature_version,
            "llm_model_id": self.llm_model_id,
            "prompt_hash": self.prompt_hash,
            "start_timestamp": self.start_timestamp.isoformat(),
            "starting_bankroll": str(self.starting_bankroll),
        }

    @property
    def identity_hash(self) -> str:
        """Sha256 (first 12 hex chars) over every field. Deterministic across processes."""
        digest = hashlib.sha256(canonical_json(self._fields_for_hash()).encode("utf-8"))
        return digest.hexdigest()[:12]

    def _fields_for_cohort(self) -> dict[str, Any]:
        """Every identity field EXCEPT ``start_timestamp``.

        Two runs of the same strategy, version, parameters, universe, data version and
        bankroll are the same *experiment continued*, not two experiments - even though
        they started at different wall-clock instants. That distinction is what makes
        crash recovery work: on restart the runner looks up a live sleeve by cohort key
        and resumes it, instead of minting a new $50 bankroll and orphaning the old one.
        """
        fields = self._fields_for_hash()
        fields.pop("start_timestamp")
        return fields

    @property
    def cohort_key(self) -> str:
        """Stable across restarts; changes only when something research-relevant changes.

        A *new* cohort is created deliberately - when a sleeve dies and a fresh $50 run
        begins, or when a version/parameter changes - never merely because the process
        was restarted.
        """
        digest = hashlib.sha256(canonical_json(self._fields_for_cohort()).encode("utf-8"))
        return digest.hexdigest()[:16]

    @property
    def experiment_id(self) -> str:
        """A stable, human-legible id: ``STRATEGY_UNIVERSE__<12-hex-hash-of-everything>``.

        The hash covers every field (including ``parameter_hash``, versions and the
        start timestamp), so the prefix is a readability aid, not the uniqueness
        guarantee - two identities can share a prefix only if they also share the hash,
        which means they are the same experiment.
        """
        prefix = f"{_slug(self.strategy_name)}_{_slug(self.market_universe)}"
        return f"{prefix}__{self.identity_hash}"
