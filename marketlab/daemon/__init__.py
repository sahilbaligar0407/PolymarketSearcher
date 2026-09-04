"""The MarketLab daemon: ingestion pipeline, supervisor, health, and recovery.

Owned by Team DAEMON. This package turns the frozen core/adapters/execution modules
into a process that runs forever: ``marketlab paper start`` boots the :class:`Supervisor`,
which wires an :class:`~marketlab.daemon.ingest.IngestService` into shared
:class:`~marketlab.daemon.registry.MarketRegistry` / :class:`~marketlab.daemon.registry.BookRegistry`
instances, feeds a :class:`~marketlab.execution.paper_broker.PaperBroker`, and drives the
strategy tournament via ``marketlab.experiments`` (imported lazily; the engine degrades
cleanly and keeps running if that module is not yet available).
"""

from __future__ import annotations
