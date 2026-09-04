"""Deterministic latency simulation for the PaperBroker.

A real strategy decides to trade at ``decision_time``, but the order does not reach the
exchange instantaneously.  We model three additive delays (all in milliseconds):

* ``signal_to_order_ms`` - time for the strategy/engine to turn a decision into an order
  object (feature computation already happened; this is serialization + queueing).
* ``network_latency_ms`` - round-trip-ish delay to the venue.
* ``processing_latency_ms`` - time the venue itself takes to accept and post the order.

Optional ``jitter_ms`` adds bounded random noise to the total, so replay can study how
sensitive a strategy's edge is to latency variance (see ``latency_sweep_ms`` in
``ExecutionConfig``).  Determinism matters more than realism here: two ``LatencyModel``
instances built with the same ``seed`` and driven with calls in the same order MUST
produce the same jitter sequence, because backtest replay depends on byte-identical
results across runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class LatencyModel:
    #: Milliseconds to turn a decision into an outbound order.
    signal_to_order_ms: int
    #: Milliseconds of network transit to the venue.
    network_latency_ms: int
    #: Milliseconds the venue takes to process/post the order.
    processing_latency_ms: int
    #: +/- bound (in ms) for uniform jitter added to the total latency. 0 disables jitter.
    jitter_ms: int = 0
    #: Seed for the jitter RNG. Same seed + same call sequence -> same jitter sequence.
    seed: int | None = None

    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # A private Random instance (never the global `random` module) so that unrelated
        # code drawing random numbers elsewhere can never perturb replay determinism, and
        # so that two LatencyModel instances with the same seed are fully independent.
        self._rng = random.Random(self.seed)

    @property
    def total_latency_ms(self) -> int:
        return self.signal_to_order_ms + self.network_latency_ms + self.processing_latency_ms

    def _draw_jitter_ms(self) -> float:
        if self.jitter_ms <= 0:
            return 0.0
        return self._rng.uniform(-float(self.jitter_ms), float(self.jitter_ms))

    def send_time(self, decision_time: datetime) -> datetime:
        """When the order leaves the engine, i.e. after signal-to-order latency only."""
        return decision_time + timedelta(milliseconds=self.signal_to_order_ms)

    def arrival_time(self, decision_time: datetime) -> datetime:
        """When the exchange actually sees the order (decision + full latency + jitter).

        Each call draws one jitter sample from this model's private RNG, so calling this
        repeatedly for a stream of orders (in a fixed order) is what "same seed -> same
        jitter sequence" means: replay must call it in the same sequence to reproduce the
        same arrival times.
        """
        jitter = self._draw_jitter_ms()
        total_ms = self.total_latency_ms + jitter
        return decision_time + timedelta(milliseconds=total_ms)
