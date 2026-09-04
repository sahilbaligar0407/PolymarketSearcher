"""The honest scorekeeper: what actually worked, after costs, out of sample.

Submodules:

* :mod:`marketlab.analytics.metrics` -- trading, execution, copy, cross-market, news and
  risk-adjusted metrics, computed from orders/fills/forecasts/portfolio snapshots.
* :mod:`marketlab.analytics.calibration` -- probability quality, scored separately from
  P&L (Murphy decomposition, ECE/MCE, Brier skill score vs the market, AUC).
* :mod:`marketlab.analytics.bootstrap` -- uncertainty quantification (seeded, deterministic
  bootstrap CIs, cluster-robust resampling, effective sample size).
* :mod:`marketlab.analytics.attribution` -- P&L attribution, latency sensitivity, overfit
  scoring, and strategy correlation clustering.
* :mod:`marketlab.analytics.reports` -- terminal (rich) and JSON reporting, with
  IN_SAMPLE / VALIDATION / FORWARD_PAPER / LIVE kept strictly separate.

Nothing in this package calls ``datetime.now()``; every timestamp is supplied by the
caller. Nothing here fabricates a metric for an empty sample -- nonexistent evidence is
represented as ``None``/an empty result, never as ``0.0``.
"""

from __future__ import annotations
