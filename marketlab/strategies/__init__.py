"""Strategy plugins.

Each module here exposes one or more :class:`marketlab.core.strategy.Strategy`
subclasses, referenced from ``configs/strategies.yaml`` by dotted path
(``module:ClassName``). Nothing is re-exported at package level - the runner imports each
strategy module directly by the path the config gives it.
"""

from __future__ import annotations
