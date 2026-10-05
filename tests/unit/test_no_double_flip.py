"""Guard: a NO candidate's edge must be computed from P(YES), never from 1 - P(YES).

``edge_after_costs`` -> ``expected_edge`` converts P(YES) to P(NO) itself. Passing
``1 - p`` flipped it twice, and seven call sites bought NO whenever the model favoured
YES (FINDINGS 57). Any line that hands ``edge_after_costs`` a ``1 - model_probability``
reintroduces that bug.
"""

from __future__ import annotations

import re
from pathlib import Path

STRATEGIES = Path(__file__).resolve().parents[2] / "marketlab" / "strategies"
_FLIPPED = re.compile(r"(ONE|Decimal\(1\)|1)\s*-\s*(model_probability|model_p|ensemble_p|mid)\b")


def test_no_strategy_flips_p_yes_before_the_edge_helper() -> None:
    offenders = []
    for path in STRATEGIES.glob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            window = " ".join(lines[i : i + 3])
            if "edge_after_costs(" in window and _FLIPPED.search(window):
                offenders.append(f"{path.name}:{i + 1}: {line.strip()}")
    assert offenders == []
