"""Variant generation: expand ``configs/strategies.yaml`` + ``configs/universes.yaml``
into the concrete list of sleeves the tournament will run.

Two entry points exist deliberately:

* :func:`generate_variants` - the real pipeline.  It validates every strategy's class
  path (a strategy module that doesn't exist yet, or exists but is broken, must not stop
  the rest of the tournament from being generated - see ``load_strategy_class``) and
  enforces every guardrail in ``meta``.
* :func:`count_variants` - a pure, import-free dry run over the YAML alone, for a CLI
  that wants to answer "how big would this grid be" without touching
  ``marketlab.strategies`` at all (that package may be under active development, or not
  installed, in whatever environment the CLI runs in).

Both are fully deterministic: dict iteration order comes straight from the YAML file
(regular ``dict`` preserves insertion order; only *set* iteration would be a problem, and
none is used here), and any place a stable subset must be chosen (``variant_limit``)
sorts on the canonical JSON of the candidate first so reruns always pick the same subset.
"""

from __future__ import annotations

import importlib
import itertools
from dataclasses import dataclass
from typing import Any

from marketlab.experiments.identity import canonical_json
from marketlab.logging import get_logger

log = get_logger(__name__)


class SweepError(Exception):
    """Base class for sweep-generation failures that name exactly what went wrong."""


class SweepLimitExceeded(SweepError):
    """Raised when a generated grid would exceed ``meta.max_total_variants``."""


@dataclass(frozen=True)
class VariantSpec:
    """One concrete tournament entry: one strategy, one universe, one parameter set."""

    strategy_name: str
    strategy_class_path: str
    universe: str
    params: dict[str, Any]
    evidence_class: str
    trades: bool
    requires_ai: bool


# ---------------------------------------------------------------------------
# Class-path resolution
# ---------------------------------------------------------------------------


def load_strategy_class(path: str) -> type:
    """Resolve ``"marketlab.strategies.momentum:MomentumStrategy"`` via importlib.

    Raises :class:`SweepError` naming the path on any failure (missing module, missing
    attribute, or the module itself raising on import) - callers are expected to catch
    this and skip just that one strategy, never propagate it into a crash.
    """
    if ":" not in path:
        raise SweepError(f"strategy class path must look like 'module.sub:ClassName', got {path!r}")
    module_path, _, class_name = path.partition(":")
    if not module_path or not class_name:
        raise SweepError(f"strategy class path must look like 'module.sub:ClassName', got {path!r}")
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # ModuleNotFoundError, ImportError, SyntaxError, etc.
        raise SweepError(
            f"cannot import module {module_path!r} for class path {path!r}: {exc}"
        ) from exc
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise SweepError(
            f"module {module_path!r} has no attribute {class_name!r} (class path {path!r})"
        ) from exc
    return cls


# ---------------------------------------------------------------------------
# Parameter-grid expansion
# ---------------------------------------------------------------------------


def _expand_params(
    variants_block: dict[str, Any] | None, variant_limit: int | None
) -> list[dict[str, Any]]:
    """Cartesian product of a strategy's ``variants:`` block, deterministically truncated.

    Value-list order (as authored in the YAML) is preserved for the un-truncated product
    so that e.g. ``threshold: [0.01, 0.02, 0.04]`` keeps its meaningful ordering.  When a
    ``variant_limit`` forces a subset, the subset is chosen by sorting every candidate's
    canonical JSON first - arbitrary, but exactly reproducible across runs and machines,
    which is the actual requirement ("truncates deterministically").
    """
    block = variants_block or {}
    if not block:
        return [{}]
    names = list(block.keys())
    value_lists: list[list[Any]] = []
    for name in names:
        values = block[name]
        value_lists.append(values if isinstance(values, list) else [values])
    combos = [dict(zip(names, combo, strict=True)) for combo in itertools.product(*value_lists)]
    if variant_limit is not None and len(combos) > variant_limit:
        combos = sorted(combos, key=canonical_json)[:variant_limit]
    return combos


def _universe_available(udef: dict[str, Any] | None) -> bool:
    if udef is None:
        return False
    return bool(udef.get("available", True))


# ---------------------------------------------------------------------------
# Real pipeline
# ---------------------------------------------------------------------------


def generate_variants(
    strategies_config: dict[str, Any],
    universes_config: dict[str, Any],
    *,
    ai_enabled: bool = True,
) -> list[VariantSpec]:
    """Expand the full tournament grid, skipping and logging anything that can't run.

    Guardrails applied, in order: ``enabled: false`` strategies are skipped; a strategy
    whose class path cannot be imported is skipped (logged) but the rest of the sweep
    still proceeds; ``requires_ai: true`` variants are skipped when ``ai_enabled`` is
    False; universes not declared, or declared with ``available: false``, are skipped;
    ``variant_limit`` truncates deterministically; ``meta.max_total_variants`` is a hard
    ceiling the whole sweep refuses to exceed.
    """
    meta = strategies_config.get("meta", {}) or {}
    max_total = meta.get("max_total_variants")
    universes = universes_config.get("universes", {}) or {}

    variants: list[VariantSpec] = []
    for strategy_name, sdef in (strategies_config.get("strategies", {}) or {}).items():
        sdef = sdef or {}
        if not sdef.get("enabled", True):
            log.info("sweep.strategy_disabled", strategy=strategy_name)
            continue

        class_path = sdef.get("class", "")
        try:
            load_strategy_class(class_path)
        except SweepError as exc:
            log.error(
                "sweep.bad_class_path", strategy=strategy_name, class_path=class_path, error=str(exc)
            )
            continue

        requires_ai = bool(sdef.get("requires_ai", False))
        if requires_ai and not ai_enabled:
            log.warning("sweep.ai_disabled_skip", strategy=strategy_name)
            continue

        param_dicts = _expand_params(sdef.get("variants"), sdef.get("variant_limit"))
        evidence_class = str(sdef.get("evidence", "E"))
        trades = bool(sdef.get("trades", True))

        for universe_name in sdef.get("universes", []) or []:
            udef = universes.get(universe_name)
            if udef is None:
                log.warning(
                    "sweep.unknown_universe", strategy=strategy_name, universe=universe_name
                )
                continue
            if not _universe_available(udef):
                log.info(
                    "sweep.universe_unavailable",
                    strategy=strategy_name,
                    universe=universe_name,
                    reason=udef.get("unavailable_reason", ""),
                )
                continue
            for params in param_dicts:
                variants.append(
                    VariantSpec(
                        strategy_name=strategy_name,
                        strategy_class_path=class_path,
                        universe=universe_name,
                        params=dict(params),
                        evidence_class=evidence_class,
                        trades=trades,
                        requires_ai=requires_ai,
                    )
                )

    if max_total is not None and len(variants) > max_total:
        dropped = variants[max_total:]
        overflow_names = sorted({f"{v.strategy_name}/{v.universe}" for v in dropped})
        raise SweepLimitExceeded(
            f"generated {len(variants)} variants but meta.max_total_variants={max_total}; "
            f"would need to drop {len(dropped)} variants, including: "
            f"{', '.join(overflow_names[:10])}"
            + (" ..." if len(overflow_names) > 10 else "")
        )
    return variants


# ---------------------------------------------------------------------------
# Dry-run counting (no imports)
# ---------------------------------------------------------------------------


def count_variants(
    strategies_config: dict[str, Any], universes_config: dict[str, Any]
) -> dict[str, int]:
    """Variant count per strategy, computed from YAML alone - never imports anything.

    Does not know whether a strategy's class actually exists (that requires an import,
    which this function deliberately avoids), and does not apply ``requires_ai`` gating
    (AI availability is a runtime fact, not something the YAML alone can answer). Use
    this for "how big would the grid be if every strategy were implemented and AI were
    available" - a useful upper bound for planning, distinct from
    :func:`generate_variants`'s "how big is the grid right now."
    """
    universes = universes_config.get("universes", {}) or {}
    counts: dict[str, int] = {}
    for strategy_name, sdef in (strategies_config.get("strategies", {}) or {}).items():
        sdef = sdef or {}
        if not sdef.get("enabled", True):
            continue
        param_dicts = _expand_params(sdef.get("variants"), sdef.get("variant_limit"))
        n_universes = sum(
            1
            for universe_name in (sdef.get("universes", []) or [])
            if _universe_available(universes.get(universe_name))
        )
        counts[strategy_name] = len(param_dicts) * n_universes
    return counts
