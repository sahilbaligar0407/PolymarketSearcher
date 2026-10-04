"""Approved cross-venue matches, as the strategies consume them.

``cross_venue`` and ``copy_trader`` were written to read ``params["matches"]``, but the
runner never injected it, so neither family could trade even if a match had been
approved. :class:`MatchBook` is the single live set (refreshed from storage by the
runner); each sleeve gets a :class:`MatchView` restricted to its own universe so that,
say, the five ``cross_venue/sports_mlb`` variants trade MLB pairs and nothing else.

The view behaves like both shapes the strategies expect: iterable (``cross_venue``
loops over match records) and ``get(key)`` by either leg's id (``copy_trader`` looks a
match up by Polymarket condition id or Kalshi canonical id).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any


class MatchBook:
    def __init__(self) -> None:
        self._by_kalshi: dict[str, Any] = {}
        self._by_poly: dict[str, Any] = {}

    def replace(self, matches: list[Any]) -> None:
        by_kalshi: dict[str, Any] = {}
        by_poly: dict[str, Any] = {}
        for m in matches:
            # One Kalshi contract per Polymarket market and vice versa: if storage ever
            # holds two approved twins for the same leg, keep the most confident.
            for key, index in ((m.canonical_id_a, by_kalshi), (m.canonical_id_b, by_poly)):
                prev = index.get(key)
                if prev is None or m.match_confidence > prev.match_confidence:
                    index[key] = m
        self._by_kalshi, self._by_poly = by_kalshi, by_poly

    def for_poly(self, poly_canonical_id: str) -> Any | None:
        return self._by_poly.get(poly_canonical_id)

    def for_kalshi(self, kalshi_canonical_id: str) -> Any | None:
        return self._by_kalshi.get(kalshi_canonical_id)

    def all(self) -> list[Any]:
        return list(self._by_kalshi.values())

    def __len__(self) -> int:
        return len(self._by_kalshi)


class MatchView:
    """A sleeve's universe-filtered, always-current window onto the shared book."""

    def __init__(self, book: MatchBook, universe: str, universes_for: Callable[[str], Any]) -> None:
        self._book = book
        self._universe = universe
        self._universes_for = universes_for

    def _in_universe(self, match: Any) -> bool:
        try:
            return self._universe in set(self._universes_for(match.canonical_id_a))
        except Exception:  # noqa: BLE001 - an unroutable market is simply out of scope
            return False

    def __iter__(self) -> Iterator[Any]:
        return (m for m in self._book.all() if self._in_universe(m))

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __bool__(self) -> bool:
        return any(True for _ in self)

    def get(self, key: str, default: Any = None) -> Any:
        if not key:
            return default
        match = self._book.for_kalshi(key) or self._book.for_poly(key) or self._book.for_poly(f"poly:{key}")
        return match if match is not None and self._in_universe(match) else default


__all__ = ["MatchBook", "MatchView"]
