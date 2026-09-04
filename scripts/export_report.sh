#!/usr/bin/env bash
# Export the current research state to data/reports/ for offline reading.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null 2>&1 || export PATH="/c/Python314/Scripts:$PATH"

STAMP="$(date +%Y-%m-%d_%H%M%S)"
OUT="data/reports/export_$STAMP"
mkdir -p "$OUT"

echo "Exporting to $OUT"
uv run marketlab report daily      > "$OUT/daily.txt"      2>&1 || true
uv run marketlab report strategies > "$OUT/strategies.txt" 2>&1 || true
uv run marketlab report categories > "$OUT/categories.txt" 2>&1 || true
uv run marketlab report traders    > "$OUT/traders.txt"    2>&1 || true
uv run marketlab report risk       > "$OUT/risk.txt"       2>&1 || true
uv run marketlab paper leaderboard > "$OUT/leaderboard.txt" 2>&1 || true
uv run marketlab experiments list  > "$OUT/experiments.txt" 2>&1 || true

# The machine-readable daily reports are the durable artefact; copy the recent ones too.
find data/reports -maxdepth 1 -name 'daily_*.json' -mtime -7 -exec cp {} "$OUT/" \; 2>/dev/null || true

echo "Done. Files:"
ls -la "$OUT"
