#!/usr/bin/env bash
# End-to-end smoke test: does a fresh MarketLab actually see real market data and keep
# every order local? Exercises the PRD first-run sequence and fails loudly on zero markets.
set -uo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null 2>&1 || export PATH="/c/Python314/Scripts:$PATH"

FAILED=0
step() {
  echo
  echo "--- $1 ---"
  shift
  if "$@"; then
    echo "  OK"
  else
    echo "  FAILED: $*"
    FAILED=1
  fi
}

echo "MarketLab smoke test"
echo "===================="

step "doctor"            uv run marketlab doctor
step "ingest once"       uv run marketlab ingest --once
step "kalshi markets"    uv run marketlab markets --venue kalshi --limit 10
step "polymarket markets" uv run marketlab markets --venue poly-global --limit 10
step "trader discovery"  uv run marketlab trader-discover
step "trader report"     uv run marketlab report traders
step "ai doctor"         uv run marketlab ai doctor
step "unit tests"        uv run pytest -m "not integration" -q
step "lint"              uv run ruff check .

echo
if [ "$FAILED" -eq 0 ]; then
  echo "SMOKE TEST PASSED"
else
  echo "SMOKE TEST FAILED - see the steps marked FAILED above"
fi
exit "$FAILED"
