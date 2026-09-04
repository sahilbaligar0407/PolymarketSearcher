#!/usr/bin/env bash
# Bootstrap a MarketLab environment from a clean checkout.
#
# Safe to re-run. Does not overwrite an existing .env and never touches data/.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "MarketLab bootstrap"
echo "==================="
echo "repo: $ROOT"

# --- environment inspection (the PRD asks for this before writing anything) -------
echo
echo "-- environment --"
uname -a 2>/dev/null || echo "uname unavailable (Windows shell)"
python --version 2>&1 || true
git --version 2>&1 || true

if ! command -v uv >/dev/null 2>&1; then
  # uv frequently lives next to the Python that installed it and is not on PATH.
  for candidate in /c/Python314/Scripts /c/Python313/Scripts "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    if [ -x "$candidate/uv" ] || [ -x "$candidate/uv.exe" ]; then
      export PATH="$candidate:$PATH"
      break
    fi
  done
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it with:  python -m pip install uv"
  exit 1
fi
echo "uv: $(uv --version)"

# --- local AI: inspect, never download -------------------------------------------
echo
echo "-- local AI --"
if command -v ollama >/dev/null 2>&1; then
  echo "ollama: found"
  ollama list 2>/dev/null || true
else
  echo "ollama: not installed (AI layer will run disabled; the engine still works)"
fi
curl -s -m 3 http://localhost:11434/api/tags >/dev/null 2>&1 \
  && echo "ollama api: reachable at localhost:11434" \
  || echo "ollama api: not reachable (fine - AI degrades to disabled)"

# --- python environment ----------------------------------------------------------
echo
echo "-- dependencies --"
uv python install 3.12 >/dev/null 2>&1 || true
uv sync --extra dev

# --- config ----------------------------------------------------------------------
echo
echo "-- config --"
if [ -f .env ]; then
  echo ".env: exists (left untouched)"
else
  cp .env.example .env
  echo ".env: created from .env.example"
  echo "      No credentials are required for PAPER mode. Fill in what you have."
fi

mkdir -p data/raw data/normalized data/parquet data/reports data/logs

# --- verify ----------------------------------------------------------------------
echo
echo "-- verification --"
uv run python -c "import marketlab; print('marketlab', marketlab.__version__)"
uv run pytest -m "not integration" -q 2>&1 | tail -3

echo
echo "Bootstrap complete. Next:"
echo "  uv run marketlab doctor"
echo "  uv run marketlab ingest --once"
echo "  uv run marketlab paper start --bankroll-per-strategy 50"
