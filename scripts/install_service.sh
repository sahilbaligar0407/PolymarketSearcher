#!/usr/bin/env bash
# Install MarketLab as a long-running background service.
#
# Prefers a systemd --user unit where systemd exists. On a machine without systemd
# (including this project's Windows host) it prints the documented equivalent instead of
# pretending to succeed.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BANKROLL="${1:-50}"

if ! command -v systemctl >/dev/null 2>&1 || ! systemctl --user show-environment >/dev/null 2>&1; then
  cat <<EOF
systemd --user is not available on this machine.

Windows
-------
  powershell -ExecutionPolicy Bypass -File scripts\\install_service.ps1

  Registers a Scheduled Task that runs at logon and restarts on failure.
    Start-ScheduledTask -TaskName MarketLab
    Stop-ScheduledTask  -TaskName MarketLab
    Get-ScheduledTask   -TaskName MarketLab | Get-ScheduledTaskInfo

Foreground / tmux (any platform)
--------------------------------
  tmux new -s marketlab
  cd "$ROOT" && uv run marketlab run --bankroll-per-strategy $BANKROLL
  # detach with ctrl-b d, reattach with: tmux attach -t marketlab

  Or with nohup:
  nohup uv run marketlab run --bankroll-per-strategy $BANKROLL >> data/logs/marketlab.out 2>&1 &

Logs land in data/logs/marketlab.jsonl regardless of how it is launched.
EOF
  exit 0
fi

UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR" "$ROOT/data/logs"

cat > "$UNIT_DIR/marketlab.service" <<EOF
[Unit]
Description=MarketLab prediction-market research engine
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$ROOT
ExecStart=$UV run marketlab run --bankroll-per-strategy $BANKROLL
Restart=always
RestartSec=10
# The engine is meant to run continuously; do not let systemd give up on restarts.
StartLimitIntervalSec=0
StandardOutput=journal
StandardError=journal
# Defence in depth: the research process never needs to write outside the repo.
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable marketlab.service

cat <<EOF
Installed systemd --user unit at $UNIT_DIR/marketlab.service

  systemctl --user start marketlab
  systemctl --user stop marketlab
  systemctl --user restart marketlab
  systemctl --user status marketlab
  journalctl --user -u marketlab -f

Consider 'loginctl enable-linger $USER' so it survives logout.
Live trading stays hard-disabled unless every gate in docs/live_safety.md is satisfied.
EOF
