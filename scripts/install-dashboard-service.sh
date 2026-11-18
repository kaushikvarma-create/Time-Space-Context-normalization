#!/usr/bin/env bash
# Run the dashboard as a user service so it survives restarts. Re-running replaces the service.
#   remove:  systemctl --user disable --now sbi-dashboard && rm ~/.config/systemd/user/sbi-dashboard.service
#   status:  systemctl --user status sbi-dashboard        log:  journalctl --user -u sbi-dashboard -n 50
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"
sed "s|@ROOT@|$ROOT|g" "$ROOT/scripts/systemd/sbi-dashboard.service" > "$UNIT_DIR/sbi-dashboard.service"
systemctl --user daemon-reload
systemctl --user enable --now sbi-dashboard.service
# start at boot without anyone logging in to Linux (WSL is kept running by the Windows task)
loginctl enable-linger "$USER" 2>/dev/null || echo "note: could not enable lingering; the dashboard starts at the first login instead"
systemctl --user --no-pager status sbi-dashboard.service | head -5
