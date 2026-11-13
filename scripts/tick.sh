#!/usr/bin/env bash
# One scheduled tick for all spaces (config/spaces/*.yaml). Safe from cron:
#   * overlapping runs are skipped (flock), a hung run is killed after 10 minutes
#   * all times are IST, output is appended to data/logs/tick.log (rotated at 10 MB)
set -uo pipefail
cd "$(dirname "$0")/.."
export TZ=Asia/Kolkata PYTHONPATH=src
LOG=data/logs/tick.log
mkdir -p data/logs
[ -f "$LOG" ] && [ "$(stat -c %s "$LOG")" -gt 10485760 ] && mv "$LOG" "$LOG.1"

exec 9>data/.tick.lock
if ! flock -n 9; then
  echo "$(date -Is) WARNING tick skipped: previous tick still running" >> "$LOG"
  exit 0
fi

timeout 600 python3 -m sbi tick >> "$LOG" 2>&1
rc=$?
case $rc in
  0) ;;                                                    # all spaces ok
  1) echo "$(date -Is) WARNING tick finished with failed space(s), rc=1" >> "$LOG" ;;
  124) echo "$(date -Is) ERROR tick killed after 600s timeout" >> "$LOG" ;;
  *) echo "$(date -Is) ERROR tick failed, rc=$rc" >> "$LOG" ;;
esac
exit $rc
