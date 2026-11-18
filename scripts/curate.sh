#!/usr/bin/env bash
# Vocabulary curation: link synonyms ('same') and hierarchy ('is_a') among object names the VLM discovered,
# apply them, and re-map history. Separate lock from tick.sh so it never delays a capture.
# Review afterwards with: PYTHONPATH=src python3 -m sbi links   (undo: python3 -m sbi link-set ID rejected)
# To run nightly at 03:00 IST, add to crontab (cron runs in UTC on this machine):
#   30 21 * * * /home/teja-clove/Time-Space-Context-normalization/scripts/curate.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export TZ=Asia/Kolkata PYTHONPATH=src
LOG=data/logs/curate.log
mkdir -p data/logs
exec 9>data/.curate.lock
flock -n 9 || { echo "$(date -Is) WARNING curation skipped: previous run still active" >> "$LOG"; exit 0; }
[ -f "$LOG" ] && [ "$(stat -c %s "$LOG")" -gt 10485760 ] && mv "$LOG" "$LOG.1"
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
echo "$(date -Is) curation start" >> "$LOG"
timeout 3600 "$PY" -m sbi curate >> "$LOG" 2>&1
rc=$?
echo "$(date -Is) curation end rc=$rc" >> "$LOG"
exit $rc
