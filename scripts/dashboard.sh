#!/usr/bin/env bash
# Launch the read-only v1.0 dashboard on http://localhost:8502 (this machine only: bound to 127.0.0.1).
set -euo pipefail
cd "$(dirname "$0")/.."
export TZ=Asia/Kolkata PYTHONPATH=src
exec .venv/bin/streamlit run src/sbi/dashboard/app.py --server.address 127.0.0.1 --server.port 8502 \
  --server.headless true --browser.gatherUsageStats false "$@"
