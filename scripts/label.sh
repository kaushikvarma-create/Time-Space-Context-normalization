#!/usr/bin/env bash
# Launch the blind labelling UI (needs the .venv with streamlit). Opens on http://localhost:8501
set -euo pipefail
cd "$(dirname "$0")/.."
export TZ=Asia/Kolkata PYTHONPATH=src
exec .venv/bin/streamlit run src/sbi/labeling/app.py "$@"
