#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1
cd "$SCRIPT_DIR/.."
.venv/bin/python gradio_failure_viewer.py --port 7865
