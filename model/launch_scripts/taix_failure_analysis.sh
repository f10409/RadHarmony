#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_HOME:?set HF_HOME to the Hugging Face cache folder}"
export HF_HOME
export PYTHONUNBUFFERED=1
cd "$SCRIPT_DIR/.."
.venv/bin/python analyze_failures.py
