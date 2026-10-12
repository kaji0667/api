#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${TMPDIR:-/tmp}}"

exec "$PYTHON_BIN" -m tasks.detection.test "$@"
