#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

exec "$PYTHON_BIN" -u -m tasks.classification.test \
  --checkpoint "${CHECKPOINT:-runs/classification/vrsbench_scene/weights/best.pt}" \
  --data-dir "${DATA_DIR:-data/VRSBench_scene}" \
  --weights "${WEIGHTS:-pretrain_model/yolov12m.pt}" \
  --split "${SPLIT:-val}" \
  --save-dir "${SAVE_DIR:-runs/classification/vrsbench_scene_eval}" \
  --device "${DEVICE:-cuda:0}" \
  --batch "${BATCH:-32}" \
  --imgsz "${IMGSZ:-640}" \
  --workers "${WORKERS:-2}" \
  --seed "${SEED:-42}" \
  "$@"
