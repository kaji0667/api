#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data/VRSBench_scene}"
VOC_ROOT="${VOC_ROOT:-data/VRSBench}"
LINK_MODE="${LINK_MODE:-hardlink}"

scene_dataset_ready() {
  [[ -d "$DATA_DIR" ]] || return 1
  local first_image
  first_image="$(find "$DATA_DIR" -mindepth 2 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print -quit)"
  [[ -n "$first_image" ]]
}

if ! scene_dataset_ready; then
  if [[ -d "$DATA_DIR" ]] && [[ -n "$(find "$DATA_DIR" -mindepth 1 -print -quit)" ]]; then
    echo "Scene-classification data directory is non-empty but contains no class-folder images: $DATA_DIR" >&2
    echo "Remove or repair that incomplete directory before rerunning this script." >&2
    exit 1
  fi
  echo "Scene-classification data not found at $DATA_DIR. Preparing it once from $VOC_ROOT ..."
  echo "This scans all VRSBench XML annotations and may take several minutes on /mnt drives."
  "$PYTHON_BIN" -u -m tasks.classification.prepare \
    --voc-root "$VOC_ROOT" \
    --output-dir "$DATA_DIR" \
    --link-mode "$LINK_MODE"
fi

exec "$PYTHON_BIN" -u -m tasks.classification.train \
  --data-dir "$DATA_DIR" \
  --model "${MODEL:-ultralytics/cfg/models/v12/yolov12m-classification.yaml}" \
  --weights "${WEIGHTS:-pretrain_model/yolov12m.pt}" \
  --save-dir "${SAVE_DIR:-runs/classification/vrsbench_scene}" \
  --device "${DEVICE:-cuda:0}" \
  --batch "${BATCH:-32}" \
  --imgsz "${IMGSZ:-640}" \
  --epochs "${EPOCHS:-20}" \
  --workers "${WORKERS:-2}" \
  --seed "${SEED:-42}" \
  "$@"
