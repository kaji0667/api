#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-runs/counting/counting-VRSBench-ViT-L-14/weights/best.pt}"
SAVE_DIR="${SAVE_DIR:-runs/counting/counting-VRSBench-ViT-L-14-eval}"

exec "$PYTHON_BIN" -m tasks.counting.test \
  --weights "$CHECKPOINT" \
  --voc-root "${VOC_ROOT:-data/VRSBench}" \
  --dataset "${DATASET:-VRSBench}" \
  --split-file "${SPLIT_FILE:-test.txt}" \
  --device "${DEVICE:-cuda:0}" \
  --imgsz "${IMGSZ:-800}" \
  --conf-thres "${CONF_THRES:-0.15}" \
  --iou-thres "${IOU_THRES:-0.50}" \
  --max-det "${MAX_DET:-300}" \
  --max-samples "${MAX_SAMPLES:-0}" \
  --max-save-vis "${MAX_SAVE_VIS:-30}" \
  --save-dir "$SAVE_DIR" \
  --text-model-name "${TEXT_MODEL_NAME:-ViT-L-14}" \
  --text-pretrained "${TEXT_PRETRAINED:-openai}" \
  "$@"
