#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-runs/semseg/srp_yolov12m_axis/weights/best_raw.pt}"
SAVE_DIR="${SAVE_DIR:-runs/semseg/srp_yolov12m_axis_eval}"

exec "$PYTHON_BIN" -m tasks.refseg.engine \
  --eval-only \
  --checkpoint "$CHECKPOINT" \
  --empty-mask-policy "${EMPTY_MASK_POLICY:-drop}" \
  --data "${DATA:-pre_datasets/RRSIS-D_refseg/data.yaml}" \
  --model "${MODEL:-ultralytics/cfg/models/v12/yolov12m-semseg.yaml}" \
  --device "${DEVICE:-cuda:0}" \
  --imgsz "${IMGSZ:-512}" \
  --batch "${BATCH:-4}" \
  --workers "${WORKERS:-2}" \
  --max-test-batches "${MAX_TEST_BATCHES:-0}" \
  --test-preview-batches "${TEST_PREVIEW_BATCHES:-5}" \
  --save-dir "$SAVE_DIR" \
  --text-queries \
  --text-encoder openclip \
  --text-model-name ViT-L-14 \
  --text-pretrained openai \
  --text-precision fp32 \
  --freeze-backbone \
  "$@"
