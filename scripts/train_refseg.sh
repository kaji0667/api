#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${TMPDIR:-/tmp}}"
if [[ -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

exec "$PYTHON_BIN" -m tasks.refseg.engine \
  --data "${DATA:-pre_datasets/RRSIS-D_refseg/data.yaml}" \
  --model "${MODEL:-ultralytics/cfg/models/v12/yolov12m-semseg.yaml}" \
  --weights "${WEIGHTS:-pretrain_model/yolov12m.pt}" \
  --device "${DEVICE:-cuda:0}" \
  --imgsz "${IMGSZ:-512}" \
  --batch "${BATCH:-4}" \
  --epochs "${EPOCHS:-60}" \
  --max-batches 0 \
  --max-val-batches 0 \
  --workers "${WORKERS:-2}" \
  --print-interval "${PRINT_INTERVAL:-200}" \
  --text-queries \
  --text-encoder openclip \
  --text-model-name ViT-L-14 \
  --text-pretrained openai \
  --text-precision fp32 \
  --scheduler cosine \
  --min-lr 1e-6 \
  --freeze-backbone \
  --patience "${PATIENCE:-8}" \
  --min-delta 0.001 \
  --augment \
  --augment-hflip 0.5 \
  --augment-vflip 0.5 \
  --augment-color-jitter 0.15 \
  --augment-direction-policy "${AUGMENT_DIRECTION_POLICY:-axis-aware}" \
  --empty-mask-policy "${EMPTY_MASK_POLICY:-drop}" \
  --seed "${SEED:-42}" \
  --pos-weight-max 10 \
  --small-target-boost 1.5 \
  --small-target-area 0.0025 \
  --loss-small-target-weight 1.5 \
  --loss-small-target-area 0.0025 \
  --val-thresholds 0.6,0.7,0.8,0.85,0.9,0.95 \
  --val-select-metric miou \
  --save-dir "${SAVE_DIR:-runs/semseg/srp_yolov12m_axis}" \
  "$@"
