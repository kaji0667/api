#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${TMPDIR:-/tmp}}"
if [[ -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

data_args=(
  --voc-root "${VOC_ROOT:-data/VRSBench}"
  --prepared-dir "${PREPARED_DIR:-pre_datasets/VRSBench}"
)
if [[ -n "${DATA:-}" ]]; then
  data_args=(--data "$DATA" --prepared-dir "${PREPARED_DIR:-pre_datasets/VRSBench}")
fi

val_args=()
if [[ "${RUN_VAL:-1}" == "1" ]]; then
  val_args+=(--run-val)
fi

exec "$PYTHON_BIN" -m tasks.counting.train \
  --dataset "${DATASET:-VRSBench}" \
  --model "${MODEL:-ultralytics/cfg/models/v12/yolov12m-counting.yaml}" \
  --weights "${WEIGHTS:-pretrain_model/yolov12m.pt}" \
  --device "${DEVICE:-cuda:0}" \
  --imgsz "${IMGSZ:-800}" \
  --batch "${BATCH:-4}" \
  --epochs "${EPOCHS:-50}" \
  --workers "${WORKERS:-2}" \
  --patience "${PATIENCE:-15}" \
  --seed "${SEED:-0}" \
  --project "${PROJECT:-runs/counting}" \
  --name "${RUN_NAME:-counting-VRSBench-ViT-L-14}" \
  --text-model-name "${TEXT_MODEL_NAME:-ViT-L-14}" \
  --text-pretrained "${TEXT_PRETRAINED:-openai}" \
  "${data_args[@]}" \
  "${val_args[@]}" \
  "$@"
