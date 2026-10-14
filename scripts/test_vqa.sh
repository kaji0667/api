#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
VISUAL_BACKBONE="${VISUAL_BACKBONE:-pretrain_model/yolov12m.pt}"
TEXT_ENCODER="${TEXT_ENCODER:-openclip/ViT-L-14.pt}"
python -m tasks.vqa.evaluate --bundle-dir "${BUNDLE_DIR:-runs/vqa/qformer}" --jsonl "${VAL_JSONL:-../vqa_work_20261011/data/val_final.jsonl}" --visual-backbone "$VISUAL_BACKBONE" --text-encoder "$TEXT_ENCODER" --device "${DEVICE:-auto}" --out "${OUT_DIR:-runs/vqa/qformer/evaluation}" "$@"
