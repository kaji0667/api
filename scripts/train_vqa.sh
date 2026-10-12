#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m tasks.vqa.train --train-jsonl "${TRAIN_JSONL:-../vqa_work_20261011/data/train_final.jsonl}" --val-jsonl "${VAL_JSONL:-../vqa_work_20261011/data/val_final.jsonl}" --config "${CONFIG:-config/vqa/qformer.json}" --device "${DEVICE:-auto}" --save-dir "${SAVE_DIR:-runs/vqa/qformer}" "$@"
