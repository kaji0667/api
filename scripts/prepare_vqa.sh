#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${ANNOTATIONS_TRAIN:?Set original VRSBench train ZIP/directory}"
: "${ANNOTATIONS_VAL:?Set original VRSBench val ZIP/directory}"
: "${IMAGES_DIR:?Set existing VRSBench JPEGImages directory}"
OUT_DIR="${OUT_DIR:-data/vqa}"
python -m tasks.vqa.data --annotations-dir "$ANNOTATIONS_TRAIN" --images-dir "$IMAGES_DIR" --split train --out "$OUT_DIR/train_raw.jsonl"
python -m tasks.vqa.data --annotations-dir "$ANNOTATIONS_VAL" --images-dir "$IMAGES_DIR" --split val --out "$OUT_DIR/val_raw.jsonl"
python -m tasks.vqa.data --clean-splits "$OUT_DIR/train_raw.jsonl" "$OUT_DIR/val_raw.jsonl" --out-dir "$OUT_DIR"
