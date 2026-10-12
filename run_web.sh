#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

required_files=(
  "pretrain_model/yolov12m.pt"
  "runs/classification/vrsbench_scene/weights/best.pt"
  "runs/detect/DIOR-RSVG-ViT-L-14/weights/best.pt"
  "runs/counting/counting-VRSBench-ViT-L-14/weights/best.pt"
  "runs/semseg/srp_yolov12m_axis/weights/best_raw.pt"
  "openclip/ViT-L-14.pt"
)

for file_path in "${required_files[@]}"; do
  if [[ ! -f "$file_path" ]]; then
    echo "缺少模型文件：$file_path" >&2
    exit 1
  fi
done

clip_cache="${XDG_CACHE_HOME:-${HOME}/.cache}/clip"
mkdir -p "$clip_cache"
if [[ ! -f "$clip_cache/ViT-L-14.pt" ]] || ! cmp -s "openclip/ViT-L-14.pt" "$clip_cache/ViT-L-14.pt"; then
  cp "openclip/ViT-L-14.pt" "$clip_cache/ViT-L-14.pt"
fi

exec "${PYTHON_BIN:-python}" web_app.py \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-7860}" \
  "$@"
