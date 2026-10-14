#!/usr/bin/env bash
# Export MODEL_API_KEY in the launch environment before starting the frozen API.
set -euo pipefail
hfsa_code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$hfsa_code_dir"
if [[ -z ${MODEL_API_KEY:-} ]] || [[ ${#MODEL_API_KEY} -lt 32 ]]; then
  echo 'Export the existing MODEL_API_KEY in this terminal before starting the API.' >&2
  exit 2
fi
export MODEL_API_KEY MODEL_MODE=real
export HFSA_ROUTER_URL="${HFSA_ROUTER_URL:-http://127.0.0.1:11434}"
export HFSA_ROUTER_MODEL="${HFSA_ROUTER_MODEL:-qwen3:1.7b}"
export HFSA_MODELS_DIR="${HFSA_MODELS_DIR:-$hfsa_code_dir}"
export HFSA_API_DEVICE="${HFSA_API_DEVICE:-auto}"
export HFSA_VQA_DEVICE="${HFSA_VQA_DEVICE:-cpu}"
export HFSA_GROUNDING_BACKEND="${HFSA_GROUNDING_BACKEND:-detection}"
if [[ $HFSA_GROUNDING_BACKEND == detection ]]; then
  # Migrate the previously exported experimental crop-reranking policy.
  if [[ -z ${HFSA_GROUNDING_POLICY:-} || $HFSA_GROUNDING_POLICY == clip_guarded ]]; then
    export HFSA_GROUNDING_POLICY=spatial_reference
  fi
else
  export HFSA_GROUNDING_POLICY="${HFSA_GROUNDING_POLICY:-native}"
fi
export HFSA_GROUNDING_SPATIAL_WEIGHT="${HFSA_GROUNDING_SPATIAL_WEIGHT:-0.15}"
export HFSA_API_PACKAGE_DIR="${HFSA_API_PACKAGE_DIR:-$hfsa_code_dir/../../rsu-formal800-v3-images}"
export HFSA_VQA_BUNDLE_DIR="$hfsa_code_dir/runs/vqa/qformer"
# Private, portable JSONL diagnostics; explicitly export an empty value to disable.
export HFSA_API_DIAGNOSTIC_FILE="${HFSA_API_DIAGNOSTIC_FILE-${XDG_DATA_HOME:-$HOME/.local/share}/hfsa/prediction-diagnostics.jsonl}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "$hfsa_code_dir/../../hfsa_env/bin/python" -B -u participant_api_starter-main/server.py \
  --package "$HFSA_API_PACKAGE_DIR" \
  --host 127.0.0.1 --port 9001
