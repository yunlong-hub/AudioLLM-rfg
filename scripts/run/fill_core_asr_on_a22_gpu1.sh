#!/usr/bin/env bash
# Fill reusable three-ASR gaps from core completed runs on A22 GPU1.
#
# GPU1 may be shared with an existing low-memory job.  The launcher owns the
# host-local GPU lock and checks available memory before invoking this script.
# Every readback command is resumable and skips successful observations.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
audio_python=${AUDIO_LLM_PYTHON:-/workspace/yunlong/anaconda3/envs/audio-llm/bin/python}
funasr_python=${FUNASR_PYTHON:-$repo_root/.venvs/funasr/bin/python}

fill_run() {
  local run_id=$1 model=$2
  echo "[$(date --iso-8601=seconds)] filling $run_id / $model"
  PYTHONPATH=src "$audio_python" -u scripts/facts/readback.py \
    --run-id "$run_id" --model "$model" --device cuda:0 \
    --chunk-seconds 25 --low-memory
  PYTHONPATH=src:third_party/FunASR "$funasr_python" -u \
    scripts/facts/readback_funasr.py \
    --run-id "$run_id" --model "$model" --device cuda:0 \
    --chunk-seconds 25
}

cd "$repo_root"
fill_run d1_main Qwen2.5-Omni-3B
fill_run d0_pilot Qwen2.5-Omni-7B
fill_run d0_pilot Qwen2.5-Omni-3B
fill_run d3_7b_stack Qwen2.5-Omni-7B
