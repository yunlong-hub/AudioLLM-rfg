#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 GPU RUN_ID ITEMS OFFSET LIMIT CONDITIONS [OUT_ROOT]" >&2
  exit 2
fi

gpu=$1
run_id=$2
items=$3
offset=$4
limit=$5
conditions=$6
out_root=${7:-exp}

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin="$root/.venvs/minicpmo45/bin/python"
model="$root/pretrain_model/Audio/MiniCPM-o-4_5"

if [[ ! -x "$python_bin" ]]; then
  echo "missing MiniCPM environment: $python_bin" >&2
  exit 1
fi

cd "$root"
exec env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
  "$python_bin" -u scripts/infer/infer.py \
    --model "$model" \
    --items "$items" \
    --run-id "$run_id" \
    --out-root "$out_root" \
    --offset "$offset" \
    --limit "$limit" \
    --conditions "$conditions" \
    --max-new-tokens 512 \
    --talker-max-new-tokens 2048 \
    --temperature 0.0 \
    --device-map auto
