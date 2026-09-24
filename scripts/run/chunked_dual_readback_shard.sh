#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <gpu> <run-id> <model> <offset> <limit>" >&2
  exit 2
fi

gpu=$1
run_id=$2
model=$3
offset=$4
limit=$5
python_bin=/workspace/yunlong/anaconda3/envs/audio-llm/bin/python

common=(
  scripts/facts/readback.py
  --run-id "$run_id"
  --model "$model"
  --offset "$offset"
  --limit "$limit"
  --chunk-seconds 25
  --force
  --device cuda:0
)

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$python_bin" -u "${common[@]}" --channels asr2 --low-memory
"$python_bin" -u "${common[@]}" --channels asr1
