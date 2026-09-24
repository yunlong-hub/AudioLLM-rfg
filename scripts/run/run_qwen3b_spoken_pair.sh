#!/usr/bin/env bash
# Run two disjoint Qwen2.5-Omni-3B Spoken-MQA shards on separate physical GPUs.
# Existing per-item outputs are reused, so the command is safe to resume.
set -euo pipefail

gpu_a=${1:?first physical GPU id required}
offset_a=${2:?first shard offset required}
limit_a=${3:?first shard limit required}
gpu_b=${4:?second physical GPU id required}
offset_b=${5:?second shard offset required}
limit_b=${6:?second shard limit required}
readback_tag=${7:-}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${AUDIO_LLM_PYTHON:-/workspace/yunlong/anaconda3/envs/audio-llm/bin/python}
model_path=/workspace/yunlong/LLM/pretrain_model/Audio/Qwen2.5-Omni-3B
items_path=data/omg_benchmarks/prepared/spoken_mqa/items.jsonl

run_shard() {
  local gpu=$1 offset=$2 limit=$3
  # Each worker sees one GPU; keep the canonical natural-run device-map value.
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
    "$python_bin" -u scripts/infer/infer.py \
      --model "$model_path" \
      --items "$items_path" \
      --run-id omg_spoken_mqa \
      --conditions LISTEN,SPEAK \
      --offset "$offset" \
      --limit "$limit" \
      --max-new-tokens 512 \
      --talker-max-new-tokens 2048 \
      --device-map cuda:0
}

cd "$repo_root"

run_shard "$gpu_a" "$offset_a" "$limit_a" &
pid_a=$!
run_shard "$gpu_b" "$offset_b" "$limit_b" &
pid_b=$!

status=0
wait "$pid_a" || status=$?
wait "$pid_b" || status=$?
if (( status != 0 )) || [[ -z "$readback_tag" ]]; then
  exit "$status"
fi

# Exactly one inference pair receives a barrier tag.  Its two workers wait for
# full 1,402-item coverage (including shards produced on other hosts), then run
# disjoint final ASR shards and publish the coverage-gated benchmark metrics.
env CUDA_VISIBLE_DEVICES="$gpu_a" \
  scripts/run/natural_readback_chain.sh \
    Qwen2.5-Omni-3B 0 "$readback_tag" spoken &
readback_pid_a=$!
env CUDA_VISIBLE_DEVICES="$gpu_b" \
  scripts/run/natural_readback_chain.sh \
    Qwen2.5-Omni-3B 1 "$readback_tag" spoken &
readback_pid_b=$!

wait "$readback_pid_a" || status=$?
wait "$readback_pid_b" || status=$?
exit "$status"
