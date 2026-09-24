#!/usr/bin/env bash
# Resume two disjoint Step-Audio Spoken-MQA shards on separate physical GPUs.
# Existing per-item outputs are reused, so interrupted runs remain resumable.
set -euo pipefail

gpu_a=${1:?first physical GPU id required}
offset_a=${2:?first shard offset required}
limit_a=${3:?first shard limit required}
gpu_b=${4:?second physical GPU id required}
offset_b=${5:?second shard offset required}
limit_b=${6:?second shard limit required}
readback_tag=${7:-}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
items_path=data/omg_benchmarks/prepared/spoken_mqa/items.jsonl

run_shard() {
  local gpu=$1 offset=$2 limit=$3
  env CUDA_VISIBLE_DEVICES="$gpu" \
    scripts/run/run_stepaudio.sh \
      --items "$items_path" \
      --run-id omg_spoken_mqa \
      --conditions LISTEN,SPEAK \
      --offset "$offset" \
      --limit "$limit"
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

# The final pair waits for all six inference shards to reach full coverage,
# then publishes disjoint three-ASR readback and coverage-gated metrics.
env CUDA_VISIBLE_DEVICES="$gpu_a" \
  scripts/run/natural_readback_chain.sh \
    Step-Audio-2-mini 0 "$readback_tag" spoken &
readback_pid_a=$!
env CUDA_VISIBLE_DEVICES="$gpu_b" \
  scripts/run/natural_readback_chain.sh \
    Step-Audio-2-mini 1 "$readback_tag" spoken &
readback_pid_b=$!

wait "$readback_pid_a" || status=$?
wait "$readback_pid_b" || status=$?
exit "$status"
