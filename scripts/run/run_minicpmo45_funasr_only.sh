#!/usr/bin/env bash
# Complete one FunASR readback shard after all dual-ASR shards are available.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU SHARD_INDEX BARRIER_TAG NUM_SHARDS" >&2
  exit 2
fi

gpu=$1
shard=$2
barrier_tag=$3
num_shards=$4
if (( num_shards < 1 || shard < 0 || shard >= num_shards )); then
  echo "require NUM_SHARDS >= 1 and 0 <= SHARD_INDEX < NUM_SHARDS" >&2
  exit 2
fi
if [[ ! "$barrier_tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "BARRIER_TAG may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

model=MiniCPM-o-4_5
funasr_python="$root/.venvs/funasr-v100/bin/python"
barrier_dir="$root/output/minicpmo45/readback-barriers/$barrier_tag"
mkdir -p "$barrier_dir"

export CUDA_VISIBLE_DEVICES="$gpu"

wait_for_barrier() {
  local pattern=$1 expected=$2 label=$3 count
  while true; do
    count=$(find "$barrier_dir" -maxdepth 1 -type f -name "$pattern" | wc -l)
    (( count == expected )) && return 0
    echo "[$(date --iso-8601=seconds)] waiting for $label barrier: $count/$expected" >&2
    sleep 30
  done
}

run_funasr() {
  local run_id=$1 conditions=$2 attempt
  for attempt in 1 2 3; do
    if PYTHONPATH="src:third_party/v100_compat:third_party/FunASR" \
      "$funasr_python" -u scripts/facts/readback_funasr.py \
        --run-id "$run_id" --model "$model" --conditions "$conditions" \
        --device cuda:0 --chunk-seconds 25 \
        --num-shards "$num_shards" --shard-index "$shard"; then
      return 0
    fi
    echo "[readback] FunASR retry $attempt/3 for $run_id shard=$shard" >&2
  done
  return 1
}

echo "[funasr-only] host=$(hostname) gpu=$gpu shard=$shard started=$(date --iso-8601=seconds)"
wait_for_barrier 'dual-*.done' "$num_shards" dual-ASR

run_funasr omg_spoken_mqa SPEAK
run_funasr omg_voicebench_short SPEAK
run_funasr d1_main SPEAK,ECHO,EF,EFA,EFB,EFW

touch "$barrier_dir/funasr-$shard.done"
wait_for_barrier 'funasr-*.done' "$num_shards" FunASR
echo "[funasr-only] host=$(hostname) gpu=$gpu shard=$shard completed=$(date --iso-8601=seconds)"
