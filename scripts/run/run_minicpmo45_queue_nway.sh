#!/usr/bin/env bash
# Run one non-overlapping shard of the full MiniCPM-o-4.5 generation matrix.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU SHARD_INDEX NUM_SHARDS" >&2
  exit 2
fi

gpu=$1
shard=$2
num_shards=$3
if (( num_shards < 1 || shard < 0 || shard >= num_shards )); then
  echo "require NUM_SHARDS >= 1 and 0 <= SHARD_INDEX < NUM_SHARDS" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
launcher="$root/scripts/run/run_minicpmo45_shard.sh"

run_slice() {
  local run_id=$1 items=$2 total=$3 conditions=$4
  local offset=$(( total * shard / num_shards ))
  local end=$(( total * (shard + 1) / num_shards ))
  local limit=$(( end - offset ))
  "$launcher" "$gpu" "$run_id" "$items" "$offset" "$limit" "$conditions"
}

echo "[queue-nway] host=$(hostname) gpu=$gpu shard=$shard/$num_shards started=$(date --iso-8601=seconds)"

run_slice omg_spoken_mqa \
  data/omg_benchmarks/prepared/spoken_mqa/items.jsonl 1402 LISTEN,SPEAK
run_slice omg_voicebench_short \
  data/omg_benchmarks/prepared/voicebench_bbh/items_short.jsonl 1000 LISTEN,SPEAK
run_slice d1_main data/main600/items.jsonl 600 \
  READ,LISTEN,SPEAK,ECHO,EF,EFA,EFB,EFW

echo "[queue-nway] host=$(hostname) gpu=$gpu shard=$shard/$num_shards completed=$(date --iso-8601=seconds)"
