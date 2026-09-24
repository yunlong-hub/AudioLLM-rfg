#!/usr/bin/env bash
# Persistent generation + three-ASR worker for one n-way shard.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU SHARD_INDEX NUM_SHARDS BARRIER_TAG" >&2
  exit 2
fi

gpu=$1
shard=$2
num_shards=$3
barrier_tag=$4
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

"$root/scripts/run/run_minicpmo45_queue_nway.sh" "$gpu" "$shard" "$num_shards"
"$root/scripts/run/run_minicpmo45_readback_queue.sh" \
  "$gpu" "$shard" "$barrier_tag" "$num_shards"
