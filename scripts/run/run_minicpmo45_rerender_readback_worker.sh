#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU SHARD_INDEX BARRIER_TAG" >&2
  exit 2
fi
gpu=$1
shard=$2
tag=$3
if [[ "$shard" != 0 && "$shard" != 1 ]]; then
  echo "SHARD_INDEX must be 0 or 1" >&2
  exit 2
fi
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
model=MiniCPM-o-4_5
offset=$(( shard * 100 ))
barrier="$root/output/minicpmo45/rerender-readback-barriers/$tag"
mkdir -p "$barrier"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH=src

"$root/.venvs/minicpmo45/bin/python" -u scripts/facts/readback_resample.py \
  --model "$model" --root exp/d4_rerender --items data/main600/items.jsonl \
  --offset "$offset" --limit 100 --prefixes ORIG,SPEAK \
  --channels asr1,asr2 --chunk-seconds 25 --low-memory
touch "$barrier/dual-$shard.done"
while (( $(find "$barrier" -maxdepth 1 -type f -name 'dual-*.done' | wc -l) < 2 )); do
  sleep 20
done
PYTHONPATH="src:third_party/v100_compat:third_party/FunASR" \
  "$root/.venvs/funasr-v100/bin/python" -u scripts/facts/readback_funasr_resample.py \
    --model "$model" --root exp/d4_rerender --prefixes ORIG,SPEAK \
    --num-shards 2 --shard-index "$shard"
touch "$barrier/funasr-$shard.done"
