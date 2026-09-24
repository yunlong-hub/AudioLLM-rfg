#!/usr/bin/env bash
# Incrementally extend MiniCPM-o FRR from N=4 to N=8 and run three-ASR readback.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU SHARD_INDEX NUM_SHARDS BARRIER_TAG" >&2
  exit 2
fi

gpu=$1
shard=$2
num_shards=$3
barrier_tag=$4
if (( num_shards < 1 || shard < 0 || shard >= num_shards )); then
  echo "require NUM_SHARDS >= 1 and 0 <= SHARD_INDEX < NUM_SHARDS" >&2
  exit 2
fi
if [[ ! "$barrier_tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid barrier tag" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
model=MiniCPM-o-4_5
total=200
offset=$(( total * shard / num_shards ))
end=$(( total * (shard + 1) / num_shards ))
limit=$(( end - offset ))
barrier_dir="$root/output/minicpmo45/resample-barriers/$barrier_tag"
mkdir -p "$barrier_dir"

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH=src

mini_python="$root/.venvs/minicpmo45/bin/python"
funasr_python="$root/.venvs/funasr-v100/bin/python"

echo "[resample-n8-worker] host=$(hostname) gpu=$gpu shard=$shard/$num_shards range=$offset:$end"
"$mini_python" -u scripts/analyze/generate_minicpmo_resamples.py \
  --offset "$offset" --limit "$limit" --n 4 \
  --seeds 404,505,606,707 --candidate-offset 3
"$mini_python" -u scripts/facts/readback_resample.py \
  --model "$model" --items data/main600/items.jsonl \
  --offset "$offset" --limit "$limit" --n 7 \
  --channels asr1,asr2 --chunk-seconds 25 --low-memory

touch "$barrier_dir/dual-$shard.done"
while (( $(find "$barrier_dir" -maxdepth 1 -type f -name 'dual-*.done' | wc -l) < num_shards )); do
  sleep 20
done

PYTHONPATH="src:third_party/v100_compat:third_party/FunASR" \
  "$funasr_python" -u scripts/facts/readback_funasr_resample.py \
    --model "$model" --num-shards "$num_shards" --shard-index "$shard"
touch "$barrier_dir/funasr-$shard.done"

echo "[resample-n8-worker] completed host=$(hostname) shard=$shard/$num_shards"
