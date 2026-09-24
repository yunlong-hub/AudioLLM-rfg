#!/usr/bin/env bash
# Run one disjoint FunASR resample shard for a candidate filename prefix.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU PREFIX SHARD_INDEX NUM_SHARDS" >&2
  exit 2
fi
gpu=$1
prefix=$2
shard=$3
num_shards=$4
if (( num_shards < 1 || shard < 0 || shard >= num_shards )); then
  echo "require 0 <= SHARD_INDEX < NUM_SHARDS" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
python_bin="$root/.venvs/funasr-v100/bin/python"
exec {mini_fd}>"/tmp/minicpmo45_n11_gpu${gpu}.lock"
flock "$mini_fd"
exec {qwen_fd}>"/tmp/qwen25_7b_v100_gpu${gpu}.lock"
flock "$qwen_fd"
export CUDA_VISIBLE_DEVICES="$gpu"

run_funasr() {
  env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH="src:third_party/v100_compat:third_party/FunASR" \
    "$python_bin" -u scripts/facts/readback_funasr_resample.py \
    --model Qwen2.5-Omni-7B --device cuda:0 --prefixes "$prefix" \
    --num-shards "$num_shards" --shard-index "$shard"
}
if ! run_funasr; then
  echo "[FunASR sidecar prefix=$prefix shard=$shard] retrying" >&2
  run_funasr
fi
echo "[FunASR sidecar prefix=$prefix shard=$shard] complete=$(date --iso-8601=seconds)"
