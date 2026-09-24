#!/usr/bin/env bash
# Precompute three-ASR readback for one completed resample candidate pool.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 GPU SHARD_INDEX NUM_SHARDS [CANDIDATE_INDEX]" >&2
  exit 2
fi
gpu=$1
shard=$2
num_shards=$3
candidate=${4:-0}
if (( num_shards < 1 || shard < 0 || shard >= num_shards || candidate < 0 )); then
  echo "require 0 <= SHARD_INDEX < NUM_SHARDS and CANDIDATE_INDEX >= 0" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
asr_python="$root/.venvs/minicpmo45/bin/python"
funasr_python="$root/.venvs/funasr-v100/bin/python"

exec {mini_fd}>"/tmp/minicpmo45_n11_gpu${gpu}.lock"
flock "$mini_fd"
exec {qwen_fd}>"/tmp/qwen25_7b_v100_gpu${gpu}.lock"
flock "$qwen_fd"
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | sed -n "$((gpu + 1))p" | tr -d ' ')
  (( used <= 1024 )) && break
  echo "[R${candidate} sidecar shard=$shard] waiting gpu=$gpu used=${used}MiB"
  sleep 20
done

total=200
offset=$(( total * shard / num_shards ))
end=$(( total * (shard + 1) / num_shards ))
limit=$(( end - offset ))
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

"$asr_python" -u scripts/facts/readback_resample.py \
  --model Qwen2.5-Omni-7B --items data/pilot/items.jsonl \
  --offset "$offset" --limit "$limit" --prefixes "R${candidate}" \
  --channels asr1,asr2 \
  --device cuda:0 --chunk-seconds 25 --low-memory

run_funasr() {
  env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH="src:third_party/v100_compat:third_party/FunASR" \
    "$funasr_python" -u scripts/facts/readback_funasr_resample.py \
    --model Qwen2.5-Omni-7B --device cuda:0 --prefixes "R${candidate}" \
    --num-shards "$num_shards" --shard-index "$shard"
}
if ! run_funasr; then
  echo "[R${candidate} sidecar shard=$shard] retrying missing FunASR readbacks" >&2
  run_funasr
fi
echo "[R${candidate} sidecar shard=$shard] complete=$(date --iso-8601=seconds)"
