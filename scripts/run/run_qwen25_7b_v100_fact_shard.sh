#!/usr/bin/env bash
# Extract one disjoint fact-cache shard after all readback workers complete.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU SHARD_INDEX NUM_SHARDS" >&2
  exit 2
fi

gpu=$1
shard_index=$2
num_shards=$3
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

barrier_tag=${QWEN25_V100_BARRIER_TAG:-full-20260920-n7-v1}
barrier_root="$root/output/qwen25_7b_v100/barriers/$barrier_tag"
readback_shards=${QWEN25_V100_READBACK_SHARDS:-7}
mkdir -p "$barrier_root" "$root/output/qwen25_7b_v100/logs"

echo "[facts shard=$shard_index] waiting for $readback_shards readback markers"
while (( $(find "$barrier_root" -maxdepth 1 -name "readback-s*-of-${readback_shards}.done" | wc -l) < readback_shards )); do
  sleep 15
done

exec {mini_fd}>"/tmp/minicpmo45_n11_gpu${gpu}.lock"
flock "$mini_fd"
exec {qwen_fd}>"/tmp/qwen25_7b_v100_gpu${gpu}.lock"
flock "$qwen_fd"

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH=src
python_bin="$root/.venvs/minicpmo45/bin/python"
model=Qwen2.5-Omni-7B
llm_model=${LLM_MODEL:-/workspace/yunlong/LLM/pretrain_model/Qwen/Qwen2.5-7B-Instruct}

"$python_bin" -u scripts/facts/extract_facts.py \
  --run-id d1_main --model "$model" --with-llm \
  --llm-device cuda:0 --llm-model "$llm_model" \
  --num-shards "$num_shards" --shard-index "$shard_index"

printf 'host=%s gpu=%s shard=%s completed=%s\n' \
  "$(hostname)" "$gpu" "$shard_index" "$(date --iso-8601=seconds)" \
  >"$barrier_root/facts-s${shard_index}-of-${num_shards}.done"
echo "[facts shard=$shard_index] completed=$(date --iso-8601=seconds)"
