#!/usr/bin/env bash
# Use the second GPU of a generation pair to precompute non-Spoken-MQA readback.
# The primary worker processes Spoken-MQA first, so the two processes write to
# different run directories.  Later primary passes reuse these observations.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU SHARD_INDEX NUM_SHARDS" >&2
  exit 2
fi

gpu=$1
shard=$2
num_shards=$3
if (( num_shards < 1 || shard < 0 || shard >= num_shards )); then
  echo "require 0 <= SHARD_INDEX < NUM_SHARDS" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

barrier_tag=${QWEN25_V100_BARRIER_TAG:-full-20260920-n7-v1}
barrier_root="$root/output/qwen25_7b_v100/barriers/$barrier_tag"
asr_python="$root/.venvs/minicpmo45/bin/python"
funasr_python="$root/.venvs/funasr-v100/bin/python"
model=Qwen2.5-Omni-7B
mkdir -p "$barrier_root"

exec 9>"/tmp/qwen25_7b_v100_sidecar_gpu${gpu}.lock"
if ! flock -n 9; then
  echo "[sidecar] another sidecar holds gpu=$gpu host=$(hostname)"
  exit 0
fi

echo "[sidecar] host=$(hostname) gpu=$gpu shard=$shard/$num_shards waiting for generation"
while (( $(find "$barrier_root" -maxdepth 1 -name "generation-s*-of-${num_shards}.done" | wc -l) < num_shards )); do
  sleep 15
done

# GPU_B is reserved by the paired primary worker, but model teardown may lag
# behind the generation marker briefly.  Do not start ASR until it is idle.
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | sed -n "$((gpu + 1))p" | tr -d ' ')
  (( used <= 1024 )) && break
  echo "[sidecar] waiting for idle gpu=$gpu used=${used}MiB"
  sleep 15
done

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_dual() {
  local run_id=$1 total=$2
  local offset=$(( total * shard / num_shards ))
  local end=$(( total * (shard + 1) / num_shards ))
  local limit=$(( end - offset ))
  "$asr_python" -u scripts/facts/readback.py \
    --run-id "$run_id" --model "$model" --offset "$offset" --limit "$limit" \
    --device cuda:0 --channels asr1,asr2 --low-memory --chunk-seconds 25
}

run_funasr() {
  local run_id=$1 conditions=$2
  env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH="src:third_party/v100_compat:third_party/FunASR" \
    "$funasr_python" -u scripts/facts/readback_funasr.py \
    --run-id "$run_id" --model "$model" --conditions "$conditions" \
    --device cuda:0 --num-shards "$num_shards" --shard-index "$shard" \
    --chunk-seconds 25
}

# The primary worker spends its first readback phase on Spoken-MQA.  Sidecars
# consume the otherwise-idle paired GPUs and finish the other two runs first.
run_dual omg_voicebench_short 1000
run_dual d1_main 600
printf 'host=%s gpu=%s completed=%s\n' "$(hostname)" "$gpu" "$(date --iso-8601=seconds)" \
  >"$barrier_root/sidecar-dual-s${shard}-of-${num_shards}.done"

# FunASR writes only after every dual-ASR primary marker exists, matching the
# main pipeline's write-order barrier.  The primary starts with Spoken-MQA.
while (( $(find "$barrier_root" -maxdepth 1 -name "dual-asr-s*-of-${num_shards}.done" | wc -l) < num_shards )); do
  sleep 15
done

run_funasr omg_voicebench_short SPEAK
run_funasr d1_main SPEAK,ECHO,EF,EFA,EFB,EFW
printf 'host=%s gpu=%s completed=%s\n' "$(hostname)" "$gpu" "$(date --iso-8601=seconds)" \
  >"$barrier_root/sidecar-readback-s${shard}-of-${num_shards}.done"
echo "[sidecar] complete shard=$shard/$num_shards completed=$(date --iso-8601=seconds)"
