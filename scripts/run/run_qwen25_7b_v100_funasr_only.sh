#!/usr/bin/env bash
# Resume one Qwen2.5-Omni-7B FunASR shard after dual ASR is complete.
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
funasr_python="$root/.venvs/funasr-v100/bin/python"
model=Qwen2.5-Omni-7B
mkdir -p "$barrier_root"

while (( $(find "$barrier_root" -maxdepth 1 -name "dual-asr-s*-of-${num_shards}.done" | wc -l) < num_shards )); do
  echo "[funasr-only] waiting for dual-ASR barrier" >&2
  sleep 15
done

exec 9>"/tmp/qwen25_7b_v100_gpu${gpu}.lock"
flock 9
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | sed -n "$((gpu + 1))p" | tr -d ' ')
  (( used <= 1024 )) && break
  echo "[funasr-only] waiting for idle gpu=$gpu used=${used}MiB"
  sleep 15
done

export CUDA_VISIBLE_DEVICES="$gpu"

run_funasr() {
  local run_id=$1 conditions=$2 attempt
  for attempt in 1 2 3; do
    if env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      PYTHONPATH="src:third_party/v100_compat:third_party/FunASR" \
      "$funasr_python" -u scripts/facts/readback_funasr.py \
        --run-id "$run_id" --model "$model" --conditions "$conditions" \
        --device cuda:0 --num-shards "$num_shards" --shard-index "$shard" \
        --chunk-seconds 25; then
      return 0
    fi
    echo "[funasr-only] retry $attempt/3 run=$run_id shard=$shard" >&2
  done
  return 1
}

echo "[funasr-only] host=$(hostname) gpu=$gpu shard=$shard/$num_shards started=$(date --iso-8601=seconds)"
run_funasr omg_spoken_mqa SPEAK
run_funasr omg_voicebench_short SPEAK
run_funasr d1_main SPEAK,ECHO,EF,EFA,EFB,EFW
printf 'host=%s gpu=%s completed=%s\n' "$(hostname)" "$gpu" "$(date --iso-8601=seconds)" \
  >"$barrier_root/readback-s${shard}-of-${num_shards}.done"
echo "[funasr-only] complete shard=$shard/$num_shards completed=$(date --iso-8601=seconds)"
