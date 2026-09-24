#!/usr/bin/env bash
# Keep the seven Qwen2.5-Omni-7B shards and the finalizer alive until final.done.
set -euo pipefail

shared_root=${QWEN25_SHARED_ROOT:-/nfs/yunlong/LLM/AudioLLM-rfg}
remote_root=${QWEN25_REMOTE_ROOT:-/workspace/yunlong/LLM/AudioLLM-rfg}
barrier_tag=${QWEN25_V100_BARRIER_TAG:-full-20260920-n7-v1}
interval=${QWEN25_SUPERVISOR_INTERVAL:-60}
run_once=${QWEN25_SUPERVISOR_ONCE:-0}
num_shards=7
barrier_root="$shared_root/output/qwen25_7b_v100/barriers/$barrier_tag"
log_root="$shared_root/output/qwen25_7b_v100/logs"
mkdir -p "$barrier_root" "$log_root"

exec 9>/tmp/qwen25_7b_v100_supervisor.lock
if ! flock -n 9; then
  echo "[supervisor] another supervisor already holds the lock"
  exit 0
fi

jobs=(
  "V96 3 4 0 v96_g3-4_s0.log"
  "V96 0 1 1 v96_g0-1_s1.log"
  "V96 2 5 2 v96_g2-5_s2.log"
  "V98 4 5 3 v98_g4-5_s3.log"
  "V99 0 1 4 v99_g0-1_s4.log"
  "V99 2 3 5 v99_g2-3_s5.log"
  "V99 4 5 6 v99_g4-5_s6.log"
)

worker_running() {
  local host=$1 gpu_a=$2 gpu_b=$3 shard=$4
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "pgrep -f 'bash scripts/run/run_qwen25_7b_v100_worker[.]sh $gpu_a $gpu_b $shard $num_shards' >/dev/null"
}

launch_worker() {
  local host=$1 gpu_a=$2 gpu_b=$3 shard=$4 log_name=$5
  echo "[supervisor] launch host=$host gpu=$gpu_a,$gpu_b shard=$shard/$num_shards"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "cd '$remote_root'; nohup env QWEN25_V100_BATCH_SIZE=1 QWEN25_V100_TALKER_MAX_NEW_TOKENS=512 QWEN25_V100_BARRIER_TAG='$barrier_tag' bash scripts/run/run_qwen25_7b_v100_worker.sh '$gpu_a' '$gpu_b' '$shard' '$num_shards' >> 'output/qwen25_7b_v100/logs/$log_name' 2>&1 < /dev/null &"
}

finalizer_running() {
  ssh -o BatchMode=yes -o ConnectTimeout=8 V96 "pgrep -f 'bash scripts/run/run_qwen25_7b_v100_finalize[.]sh 5 $num_shards' >/dev/null"
}

launch_finalizer() {
  echo "[supervisor] launch finalizer host=V96 gpu=5"
  ssh -o BatchMode=yes -o ConnectTimeout=8 V96 "cd '$remote_root'; nohup env QWEN25_V100_BARRIER_TAG='$barrier_tag' bash scripts/run/run_qwen25_7b_v100_finalize.sh 5 '$num_shards' >> output/qwen25_7b_v100/logs/finalize_v96_g5.log 2>&1 < /dev/null &"
}

while [[ ! -f "$barrier_root/final.done" ]]; do
  active=0
  for row in "${jobs[@]}"; do
    read -r host gpu_a gpu_b shard log_name <<<"$row"
    if [[ -f "$barrier_root/readback-s${shard}-of-${num_shards}.done" ]]; then
      continue
    fi
    if worker_running "$host" "$gpu_a" "$gpu_b" "$shard"; then
      ((active += 1))
    else
      launch_worker "$host" "$gpu_a" "$gpu_b" "$shard" "$log_name" || true
    fi
  done

  readback_count=$(find "$barrier_root" -maxdepth 1 -name "readback-s*-of-${num_shards}.done" | wc -l)
  generation_count=$(find "$barrier_root" -maxdepth 1 -name "generation-s*-of-${num_shards}.done" | wc -l)
  echo "[supervisor] ts=$(date --iso-8601=seconds) active=$active generation=$generation_count/$num_shards readback=$readback_count/$num_shards"

  if (( readback_count == num_shards )) && ! finalizer_running; then
    launch_finalizer || true
  fi
  if [[ "$run_once" == 1 ]]; then
    break
  fi
  sleep "$interval"
done

if [[ -f "$barrier_root/final.done" ]]; then
  echo "[supervisor] final.done observed at $(date --iso-8601=seconds)"
fi
