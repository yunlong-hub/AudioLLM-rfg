#!/usr/bin/env bash
# Four-way V100 readback queue for the MiniCPM-o-4_5 evaluation.
#
# Launch one process per inference shard after generation has completed:
#   run_minicpmo45_readback_queue.sh GPU SHARD_INDEX BARRIER_TAG [NUM_SHARDS]
#
# Stage 1 partitions sorted item directories into disjoint contiguous ranges for
# Whisper + Seamless.  Stage 2 starts only after all four stage-1 workers have
# finished; FunASR uses its own deterministic modulo partition.  The barrier is
# required because both stages update the same prediction JSON files.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 GPU SHARD_INDEX BARRIER_TAG [NUM_SHARDS]" >&2
  exit 2
fi

gpu=$1
shard=$2
barrier_tag=$3
num_shards=${4:-4}
if (( num_shards < 1 || shard < 0 || shard >= num_shards )); then
  echo "require NUM_SHARDS >= 1 and 0 <= SHARD_INDEX < NUM_SHARDS" >&2
  exit 2
fi
if [[ ! "$barrier_tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "BARRIER_TAG may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

model=MiniCPM-o-4_5
dual_python="$root/.venvs/minicpmo45/bin/python"
funasr_python="$root/.venvs/funasr-v100/bin/python"
barrier_dir="$root/output/minicpmo45/readback-barriers/$barrier_tag"
mkdir -p "$barrier_dir"

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH=src

wait_for_count() {
  local run_id=$1 pattern=$2 expected=$3
  local pred_root="$root/exp/$run_id/predictions/$model"
  local count
  while true; do
    if [[ -d "$pred_root" ]]; then
      count=$(find "$pred_root" -mindepth 2 -maxdepth 2 -type f -name "$pattern" | wc -l)
    else
      count=0
    fi
    (( count >= expected )) && break
    echo "[$(date --iso-8601=seconds)] waiting for $run_id/$pattern: $count/$expected" >&2
    sleep 300
  done
}

run_dual() {
  local run_id=$1 total=$2
  local offset=$(( total * shard / num_shards ))
  local end=$(( total * (shard + 1) / num_shards ))
  local limit=$(( end - offset ))
  local tag=""
  (( offset > 0 )) && tag="_off$offset"
  local summary="$root/exp/$run_id/metrics/readback_${model}${tag}.json"
  local attempt
  for attempt in 1 2 3; do
    "$dual_python" -u scripts/facts/readback.py \
      --run-id "$run_id" --model "$model" \
      --offset "$offset" --limit "$limit" --device cuda:0 \
      --chunk-seconds 25 --low-memory --channels asr1,asr2
    if "$dual_python" - "$summary" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
raise SystemExit(0 if summary.get("n_errors_total") == 0 else 1)
PY
    then
      return 0
    fi
    echo "[readback] dual-ASR retry $attempt/3 for $run_id shard=$shard" >&2
  done
  return 1
}

run_funasr() {
  local run_id=$1 conditions=$2
  local attempt
  for attempt in 1 2 3; do
    if PYTHONPATH="src:third_party/v100_compat:third_party/FunASR" \
      "$funasr_python" -u scripts/facts/readback_funasr.py \
        --run-id "$run_id" --model "$model" --conditions "$conditions" \
        --device cuda:0 --chunk-seconds 25 \
        --num-shards "$num_shards" --shard-index "$shard"; then
      return 0
    fi
    echo "[readback] FunASR retry $attempt/3 for $run_id shard=$shard" >&2
  done
  return 1
}

echo "[readback] host=$(hostname) gpu=$gpu shard=$shard started=$(date --iso-8601=seconds)"

for pattern in LISTEN.json SPEAK.json SPEAK.wav; do
  wait_for_count omg_spoken_mqa "$pattern" 1402
  wait_for_count omg_voicebench_short "$pattern" 1000
done
for condition in READ LISTEN SPEAK ECHO EF EFA EFB EFW; do
  wait_for_count d1_main "$condition.json" 600
done
for condition in SPEAK ECHO EF EFA EFB EFW; do
  wait_for_count d1_main "$condition.wav" 600
done

run_dual omg_spoken_mqa 1402
run_dual omg_voicebench_short 1000
run_dual d1_main 600

touch "$barrier_dir/dual-$shard.done"
while true; do
  dual_done=$(find "$barrier_dir" -maxdepth 1 -type f -name 'dual-*.done' | wc -l)
  (( dual_done == num_shards )) && break
  echo "[$(date --iso-8601=seconds)] waiting for dual-ASR barrier: $dual_done/$num_shards" >&2
  sleep 30
done

run_funasr omg_spoken_mqa SPEAK
run_funasr omg_voicebench_short SPEAK
run_funasr d1_main SPEAK,ECHO,EF,EFA,EFB,EFW

touch "$barrier_dir/funasr-$shard.done"
while true; do
  funasr_done=$(find "$barrier_dir" -maxdepth 1 -type f -name 'funasr-*.done' | wc -l)
  (( funasr_done == num_shards )) && break
  echo "[$(date --iso-8601=seconds)] waiting for FunASR barrier: $funasr_done/$num_shards" >&2
  sleep 30
done

if [[ "$shard" == 0 ]]; then
  "$dual_python" scripts/analyze/score_omg_benchmark.py \
    --items data/omg_benchmarks/prepared/spoken_mqa/items.jsonl \
    --run-id omg_spoken_mqa --model-slug "$model" \
    --speech-asrs asr1,asr2,asr3
  "$dual_python" scripts/analyze/score_omg_benchmark.py \
    --items data/omg_benchmarks/prepared/voicebench_bbh/items_short.jsonl \
    --run-id omg_voicebench_short --model-slug "$model" \
    --speech-asrs asr1,asr2,asr3
fi

echo "[readback] host=$(hostname) gpu=$gpu shard=$shard completed=$(date --iso-8601=seconds)"
