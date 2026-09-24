#!/usr/bin/env bash
# Two-way sharded three-ASR readback for the retained natural benchmarks.
# Launch two copies with shard_index 0 and 1 and a shared, unique barrier_tag.
set -euo pipefail

model=${1:?model slug required}
shard_index=${2:?shard index 0 or 1 required}
barrier_tag=${3:?unique barrier tag required}
scope=${4:-both}

if [[ "$shard_index" != "0" && "$shard_index" != "1" ]]; then
  echo "shard_index must be 0 or 1" >&2
  exit 2
fi
if [[ "$scope" != "both" && "$scope" != "spoken" && "$scope" != "voice" ]]; then
  echo "scope must be both, spoken, or voice" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

python_bin=${AUDIO_LLM_PYTHON:-/workspace/yunlong/anaconda3/envs/audio-llm/bin/python}
funasr_python=${FUNASR_PYTHON:-$repo_root/.venvs/funasr/bin/python}
barrier_dir="$repo_root/output/experiment-extension/readback-barriers/$barrier_tag"
mkdir -p "$barrier_dir"

if [[ "$scope" == "spoken" ]]; then
  runs=(omg_spoken_mqa)
  item_files=(data/omg_benchmarks/prepared/spoken_mqa/items.jsonl)
  totals=(1402)
elif [[ "$scope" == "voice" ]]; then
  runs=(omg_voicebench_short)
  item_files=(data/omg_benchmarks/prepared/voicebench_bbh/items_short.jsonl)
  totals=(1000)
else
  runs=(omg_spoken_mqa omg_voicebench_short)
  item_files=(
    data/omg_benchmarks/prepared/spoken_mqa/items.jsonl
    data/omg_benchmarks/prepared/voicebench_bbh/items_short.jsonl
  )
  totals=(1402 1000)
fi

for index in "${!runs[@]}"; do
  run_id=${runs[$index]}
  total=${totals[$index]}
  first_limit=$(( (total + 1) / 2 ))
  if [[ "$shard_index" == "0" ]]; then
    offset=0
    limit=$first_limit
  else
    offset=$first_limit
    limit=$(( total - first_limit ))
  fi

  pred_root="$repo_root/exp/$run_id/predictions/$model"
  while true; do
    wav_count=$(find "$pred_root" -mindepth 2 -maxdepth 2 -type f -name SPEAK.wav 2>/dev/null | wc -l)
    if (( wav_count >= total )); then
      break
    fi
    echo "[$(date --iso-8601=seconds)] waiting for $run_id SPEAK WAVs: $wav_count/$total" >&2
    sleep 300
  done

  PYTHONPATH=src "$python_bin" -u scripts/facts/readback.py \
    --run-id "$run_id" --model "$model" \
    --offset "$offset" --limit "$limit" --device cuda:0 \
    --chunk-seconds 25 --low-memory
done

touch "$barrier_dir/dual-$shard_index.done"
while [[ ! -f "$barrier_dir/dual-0.done" || ! -f "$barrier_dir/dual-1.done" ]]; do
  sleep 30
done

for run_id in "${runs[@]}"; do
  PYTHONPATH=src:third_party/FunASR "$funasr_python" -u scripts/facts/readback_funasr.py \
    --run-id "$run_id" --model "$model" --conditions SPEAK --device cuda:0 \
    --chunk-seconds 25 --num-shards 2 --shard-index "$shard_index"
done

touch "$barrier_dir/funasr-$shard_index.done"
if [[ "$shard_index" == "0" ]]; then
  while [[ ! -f "$barrier_dir/funasr-1.done" ]]; do
    sleep 30
  done
  for index in "${!runs[@]}"; do
    PYTHONPATH=src "$python_bin" scripts/analyze/score_omg_benchmark.py \
      --items "${item_files[$index]}" --run-id "${runs[$index]}" \
      --model-slug "$model" --speech-asrs asr1,asr2,asr3
  done
fi
