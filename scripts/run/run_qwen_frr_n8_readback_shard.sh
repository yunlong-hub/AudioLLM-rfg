#!/usr/bin/env bash
# Wait for a complete Qwen N=8 candidate pool, then run one disjoint
# dual-ASR/FunASR shard. Shard zero owns the final fact extraction and metrics.
set -euo pipefail

if [[ $# -ne 10 ]]; then
  echo "usage: $0 <gpu> <model-slug> <offset> <limit> <shard-index> <num-shards> <tag> <pilot-pred> <facts-root> <model-path>" >&2
  exit 2
fi

gpu=$1
model=$2
offset=$3
limit=$4
shard_index=$5
num_shards=$6
tag=$7
pilot_pred=$8
facts_root=$9
model_path=${10}

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

audio_python=/workspace/yunlong/anaconda3/envs/audio-llm/bin/python
funasr_python="$root/.venvs/funasr/bin/python"
barrier="$root/output/qwen_frr_n8/barriers/$tag"
mkdir -p "$barrier" "$root/output/qwen_frr_n8/logs"

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# N=8 consists of the original SPEAK sample plus R0--R6.  Wait for all
# generated JSON/WAV pairs before any recognizer mutates their records.
while :; do
  json_count=$(find "exp/d0_resample/$model" -mindepth 2 -maxdepth 2 \
    -type f -name 'R[0-6].json' | wc -l)
  wav_count=$(find "exp/d0_resample/$model" -mindepth 2 -maxdepth 2 \
    -type f -name 'R[0-6].wav' | wc -l)
  if (( json_count == 1400 && wav_count == 1400 )); then
    break
  fi
  echo "[n8 readback] waiting model=$model json=$json_count/1400 wav=$wav_count/1400"
  sleep 30
done

"$audio_python" -u scripts/facts/readback_resample.py \
  --model "$model" --items data/pilot/items.jsonl \
  --offset "$offset" --limit "$limit" --n 7 --channels asr1,asr2 \
  --device cuda:0 --chunk-seconds 25 --low-memory
touch "$barrier/dual-$shard_index.done"

for ((index=0; index<num_shards; index++)); do
  while [[ ! -f "$barrier/dual-$index.done" ]]; do sleep 20; done
done

PYTHONPATH=src:third_party/FunASR "$funasr_python" -u \
  scripts/facts/readback_funasr_resample.py \
  --model "$model" --device cuda:0 \
  --num-shards "$num_shards" --shard-index "$shard_index"
touch "$barrier/funasr-$shard_index.done"

if [[ "$shard_index" == "0" ]]; then
  for ((index=0; index<num_shards; index++)); do
    while [[ ! -f "$barrier/funasr-$index.done" ]]; do sleep 20; done
  done

  "$audio_python" - <<PY
import json
from pathlib import Path

root = Path("exp/d0_resample") / "$model"
paths = sorted(root.glob("*/R[0-6].json"))
missing = []
for path in paths:
    record = json.loads(path.read_text())
    for channel in ("asr1", "asr2", "asr3"):
        if not record.get(channel):
            missing.append(f"{path}:{channel}")
if len(paths) != 1400 or missing:
    raise SystemExit(
        f"incomplete N=8 readback: records={len(paths)}/1400 "
        f"missing={len(missing)} examples={missing[:20]}"
    )
print(f"validated N=8 readback: records={len(paths)} channels=3")
PY

  "$audio_python" -u scripts/analyze/probe_resample.py \
    --model "$model_path" --pilot-pred "$pilot_pred" \
    --facts-root "$facts_root" --out exp/d0_resample \
    --n 7 --seeds 101,202,303,404,505,606,707 \
    --selector-asr asr1 --evaluator-asr asr2 --llm-device cuda:0
  "$audio_python" -u scripts/analyze/evaluate_frr_loo.py \
    --model-slug "$model" --pilot-pred "$pilot_pred" \
    --resample-root exp/d0_resample \
    --output "exp/d0_resample/frr_loo_${model}.json"
  "$audio_python" -u scripts/analyze/evaluate_frr_curve.py \
    --model-slug "$model" --pilot-pred "$pilot_pred" \
    --resample-root exp/d0_resample --candidate-counts 1,2,4,8 \
    --output "exp/d0_resample/frr_curve_${model}.json"
  touch "$barrier/final.done"
fi

echo "[n8 readback] complete model=$model shard=$shard_index/$num_shards at $(date --iso-8601=seconds)"
