#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU BARRIER_TAG NUM_SHARDS" >&2
  exit 2
fi
gpu=$1
tag=$2
num_shards=$3
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
model=MiniCPM-o-4_5
targets="exp/d4_rerender/metrics/targets_${model}.json"
llm_model=pretrain_model/Qwen/Qwen2.5-7B-Instruct
barrier="$root/output/minicpmo45/rerender-barriers/$tag"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH=src
while (( $(find "$barrier" -maxdepth 1 -type f -name 'generation-*.done' | wc -l) < num_shards )); do
  sleep 20
done

"$root/.venvs/minicpmo45/bin/python" -u scripts/facts/readback_resample.py \
  --model "$model" --root exp/d4_rerender --items data/main600/items.jsonl \
  --limit 200 --prefixes ORIG,SPEAK --channels asr1,asr2 --chunk-seconds 25 --low-memory
PYTHONPATH="src:third_party/v100_compat:third_party/FunASR" \
  "$root/.venvs/funasr-v100/bin/python" -u scripts/facts/readback_funasr_resample.py \
    --model "$model" --root exp/d4_rerender --prefixes ORIG,SPEAK
for mode in asr1 asr2 asr3; do
  "$root/.venvs/minicpmo45/bin/python" -u scripts/analyze/judge_rerender.py \
    --run-id d4_rerender --model "$model" --readback-mode "$mode" \
    --targets "$targets" --out "reports/rerender_${model}_${mode}.md" \
    --metrics-out "exp/d4_rerender/metrics/rerender_judge_${model}_${mode}.json" \
    --llm-model "$llm_model" --llm-device cuda:0 --llm-batch-size 32
done
echo "[minicpmo-rerender-finalize] completed=$(date --iso-8601=seconds)"
