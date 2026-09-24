#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU BARRIER_TAG" >&2
  exit 2
fi
gpu=$1
tag=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
model=MiniCPM-o-4_5
targets="exp/d4_rerender/metrics/targets_${model}.json"
llm_model=pretrain_model/Qwen/Qwen2.5-7B-Instruct
barrier="$root/output/minicpmo45/rerender-readback-barriers/$tag"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH=src
while (( $(find "$barrier" -maxdepth 1 -type f -name 'funasr-*.done' | wc -l) < 2 )); do
  sleep 20
done
for mode in asr1 asr2 asr3; do
  "$root/.venvs/minicpmo45/bin/python" -u scripts/analyze/judge_rerender.py \
    --run-id d4_rerender --model "$model" --readback-mode "$mode" \
    --targets "$targets" --out "reports/rerender_${model}_${mode}.md" \
    --metrics-out "exp/d4_rerender/metrics/rerender_judge_${model}_${mode}.json" \
    --llm-model "$llm_model" --llm-device cuda:0 --llm-batch-size 32
done
echo "[minicpmo-rerender-judge] completed=$(date --iso-8601=seconds)"
