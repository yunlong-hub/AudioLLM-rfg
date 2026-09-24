#!/usr/bin/env bash
# Finish MiniCPM-o N=4, independent FRR, N curve, and targeted rerendering.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU BARRIER_TAG NUM_SHARDS" >&2
  exit 2
fi

gpu=$1
barrier_tag=$2
num_shards=$3
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

model=MiniCPM-o-4_5
model_path=pretrain_model/Audio/MiniCPM-o-4_5
python_bin="$root/.venvs/minicpmo45/bin/python"
funasr_python="$root/.venvs/funasr-v100/bin/python"
barrier_dir="$root/output/minicpmo45/resample-barriers/$barrier_tag"
targets="exp/d4_rerender/metrics/targets_${model}.json"
llm_model=pretrain_model/Qwen/Qwen2.5-7B-Instruct
mkdir -p "$barrier_dir" output/minicpmo45/resample-logs/"$barrier_tag"

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH=src

while (( $(find "$barrier_dir" -maxdepth 1 -type f -name 'funasr-*.done' | wc -l) < num_shards )); do
  sleep 30
done

"$python_bin" scripts/analyze/validate_minicpmo_resamples.py

"$python_bin" -u scripts/analyze/probe_resample.py \
  --model "$model_path" --items data/main600/items.jsonl --limit 200 \
  --pilot-pred exp/d1_main/predictions --facts-root exp/d1_main/facts \
  --out exp/d0_resample --llm-model "$llm_model" --llm-device cuda:0 \
  --llm-batch-size 32

"$python_bin" scripts/analyze/evaluate_frr_loo.py \
  --model-slug "$model" --pilot-pred exp/d1_main/predictions \
  --resample-root exp/d0_resample \
  --output "exp/d0_resample/frr_loo_${model}.json"
"$python_bin" scripts/analyze/evaluate_frr_curve.py \
  --model-slug "$model" --pilot-pred exp/d1_main/predictions \
  --resample-root exp/d0_resample \
  --output "exp/d0_resample/frr_curve_${model}.json"

"$python_bin" -u scripts/analyze/probe_rerender.py \
  --model "$model_path" --resample-root exp/d0_resample \
  --pilot exp/d1_main/predictions --facts-root exp/d1_main/facts \
  --items data/main600/items.jsonl --targets-out "$targets" \
  --llm-model "$llm_model" --llm-device cuda:0 --llm-batch-size 32

"$python_bin" -u scripts/facts/readback_resample.py \
  --model "$model" --root exp/d4_rerender --items data/main600/items.jsonl \
  --limit 200 --prefixes ORIG,SPEAK --channels asr1,asr2 \
  --chunk-seconds 25 --low-memory
PYTHONPATH="src:third_party/v100_compat:third_party/FunASR" \
  "$funasr_python" -u scripts/facts/readback_funasr_resample.py \
    --model "$model" --root exp/d4_rerender --prefixes ORIG,SPEAK

for mode in asr1 asr2 asr3; do
  "$python_bin" -u scripts/analyze/judge_rerender.py \
    --run-id d4_rerender --model "$model" --readback-mode "$mode" \
    --targets "$targets" \
    --out "reports/rerender_${model}_${mode}.md" \
    --metrics-out "exp/d4_rerender/metrics/rerender_judge_${model}_${mode}.json" \
    --llm-model "$llm_model" --llm-device cuda:0 --llm-batch-size 32
done

echo "[minicpmo-resample-finalize] completed=$(date --iso-8601=seconds)"
