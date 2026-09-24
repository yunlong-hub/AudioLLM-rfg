#!/usr/bin/env bash
# Validate and score the completed MiniCPM-o N=8 fixed-text candidate pool.
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
barrier_dir="$root/output/minicpmo45/resample-barriers/$barrier_tag"
llm_model=pretrain_model/Qwen/Qwen2.5-7B-Instruct
mkdir -p "$barrier_dir"

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH=src

while (( $(find "$barrier_dir" -maxdepth 1 -type f -name 'funasr-*.done' | wc -l) < num_shards )); do
  sleep 30
done

"$python_bin" scripts/analyze/validate_minicpmo_resamples.py --n 7
"$python_bin" -u scripts/analyze/probe_resample.py \
  --model "$model_path" --items data/main600/items.jsonl --limit 200 \
  --n 7 --seeds 101,202,303,404,505,606,707 \
  --pilot-pred exp/d1_main/predictions --facts-root exp/d1_main/facts \
  --out exp/d0_resample --llm-model "$llm_model" --llm-device cuda:0 \
  --llm-batch-size 32
"$python_bin" scripts/analyze/evaluate_frr_loo.py \
  --model-slug "$model" --pilot-pred exp/d1_main/predictions \
  --resample-root exp/d0_resample \
  --output "exp/d0_resample/frr_loo_${model}.json"
"$python_bin" scripts/analyze/evaluate_frr_curve.py \
  --model-slug "$model" --pilot-pred exp/d1_main/predictions \
  --resample-root exp/d0_resample --candidate-counts 1,2,4,8 \
  --output "exp/d0_resample/frr_curve_${model}.json"
"$python_bin" scripts/analyze/summarize_minicpmo_extensions.py

touch "$barrier_dir/final.done"
echo "[minicpmo-resample-n8-finalize] completed=$(date --iso-8601=seconds)"
