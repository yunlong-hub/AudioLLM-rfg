#!/usr/bin/env bash
# Extract facts and compute independent FRR metrics after all readback shards.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU NUM_SHARDS" >&2
  exit 2
fi
gpu=$1
num_shards=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
barrier_tag=${QWEN25_RESAMPLE_BARRIER_TAG:-resample-20260920-n4-v1}
barrier_root="$root/output/qwen25_7b_v100/barriers/$barrier_tag"

while (( $(find "$barrier_root" -maxdepth 1 -name "readback-s*-of-${num_shards}.done" | wc -l) < num_shards )); do
  sleep 20
done

exec {mini_fd}>"/tmp/minicpmo45_n11_gpu${gpu}.lock"
flock "$mini_fd"
exec {qwen_fd}>"/tmp/qwen25_7b_v100_gpu${gpu}.lock"
flock "$qwen_fd"
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src
python_bin="$root/.venvs/minicpmo45/bin/python"
model_slug=Qwen2.5-Omni-7B
model_path=/workspace/yunlong/LLM/pretrain_model/Qwen/Qwen2.5-Omni-7B/Qwen2.5-Omni-7B

"$python_bin" -u scripts/analyze/probe_resample.py \
  --model "$model_path" --n 3 --seeds 101,202,303 \
  --pilot-pred exp/d3_7b_stack/predictions \
  --facts-root exp/d3_7b_stack/facts --out exp/d0_resample \
  --selector-asr asr1 --evaluator-asr asr2 --llm-device cuda:0
"$python_bin" scripts/analyze/evaluate_frr_loo.py \
  --model-slug "$model_slug" --pilot-pred exp/d3_7b_stack/predictions \
  --resample-root exp/d0_resample \
  --output "exp/d0_resample/frr_loo_${model_slug}.json"
"$python_bin" scripts/analyze/evaluate_frr_curve.py \
  --model-slug "$model_slug" --pilot-pred exp/d3_7b_stack/predictions \
  --resample-root exp/d0_resample --candidate-counts 1,2,4 \
  --output "exp/d0_resample/frr_curve_${model_slug}.json"
"$python_bin" scripts/analyze/probe_randomness.py \
  --run-id d0_resample --model "$model_slug" \
  --pilot exp/d3_7b_stack/predictions --facts-root exp/d3_7b_stack/facts \
  --out reports/randomness_qwen25_7b_vllm.md \
  --metrics-out exp/d0_resample/randomness_Qwen2.5-Omni-7B.json \
  --llm-device cuda:0

printf 'host=%s gpu=%s completed=%s\n' \
  "$(hostname)" "$gpu" "$(date --iso-8601=seconds)" >"$barrier_root/final.done"
echo "[resample finalize] completed=$(date --iso-8601=seconds)"
