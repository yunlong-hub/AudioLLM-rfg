#!/usr/bin/env bash
# Extract facts and compute the independent Qwen2.5-Omni-7B N=8 FRR metrics.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU NUM_SHARDS" >&2
  exit 2
fi
gpu=$1
num_shards=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
barrier_tag=${QWEN25_RESAMPLE_BARRIER_TAG:-resample-20260920-n8-v1}
barrier_root="$root/output/qwen25_7b_v100/barriers/$barrier_tag"

while (( $(find "$barrier_root" -maxdepth 1 -name "readback-s*-of-${num_shards}.done" | wc -l) < num_shards )); do
  sleep 20
done

python_bin="$root/.venvs/minicpmo45/bin/python"
model_slug=Qwen2.5-Omni-7B
model_path=/workspace/yunlong/LLM/pretrain_model/Qwen/Qwen2.5-Omni-7B/Qwen2.5-Omni-7B

"$python_bin" - <<'PY'
import json
from pathlib import Path

root = Path("exp/d0_resample/Qwen2.5-Omni-7B")
paths = sorted(root.glob("*/R[0-6].json"))
missing = []
for path in paths:
    record = json.loads(path.read_text())
    if not path.with_suffix(".wav").is_file():
        missing.append(f"{path}:wav")
    for channel in ("asr1", "asr2", "asr3"):
        if not record.get(channel):
            missing.append(f"{path}:{channel}")
if len(paths) != 1400 or missing:
    raise SystemExit(
        f"incomplete Qwen7 N=8 pool: records={len(paths)}/1400 "
        f"missing={len(missing)} examples={missing[:20]}"
    )
print("validated Qwen7 N=8 pool: 1400 records, three ASR channels")
PY

exec {mini_fd}>"/tmp/minicpmo45_n11_gpu${gpu}.lock"
flock "$mini_fd"
exec {qwen_fd}>"/tmp/qwen25_7b_v100_gpu${gpu}.lock"
flock "$qwen_fd"
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src

"$python_bin" -u scripts/analyze/probe_resample.py \
  --model "$model_path" --n 7 --seeds 101,202,303,404,505,606,707 \
  --pilot-pred exp/d3_7b_stack/predictions \
  --facts-root exp/d3_7b_stack/facts --out exp/d0_resample \
  --selector-asr asr1 --evaluator-asr asr2 --llm-device cuda:0
"$python_bin" scripts/analyze/evaluate_frr_loo.py \
  --model-slug "$model_slug" --pilot-pred exp/d3_7b_stack/predictions \
  --resample-root exp/d0_resample \
  --output "exp/d0_resample/frr_loo_${model_slug}.json"
"$python_bin" scripts/analyze/evaluate_frr_curve.py \
  --model-slug "$model_slug" --pilot-pred exp/d3_7b_stack/predictions \
  --resample-root exp/d0_resample --candidate-counts 1,2,4,8 \
  --output "exp/d0_resample/frr_curve_${model_slug}.json"
"$python_bin" scripts/analyze/probe_randomness.py \
  --run-id d0_resample --model "$model_slug" \
  --pilot exp/d3_7b_stack/predictions --facts-root exp/d3_7b_stack/facts \
  --out reports/randomness_qwen25_7b_vllm.md \
  --metrics-out exp/d0_resample/randomness_Qwen2.5-Omni-7B.json \
  --llm-device cuda:0

printf 'host=%s gpu=%s completed=%s\n' \
  "$(hostname)" "$gpu" "$(date --iso-8601=seconds)" >"$barrier_root/final.done"
echo "[resample n8 finalize] completed=$(date --iso-8601=seconds)"
