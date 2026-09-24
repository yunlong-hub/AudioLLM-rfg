#!/usr/bin/env bash
# Precompute one fixed-seed candidate over a non-overlapping item slice.
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 GPU_A GPU_B OFFSET LIMIT CANDIDATE_INDEX SEED TAG" >&2
  exit 2
fi
gpu_a=$1
gpu_b=$2
offset=$3
limit=$4
candidate=$5
seed=$6
tag=$7
if (( gpu_a == gpu_b || offset < 0 || limit < 1 || candidate < 0 )); then
  echo "require distinct GPUs, OFFSET >= 0, LIMIT >= 1, and CANDIDATE_INDEX >= 0" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
image=${QWEN25_V100_IMAGE:-/workspace/yunlong/apptainer/images/codex_ssh.sif}
model=${QWEN25_7B_MODEL:-/workspace/yunlong/LLM/pretrain_model/Qwen/Qwen2.5-Omni-7B/Qwen2.5-Omni-7B}
python_bin="$root/.venvs/vllm_omni_v100/bin/python"
stage_config=${QWEN25_V100_STAGE_CONFIG:-$root/configs/infer/qwen25_omni_v100.yaml}

mapfile -t lock_gpus < <(printf '%s\n' "$gpu_a" "$gpu_b" | sort -n)
for gpu in "${lock_gpus[@]}"; do
  exec {fd}>"/tmp/minicpmo45_n11_gpu${gpu}.lock"
  flock "$fd"
  exec {fd}>"/tmp/qwen25_7b_v100_gpu${gpu}.lock"
  flock "$fd"
done

gpu_memory_used() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | sed -n "$(( $1 + 1 ))p" | tr -d ' '
}
while :; do
  used_a=$(gpu_memory_used "$gpu_a")
  used_b=$(gpu_memory_used "$gpu_b")
  if (( used_a <= 1024 && used_b <= 1024 )); then break; fi
  echo "[resample slice=$tag] GPUs busy: g${gpu_a}=${used_a}MiB g${gpu_b}=${used_b}MiB"
  sleep 30
done

export CUDA_VISIBLE_DEVICES="$gpu_a,$gpu_b"
export QWEN_ROOT="$root" QWEN_MODEL="$model" QWEN_PYTHON="$python_bin"
export QWEN_CC="$root/.venvs/vllm_omni_v100/bin/zigcc"
export QWEN_CXX="$root/.venvs/vllm_omni_v100/bin/zigcxx"
export QWEN_STAGE_CONFIG="$stage_config" QWEN_OFFSET="$offset" QWEN_LIMIT="$limit"
export QWEN_CANDIDATE="$candidate" QWEN_SEED="$seed" QWEN_METRICS_TAG="$tag"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/triton_qwen25_7b_resample_${USER}_${tag}"

run_generation() {
  apptainer exec --nv --bind /workspace:/workspace "$image" bash -lc '
    set -euo pipefail
    cd "$QWEN_ROOT"
    export PATH="$QWEN_ROOT/.venvs/vllm_omni_v100/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    export PYTHONPATH=src:.
    export CC="$QWEN_CC" CXX="$QWEN_CXX"
    exec "$QWEN_PYTHON" -u scripts/analyze/generate_vllm_omni_resamples.py \
      --model "$QWEN_MODEL" --items data/pilot/items.jsonl \
      --pilot-pred exp/d3_7b_stack/predictions --out-root exp/d0_resample \
      --offset "$QWEN_OFFSET" --limit "$QWEN_LIMIT" --seeds "$QWEN_SEED" \
      --candidate-offset "$QWEN_CANDIDATE" --batch-size 1 \
      --talker-max-new-tokens 512 --stage-configs-path "$QWEN_STAGE_CONFIG" \
      --metrics-tag "$QWEN_METRICS_TAG"
  '
}
if ! run_generation; then
  echo "[resample slice=$tag] retrying missing candidates" >&2
  run_generation
fi
echo "[resample slice=$tag] complete=$(date --iso-8601=seconds)"
