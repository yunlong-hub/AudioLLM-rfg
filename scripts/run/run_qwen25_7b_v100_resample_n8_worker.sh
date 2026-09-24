#!/usr/bin/env bash
# Extend one Qwen2.5-Omni-7B resampling shard from N=4 to N=8.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU_A GPU_B SHARD_INDEX NUM_SHARDS" >&2
  exit 2
fi
gpu_a=$1
gpu_b=$2
shard=$3
num_shards=$4
if (( gpu_a == gpu_b || num_shards < 1 || shard < 0 || shard >= num_shards )); then
  echo "require distinct GPUs and 0 <= SHARD_INDEX < NUM_SHARDS" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
image=${QWEN25_V100_IMAGE:-/workspace/yunlong/apptainer/images/codex_ssh.sif}
model=${QWEN25_7B_MODEL:-/workspace/yunlong/LLM/pretrain_model/Qwen/Qwen2.5-Omni-7B/Qwen2.5-Omni-7B}
python_bin="$root/.venvs/vllm_omni_v100/bin/python"
asr_python="$root/.venvs/minicpmo45/bin/python"
funasr_python="$root/.venvs/funasr-v100/bin/python"
stage_config=${QWEN25_V100_STAGE_CONFIG:-$root/configs/infer/qwen25_omni_v100.yaml}
barrier_tag=${QWEN25_RESAMPLE_BARRIER_TAG:-resample-20260920-n8-v1}
barrier_root="$root/output/qwen25_7b_v100/barriers/$barrier_tag"
mkdir -p "$barrier_root" "$root/output/qwen25_7b_v100/logs"

mapfile -t lock_gpus < <(printf '%s\n' "$gpu_a" "$gpu_b" | sort -n)
for gpu in "${lock_gpus[@]}"; do
  exec {fd}>"/tmp/minicpmo45_n11_gpu${gpu}.lock"
  echo "[resample n8 shard=$shard] waiting MiniCPM lock gpu=$gpu host=$(hostname)"
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
  echo "[resample n8 shard=$shard] GPUs busy: g${gpu_a}=${used_a}MiB g${gpu_b}=${used_b}MiB"
  sleep 30
done

total=200
offset=$(( total * shard / num_shards ))
end=$(( total * (shard + 1) / num_shards ))
limit=$(( end - offset ))
tag="$(hostname -s)_g${gpu_a}-${gpu_b}_s${shard}of${num_shards}_n8"
export CUDA_VISIBLE_DEVICES="$gpu_a,$gpu_b"
export QWEN_ROOT="$root" QWEN_MODEL="$model" QWEN_PYTHON="$python_bin"
export QWEN_CC="$root/.venvs/vllm_omni_v100/bin/zigcc"
export QWEN_CXX="$root/.venvs/vllm_omni_v100/bin/zigcxx"
export QWEN_STAGE_CONFIG="$stage_config" QWEN_OFFSET="$offset" QWEN_LIMIT="$limit"
export QWEN_METRICS_TAG="$tag"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/triton_qwen25_7b_resample_${USER}_g${gpu_a}-${gpu_b}_n8"

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
      --offset "$QWEN_OFFSET" --limit "$QWEN_LIMIT" \
      --seeds 404,505,606,707 --candidate-offset 3 \
      --batch-size 1 --talker-max-new-tokens 512 \
      --stage-configs-path "$QWEN_STAGE_CONFIG" --metrics-tag "$QWEN_METRICS_TAG"
  '
}
if ! run_generation; then
  echo "[resample n8 shard=$shard] retrying missing candidates" >&2
  run_generation
fi
printf 'host=%s gpu_a=%s gpu_b=%s completed=%s\n' \
  "$(hostname)" "$gpu_a" "$gpu_b" "$(date --iso-8601=seconds)" \
  >"$barrier_root/generation-s${shard}-of-${num_shards}.done"

while (( $(find "$barrier_root" -maxdepth 1 -name "generation-s*-of-${num_shards}.done" | wc -l) < num_shards )); do
  sleep 20
done

export CUDA_VISIBLE_DEVICES="$gpu_a" PYTHONPATH=src
"$asr_python" -u scripts/facts/readback_resample.py \
  --model Qwen2.5-Omni-7B --items data/pilot/items.jsonl \
  --offset "$offset" --limit "$limit" --n 7 --prefixes R3,R4,R5,R6 \
  --channels asr1,asr2 --device cuda:0 --chunk-seconds 25 --low-memory
printf 'host=%s gpu=%s completed=%s\n' \
  "$(hostname)" "$gpu_a" "$(date --iso-8601=seconds)" \
  >"$barrier_root/dual-asr-s${shard}-of-${num_shards}.done"

while (( $(find "$barrier_root" -maxdepth 1 -name "dual-asr-s*-of-${num_shards}.done" | wc -l) < num_shards )); do
  sleep 20
done

run_funasr() {
  env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH="src:third_party/v100_compat:third_party/FunASR" \
    "$funasr_python" -u scripts/facts/readback_funasr_resample.py \
    --model Qwen2.5-Omni-7B --prefixes R3,R4,R5,R6 --device cuda:0 \
    --num-shards "$num_shards" --shard-index "$shard"
}
if ! run_funasr; then
  echo "[resample n8 shard=$shard] retrying missing FunASR readbacks" >&2
  run_funasr
fi
printf 'host=%s gpu=%s completed=%s\n' \
  "$(hostname)" "$gpu_a" "$(date --iso-8601=seconds)" \
  >"$barrier_root/readback-s${shard}-of-${num_shards}.done"
echo "[resample n8 shard=$shard] complete=$(date --iso-8601=seconds)"
