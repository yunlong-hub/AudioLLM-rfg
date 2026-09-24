#!/usr/bin/env bash
# Run one complete Qwen2.5-Omni-7B shard on a pair of V100 GPUs.
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
# token2wav exceeds 32 GiB on V100 for long speech even at batch one unless the
# thinker KV cache is reduced and the talker output is bounded.
batch_size=${QWEN25_V100_BATCH_SIZE:-1}
talker_max_new_tokens=${QWEN25_V100_TALKER_MAX_NEW_TOKENS:-512}
stage_config=${QWEN25_V100_STAGE_CONFIG:-$root/configs/infer/qwen25_omni_v100.yaml}
barrier_tag=${QWEN25_V100_BARRIER_TAG:-full-20260920-n7-v1}
barrier_root="$root/output/qwen25_7b_v100/barriers/$barrier_tag"
mkdir -p "$barrier_root" "$root/output/qwen25_7b_v100/logs"

if [[ ! -s "$image" || ! -f "$model/config.json" || ! -x "$python_bin" || ! -f "$stage_config" ]]; then
  echo "missing image, model, environment, or stage config: image=$image model=$model python=$python_bin stage_config=$stage_config" >&2
  exit 1
fi

# Coordinate with the MiniCPM task by holding its per-GPU locks as well as our own.
mapfile -t lock_gpus < <(printf '%s\n' "$gpu_a" "$gpu_b" | sort -n)
lock_fds=()
for gpu in "${lock_gpus[@]}"; do
  exec {fd}>"/tmp/minicpmo45_n11_gpu${gpu}.lock"
  echo "[worker] waiting MiniCPM lock gpu=$gpu host=$(hostname)"
  flock "$fd"
  lock_fds+=("$fd")
  exec {fd}>"/tmp/qwen25_7b_v100_gpu${gpu}.lock"
  flock "$fd"
  lock_fds+=("$fd")
done

gpu_memory_used() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | sed -n "$(( $1 + 1 ))p" | tr -d ' '
}

while :; do
  used_a=$(gpu_memory_used "$gpu_a")
  used_b=$(gpu_memory_used "$gpu_b")
  if (( used_a <= 1024 && used_b <= 1024 )); then
    break
  fi
  echo "[worker] locks acquired but GPUs still busy: g${gpu_a}=${used_a}MiB g${gpu_b}=${used_b}MiB"
  sleep 30
done

export CUDA_VISIBLE_DEVICES="$gpu_a,$gpu_b"
export QWEN_ROOT="$root"
export QWEN_MODEL="$model"
export QWEN_PYTHON="$python_bin"
export QWEN_CC="$root/.venvs/vllm_omni_v100/bin/zigcc"
export QWEN_CXX="$root/.venvs/vllm_omni_v100/bin/zigcxx"
export QWEN_STAGE_CONFIG="$stage_config"
export QWEN_TALKER_MAX_NEW_TOKENS="$talker_max_new_tokens"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/triton_qwen25_7b_v100_${USER}_g${gpu_a}-${gpu_b}"

run_infer() {
  local run_id=$1 items=$2 total=$3 conditions=$4 requested_batch=$5
  local offset=$(( total * shard / num_shards ))
  local end=$(( total * (shard + 1) / num_shards ))
  local limit=$(( end - offset ))
  local tag="$(hostname -s)_g${gpu_a}-${gpu_b}_s${shard}of${num_shards}"
  export QWEN_RUN_ID="$run_id" QWEN_ITEMS="$items" QWEN_OFFSET="$offset"
  export QWEN_LIMIT="$limit" QWEN_CONDITIONS="$conditions"
  export QWEN_BATCH_SIZE="$requested_batch" QWEN_METRICS_TAG="$tag"
  apptainer exec --nv --bind /workspace:/workspace "$image" bash -lc '
    set -euo pipefail
    cd "$QWEN_ROOT"
    export PATH="$QWEN_ROOT/.venvs/vllm_omni_v100/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    export PYTHONPATH=src
    export CC="$QWEN_CC"
    export CXX="$QWEN_CXX"
    exec "$QWEN_PYTHON" -u scripts/infer/infer_vllm_omni.py \
      --model "$QWEN_MODEL" --items "$QWEN_ITEMS" --run-id "$QWEN_RUN_ID" \
      --offset "$QWEN_OFFSET" --limit "$QWEN_LIMIT" \
      --conditions "$QWEN_CONDITIONS" --batch-size "$QWEN_BATCH_SIZE" \
      --talker-max-new-tokens "$QWEN_TALKER_MAX_NEW_TOKENS" \
      --stage-configs-path "$QWEN_STAGE_CONFIG" \
      --metrics-tag "$QWEN_METRICS_TAG"
  '
}

run_with_retry() {
  local run_id=$1 items=$2 total=$3 conditions=$4
  echo "[worker] generation run=$run_id shard=$shard/$num_shards batch=$batch_size"
  if ! run_infer "$run_id" "$items" "$total" "$conditions" "$batch_size"; then
    echo "[worker] retrying missing/error records with batch=1: run=$run_id" >&2
    run_infer "$run_id" "$items" "$total" "$conditions" 1
  fi
}

run_with_retry omg_spoken_mqa \
  data/omg_benchmarks/prepared/spoken_mqa/items.jsonl 1402 LISTEN,SPEAK
run_with_retry omg_voicebench_short \
  data/omg_benchmarks/prepared/voicebench_bbh/items_short.jsonl 1000 LISTEN,SPEAK
run_with_retry d1_main data/main600/items.jsonl 600 \
  READ,LISTEN,SPEAK,ECHO,EF,EFA,EFB,EFW

printf 'host=%s gpu_a=%s gpu_b=%s completed=%s\n' \
  "$(hostname)" "$gpu_a" "$gpu_b" "$(date --iso-8601=seconds)" \
  >"$barrier_root/generation-s${shard}-of-${num_shards}.done"

echo "[worker] generation complete; waiting for $num_shards generation markers"
while (( $(find "$barrier_root" -maxdepth 1 -name "generation-s*-of-${num_shards}.done" | wc -l) < num_shards )); do
  sleep 30
done

export CUDA_VISIBLE_DEVICES="$gpu_a"
export PYTHONPATH=src
asr_python="$root/.venvs/minicpmo45/bin/python"
funasr_python="$root/.venvs/funasr-v100/bin/python"
model_slug=Qwen2.5-Omni-7B

run_dual_asr() {
  local run_id=$1 total=$2
  local offset=$(( total * shard / num_shards ))
  local end=$(( total * (shard + 1) / num_shards ))
  local limit=$(( end - offset ))
  "$asr_python" -u scripts/facts/readback.py \
    --run-id "$run_id" --model "$model_slug" --offset "$offset" --limit "$limit" \
    --device cuda:0 --channels asr1,asr2 --low-memory --chunk-seconds 25
}

run_funasr() {
  local run_id=$1 conditions=$2
  if ! env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH="src:third_party/v100_compat:third_party/FunASR" \
    "$funasr_python" -u scripts/facts/readback_funasr.py \
    --run-id "$run_id" --model "$model_slug" --conditions "$conditions" \
    --device cuda:0 --num-shards "$num_shards" --shard-index "$shard" \
    --chunk-seconds 25; then
    echo "[worker] retrying failed FunASR observations: run=$run_id" >&2
    env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      PYTHONPATH="src:third_party/v100_compat:third_party/FunASR" \
      "$funasr_python" -u scripts/facts/readback_funasr.py \
      --run-id "$run_id" --model "$model_slug" --conditions "$conditions" \
      --device cuda:0 --num-shards "$num_shards" --shard-index "$shard" \
      --chunk-seconds 25
  fi
}

run_dual_asr omg_spoken_mqa 1402
run_dual_asr omg_voicebench_short 1000
run_dual_asr d1_main 600

printf 'host=%s gpu=%s completed=%s\n' \
  "$(hostname)" "$gpu_a" "$(date --iso-8601=seconds)" \
  >"$barrier_root/dual-asr-s${shard}-of-${num_shards}.done"
echo "[worker] dual ASR complete; waiting before FunASR writes"
while (( $(find "$barrier_root" -maxdepth 1 -name "dual-asr-s*-of-${num_shards}.done" | wc -l) < num_shards )); do
  sleep 30
done

run_funasr omg_spoken_mqa SPEAK
run_funasr omg_voicebench_short SPEAK
run_funasr d1_main SPEAK,ECHO,EF,EFA,EFB,EFW

printf 'host=%s gpu=%s completed=%s\n' \
  "$(hostname)" "$gpu_a" "$(date --iso-8601=seconds)" \
  >"$barrier_root/readback-s${shard}-of-${num_shards}.done"
echo "[worker] complete shard=$shard/$num_shards completed=$(date --iso-8601=seconds)"
