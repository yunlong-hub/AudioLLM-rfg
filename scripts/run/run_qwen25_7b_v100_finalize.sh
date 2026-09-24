#!/usr/bin/env bash
# Wait for all Qwen2.5-Omni-7B readback shards, validate, and compute metrics.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU NUM_SHARDS" >&2
  exit 2
fi

gpu=$1
num_shards=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

barrier_tag=${QWEN25_V100_BARRIER_TAG:-full-20260920-n7-v1}
barrier_root="$root/output/qwen25_7b_v100/barriers/$barrier_tag"
mkdir -p "$barrier_root" "$root/output/qwen25_7b_v100/logs"

echo "[finalize] waiting for $num_shards readback markers"
while (( $(find "$barrier_root" -maxdepth 1 -name "readback-s*-of-${num_shards}.done" | wc -l) < num_shards )); do
  sleep 30
done

exec {mini_fd}>"/tmp/minicpmo45_n11_gpu${gpu}.lock"
flock "$mini_fd"
exec {qwen_fd}>"/tmp/qwen25_7b_v100_gpu${gpu}.lock"
flock "$qwen_fd"

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH=src
python_bin="$root/.venvs/minicpmo45/bin/python"
funasr_python="$root/.venvs/funasr-v100/bin/python"
model=Qwen2.5-Omni-7B
llm_model=${LLM_MODEL:-/workspace/yunlong/LLM/pretrain_model/Qwen/Qwen2.5-7B-Instruct}

if [[ ! -f "$llm_model/config.json" ]]; then
  echo "[finalize] missing fact extractor: $llm_model/config.json" >&2
  exit 1
fi

# Idempotent catch-up passes repair any observations that failed in a shard worker.
for run_id in omg_spoken_mqa omg_voicebench_short d1_main; do
  for attempt in 1 2; do
    echo "[finalize] dual-ASR catch-up run=$run_id attempt=$attempt"
    "$python_bin" -u scripts/facts/readback.py \
      --run-id "$run_id" --model "$model" --device cuda:0 \
      --channels asr1,asr2 --low-memory --chunk-seconds 25
  done
done

run_funasr_catchup() {
  local run_id=$1 conditions=$2
  for attempt in 1 2; do
    echo "[finalize] FunASR catch-up run=$run_id attempt=$attempt"
    if env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      PYTHONPATH="src:third_party/v100_compat:third_party/FunASR" \
      "$funasr_python" -u scripts/facts/readback_funasr.py \
      --run-id "$run_id" --model "$model" --conditions "$conditions" \
      --device cuda:0 --chunk-seconds 25; then
      return 0
    fi
  done
  return 1
}

run_funasr_catchup omg_spoken_mqa SPEAK
run_funasr_catchup omg_voicebench_short SPEAK
run_funasr_catchup d1_main SPEAK,ECHO,EF,EFA,EFB,EFW

"$python_bin" scripts/analyze/validate_qwen25_7b_v100.py --deep-audio \
  --output output/qwen25_7b_v100/validation_generation.json
"$python_bin" scripts/analyze/validate_qwen25_7b_v100.py --require-readback \
  --output output/qwen25_7b_v100/validation_readback.json

"$python_bin" scripts/analyze/score_omg_benchmark.py \
  --items data/omg_benchmarks/prepared/spoken_mqa/items.jsonl \
  --run-id omg_spoken_mqa --model-slug "$model" \
  --conditions LISTEN,SPEAK --speech-asrs asr1,asr2,asr3
"$python_bin" scripts/analyze/score_omg_benchmark.py \
  --items data/omg_benchmarks/prepared/voicebench_bbh/items_short.jsonl \
  --run-id omg_voicebench_short --model-slug "$model" \
  --conditions LISTEN,SPEAK --speech-asrs asr1,asr2,asr3

"$python_bin" -u scripts/facts/extract_facts.py \
  --run-id d1_main --model "$model" --with-llm \
  --llm-device cuda:0 --llm-model "$llm_model"

for mode in asr1 asr2 asr3 union12 union13 union123; do
  "$python_bin" scripts/facts/score.py \
    --run-id d1_main --model "$model" --items data/main600/items.jsonl \
    --readback-mode "$mode"
  "$python_bin" scripts/analyze/probe_ef.py \
    --run-id d1_main --models "$model" --readback-mode "$mode" \
    --out "reports/probe_ef_d1_main_qwen25_7b_vllm_${mode}.md"
done

"$python_bin" scripts/analyze/probe_grid.py \
  --run-id d1_main --models "$model" \
  --out reports/mechanism_d1_main_qwen25_7b_vllm.md
"$python_bin" scripts/analyze/probe_length.py \
  --run-id d1_main --models "$model" \
  --out reports/length_control_d1_main_qwen25_7b_vllm.md
"$python_bin" scripts/analyze/probe_taxonomy.py \
  --run-id d1_main --models "$model" \
  --out reports/taxonomy_d1_main_qwen25_7b_vllm.md

printf 'host=%s gpu=%s completed=%s\n' \
  "$(hostname)" "$gpu" "$(date --iso-8601=seconds)" >"$barrier_root/final.done"
echo "[finalize] completed=$(date --iso-8601=seconds)"
