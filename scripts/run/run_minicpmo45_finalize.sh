#!/usr/bin/env bash
# Validate completed outputs, extract facts, and produce all MiniCPM-o-4.5 metrics.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 GPU" >&2
  exit 2
fi

gpu=$1
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

python_bin="$root/.venvs/minicpmo45/bin/python"
model=MiniCPM-o-4_5
all_models=Qwen3-Omni-30B-A3B-Instruct,Qwen2.5-Omni-3B,MiniCPM-o-4_5
llm_model=${LLM_MODEL:-/workspace/yunlong/LLM/pretrain_model/Qwen/Qwen2.5-7B-Instruct}
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH=src

if [[ ! -f "$llm_model/config.json" ]]; then
  echo "[finalize] missing fact-extraction model: $llm_model/config.json" >&2
  exit 1
fi

echo "[finalize] generation/audio validation"
"$python_bin" scripts/analyze/validate_minicpmo45.py --deep-audio \
  --output output/minicpmo45/validation_generation.json

echo "[finalize] three-ASR coverage validation"
if ! "$python_bin" scripts/analyze/validate_minicpmo45.py \
  --require-readback --output output/minicpmo45/validation_readback.json; then
  echo "[finalize] readback coverage has empty/error observations; continuing so they remain measured outcomes" >&2
fi

echo "[finalize] dual-channel fact extraction"
"$python_bin" -u scripts/facts/extract_facts.py \
  --run-id d1_main --model "$model" --with-llm \
  --llm-device cuda:0 \
  --llm-model "$llm_model"

for mode in asr1 asr2 asr3 union12 union13 union123; do
  echo "[finalize] score/probe mode=$mode"
  "$python_bin" scripts/facts/score.py \
    --run-id d1_main --model "$model" --items data/main600/items.jsonl \
    --readback-mode "$mode"
  "$python_bin" scripts/analyze/probe_ef.py \
    --run-id d1_main --models "$all_models" --readback-mode "$mode" \
    --out "reports/probe_ef_d1_main_${mode}.md"
done

"$python_bin" scripts/analyze/probe_grid.py \
  --run-id d1_main --models "$all_models" \
  --out reports/mechanism_d1_main.md
"$python_bin" scripts/analyze/probe_length.py \
  --run-id d1_main --models "$all_models" \
  --out reports/length_control_d1_main.md
"$python_bin" scripts/analyze/probe_taxonomy.py \
  --run-id d1_main --models "$all_models" \
  --out reports/taxonomy_d1_main.md

echo "[finalize] completed=$(date --iso-8601=seconds)"
