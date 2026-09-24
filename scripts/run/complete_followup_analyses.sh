#!/usr/bin/env bash
# 在一张卡上顺序完成依赖前序产物的事实抽取与独立 ASR 复核。
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

python_bin=${AUDIO_LLM_PYTHON:-/workspace/yunlong/anaconda3/envs/audio-llm/bin/python}

PYTHONPATH=src "$python_bin" -u scripts/facts/extract_facts.py \
  --run-id d3_7b_stack \
  --model Qwen2.5-Omni-7B \
  --with-llm \
  --llm-device cuda:0

for readback_mode in asr3 union123; do
  PYTHONPATH=src "$python_bin" -u scripts/facts/score.py \
    --run-id d3_7b_stack \
    --model Qwen2.5-Omni-7B \
    --readback-mode "$readback_mode"
  PYTHONPATH=src "$python_bin" -u scripts/analyze/probe_ef.py \
    --run-id d3_7b_stack \
    --models Qwen2.5-Omni-7B \
    --readback-mode "$readback_mode"
done

PYTHONPATH=src "$python_bin" -u scripts/analyze/judge_rerender.py \
  --model Qwen3-Omni-30B-A3B-Instruct \
  --readback-mode asr3 \
  --llm-device cuda:0

PYTHONPATH=src "$python_bin" -u scripts/facts/readback_funasr_resample.py \
  --model Qwen2.5-Omni-3B \
  --device cuda:0

PYTHONPATH=src "$python_bin" -u scripts/analyze/probe_resample.py \
  --model /workspace/yunlong/LLM/pretrain_model/Audio/Qwen2.5-Omni-3B \
  --n 3 \
  --seeds 101,202,303 \
  --out exp/d0_resample \
  --selector-asr asr1 \
  --evaluator-asr asr3 \
  --llm-device cuda:0
