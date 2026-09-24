#!/usr/bin/env bash
# D0 后处理流水线：回读 → 双通道抽取 → 打分 → 闸门判定。
#
# 用法：scripts/pipelines/run_d0_pipeline.sh <run_id> [model_slug] [gpu]
#   model_slug 省略时处理该 run 下全部模型；gpu 默认 0（跑前先确认该卡空闲）。
#   PY / LLM_MODEL 环境变量可覆盖默认解释器与抽取用 LLM。
#
# 前置：推理产物已存在于 exp/<run_id>/predictions/（由 scripts/infer/ 生成）。
set -euo pipefail

RUN_ID="${1:?用法: run_d0_pipeline.sh <run_id> [model_slug] [gpu]}"
MODEL="${2:-}"
GPU="${3:-0}"
PY="${PY:-/workspace/yunlong/anaconda3/envs/audio-llm/bin/python}"
LLM_MODEL="${LLM_MODEL:-/workspace/yunlong/LLM/pretrain_model/Qwen/Qwen2.5-7B-Instruct}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL_ARG=()
[ -n "$MODEL" ] && MODEL_ARG=(--model "$MODEL")

echo "=== [1/4] 双 ASR 回读 (run=$RUN_ID model=${MODEL:-ALL} gpu=$GPU) ==="
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u scripts/facts/readback.py --run-id "$RUN_ID" "${MODEL_ARG[@]}"

echo "=== [2/4] 双通道事实抽取（规则 + LLM） ==="
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u scripts/facts/extract_facts.py --run-id "$RUN_ID" \
  "${MODEL_ARG[@]}" --with-llm --llm-model "$LLM_MODEL" --llm-device "cuda:0"

echo "=== [3/4] 打分（PG/RFG/RG/RFG_EF/分项/RFG_corr + 按题聚类 bootstrap） ==="
"$PY" -u scripts/facts/score.py --run-id "$RUN_ID" "${MODEL_ARG[@]}"

echo "=== [4/4] 生成闸门报告 ==="
"$PY" -u scripts/analyze/d0_gate.py --run-id "$RUN_ID" --out reports/d0_gate.md

echo "完成：reports/d0_gate.md"
