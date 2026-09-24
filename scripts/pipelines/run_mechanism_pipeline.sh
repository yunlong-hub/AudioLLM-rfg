#!/usr/bin/env bash
# 机制分析流水线：回读 → 双通道抽取 → 主指标 → 分解 / 2×2 / 长度三探针。
#
# 用法：scripts/pipelines/run_mechanism_pipeline.sh <run_id> <model_slug> [gpu] [items]
#   PY / LLM_MODEL 环境变量可覆盖默认解释器与抽取用 LLM。
#
# 前置：推理产物已存在于 exp/<run_id>/predictions/（由 scripts/infer/ 生成）。
set -euo pipefail

RUN_ID="${1:?用法: run_mechanism_pipeline.sh <run_id> <model_slug> [gpu] [items]}"
MODEL="${2:?需要 model_slug}"
GPU="${3:-0}"
ITEMS="${4:-}"
PY="${PY:-/workspace/yunlong/anaconda3/envs/audio-llm/bin/python}"
LLM_MODEL="${LLM_MODEL:-/workspace/yunlong/LLM/pretrain_model/Qwen/Qwen2.5-7B-Instruct}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"

ITEM_ARG=()
[ -n "$ITEMS" ] && ITEM_ARG=(--items "$ITEMS")

echo "=== [1/6] 双 ASR 回读 ==="
$PY -u scripts/facts/readback.py --run-id "$RUN_ID" --model "$MODEL"

echo "=== [2/6] 双通道事实抽取（规则 + LLM） ==="
$PY -u scripts/facts/extract_facts.py --run-id "$RUN_ID" --model "$MODEL" \
  --with-llm --llm-device cuda:0 --llm-model "$LLM_MODEL"

echo "=== [3/6] 主指标打分 ==="
$PY -u scripts/facts/score.py --run-id "$RUN_ID" --model "$MODEL" "${ITEM_ARG[@]}"

echo "=== [4/6] 三段分解（plan vs render） ==="
$PY -u scripts/analyze/probe_ef.py --run-id "$RUN_ID" --models "$MODEL" \
  --out "reports/probe_ef_${RUN_ID}.md"

echo "=== [5/6] 2×2 机制（模态 × 内容来源） ==="
$PY -u scripts/analyze/probe_grid.py --run-id "$RUN_ID" --models "$MODEL" \
  --out "reports/mechanism_${RUN_ID}.md"

echo "=== [6/6] 长度控制 ==="
$PY -u scripts/analyze/probe_length.py --run-id "$RUN_ID" --models "$MODEL" \
  --out "reports/length_control_${RUN_ID}.md"

echo "完成：reports/{probe_ef,mechanism,length_control}_${RUN_ID}.md"
