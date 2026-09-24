#!/usr/bin/env bash
# 推理启动器：按 run / 模型 / 分片 / 环境 启动条件推理。
#
# 用法：
#   scripts/pipelines/run_infer.sh <env> <model_path> <run_id> [conditions] [offset] [limit] [gpu]
#
#   env          audio-llm（默认，py3.12/cu130）| llm_a42（py3.12/cu118，仅 A42 节点）
#   model_path   模型目录绝对路径
#   run_id       产物写入 exp/<run_id>/predictions/
#   conditions   逗号分隔；留空 "-" 表示全部条件
#   offset/limit 分片参数（共享 NFS 上并行分片常慢于单进程，非必要不分片）
#   gpu          默认 0；跑前务必确认该卡空闲（nvidia-smi）
#
# 例：
#   scripts/pipelines/run_infer.sh audio-llm /path/Qwen3-Omni-30B-A3B-Instruct d0_pilot - 0 100 3
#   scripts/pipelines/run_infer.sh llm_a42   /path/Qwen2.5-Omni-3B          d1_main  - 300 300 0
set -euo pipefail

ENV_NAME="${1:?用法: run_infer.sh <env> <model_path> <run_id> [conditions] [offset] [limit] [gpu]}"
MODEL_PATH="${2:?需要模型路径}"
RUN_ID="${3:?需要 run_id}"
CONDITIONS="${4:--}"
OFFSET="${5:-0}"
LIMIT="${6:-0}"
GPU="${7:-0}"

case "$ENV_NAME" in
  audio-llm) PY="${PY:-/workspace/yunlong/anaconda3/envs/audio-llm/bin/python}" ;;
  llm_a42)   PY="${PY:-/workspace/yunlong/LLM/AudioLLM-rfg/.venvs/audio_llm_cu12/bin/python}" ;;
  *)         echo "未知环境: $ENV_NAME（可选 audio-llm | llm_a42）" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ARG=(--model "$MODEL_PATH" --run-id "$RUN_ID")
[ "$CONDITIONS" != "-" ] && [ -n "$CONDITIONS" ] && ARG+=(--conditions "$CONDITIONS")
[ "$OFFSET" != "0" ] && ARG+=(--offset "$OFFSET")
[ "$LIMIT"  != "0" ] && ARG+=(--limit "$LIMIT")

echo "=== 推理 env=$ENV_NAME gpu=$GPU run=$RUN_ID conditions=$CONDITIONS offset=$OFFSET limit=${LIMIT:-ALL} ==="
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u scripts/infer/infer.py "${ARG[@]}"
