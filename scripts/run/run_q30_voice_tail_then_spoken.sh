#!/usr/bin/env bash
# Reuse one released two-GPU Q30 replica for a disjoint VoiceBench tail, then
# immediately continue with a disjoint Spoken-MQA shard on the same GPU pair.
set -euo pipefail

gpu_ids=${1:?comma-separated physical GPU ids required}
voice_offset=${2:?VoiceBench offset required}
voice_limit=${3:?VoiceBench limit required}
spoken_offset=${4:?Spoken-MQA offset required}
spoken_limit=${5:?Spoken-MQA limit required}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${AUDIO_LLM_PYTHON:-/workspace/yunlong/anaconda3/envs/audio-llm/bin/python}
model_path=/workspace/yunlong/LLM/pretrain_model/Audio/Qwen3-Omni-30B-A3B-Instruct

cd "$repo_root"

env CUDA_VISIBLE_DEVICES="$gpu_ids" PYTHONPATH=src \
  "$python_bin" -u scripts/infer/infer.py \
    --model "$model_path" \
    --items data/omg_benchmarks/prepared/voicebench_bbh/items_short.jsonl \
    --run-id omg_voicebench_short \
    --conditions LISTEN,SPEAK \
    --offset "$voice_offset" \
    --limit "$voice_limit" \
    --max-new-tokens 512 \
    --talker-max-new-tokens 2048 \
    --device-map auto

exec env CUDA_VISIBLE_DEVICES="$gpu_ids" PYTHONPATH=src \
  "$python_bin" -u scripts/infer/infer.py \
    --model "$model_path" \
    --items data/omg_benchmarks/prepared/spoken_mqa/items.jsonl \
    --run-id omg_spoken_mqa \
    --conditions LISTEN,SPEAK \
    --offset "$spoken_offset" \
    --limit "$spoken_limit" \
    --max-new-tokens 512 \
    --talker-max-new-tokens 2048 \
    --device-map auto
