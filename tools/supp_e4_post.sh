#!/usr/bin/env bash
# E4 收尾：等生成结束 → 最后一轮回读（Whisper+Seamless）→ 事实抽取与打分
set -uo pipefail
ROOT=/workspace/yunlong/LLM/AudioLLM-rfg
cd "$ROOT"
PY=/workspace/yunlong/anaconda3/envs/audio-llm/bin/python
RB=.venvs/audio_llm_cu12/bin/python
export PYTHONPATH=src:. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

wait_gen() {
  local model="$1" want=720
  for _ in $(seq 1 240); do
    n=$(find "exp/d5_vb_resample/$model" -name 'R*.json' 2>/dev/null | wc -l)
    running=$(pgrep -f "generate_vllm_omni.*$model" | wc -l)
    if [ "$n" -ge "$want" ] || [ "$running" -eq 0 ]; then
      echo "[post] $model generation settled: R-files=$n running=$running"
      return 0
    fi
    sleep 30
  done
  echo "[post] $model wait timed out"
}

for model in Qwen2.5-Omni-3B Qwen2.5-Omni-7B; do
  wait_gen "$model"
done

for model in Qwen2.5-Omni-3B Qwen2.5-Omni-7B; do
  echo "[post] final readback $model"
  CUDA_VISIBLE_DEVICES=0 $RB -u scripts/facts/readback_resample.py \
    --model "$model" --root exp/d5_vb_resample \
    --items data/omg_benchmarks/prepared/voicebench_bbh/items_subset300.jsonl \
    --offset 0 --limit 240 --n 3 --channels asr1,asr2 --device cuda:0 \
    --chunk-seconds 25 --low-memory > "logs/supp/e4_rb_final_${model}.log" 2>&1
  echo "[post] readback done $model rc=$?"
done

for model in Qwen2.5-Omni-3B Qwen2.5-Omni-7B; do
  echo "[post] scoring $model"
  CUDA_VISIBLE_DEVICES=0 $PY -u tools/supp_e4_e2e.py \
    --model-slug "$model" --resample-root exp/d5_vb_resample \
    > "logs/supp/e4_score_${model}.log" 2>&1
  echo "[post] score done $model rc=$?"
done
echo "[post] ALL DONE $(date -u +%FT%TZ)"
