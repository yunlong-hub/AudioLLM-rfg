#!/usr/bin/env bash
# Fill the final d1_main Q30 dual-ASR gap, then run the upper Spoken-MQA shard.
# The faster-releasing A22/A31 replicas own the remaining VoiceBench 0:350
# tail, while A31 owns Spoken 809:1100; this shard therefore starts at 1100.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${AUDIO_LLM_PYTHON:-/workspace/yunlong/anaconda3/envs/audio-llm/bin/python}
model_slug=Qwen3-Omni-30B-A3B-Instruct
model_path=/workspace/yunlong/LLM/pretrain_model/Audio/Qwen3-Omni-30B-A3B-Instruct
next_gpu_ids=${NEXT_GPU_IDS:-1,0}

cd "$repo_root"

# The queue exposes one physical GPU as cuda:0 for the two ASR models.  A
# spillover card may have completed this gap while this queue was waiting, so
# check the lightweight artifact state before loading either recognizer.
dual_pending=$(PYTHONPATH=src "$python_bin" - <<'PY'
from pathlib import Path
from rfg.run.natural_supervisor import readback_progress

root = Path("exp/d1_main/predictions/Qwen3-Omni-30B-A3B-Instruct")
print(readback_progress(root)["dual_pending"])
PY
)
if (( dual_pending > 0 )); then
  PYTHONPATH=src "$python_bin" -u scripts/facts/readback.py \
    --run-id d1_main --model "$model_slug" --device cuda:0
else
  echo "[queue] d1_main dual ASR already complete; skipping recognizer load"
fi

# The same reverse card order keeps the second Q30 instance's active stages
# offset from the full Spoken worker.
exec env CUDA_VISIBLE_DEVICES="$next_gpu_ids" PYTHONPATH=src \
  "$python_bin" -u scripts/infer/infer.py \
  --model "$model_path" \
  --items data/omg_benchmarks/prepared/spoken_mqa/items.jsonl \
  --run-id omg_spoken_mqa --conditions LISTEN,SPEAK \
  --offset 1100 --limit 302 \
  --max-new-tokens 512 --talker-max-new-tokens 2048 --device-map auto
