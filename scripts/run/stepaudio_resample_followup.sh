#!/usr/bin/env bash
# Wait for one physical GPU's generation shards, run dual-ASR, join a global
# barrier, run one FunASR shard, then let shard zero finalize FRR and N curves.
set -euo pipefail

if [[ $# -lt 7 || $# -gt 10 ]]; then
  echo "usage: $0 <gpu> <after-pids> <offset> <limit> <shard-index> <num-shards> <tag> [new-candidates] [seeds] [curve-counts]" >&2
  exit 2
fi

gpu=$1
after_pids=$2
offset=$3
limit=$4
shard_index=$5
num_shards=$6
tag=$7
new_candidates=${8:-3}
seeds=${9:-101,202,303}
curve_counts=${10:-1,2,4}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

pid_is_running() {
  local pid=$1 state
  [[ -r "/proc/$pid/stat" ]] || return 1
  state=$(awk '{print $3}' "/proc/$pid/stat")
  [[ "$state" != "Z" && "$state" != "X" ]]
}

IFS=',' read -r -a pid_array <<< "$after_pids"
for pid in "${pid_array[@]}"; do
  while pid_is_running "$pid"; do sleep 5; done
done

audio_python=/workspace/yunlong/anaconda3/envs/audio-llm/bin/python
funasr_python="$repo_root/.venvs/funasr/bin/python"
model=Step-Audio-2-mini
barrier="$repo_root/output/experiment-extension/readback-barriers/$tag"
mkdir -p "$barrier"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# A cheap no-op when the first pass is complete; otherwise resume any failed or
# interrupted candidate before exposing the range to ASR and the global barrier.
"$repo_root/.venvs/stepaudio/bin/python" -u \
  scripts/analyze/generate_stepaudio_resamples.py --offset "$offset" --limit "$limit" \
  --n "$new_candidates" --seeds "$seeds"

"$audio_python" -u scripts/facts/readback_resample.py \
  --model "$model" --offset "$offset" --limit "$limit" \
  --n "$new_candidates" --channels asr1,asr2 --chunk-seconds 25 --low-memory --device cuda:0
touch "$barrier/dual-$shard_index.done"
for ((index=0; index<num_shards; index++)); do
  while [[ ! -f "$barrier/dual-$index.done" ]]; do sleep 20; done
done

PYTHONPATH=src:third_party/FunASR "$funasr_python" -u \
  scripts/facts/readback_funasr_resample.py --model "$model" --device cuda:0 \
  --num-shards "$num_shards" --shard-index "$shard_index"
touch "$barrier/funasr-$shard_index.done"

if [[ "$shard_index" == "0" ]]; then
  for ((index=0; index<num_shards; index++)); do
    while [[ ! -f "$barrier/funasr-$index.done" ]]; do sleep 20; done
  done
  "$audio_python" -u scripts/analyze/probe_resample.py \
    --model "$repo_root/pretrain_model/Audio/$model" \
    --pilot-pred exp/d2_stepaudio/predictions --facts-root exp/d2_stepaudio/facts \
    --out exp/d0_resample --n "$new_candidates" --seeds "$seeds" \
    --selector-asr asr1 --evaluator-asr asr2
  "$audio_python" -u scripts/analyze/evaluate_frr_loo.py \
    --model-slug "$model" --pilot-pred exp/d2_stepaudio/predictions \
    --resample-root exp/d0_resample \
    --output exp/d0_resample/frr_loo_${model}.json
  "$audio_python" -u scripts/analyze/evaluate_frr_curve.py \
    --model-slug "$model" --pilot-pred exp/d2_stepaudio/predictions \
    --resample-root exp/d0_resample --candidate-counts "$curve_counts" \
    --output exp/d0_resample/frr_curve_${model}.json
fi
