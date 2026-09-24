#!/usr/bin/env bash
# Merge parallel fact shards, then run the idempotent validation/scoring finalizer.
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
echo "[parallel finalize] waiting for $num_shards fact markers"
while (( $(find "$barrier_root" -maxdepth 1 -name "facts-s*-of-${num_shards}.done" | wc -l) < num_shards )); do
  sleep 15
done

export PYTHONPATH=src
python_bin="$root/.venvs/minicpmo45/bin/python"
"$python_bin" scripts/facts/merge_fact_shards.py \
  --run-id d1_main --model Qwen2.5-Omni-7B --num-shards "$num_shards"

exec scripts/run/run_qwen25_7b_v100_finalize.sh "$gpu" 7
