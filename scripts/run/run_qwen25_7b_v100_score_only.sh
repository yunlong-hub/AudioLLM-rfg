#!/usr/bin/env bash
# Finish Qwen scoring after readback, validation, benchmarks, and facts are complete.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
export PYTHONPATH=src
python_bin="$root/.venvs/minicpmo45/bin/python"
model=Qwen2.5-Omni-7B
barrier_tag=${QWEN25_V100_BARRIER_TAG:-full-20260920-n7-v1}
barrier_root="$root/output/qwen25_7b_v100/barriers/$barrier_tag"
log_root="$root/output/qwen25_7b_v100/logs"
mkdir -p "$barrier_root" "$log_root"

"$python_bin" - <<'PY'
import json
for path in (
    "output/qwen25_7b_v100/validation_generation.json",
    "output/qwen25_7b_v100/validation_readback.json",
):
    with open(path) as fh:
        result = json.load(fh)
    if result.get("n_errors") != 0:
        raise SystemExit(f"validation is not clean: {path}")
PY

pids=()
for mode in asr1 asr2 asr3 union12 union13 union123; do
  (
    "$python_bin" scripts/facts/score.py \
      --run-id d1_main --model "$model" --model-only \
      --items data/main600/items.jsonl --readback-mode "$mode"
    "$python_bin" scripts/analyze/probe_ef.py \
      --run-id d1_main --models "$model" --readback-mode "$mode" \
      --out "reports/probe_ef_d1_main_qwen25_7b_vllm_${mode}.md"
  ) >"$log_root/score_qwen25_7b_${mode}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed )); then
  echo "one or more parallel score jobs failed" >&2
  exit 1
fi

"$python_bin" scripts/analyze/probe_grid.py \
  --run-id d1_main --models "$model" \
  --out reports/mechanism_d1_main_qwen25_7b_vllm.md
"$python_bin" scripts/analyze/probe_length.py \
  --run-id d1_main --models "$model" \
  --out reports/length_control_d1_main_qwen25_7b_vllm.md
"$python_bin" scripts/analyze/probe_taxonomy.py \
  --run-id d1_main --models "$model" \
  --out reports/taxonomy_d1_main_qwen25_7b_vllm.md

printf 'host=%s mode=parallel-score completed=%s\n' \
  "$(hostname)" "$(date --iso-8601=seconds)" >"$barrier_root/final.done"
echo "[score-only] completed=$(date --iso-8601=seconds)"
