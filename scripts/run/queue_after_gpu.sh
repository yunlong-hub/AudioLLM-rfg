#!/usr/bin/env bash
# Wait for known jobs and for their GPU(s) to become free, then run the command.
# Usage: queue_after_gpu.sh GPU_IDS AFTER_PIDS [MAX_USED_MIB] -- COMMAND ...
# AFTER_PIDS accepts one PID, comma-separated PIDs, or 0 to skip the job gate.
set -euo pipefail

gpu_ids=${1:?comma-separated GPU ids required}
after_pids=${2:?PID list to wait for required; use 0 to skip}
max_used_mib=${3:-1024}
shift 3
if [[ ${1:-} != "--" ]]; then
  echo "expected -- before command" >&2
  exit 2
fi
shift
if [[ $# -eq 0 ]]; then
  echo "command required" >&2
  exit 2
fi

pid_is_running() {
  local pid=$1 state
  [[ -r "/proc/$pid/stat" ]] || return 1
  state=$(awk '{print $3}' "/proc/$pid/stat")
  [[ "$state" != "Z" && "$state" != "X" ]]
}

IFS=',' read -r -a pid_array <<< "$after_pids"
for after_pid in "${pid_array[@]}"; do
  if [[ "$after_pid" != "0" ]]; then
    while pid_is_running "$after_pid"; do
      sleep 5
    done
  fi
done

IFS=',' read -r -a gpu_array <<< "$gpu_ids"
while true; do
  all_free=1
  for gpu in "${gpu_array[@]}"; do
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used > max_used_mib )); then
      all_free=0
      break
    fi
  done
  if (( all_free )); then
    break
  fi
  sleep 5
done

echo "[$(date --iso-8601=seconds)] starting on CUDA_VISIBLE_DEVICES=$gpu_ids: $*" >&2
exec env CUDA_VISIBLE_DEVICES="$gpu_ids" "$@"
