#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <gpu> <spoken|voice> <shard-index>" >&2
  exit 2
fi

gpu=$1
group=$2
shard=$3
runner=scripts/run/chunked_dual_readback_shard.sh

case "$group:$shard" in
  spoken:0) q_offset=0;   q_limit=180; step_offset=0;   step_limit=46 ;;
  spoken:1) q_offset=180; q_limit=180; step_offset=46;  step_limit=46 ;;
  spoken:2) q_offset=360; q_limit=180; step_offset=92;  step_limit=46 ;;
  spoken:3) q_offset=540; q_limit=180; step_offset=138; step_limit=46 ;;
  voice:0)  q_offset=0;   q_limit=500; step_offset=0;   step_limit=500 ;;
  voice:1)  q_offset=500; q_limit=500; step_offset=500; step_limit=500 ;;
  *)
    echo "unsupported group/shard: $group/$shard" >&2
    exit 2
    ;;
esac

if [[ "$group" == "spoken" ]]; then
  "$runner" "$gpu" omg_spoken_mqa Qwen2.5-Omni-3B "$q_offset" "$q_limit"
  "$runner" "$gpu" omg_spoken_mqa Step-Audio-2-mini "$step_offset" "$step_limit"
else
  "$runner" "$gpu" omg_voicebench_short Qwen2.5-Omni-3B "$q_offset" "$q_limit"
  "$runner" "$gpu" omg_voicebench_short Step-Audio-2-mini "$step_offset" "$step_limit"
fi
