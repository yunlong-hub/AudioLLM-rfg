#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU SHARD_INDEX" >&2
  exit 2
fi

gpu=$1
shard=$2
if [[ ! "$shard" =~ ^[0-3]$ ]]; then
  echo "SHARD_INDEX must be one of 0,1,2,3" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
launcher="$root/scripts/run/run_minicpmo45_shard.sh"

spoken_offsets=(0 351 702 1053)
spoken_limits=(351 351 351 349)
voice_offsets=(0 250 500 750)
voice_limits=(250 250 250 250)
main_offsets=(0 150 300 450)
main_limits=(150 150 150 150)

echo "[queue] host=$(hostname) gpu=$gpu shard=$shard started=$(date --iso-8601=seconds)"

"$launcher" "$gpu" omg_spoken_mqa \
  data/omg_benchmarks/prepared/spoken_mqa/items.jsonl \
  "${spoken_offsets[$shard]}" "${spoken_limits[$shard]}" LISTEN,SPEAK

"$launcher" "$gpu" omg_voicebench_short \
  data/omg_benchmarks/prepared/voicebench_bbh/items_short.jsonl \
  "${voice_offsets[$shard]}" "${voice_limits[$shard]}" LISTEN,SPEAK

"$launcher" "$gpu" d1_main \
  data/main600/items.jsonl \
  "${main_offsets[$shard]}" "${main_limits[$shard]}" \
  READ,LISTEN,SPEAK,ECHO,EF,EFA,EFB,EFW

echo "[queue] host=$(hostname) gpu=$gpu shard=$shard completed=$(date --iso-8601=seconds)"
