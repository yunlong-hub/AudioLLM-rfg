#!/usr/bin/env bash
# Run Step-Audio with the host-compatible, isolated Transformers 4.49 runtime.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
host_name=$(hostname -s)

case "$host_name" in
  A42*|a42*)
    default_python="$repo_root/.venvs/stepaudio_cu12/bin/python"
    ;;
  *)
    default_python="$repo_root/.venvs/stepaudio/bin/python"
    ;;
esac

python_bin=${STEP_AUDIO_PYTHON:-$default_python}
if [[ ! -x "$python_bin" ]]; then
  echo "Step-Audio interpreter is unavailable on $host_name: $python_bin" >&2
  exit 1
fi

cd "$repo_root"
exec env PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$repo_root/src" \
  "$python_bin" -u scripts/infer/infer_stepaudio.py "$@"
