"""Stop an open-ended inference worker after its assigned range is complete.

This guard is used when a legacy/full-range worker was started before the
dataset was partitioned.  It prevents that worker from entering ranges now
owned by faster, disjoint workers while preserving resumable artifacts.
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path


def process_identity(pid: int) -> tuple[str, str] | None:
    """Return Linux start ticks and NUL-separated command line for *pid*."""
    proc = Path("/proc") / str(pid)
    try:
        # ``comm`` is parenthesized and may contain spaces, so split only after
        # its closing parenthesis.  starttime is field 22, i.e. index 19 once
        # fields 1-2 have been removed.
        stat_tail = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    if stat_tail[0] in {"Z", "X"}:
        return None
    return stat_tail[19], cmdline


def load_item_ids(items_path: Path, offset: int, limit: int) -> list[str]:
    rows = [json.loads(line) for line in items_path.read_text().splitlines() if line.strip()]
    selected = rows[offset : offset + limit]
    if len(selected) != limit:
        raise ValueError(
            f"requested range {offset}:{offset + limit} has only {len(selected)} items"
        )
    return [str(row["id"]) for row in selected]


def condition_complete(item_dir: Path, condition: str) -> bool:
    record_path = item_dir / f"{condition}.json"
    try:
        record = json.loads(record_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if record.get("error"):
        return False
    if condition == "SPEAK":
        wav_path = item_dir / "SPEAK.wav"
        try:
            if wav_path.stat().st_size <= 44:
                return False
        except OSError:
            return False
    return True


def completed_count(prediction_root: Path, item_ids: list[str], conditions: tuple[str, ...]) -> int:
    return sum(
        all(condition_complete(prediction_root / item_id, cond) for cond in conditions)
        for item_id in item_ids
    )


def worker_command_matches(cmdline: str, run_id: str, model_slug: str) -> bool:
    """Return whether *cmdline* is an inference worker owned by this run/model."""
    if "--run-id" not in cmdline or run_id not in cmdline:
        return False
    if "scripts/infer/infer.py" in cmdline:
        return model_slug in cmdline
    return (
        model_slug == "Step-Audio-2-mini"
        and "scripts/infer/infer_stepaudio.py" in cmdline
    )


def guard_worker(
    *,
    pid: int,
    run_id: str,
    model_slug: str,
    items_path: Path,
    offset: int,
    limit: int,
    conditions: tuple[str, ...] = ("LISTEN", "SPEAK"),
    interval: int = 60,
    grace: int = 60,
    out_root: Path = Path("exp"),
) -> int:
    """Watch exact artifact coverage and terminate the matching worker at the boundary."""
    identity = process_identity(pid)
    if identity is None:
        print(f"[guard] pid={pid} is not running", flush=True)
        return 3
    start_ticks, cmdline = identity
    if not worker_command_matches(cmdline, run_id, model_slug):
        print(f"[guard] refusing pid={pid}; unexpected command: {cmdline}", flush=True)
        return 2

    item_ids = load_item_ids(items_path, offset, limit)
    prediction_root = out_root / run_id / "predictions" / model_slug
    previous = -1
    while True:
        current_identity = process_identity(pid)
        if current_identity is None:
            print(f"[guard] pid={pid} exited before boundary", flush=True)
            return 3
        if current_identity[0] != start_ticks or current_identity[1] != cmdline:
            print(f"[guard] refusing reused/changed pid={pid}", flush=True)
            return 2

        done = completed_count(prediction_root, item_ids, conditions)
        if done != previous:
            print(
                f"[guard] pid={pid} range={offset}:{offset + limit} "
                f"complete={done}/{limit}",
                flush=True,
            )
            previous = done
        if done == limit:
            break
        time.sleep(interval)

    # Revalidate immediately before signalling so PID reuse can never target an
    # unrelated process.
    if process_identity(pid) != (start_ticks, cmdline):
        print(f"[guard] pid={pid} changed before SIGTERM; refusing", flush=True)
        return 2
    print(f"[guard] boundary complete; sending SIGTERM to pid={pid}", flush=True)
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if process_identity(pid) is None:
            print(f"[guard] pid={pid} stopped cleanly", flush=True)
            return 0
        time.sleep(1)
    print(f"[guard] pid={pid} did not stop within {grace}s; no SIGKILL sent", flush=True)
    return 4
