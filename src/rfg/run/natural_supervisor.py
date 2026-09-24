"""Resource-aware supervisor for natural-benchmark ASR overlays."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from rfg.facts.readback_state import channel_reusable_for_chunking


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIO_LLM_PYTHON = "/workspace/yunlong/anaconda3/envs/audio-llm/bin/python"
FUNASR_PYTHON = str(PROJECT_ROOT / ".venvs" / "funasr" / "bin" / "python")
CHUNK_SECONDS = 25.0
CHUNKED_ASR_VERSIONS = {
    "asr1": "whisper-large-v3-chunk25s",
    "asr2": "seamless-m4t-v2-large-chunk25s",
    "asr3": "Fun-ASR-Nano-2512-chunk25s",
}


@dataclass(frozen=True)
class Slot:
    name: str
    host: str
    gpu: int
    run_id: str
    model: str
    max_used_mib: int = 74_000
    min_pending: int = 20
    idle_util_threshold: int = 50
    idle_min_pending: int = 1
    launch_allowed: bool = True
    allowed_modes: tuple[str, ...] = ("dual", "asr3")


SLOTS = (
    Slot("a22_gpu2_step_voice", "A22-direct", 2, "omg_voicebench_short", "Step-Audio-2-mini"),
    Slot("a22_gpu3_step_spoken", "A22-direct", 3, "omg_spoken_mqa", "Step-Audio-2-mini"),
    Slot("a31_gpu0_q30_spoken", "A31-direct", 0, "omg_spoken_mqa", "Qwen3-Omni-30B-A3B-Instruct"),
    Slot("a31_gpu1_q3_spoken", "A31-direct", 1, "omg_spoken_mqa", "Qwen2.5-Omni-3B"),
)

Q30_VOICE_SLOTS = (
    Slot(
        "a23_gpu0_q30_spoken_asr2_chunked",
        "A23-direct",
        0,
        "omg_spoken_mqa",
        "Qwen3-Omni-30B-A3B-Instruct",
        min_pending=10,
        allowed_modes=("asr2_chunked",),
    ),
    Slot(
        "a23_gpu1_q30_spoken_asr1_chunked",
        "A23-direct",
        1,
        "omg_spoken_mqa",
        "Qwen3-Omni-30B-A3B-Instruct",
        min_pending=10,
        allowed_modes=("asr1_chunked",),
    ),
    Slot(
        "a22_gpu2_q30_voice_asr3",
        "A22-direct",
        2,
        "omg_voicebench_short",
        "Qwen3-Omni-30B-A3B-Instruct",
        min_pending=10,
        allowed_modes=("asr3",),
    ),
    Slot(
        "a22_gpu3_q30_voice_asr2_chunked",
        "A22-direct",
        3,
        "omg_voicebench_short",
        "Qwen3-Omni-30B-A3B-Instruct",
        min_pending=10,
        allowed_modes=("asr2_chunked",),
    ),
    Slot(
        "a31_gpu0_q30_spoken_asr3",
        "A31-direct",
        0,
        "omg_spoken_mqa",
        "Qwen3-Omni-30B-A3B-Instruct",
        min_pending=10,
        allowed_modes=("asr3",),
    ),
    Slot(
        "a31_gpu1_q30_voice_asr1_chunked",
        "A31-direct",
        1,
        "omg_voicebench_short",
        "Qwen3-Omni-30B-A3B-Instruct",
        min_pending=10,
        allowed_modes=("asr1_chunked",),
    ),
)

SLOT_PROFILES = {
    "natural": SLOTS,
    "q30_voice": Q30_VOICE_SLOTS,
}


def readback_progress(prediction_root: Path) -> dict[str, int]:
    progress = {
        "wav": 0,
        "asr1_done": 0,
        "asr1_pending": 0,
        "asr2_done": 0,
        "asr2_pending": 0,
        "dual_done": 0,
        "dual_pending": 0,
        "asr3_done": 0,
        "asr3_pending": 0,
    }
    if not prediction_root.is_dir():
        return progress
    for json_path in prediction_root.glob("*/SPEAK.json"):
        if not json_path.with_suffix(".wav").exists():
            continue
        try:
            record = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        readback = record.get("readback") or {}
        duration = record.get("audio_duration_sec")
        current = {
            channel: channel_reusable_for_chunking(
                readback,
                channel,
                duration_sec=duration,
                chunk_seconds=CHUNK_SECONDS,
                expected_version=version,
            )
            for channel, version in CHUNKED_ASR_VERSIONS.items()
        }
        asr1_ok = current["asr1"]
        asr2_ok = current["asr2"]
        asr3_ok = current["asr3"]
        progress["wav"] += 1
        progress["asr1_done"] += int(asr1_ok)
        progress["asr2_done"] += int(asr2_ok)
        progress["dual_done"] += int(asr1_ok and asr2_ok)
        progress["asr3_done"] += int(asr3_ok)
    progress["asr1_pending"] = progress["wav"] - progress["asr1_done"]
    progress["asr2_pending"] = progress["wav"] - progress["asr2_done"]
    progress["dual_pending"] = progress["wav"] - progress["dual_done"]
    progress["asr3_pending"] = progress["wav"] - progress["asr3_done"]
    return progress


def remote_gpu_stats(slot: Slot) -> tuple[int, int] | None:
    command = [
        "ssh",
        slot.host,
        "nvidia-smi",
        "-i",
        str(slot.gpu),
        "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
        used_mib, utilization = (
            int(value.strip()) for value in result.stdout.strip().split(",")
        )
        return used_mib, utilization
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def choose_launch_mode(
    progress: dict[str, int],
    min_pending: int,
    allowed_modes: tuple[str, ...] = ("dual", "asr3"),
) -> str | None:
    if (
        "asr1_chunked" in allowed_modes
        and progress.get("asr1_pending", 0) >= min_pending
    ):
        return "asr1_chunked"
    if (
        "asr2_chunked" in allowed_modes
        and progress.get("asr2_pending", 0) >= min_pending
    ):
        return "asr2_chunked"
    if (
        "asr3" in allowed_modes
        and progress["asr3_pending"] >= 200
        and (progress["dual_pending"] < 100 or "dual" not in allowed_modes)
    ):
        return "asr3"
    if "dual" in allowed_modes and progress["dual_pending"] >= min_pending:
        return "dual"
    if "asr3" in allowed_modes and progress["asr3_pending"] >= min_pending:
        return "asr3"
    return None


def gpu_asr_lock_path(slot: Slot) -> str:
    """Return the host-local lock shared by every ASR overlay on one GPU.

    Slot-specific locks permit a fallback task and the regular supervisor slot
    to start together while the first process is still loading on CPU and has
    not raised ``memory.used`` yet.  A physical-GPU lock closes that race while
    preserving full parallelism across cards and hosts.
    """
    return f"/tmp/audio_llm_rfg_natural_asr_gpu{slot.gpu}.lock"


def remote_launch_command(
    *, lock_path: str, command: list[str], task_log: Path
) -> str:
    """Build a truthful, detached, host-local locked launch command.

    ``setsid -f flock -n ...`` detaches before the lock attempt, so SSH can
    report success even when the background process immediately loses the
    lock race.  The outer nonblocking flock below performs the observable lock
    attempt synchronously.  ``--close`` keeps its descriptor out of the
    detached child; the inner blocking flock then takes ownership as soon as
    the short outer launcher exits.
    """
    detached = (
        f"setsid -f flock {shlex.quote(lock_path)} "
        f"{' '.join(shlex.quote(value) for value in command)} "
        f">> {shlex.quote(str(task_log))} 2>&1 < /dev/null"
    )
    return (
        f"cd {shlex.quote(str(PROJECT_ROOT))} && "
        f"flock -n --close {shlex.quote(lock_path)} "
        f"sh -c {shlex.quote(detached)}"
    )


def launch_readback(slot: Slot, log_dir: Path, mode: str) -> bool:
    lock_path = gpu_asr_lock_path(slot)
    task_log = log_dir / f"{slot.name}_{mode}.log"
    if mode in ("dual", "asr1_chunked", "asr2_chunked"):
        command = [
            "env",
            f"CUDA_VISIBLE_DEVICES={slot.gpu}",
            "PYTHONPATH=src",
            AUDIO_LLM_PYTHON,
            "-u",
            "scripts/facts/readback.py",
            "--run-id",
            slot.run_id,
            "--model",
            slot.model,
            "--device",
            "cuda:0",
        ]
        if mode in ("asr1_chunked", "asr2_chunked"):
            command.extend(
                [
                    "--channels",
                    "asr1" if mode == "asr1_chunked" else "asr2",
                    "--chunk-seconds",
                    "25",
                ]
            )
            if mode == "asr2_chunked":
                command.append("--low-memory")
    elif mode == "asr3":
        command = [
            "env",
            f"CUDA_VISIBLE_DEVICES={slot.gpu}",
            "PYTHONPATH=src:third_party/FunASR",
            FUNASR_PYTHON,
            "-u",
            "scripts/facts/readback_funasr.py",
            "--run-id",
            slot.run_id,
            "--model",
            slot.model,
            "--conditions",
            "SPEAK",
            "--device",
            "cuda:0",
            "--chunk-seconds",
            "25",
        ]
    else:
        raise ValueError(f"unsupported readback mode: {mode}")
    remote = remote_launch_command(
        lock_path=lock_path,
        command=command,
        task_log=task_log,
    )
    try:
        subprocess.run(["ssh", slot.host, remote], check=True, timeout=20)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def supervise_once(
    log_dir: Path,
    *,
    slots: tuple[Slot, ...] = SLOTS,
    launch_enabled: bool = True,
) -> dict[str, object]:
    log_dir.mkdir(parents=True, exist_ok=True)
    snapshot: dict[str, object] = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "launch_enabled": launch_enabled,
        "slots": [],
    }
    for slot in slots:
        prediction_root = PROJECT_ROOT / "exp" / slot.run_id / "predictions" / slot.model
        progress = readback_progress(prediction_root)
        gpu_stats = remote_gpu_stats(slot)
        used_mib = gpu_stats[0] if gpu_stats is not None else None
        utilization_gpu = gpu_stats[1] if gpu_stats is not None else None
        effective_min_pending = slot.min_pending
        if (
            utilization_gpu is not None
            and utilization_gpu <= slot.idle_util_threshold
        ):
            effective_min_pending = slot.idle_min_pending
        launched = False
        launch_mode = None
        if (
            launch_enabled
            and slot.launch_allowed
            and used_mib is not None
            and used_mib <= slot.max_used_mib
        ):
            launch_mode = choose_launch_mode(
                progress,
                effective_min_pending,
                slot.allowed_modes,
            )
            if launch_mode:
                launched = launch_readback(slot, log_dir, launch_mode)
        snapshot["slots"].append(
            {
                **asdict(slot),
                **progress,
                "used_mib": used_mib,
                "utilization_gpu": utilization_gpu,
                "effective_min_pending": effective_min_pending,
                "launch_mode": launch_mode,
                "launched": launched,
            }
        )
    return snapshot


def cycle_sleep_seconds(interval: int, started_at: float, now: float) -> float:
    """Return the sleep needed to keep cycle start times ``interval`` apart."""
    return max(0.0, interval - (now - started_at))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--monitor-only", action="store_true")
    parser.add_argument("--profile", choices=tuple(SLOT_PROFILES), default="natural")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "experiment-extension" / "supervisor",
    )
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    event_log = args.log_dir / "natural_gpu_supervisor.jsonl"
    stop_file = args.log_dir / "STOP"
    pid_file = args.log_dir / "natural_gpu_supervisor.pid"
    pid_file.write_text(f"{os.getpid()}\n")
    try:
        while True:
            cycle_started = time.monotonic()
            snapshot = supervise_once(
                args.log_dir,
                slots=SLOT_PROFILES[args.profile],
                launch_enabled=not args.monitor_only,
            )
            with event_log.open("a") as stream:
                stream.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            print(json.dumps(snapshot, ensure_ascii=False), flush=True)
            if args.once or stop_file.exists():
                return 0
            time.sleep(
                cycle_sleep_seconds(
                    args.interval,
                    cycle_started,
                    time.monotonic(),
                )
            )
    finally:
        if pid_file.exists() and pid_file.read_text().strip() == str(os.getpid()):
            pid_file.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
