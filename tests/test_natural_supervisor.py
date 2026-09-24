from __future__ import annotations

import json
from pathlib import Path

from rfg.run.natural_supervisor import (
    Slot,
    choose_launch_mode,
    cycle_sleep_seconds,
    gpu_asr_lock_path,
    readback_progress,
    remote_launch_command,
    supervise_once,
)


def write_prediction(
    root: Path,
    item_id: str,
    readback: dict,
    *,
    duration_sec: float = 10.0,
) -> None:
    item = root / item_id
    item.mkdir(parents=True)
    (item / "SPEAK.wav").write_bytes(b"RIFF")
    (item / "SPEAK.json").write_text(
        json.dumps({"audio_duration_sec": duration_sec, "readback": readback})
    )


def test_readback_progress_counts_channels_independently(tmp_path: Path) -> None:
    write_prediction(
        tmp_path,
        "unit_0000",
        {"asr1": "one", "asr2": "one", "asr3": "one"},
    )
    write_prediction(tmp_path, "unit_0001", {"asr1": "two"})
    write_prediction(
        tmp_path,
        "unit_0002",
        {"asr3": None, "asr3_sec": 0.5, "asr3_error": None},
    )

    assert readback_progress(tmp_path) == {
        "wav": 3,
        "asr1_done": 2,
        "asr1_pending": 1,
        "asr2_done": 1,
        "asr2_pending": 2,
        "dual_done": 1,
        "dual_pending": 2,
        "asr3_done": 2,
        "asr3_pending": 1,
    }


def test_readback_progress_marks_stale_long_audio_pending(tmp_path: Path) -> None:
    write_prediction(
        tmp_path,
        "unit_0000",
        {
            "asr1": "one",
            "asr2": "one",
            "asr3": "one",
            "asr_versions": {
                "asr1": "whisper-large-v3",
                "asr2": "seamless-m4t-v2-large",
                "asr3": "Fun-ASR-Nano-2512",
            },
        },
        duration_sec=30.0,
    )

    progress = readback_progress(tmp_path)

    assert progress["wav"] == 1
    assert progress["dual_pending"] == 1
    assert progress["asr3_pending"] == 1


def test_launch_mode_drains_large_asr3_backlog() -> None:
    assert choose_launch_mode({"dual_pending": 26, "asr3_pending": 960}, 20) == "asr3"
    assert choose_launch_mode({"dual_pending": 235, "asr3_pending": 322}, 20) == "dual"
    assert choose_launch_mode({"dual_pending": 5, "asr3_pending": 10}, 20) is None
    assert (
        choose_launch_mode(
            {"dual_pending": 48, "asr3_pending": 42},
            20,
            ("asr3",),
        )
        == "asr3"
    )


def test_launch_mode_can_target_chunked_channel_backlogs() -> None:
    progress = {
        "asr1_pending": 12,
        "asr2_pending": 14,
        "dual_pending": 14,
        "asr3_pending": 0,
    }

    assert choose_launch_mode(progress, 10, ("asr1_chunked",)) == "asr1_chunked"
    assert choose_launch_mode(progress, 10, ("asr2_chunked",)) == "asr2_chunked"
    assert choose_launch_mode(progress, 20, ("asr1_chunked",)) is None


def test_asr_lock_is_shared_by_slots_on_the_same_physical_gpu() -> None:
    regular = Slot("regular", "A23-direct", 0, "run-a", "model")
    fallback = Slot("fallback", "A23-direct", 0, "run-b", "model")
    other_gpu = Slot("other", "A23-direct", 1, "run-a", "model")

    assert gpu_asr_lock_path(regular) == gpu_asr_lock_path(fallback)
    assert gpu_asr_lock_path(regular) != gpu_asr_lock_path(other_gpu)


def test_remote_launch_checks_lock_before_detaching(tmp_path: Path) -> None:
    remote = remote_launch_command(
        lock_path="/tmp/gpu lock",
        command=["python", "worker.py", "--name", "two words"],
        task_log=tmp_path / "task log.txt",
    )

    outer_lock = remote.index("flock -n --close")
    detached = remote.index("setsid -f flock")
    assert outer_lock < detached
    assert "setsid -f flock -n" not in remote
    assert "'/tmp/gpu lock'" in remote
    assert "'two words'" in remote


def test_monitor_only_never_launches(monkeypatch, tmp_path: Path) -> None:
    slot = Slot("test", "local", 0, "run", "model")
    monkeypatch.setattr("rfg.run.natural_supervisor.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("rfg.run.natural_supervisor.remote_gpu_stats", lambda _: (0, 0))

    def fail_launch(*args, **kwargs):
        raise AssertionError("monitor-only mode must not launch work")

    monkeypatch.setattr("rfg.run.natural_supervisor.launch_readback", fail_launch)
    snapshot = supervise_once(tmp_path / "logs", slots=(slot,), launch_enabled=False)

    assert snapshot["launch_enabled"] is False
    assert snapshot["slots"][0]["launched"] is False


def test_slot_can_be_monitor_only_in_launching_profile(monkeypatch, tmp_path: Path) -> None:
    slot = Slot("test", "local", 0, "run", "model", launch_allowed=False)
    monkeypatch.setattr("rfg.run.natural_supervisor.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("rfg.run.natural_supervisor.remote_gpu_stats", lambda _: (0, 0))

    def fail_launch(*args, **kwargs):
        raise AssertionError("monitor-only slot must not launch work")

    monkeypatch.setattr("rfg.run.natural_supervisor.launch_readback", fail_launch)
    snapshot = supervise_once(tmp_path / "logs", slots=(slot,), launch_enabled=True)

    assert snapshot["launch_enabled"] is True
    assert snapshot["slots"][0]["launch_allowed"] is False
    assert snapshot["slots"][0]["launched"] is False


def test_low_utilization_uses_single_item_launch_threshold(
    monkeypatch, tmp_path: Path
) -> None:
    slot = Slot(
        "test",
        "local",
        0,
        "run",
        "model",
        min_pending=10,
        allowed_modes=("asr3",),
    )
    prediction_root = tmp_path / "exp" / "run" / "predictions" / "model"
    write_prediction(prediction_root, "unit_0000", {})
    monkeypatch.setattr("rfg.run.natural_supervisor.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "rfg.run.natural_supervisor.remote_gpu_stats", lambda _: (0, 20)
    )
    launched = []
    monkeypatch.setattr(
        "rfg.run.natural_supervisor.launch_readback",
        lambda *args: launched.append(args) or True,
    )

    snapshot = supervise_once(tmp_path / "logs", slots=(slot,))

    assert snapshot["slots"][0]["effective_min_pending"] == 1
    assert snapshot["slots"][0]["launch_mode"] == "asr3"
    assert snapshot["slots"][0]["launched"] is True
    assert launched


def test_high_utilization_keeps_batch_threshold(monkeypatch, tmp_path: Path) -> None:
    slot = Slot(
        "test",
        "local",
        0,
        "run",
        "model",
        min_pending=10,
        allowed_modes=("asr3",),
    )
    prediction_root = tmp_path / "exp" / "run" / "predictions" / "model"
    write_prediction(prediction_root, "unit_0000", {})
    monkeypatch.setattr("rfg.run.natural_supervisor.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "rfg.run.natural_supervisor.remote_gpu_stats", lambda _: (0, 80)
    )
    monkeypatch.setattr(
        "rfg.run.natural_supervisor.launch_readback",
        lambda *args: (_ for _ in ()).throw(AssertionError("must not launch")),
    )

    snapshot = supervise_once(tmp_path / "logs", slots=(slot,))

    assert snapshot["slots"][0]["effective_min_pending"] == 10
    assert snapshot["slots"][0]["launch_mode"] is None
    assert snapshot["slots"][0]["launched"] is False


def test_cycle_sleep_includes_supervision_time_in_interval() -> None:
    assert cycle_sleep_seconds(300, started_at=100.0, now=124.5) == 275.5
    assert cycle_sleep_seconds(300, started_at=100.0, now=405.0) == 0.0
