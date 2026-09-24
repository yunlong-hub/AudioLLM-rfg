import json
import os
from pathlib import Path

from rfg.run.infer_guard import (
    completed_count,
    guard_worker,
    load_item_ids,
    process_identity,
    worker_command_matches,
)


def test_load_item_ids_uses_exact_requested_slice(tmp_path: Path) -> None:
    items = tmp_path / "items.jsonl"
    items.write_text("".join(json.dumps({"id": f"i{idx}"}) + "\n" for idx in range(5)))

    assert load_item_ids(items, 1, 3) == ["i1", "i2", "i3"]


def test_completed_count_requires_valid_json_and_nonempty_speak_wav(tmp_path: Path) -> None:
    root = tmp_path / "predictions"
    for item_id in ("ok", "missing_wav", "failed"):
        item_dir = root / item_id
        item_dir.mkdir(parents=True)
        (item_dir / "LISTEN.json").write_text(json.dumps({"text": "answer"}))
        (item_dir / "SPEAK.json").write_text(json.dumps({"text": "answer"}))

    (root / "ok" / "SPEAK.wav").write_bytes(b"R" * 45)
    (root / "failed" / "SPEAK.wav").write_bytes(b"R" * 45)
    (root / "failed" / "SPEAK.json").write_text(json.dumps({"error": "oom"}))

    assert completed_count(
        root, ["ok", "missing_wav", "failed"], ("LISTEN", "SPEAK")
    ) == 1


def test_process_identity_recognizes_current_process() -> None:
    identity = process_identity(os.getpid())

    assert identity is not None
    assert identity[0].isdigit()
    assert "pytest" in identity[1]


def test_worker_command_matches_stepaudio_without_model_argument() -> None:
    cmdline = (
        "python scripts/infer/infer_stepaudio.py --run-id omg_spoken_mqa "
        "--conditions LISTEN,SPEAK"
    )

    assert worker_command_matches(cmdline, "omg_spoken_mqa", "Step-Audio-2-mini")
    assert not worker_command_matches(cmdline, "other-run", "Step-Audio-2-mini")
    assert not worker_command_matches(cmdline, "omg_spoken_mqa", "other-model")


def test_guard_refuses_unrelated_live_pid(tmp_path: Path) -> None:
    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps({"id": "i0"}) + "\n")

    assert guard_worker(
        pid=os.getpid(),
        run_id="wrong-run",
        model_slug="wrong-model",
        items_path=items,
        offset=0,
        limit=1,
        interval=1,
        grace=1,
        out_root=tmp_path,
    ) == 2
