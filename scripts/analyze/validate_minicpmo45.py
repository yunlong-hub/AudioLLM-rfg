#!/usr/bin/env python3
"""Validate MiniCPM-o-4.5 generation and optional three-ASR coverage."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL = "MiniCPM-o-4_5"
SPECS = (
    (
        "omg_spoken_mqa",
        "data/omg_benchmarks/prepared/spoken_mqa/items.jsonl",
        ("LISTEN", "SPEAK"),
        ("SPEAK",),
    ),
    (
        "omg_voicebench_short",
        "data/omg_benchmarks/prepared/voicebench_bbh/items_short.jsonl",
        ("LISTEN", "SPEAK"),
        ("SPEAK",),
    ),
    (
        "d1_main",
        "data/main600/items.jsonl",
        ("READ", "LISTEN", "SPEAK", "ECHO", "EF", "EFA", "EFB", "EFW"),
        ("SPEAK", "ECHO", "EF", "EFA", "EFB", "EFW"),
    ),
)


def load_ids(path: Path) -> list[str]:
    with path.open() as handle:
        return [json.loads(line)["id"] for line in handle if line.strip()]


def validate_audio(path: Path, deep: bool) -> str | None:
    try:
        info = sf.info(path)
        if info.samplerate != 24000:
            return f"sample_rate={info.samplerate}"
        if info.frames <= 0 or info.channels <= 0:
            return f"invalid_shape=frames:{info.frames},channels:{info.channels}"
        if deep:
            for block in sf.blocks(path, blocksize=1_000_000, dtype="float32"):
                if not np.isfinite(block).all():
                    return "non_finite_samples"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--require-readback", action="store_true")
    parser.add_argument("--deep-audio", action="store_true")
    parser.add_argument(
        "--output", default="output/minicpmo45/validation.json"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    failures: list[str] = []
    runs: dict[str, dict] = {}
    for run_id, items_rel, conditions, speech_conditions in SPECS:
        ids = load_ids(root / items_rel)
        pred_root = root / "exp" / run_id / "predictions" / MODEL
        condition_counts = {condition: 0 for condition in conditions}
        wav_counts = {condition: 0 for condition in speech_conditions}
        config_hashes: Counter[str] = Counter()
        long_audio = 0
        for item_id in ids:
            item_dir = pred_root / item_id
            for condition in conditions:
                record_path = item_dir / f"{condition}.json"
                if not record_path.is_file():
                    failures.append(f"{run_id}/{item_id}/{condition}: missing JSON")
                    continue
                condition_counts[condition] += 1
                try:
                    record = json.loads(record_path.read_text())
                except Exception as exc:
                    failures.append(
                        f"{run_id}/{item_id}/{condition}: invalid JSON: {exc}"
                    )
                    continue
                if record.get("error"):
                    failures.append(
                        f"{run_id}/{item_id}/{condition}: {record['error']}"
                    )
                if record.get("config_hash"):
                    config_hashes[str(record["config_hash"])] += 1
                if condition not in speech_conditions:
                    continue
                wav_path = item_dir / f"{condition}.wav"
                if not wav_path.is_file():
                    failures.append(f"{run_id}/{item_id}/{condition}: missing WAV")
                    continue
                wav_counts[condition] += 1
                audio_error = validate_audio(wav_path, args.deep_audio)
                if audio_error:
                    failures.append(
                        f"{run_id}/{item_id}/{condition}: WAV {audio_error}"
                    )
                long_audio += int(bool(record.get("long_audio")))
                if args.require_readback:
                    readback = record.get("readback") or {}
                    for channel in ("asr1", "asr2", "asr3"):
                        value = readback.get(channel)
                        if not isinstance(value, str) or not value.strip():
                            failures.append(
                                f"{run_id}/{item_id}/{condition}: empty {channel}"
                            )
                        error = readback.get(f"{channel}_error")
                        if error:
                            failures.append(
                                f"{run_id}/{item_id}/{condition}: "
                                f"{channel}_error={error}"
                            )
        runs[run_id] = {
            "expected_items": len(ids),
            "json_counts": condition_counts,
            "wav_counts": wav_counts,
            "config_hash_counts": dict(sorted(config_hashes.items())),
            "long_audio_records": long_audio,
        }

    report = {
        "model": MODEL,
        "require_readback": args.require_readback,
        "deep_audio": args.deep_audio,
        "runs": runs,
        "n_failures": len(failures),
        "failures": failures[:500],
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    with os.fdopen(fd, "w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
