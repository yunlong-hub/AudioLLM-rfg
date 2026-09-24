#!/usr/bin/env python3
"""Validate full Qwen2.5-Omni-7B generation and three-ASR coverage."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RUNS = (
    ("omg_spoken_mqa", 1402, ("LISTEN", "SPEAK"), ("SPEAK",)),
    ("omg_voicebench_short", 1000, ("LISTEN", "SPEAK"), ("SPEAK",)),
    (
        "d1_main",
        600,
        ("READ", "LISTEN", "SPEAK", "ECHO", "EF", "EFA", "EFB", "EFW"),
        ("SPEAK", "ECHO", "EF", "EFA", "EFB", "EFW"),
    ),
)
MODEL = "Qwen2.5-Omni-7B"


def nonempty(value: object) -> bool:
    if isinstance(value, dict):
        value = value.get("text")
    return isinstance(value, str) and bool(value.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", default="exp")
    parser.add_argument("--require-readback", action="store_true")
    parser.add_argument("--deep-audio", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result: dict[str, object] = {
        "model": MODEL,
        "stack": "vllm-omni-0.18.0-v100",
        "require_readback": args.require_readback,
        "deep_audio": args.deep_audio,
        "runs": {},
        "errors": [],
    }
    errors: list[str] = result["errors"]  # type: ignore[assignment]

    for run_id, expected, conditions, speech_conditions in RUNS:
        model_root = Path(args.exp_root) / run_id / "predictions" / MODEL
        counts: dict[str, int] = {}
        wav_counts: dict[str, int] = {}
        record_errors = 0
        readback_missing = {name: 0 for name in ("asr1", "asr2", "asr3")}
        audio_checked = 0
        for condition in conditions:
            paths = sorted(model_root.glob(f"*/{condition}.json"))
            counts[condition] = len(paths)
            if len(paths) != expected:
                errors.append(f"{run_id}/{condition}: json={len(paths)} expected={expected}")
            for path in paths:
                try:
                    record = json.loads(path.read_text())
                except Exception as exc:
                    record_errors += 1
                    errors.append(f"{path}: invalid json: {exc}")
                    continue
                if record.get("error"):
                    record_errors += 1
                    errors.append(f"{path}: {record['error']}")
                if record.get("stack") != "vllm-omni":
                    errors.append(f"{path}: unexpected stack={record.get('stack')!r}")
                if condition not in speech_conditions:
                    continue
                wav_path = path.with_suffix(".wav")
                if args.require_readback:
                    readback = record.get("readback") or {}
                    for channel in readback_missing:
                        if not nonempty(readback.get(channel)):
                            readback_missing[channel] += 1
                if args.deep_audio and wav_path.exists():
                    import numpy as np
                    import soundfile as sf

                    audio, sample_rate = sf.read(wav_path, dtype="float32")
                    audio_checked += 1
                    if sample_rate != 24000 or audio.size == 0 or not np.isfinite(audio).all():
                        errors.append(
                            f"{wav_path}: sr={sample_rate} samples={audio.size} "
                            f"finite={bool(np.isfinite(audio).all())}"
                        )
                    elif math.isclose(float(np.sqrt(np.mean(audio * audio))), 0.0, abs_tol=1e-7):
                        errors.append(f"{wav_path}: silent audio")

        for condition in speech_conditions:
            wav_count = len(list(model_root.glob(f"*/{condition}.wav")))
            wav_counts[condition] = wav_count
            if wav_count != expected:
                errors.append(f"{run_id}/{condition}: wav={wav_count} expected={expected}")
        if args.require_readback:
            for channel, missing in readback_missing.items():
                if missing:
                    errors.append(f"{run_id}/{channel}: missing_or_empty={missing}")

        result["runs"][run_id] = {  # type: ignore[index]
            "expected_per_condition": expected,
            "json_counts": counts,
            "wav_counts": wav_counts,
            "record_errors": record_errors,
            "readback_missing": readback_missing,
            "audio_checked": audio_checked,
        }

    result["n_errors"] = len(errors)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
