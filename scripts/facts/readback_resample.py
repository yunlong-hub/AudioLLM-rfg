#!/usr/bin/env python3
"""Sharded Whisper/Seamless readback for fixed-seed resampling artifacts."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from rfg.facts.readback_state import channel_reusable_for_chunking
from rfg.models.asr import SeamlessReadback, WhisperReadback


PRETRAIN = "/workspace/yunlong/LLM/pretrain_model/Audio"
MODEL_PATHS = {
    "asr1": f"{PRETRAIN}/whisper-large-v3",
    "asr2": f"{PRETRAIN}/seamless-m4t-v2-large",
}


def atomic_write(path: Path, record: dict) -> None:
    temporary = path.with_name(f"{path.name}.readback.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(record, ensure_ascii=False))
    os.replace(temporary, path)


def parse_channels(value: str) -> tuple[str, ...]:
    channels = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not channels or any(channel not in MODEL_PATHS for channel in channels):
        raise argparse.ArgumentTypeError("channels must be asr1, asr2, or asr1,asr2")
    return channels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model slug under root")
    parser.add_argument("--items", default="data/pilot/items.jsonl")
    parser.add_argument("--root", default="exp/d0_resample")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--prefixes", default="R",
                        help="comma-separated filename prefixes; R uses R0..R<n>, others scan")
    parser.add_argument("--channels", type=parse_channels, default=("asr1", "asr2"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-seconds", type=float, default=25.0)
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.items).read_text().splitlines() if line.strip()]
    selected = rows[args.offset : args.offset + args.limit if args.limit else None]
    model_root = Path(args.root) / args.model
    jobs = []
    prefixes = tuple(part.strip() for part in args.prefixes.split(",") if part.strip())
    for item in selected:
        item_root = model_root / item["id"]
        if prefixes == ("R",):
            paths = [item_root / f"R{candidate}.json" for candidate in range(args.n)]
        elif item_root.is_dir():
            paths = sorted(path for path in item_root.glob("*.json")
                           if path.name.startswith(prefixes))
        else:
            paths = []
        for record_path in paths:
            wav_path = record_path.with_suffix(".wav")
            if record_path.exists() and wav_path.exists():
                jobs.append((record_path, wav_path))

    recognizers = {}
    if "asr1" in args.channels:
        recognizers["asr1"] = WhisperReadback(
            MODEL_PATHS["asr1"], device=args.device,
            chunk_seconds=args.chunk_seconds or None,
        )
    if "asr2" in args.channels:
        recognizers["asr2"] = SeamlessReadback(
            MODEL_PATHS["asr2"], device=args.device, low_memory=args.low_memory,
            chunk_seconds=args.chunk_seconds or None,
        )
    print(
        f"[resample-readback] range={args.offset}:{args.offset + len(selected)} "
        f"jobs={len(jobs)} channels={args.channels}",
        flush=True,
    )

    processed = reused = failed = 0
    started = time.time()
    for index, (record_path, wav_path) in enumerate(jobs, 1):
        record = json.loads(record_path.read_text())
        duration = record.get("audio_duration_sec")
        changed = False
        for channel, recognizer in recognizers.items():
            reusable = channel_reusable_for_chunking(
                record,
                channel,
                duration_sec=duration,
                chunk_seconds=args.chunk_seconds or None,
                expected_version=recognizer.name,
            )
            if reusable and not args.force:
                continue
            try:
                result = recognizer.transcribe(str(wav_path))
                record[channel] = result.text
                record[f"{channel}_sec"] = round(result.latency_sec, 3)
                record[f"{channel}_error"] = None
                versions = record.setdefault("asr_versions", {})
                versions[channel] = recognizer.name
                changed = True
                processed += 1
            except Exception as exc:
                record[f"{channel}_error"] = f"{type(exc).__name__}: {exc}"
                changed = True
                failed += 1
        if changed:
            atomic_write(record_path, record)
        else:
            reused += 1
        if index % 20 == 0 or index == len(jobs):
            elapsed = time.time() - started
            eta = elapsed / index * (len(jobs) - index) if index else 0
            print(
                f"  [{index}/{len(jobs)}] processed={processed} reused={reused} "
                f"errors={failed} eta={eta:.0f}s",
                flush=True,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
