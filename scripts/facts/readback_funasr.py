#!/usr/bin/env python3
"""Add Fun-ASR readback (``readback.asr3``) to existing speech records.

The script never reruns or replaces Whisper/Seamless.  Jobs are deterministically
sharded by JSON path, so two A42 GPUs can process disjoint records safely.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from rfg.facts.readback_state import channel_reusable_for_chunking
from rfg.models.funasr import FunAsrReadback

MODEL_PATH = "/workspace/yunlong/LLM/AudioLLM-rfg/pretrain_model/ASR/Fun-ASR-Nano-2512"
DEFAULT_CONDITIONS = "SPEAK,ECHO,EF"


def atomic_write_json(path: str, record: dict) -> None:
    tmp = f"{path}.asr3.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def collect_jobs(pred_root: str, models: list[str], conditions: set[str]) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    for model in models:
        model_dir = os.path.join(pred_root, model)
        if not os.path.isdir(model_dir):
            continue
        for item_id in sorted(os.listdir(model_dir)):
            item_dir = os.path.join(model_dir, item_id)
            if not os.path.isdir(item_dir):
                continue
            for condition in sorted(conditions):
                json_path = os.path.join(item_dir, f"{condition}.json")
                wav_path = os.path.join(item_dir, f"{condition}.wav")
                if os.path.exists(json_path) and os.path.exists(wav_path):
                    jobs.append((json_path, wav_path))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="d0_pilot")
    ap.add_argument("--out-root", default="exp")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.0,
        help="长音频按静音点切成不超过该秒数的块；0 保持单段推理",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        ap.error("require 0 <= shard-index < num-shards")

    run_dir = os.path.join(args.out_root, args.run_id)
    pred_root = os.path.join(run_dir, "predictions")
    models = args.model or sorted(
        name for name in os.listdir(pred_root)
        if os.path.isdir(os.path.join(pred_root, name))
    )
    conditions = {value.strip() for value in args.conditions.split(",") if value.strip()}
    all_jobs = collect_jobs(pred_root, models, conditions)
    jobs = [job for index, job in enumerate(all_jobs)
            if index % args.num_shards == args.shard_index]
    if args.limit:
        jobs = jobs[:args.limit]

    pending: list[tuple[str, str]] = []
    expected_version = FunAsrReadback.version_name(args.chunk_seconds or None)
    for json_path, wav_path in jobs:
        record = json.load(open(json_path))
        readback = record.get("readback") or {}
        duration = record.get("audio_duration_sec")
        reusable = channel_reusable_for_chunking(
            readback,
            "asr3",
            duration_sec=duration,
            chunk_seconds=args.chunk_seconds or None,
            expected_version=expected_version,
        )
        if not args.force and reusable:
            continue
        pending.append((json_path, wav_path))

    print(
        f"[funasr] run={args.run_id} shard={args.shard_index}/{args.num_shards} "
        f"jobs={len(jobs)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return 0

    recognizer = FunAsrReadback(
        args.model_path,
        device=args.device,
        chunk_seconds=args.chunk_seconds or None,
    )
    errors: list[str] = []
    latencies: list[float] = []
    started = time.time()
    for index, (json_path, wav_path) in enumerate(pending, 1):
        record = json.load(open(json_path))
        readback = dict(record.get("readback") or {})
        try:
            result = recognizer.transcribe(wav_path)
            readback["asr3"] = result.text
            readback["asr3_sec"] = round(result.latency_sec or 0.0, 3)
            readback["asr3_error"] = None
            versions = dict(readback.get("asr_versions") or {})
            versions["asr3"] = recognizer.name
            readback["asr_versions"] = versions
            latencies.append(result.latency_sec or 0.0)
        except Exception as exc:
            readback["asr3"] = None
            readback["asr3_error"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"{json_path}: {readback['asr3_error']}")
        record["readback"] = readback
        atomic_write_json(json_path, record)
        if index % 20 == 0 or index == len(pending):
            elapsed = time.time() - started
            eta = elapsed / index * (len(pending) - index)
            print(
                f"  [{index}/{len(pending)}] errors={len(errors)} "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

    summary = {
        "run_id": args.run_id,
        "models": models,
        "conditions": sorted(conditions),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_jobs": len(jobs),
        "n_processed": len(pending),
        "n_errors": len(errors),
        "mean_latency_sec": sum(latencies) / len(latencies) if latencies else None,
        "errors": errors[:100],
    }
    metrics_dir = os.path.join(run_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    output = os.path.join(
        metrics_dir,
        "readback_funasr_"
        + (models[0] + "_" if len(models) == 1 else "")
        + f"shard{args.shard_index}-of-{args.num_shards}.json",
    )
    atomic_write_json(output, summary)
    print(f"written -> {output}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
