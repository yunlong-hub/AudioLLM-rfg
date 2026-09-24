#!/usr/bin/env python3
"""Add Fun-ASR readback to existing resampling records."""
from __future__ import annotations

import argparse
import json
import os
import time

from rfg.facts.readback_state import channel_complete
from rfg.models.funasr import FunAsrReadback

MODEL_PATH = "/workspace/yunlong/LLM/AudioLLM-rfg/pretrain_model/ASR/Fun-ASR-Nano-2512"


def atomic_write(path: str, record: dict) -> None:
    tmp = f"{path}.asr3.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(record, fh, ensure_ascii=False)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model slug under the resample root")
    ap.add_argument("--root", default="exp/d0_resample")
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prefixes", default="R",
                    help="comma-separated JSON filename prefixes to include")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        ap.error("require 0 <= shard-index < num-shards")

    model_dir = os.path.join(args.root, args.model)
    prefixes = tuple(value.strip() for value in args.prefixes.split(",") if value.strip())
    paths = []
    for item_id in sorted(os.listdir(model_dir)):
        item_dir = os.path.join(model_dir, item_id)
        if not os.path.isdir(item_dir):
            continue
        paths.extend(
            os.path.join(item_dir, name)
            for name in sorted(os.listdir(item_dir))
            if name.startswith(prefixes) and name.endswith(".json")
            and os.path.exists(os.path.join(item_dir, name[:-5] + ".wav"))
        )
    jobs = [path for index, path in enumerate(paths)
            if index % args.num_shards == args.shard_index]
    pending = [
        path for path in jobs
        if args.force or not channel_complete(json.load(open(path)), "asr3")
    ]
    print(
        f"[funasr-resample] model={args.model} shard={args.shard_index}/{args.num_shards} "
        f"jobs={len(jobs)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return 0

    recognizer = FunAsrReadback(args.model_path, device=args.device)
    errors = []
    started = time.time()
    for index, path in enumerate(pending, 1):
        record = json.load(open(path))
        wav = path[:-5] + ".wav"
        try:
            result = recognizer.transcribe(wav)
            record["asr3"] = result.text
            record["asr3_sec"] = round(result.latency_sec or 0.0, 3)
            record["asr3_error"] = None
            record["asr3_version"] = recognizer.name
        except Exception as exc:
            record["asr3"] = None
            record["asr3_error"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"{path}: {record['asr3_error']}")
        atomic_write(path, record)
        if index % 20 == 0 or index == len(pending):
            elapsed = time.time() - started
            eta = elapsed / index * (len(pending) - index)
            print(f"  [{index}/{len(pending)}] errors={len(errors)} eta={eta:.0f}s", flush=True)

    summary = {
        "model": args.model,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_jobs": len(jobs),
        "n_processed": len(pending),
        "n_errors": len(errors),
        "errors": errors[:100],
    }
    output = os.path.join(
        args.root,
        f"readback_funasr_resample_{args.model}_shard{args.shard_index}-of-{args.num_shards}.json",
    )
    atomic_write(output, summary)
    print(f"written -> {output}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
