#!/usr/bin/env python3
"""Generate one deterministic item shard of MiniCPM targeted rerenders."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf

from rfg.models.registry import load_s2s_model
from rfg.run.conditions import ef_prompt
from rfg.run.resample_generation import atomic_json, candidate_complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pretrain_model/Audio/MiniCPM-o-4_5")
    parser.add_argument("--items", default="data/main600/items.jsonl")
    parser.add_argument("--targets", default="exp/d4_rerender/metrics/targets_MiniCPM-o-4_5.json")
    parser.add_argument("--root", default="exp/d4_rerender")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")

    item_rows = [json.loads(line) for line in Path(args.items).read_text().splitlines() if line.strip()]
    items = {row["id"]: row for row in item_rows}
    targets = json.loads(Path(args.targets).read_text())
    selected = [row for index, row in enumerate(targets)
                if index % args.num_shards == args.shard_index]
    model_slug = Path(args.model).name
    output_root = Path(args.root) / model_slug
    jobs = []
    for row in selected:
        for variant in ("ORIG", "SPEAK"):
            variant_text = row["variants"][f"{variant}_text"]
            for candidate in range(args.n):
                json_path = output_root / row["item_id"] / f"{variant}_{candidate}.json"
                wav_path = json_path.with_suffix(".wav")
                if not candidate_complete(json_path, wav_path):
                    jobs.append((row["item_id"], variant, variant_text, candidate,
                                 json_path, wav_path))
    print(f"[minicpmo-rerender] shard={args.shard_index}/{args.num_shards} "
          f"items={len(selected)} pending={len(jobs)}", flush=True)
    if not jobs:
        return 0

    model = load_s2s_model(args.model, device_map="auto")
    errors = []
    for index, (item_id, variant, variant_text, candidate, json_path, wav_path) in enumerate(jobs, 1):
        json_path.parent.mkdir(parents=True, exist_ok=True)
        question_audio = (items[item_id].get("question_audio") or {}).get("path")
        started = time.time()
        record = {
            "item_id": item_id,
            "variant": variant,
            "candidate": candidate,
            "seed": 1000 + candidate,
            "target_text": variant_text,
        }
        try:
            result = model.chat(
                [{"type": "audio", "audio": question_audio},
                 {"type": "text", "text": ef_prompt(variant_text)}],
                want_audio=True, seed=1000 + candidate,
            )
            temporary_wav = wav_path.with_suffix(".tmp.wav")
            model.save_wav(str(temporary_wav), result)
            info = sf.info(temporary_wav)
            temporary_wav.replace(wav_path)
            record.update({
                "text": result.text,
                "audio": str(wav_path),
                "audio_duration_sec": round(info.frames / info.samplerate, 3),
                "latency_sec": round(time.time() - started, 3),
            })
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"{item_id}/{variant}_{candidate}: {record['error']}")
        atomic_json(json_path, record)
        if index % 5 == 0 or index == len(jobs):
            print(f"  [{index}/{len(jobs)}] errors={len(errors)}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
