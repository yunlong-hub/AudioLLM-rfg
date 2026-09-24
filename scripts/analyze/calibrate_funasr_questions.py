#!/usr/bin/env python3
"""Measure Fun-ASR WER on question audio with known source text."""
from __future__ import annotations

import argparse
import json
import os
import statistics

from rfg.models.funasr import FunAsrReadback
from rfg.score.textnorm import wer_norm

MODEL_PATH = "/workspace/yunlong/LLM/AudioLLM-rfg/pretrain_model/ASR/Fun-ASR-Nano-2512"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/pilot/items.jsonl")
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="reports/funasr_question_calibration.json")
    args = ap.parse_args()

    items = [json.loads(line) for line in open(args.items)]
    recognizer = FunAsrReadback(args.model_path, device=args.device)
    rows = []
    for index, item in enumerate(items, 1):
        audio = (item.get("question_audio") or {}).get("path")
        if not audio or not os.path.exists(audio):
            continue
        result = recognizer.transcribe(audio)
        target = item["question_text"]
        rows.append({
            "item_id": item["id"],
            "target": target,
            "hypothesis": result.text,
            "wer": wer_norm(target, result.text),
        })
        if index % 20 == 0:
            print(f"[{index}/{len(items)}]", flush=True)

    report = {
        "model": recognizer.name,
        "items": args.items,
        "n_items": len(rows),
        "mean_wer": statistics.mean(row["wer"] for row in rows) if rows else None,
        "median_wer": statistics.median(row["wer"] for row in rows) if rows else None,
        "exact_rate": (sum(row["wer"] == 0 for row in rows) / len(rows)) if rows else None,
        "per_item": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps({key: value for key, value in report.items() if key != "per_item"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
