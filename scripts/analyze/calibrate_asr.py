#!/usr/bin/env python3
"""Calibrate readback ASRs on fixed-text speech conditions.

EF exposes the text passed to the speech renderer, so its audio provides an
outcome-independent calibration set for choosing the readback instrument.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from rfg.facts.rules import extract_rules
from rfg.score.textnorm import wer_norm

ASR_KEYS = ("asr1", "asr2", "asr3")
COMBINATIONS = {
    "asr1": ("asr1",),
    "asr2": ("asr2",),
    "asr3": ("asr3",),
    "union12": ("asr1", "asr2"),
    "union13": ("asr1", "asr3"),
    "union123": ("asr1", "asr2", "asr3"),
}


def evaluate_model(model_dir: str, condition: str) -> dict:
    rows: list[dict] = []
    for item_id in sorted(os.listdir(model_dir)):
        path = os.path.join(model_dir, item_id, f"{condition}.json")
        if not os.path.exists(path):
            continue
        record = json.load(open(path))
        readback = record.get("readback") or {}
        if not all(readback.get(key) for key in ASR_KEYS):
            continue
        target = record.get("text") or ""
        upstream = extract_rules(target)
        facts = {key: extract_rules(readback[key]) for key in ASR_KEYS}
        rows.append({
            "wer": {key: wer_norm(target, readback[key]) for key in ASR_KEYS},
            "upstream": upstream,
            "facts": facts,
        })

    result: dict = {"n_items": len(rows), "systems": {}}
    for name, keys in COMBINATIONS.items():
        retained = total = additions = predicted = 0
        for row in rows:
            downstream = set().union(*(row["facts"][key] for key in keys))
            upstream = row["upstream"]
            retained += len(upstream & downstream)
            total += len(upstream)
            additions += len(downstream - upstream)
            predicted += len(downstream)
        block = {
            "retained": retained,
            "total": total,
            "fact_recall": retained / total if total else None,
            "additions": additions,
            "predicted": predicted,
            "addition_rate": additions / predicted if predicted else None,
        }
        if len(keys) == 1:
            values = [row["wer"][keys[0]] for row in rows]
            block["mean_wer"] = sum(values) / len(values) if values else None
        result["systems"][name] = block
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="d0_pilot")
    ap.add_argument("--out-root", default="exp")
    ap.add_argument("--condition", default="EF")
    ap.add_argument("--out", default="reports/asr_calibration.md")
    args = ap.parse_args()

    pred_root = os.path.join(args.out_root, args.run_id, "predictions")
    report: dict = {
        "run_id": args.run_id,
        "condition": args.condition,
        "models": {},
    }
    for model in sorted(os.listdir(pred_root)):
        model_dir = os.path.join(pred_root, model)
        if os.path.isdir(model_dir):
            result = evaluate_model(model_dir, args.condition)
            if result["n_items"]:
                report["models"][model] = result

    lines = [
        "# ASR readback calibration", "",
        f"Run: `{args.run_id}`; fixed-text condition: `{args.condition}`.", "",
        "| model | system | n | mean WER | rule-fact recall | addition rate |", 
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, result in report["models"].items():
        for system, block in result["systems"].items():
            wer = block.get("mean_wer")
            lines.append(
                f"| {model} | `{system}` | {result['n_items']} | "
                f"{wer:.4f} | " if wer is not None else
                f"| {model} | `{system}` | {result['n_items']} | -- | "
            )
            lines[-1] += (
                f"{block['fact_recall']:.4f} | {block['addition_rate']:.4f} |"
                if block["fact_recall"] is not None and block["addition_rate"] is not None
                else "-- | -- |"
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    json_path = os.path.splitext(args.out)[0] + ".json"
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print("\n".join(lines))
    print(f"written -> {args.out} | {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
