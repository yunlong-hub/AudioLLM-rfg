#!/usr/bin/env python3
"""Validate the MiniCPM-o N=4 candidate pool before fact evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="data/main600/items.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--base-pred", default="exp/d1_main/predictions")
    parser.add_argument("--root", default="exp/d0_resample")
    parser.add_argument("--model", default="MiniCPM-o-4_5")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--output", default="exp/d0_resample/validation_MiniCPM-o-4_5.json")
    args = parser.parse_args()

    items = [json.loads(line) for line in Path(args.items).read_text().splitlines() if line.strip()]
    items = items[:args.limit]
    base_root = Path(args.base_pred) / args.model
    root = Path(args.root) / args.model
    errors = []
    n_json = n_wav = n_complete_asr = 0
    for item in items:
        item_id = item["id"]
        base_path = base_root / item_id / "SPEAK.json"
        try:
            baseline = json.loads(base_path.read_text())["text"].strip()
        except Exception as exc:
            errors.append(f"{base_path}: {type(exc).__name__}: {exc}")
            continue
        for candidate in range(args.n):
            path = root / item_id / f"R{candidate}.json"
            wav = path.with_suffix(".wav")
            if not path.exists():
                errors.append(f"missing {path}")
                continue
            n_json += 1
            if not wav.exists() or wav.stat().st_size <= 44:
                errors.append(f"missing/empty {wav}")
            else:
                n_wav += 1
            try:
                record = json.loads(path.read_text())
            except Exception as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
                continue
            if record.get("error"):
                errors.append(f"{path}: {record['error']}")
            if record.get("text", "").strip() != baseline or not record.get("text_matches_baseline"):
                errors.append(f"{path}: semantic text does not match baseline")
            channels_ok = True
            for channel in ("asr1", "asr2", "asr3"):
                if record.get(f"{channel}_error") or not isinstance(record.get(channel), str):
                    errors.append(f"{path}: incomplete {channel}")
                    channels_ok = False
            n_complete_asr += int(channels_ok)

    expected = len(items) * args.n
    summary = {
        "model": args.model,
        "n_items": len(items),
        "n_new_candidates": args.n,
        "expected": expected,
        "n_json": n_json,
        "n_wav": n_wav,
        "n_complete_three_asr": n_complete_asr,
        "n_errors": len(errors),
        "errors": errors[:200],
        "ready": n_json == n_wav == n_complete_asr == expected and not errors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "errors"},
                     ensure_ascii=False, indent=2))
    if errors:
        print("first errors:", *errors[:10], sep="\n")
    return 0 if summary["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
