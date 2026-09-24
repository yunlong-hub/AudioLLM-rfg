#!/usr/bin/env python3
"""Score one model on a complete prepared OMG benchmark manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rfg.score.benchmark import score_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--conditions", default="LISTEN,SPEAK")
    parser.add_argument("--speech-asrs", default="",
                        help="optional comma-separated SPEAK readbacks, e.g. asr1,asr2,asr3")
    parser.add_argument("--exp-root", default="exp")
    args = parser.parse_args()

    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    speech_asrs = [value.strip() for value in args.speech_asrs.split(",") if value.strip()]
    run_root = Path(args.exp_root) / args.run_id
    prediction_root = run_root / "predictions" / args.model_slug
    summary, details = score_benchmark(args.items, prediction_root, conditions, speech_asrs)
    summary.update(run_id=args.run_id, model_slug=args.model_slug)

    metrics_root = run_root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_root / f"benchmark_{args.model_slug}.json"
    details_path = metrics_root / f"benchmark_{args.model_slug}.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with details_path.open("w") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary={summary_path}")
    print(f"details={details_path}")
    return 0 if summary["ready"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
