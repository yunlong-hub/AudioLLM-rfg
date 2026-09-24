#!/usr/bin/env python3
"""Score a completed three-ASR human adjudication packet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rfg.audit.score import score_adjudication


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", default="data/adjudication/three_asr_sample100.jsonl")
    parser.add_argument("--output", default="output/experiment-extension/three_asr_human_audit.json")
    args = parser.parse_args()
    result = score_adjudication(args.packet)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ready"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
