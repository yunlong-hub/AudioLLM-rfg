#!/usr/bin/env python3
"""Create the blank, stratified three-ASR human adjudication packet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rfg.audit.adjudication import (AuditSource, collect_candidates,
                                    stratified_sample, summarize_packet)

DEFAULT_SOURCES = (
    "d1_main|Qwen3-Omni-30B-A3B-Instruct|exp/d1_main/facts/Qwen3-Omni-30B-A3B-Instruct.jsonl",
    "d1_main|Qwen2.5-Omni-3B|exp/d1_main/facts/Qwen2.5-Omni-3B.jsonl",
    "d2_stepaudio|Step-Audio-2-mini|exp/d2_stepaudio/facts/Step-Audio-2-mini.jsonl",
)


def _source(value: str) -> AuditSource:
    try:
        run_id, model_slug, facts_path = value.split("|", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("source must be RUN_ID|MODEL_SLUG|FACTS_JSONL") from exc
    return AuditSource(run_id, model_slug, Path(facts_path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=_source)
    parser.add_argument("--condition", default="SPEAK")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output", default="data/adjudication/three_asr_sample100.jsonl")
    args = parser.parse_args()

    sources = args.source or [_source(value) for value in DEFAULT_SOURCES]
    output = Path(args.output)
    report = output.with_suffix(".manifest.json")
    if output.exists() or report.exists():
        raise FileExistsError(
            f"refusing to overwrite an adjudication artifact that may contain human labels: {output}")

    candidates = collect_candidates(sources, args.condition)
    selected = stratified_sample(candidates, args.size, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize_packet(candidates, selected)
    summary.update(
        condition=args.condition,
        seed=args.seed,
        output=str(output),
        sources=[source.__dict__ | {"facts_path": str(source.facts_path)} for source in sources],
    )
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
