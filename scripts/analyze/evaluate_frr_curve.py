#!/usr/bin/env python3
"""Evaluate an independent FRR candidate-count curve from one candidate pool."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rfg.facts.extract import load_extractions
from rfg.facts.llm import PROMPT_SHA256
from rfg.facts.resample import load_resample_corpus
from rfg.score.frr import evaluate_loo_folds


def truncate_candidates(corpus: list[dict], n_candidates: int) -> list[dict]:
    fields = ("sample_keys", "sample_keys_asr2", "sample_keys_asr3")
    return [
        {
            **item,
            **{field: item.get(field, [])[:n_candidates] for field in fields},
        }
        for item in corpus
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--pilot-pred", default="exp/d0_pilot/predictions")
    parser.add_argument("--resample-root", default="exp/d0_resample")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--candidate-counts", default="1,2,4")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    cache = args.cache or os.path.join(
        args.resample_root, "facts_dual", f"{args.model_slug}.jsonl")
    corpus, _ = load_resample_corpus(args.pilot_pred, args.resample_root, args.model_slug)
    extracted = load_extractions(cache, prompt_sha256=PROMPT_SHA256)
    counts = [int(value) for value in args.candidate_counts.split(",")]
    curve = {}
    for index, count in enumerate(counts):
        result = evaluate_loo_folds(
            truncate_candidates(corpus, count),
            extracted,
            seed=args.seed + index * 10,
            n_boot=args.n_boot,
        )
        curve[str(count)] = {key: value for key, value in result.items() if key != "per_item"}
    output = {
        "model": args.model_slug,
        "candidate_counts": counts,
        "protocol": "prefix candidates; independent leave-one-ASR-out at each N",
        "curve": curve,
        "cache": cache,
        "prompt_sha256": PROMPT_SHA256,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
