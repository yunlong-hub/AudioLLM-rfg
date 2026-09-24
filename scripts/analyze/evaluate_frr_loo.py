#!/usr/bin/env python3
"""Evaluate FRR with each ASR held out in turn."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rfg.facts.extract import load_extractions
from rfg.facts.llm import PROMPT_SHA256
from rfg.facts.resample import load_resample_corpus
from rfg.score.frr import evaluate_loo_folds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--pilot-pred", default="exp/d0_pilot/predictions")
    parser.add_argument("--resample-root", default="exp/d0_resample")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()

    cache = args.cache or os.path.join(
        args.resample_root, "facts_dual", f"{args.model_slug}.jsonl")
    corpus, _ = load_resample_corpus(args.pilot_pred, args.resample_root, args.model_slug)
    extracted = load_extractions(cache, prompt_sha256=PROMPT_SHA256)
    result = evaluate_loo_folds(corpus, extracted, seed=args.seed, n_boot=args.n_boot)
    result.update(
        model=args.model_slug,
        cache=cache,
        prompt_sha256=PROMPT_SHA256,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    printable = {key: value for key, value in result.items() if key != "per_item"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
