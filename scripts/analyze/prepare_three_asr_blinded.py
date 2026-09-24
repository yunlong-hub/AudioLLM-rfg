#!/usr/bin/env python3
"""Export two independent, model/ASR-blind annotation sheets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rfg.audit.blinding import (ANNOTATORS, jsonl_sha256, load_jsonl,
                                make_blinded_sheet, write_new_jsonl)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", default="data/adjudication/three_asr_sample100.jsonl")
    parser.add_argument("--output-dir", default="data/adjudication/blinded")
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()

    master_path = Path(args.master)
    output_dir = Path(args.output_dir)
    outputs = {
        annotator: output_dir / f"three_asr_sample100.{annotator}.jsonl"
        for annotator in ANNOTATORS
    }
    manifest_path = output_dir / "three_asr_sample100.blinding_manifest.json"
    existing = [path for path in (*outputs.values(), manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite possible human labels: "
            + ", ".join(str(path) for path in existing)
        )

    master_rows = load_jsonl(master_path)
    for annotator, path in outputs.items():
        write_new_jsonl(path, make_blinded_sheet(master_rows, annotator, args.seed))
    manifest = {
        "master": str(master_path),
        "master_sha256": jsonl_sha256(master_path),
        "n_items": len(master_rows),
        "seed": args.seed,
        "sheets": {name: str(path) for name, path in outputs.items()},
        "blinded_fields": ["model", "run_id", "item_id", "internal_text", "asr",
                           "pairwise_wer", "extractor", "stratum", "category"],
        "annotation_schema": {
            "transcript": "string; use an empty string only when no speech is intelligible",
            "facts": "list of {type, value, polarity}; an empty list is valid",
            "notes": "null or string",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
