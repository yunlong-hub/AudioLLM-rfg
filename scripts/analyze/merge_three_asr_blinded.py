#!/usr/bin/env python3
"""Merge completed blinded annotations without overwriting source labels."""
from __future__ import annotations

import argparse

from rfg.audit.blinding import (load_jsonl, make_adjudication_sheet,
                                merge_adjudication_sheet,
                                merge_annotator_sheets, write_new_jsonl)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)

    annotators = subparsers.add_parser("annotators")
    annotators.add_argument("--master", default="data/adjudication/three_asr_sample100.jsonl")
    annotators.add_argument(
        "--annotator-1", default="data/adjudication/blinded/three_asr_sample100.annotator_1.jsonl"
    )
    annotators.add_argument(
        "--annotator-2", default="data/adjudication/blinded/three_asr_sample100.annotator_2.jsonl"
    )
    annotators.add_argument(
        "--merged-output", default="data/adjudication/three_asr_sample100.annotated.jsonl"
    )
    annotators.add_argument(
        "--adjudication-output",
        default="data/adjudication/blinded/three_asr_sample100.adjudication.jsonl",
    )
    annotators.add_argument("--seed", type=int, default=20260918)

    adjudication = subparsers.add_parser("adjudication")
    adjudication.add_argument(
        "--annotated-master", default="data/adjudication/three_asr_sample100.annotated.jsonl"
    )
    adjudication.add_argument(
        "--adjudication-sheet",
        default="data/adjudication/blinded/three_asr_sample100.adjudication.jsonl",
    )
    adjudication.add_argument(
        "--output", default="data/adjudication/three_asr_sample100.completed.jsonl"
    )
    args = parser.parse_args()

    if args.stage == "annotators":
        merged = merge_annotator_sheets(
            load_jsonl(args.master),
            load_jsonl(args.annotator_1),
            load_jsonl(args.annotator_2),
        )
        # Validate both targets before creating either artifact.
        from pathlib import Path
        targets = (Path(args.merged_output), Path(args.adjudication_output))
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite possible human labels: " + ", ".join(existing)
            )
        write_new_jsonl(targets[0], merged)
        write_new_jsonl(targets[1], make_adjudication_sheet(merged, args.seed))
        print(f"merged {len(merged)} independent annotations -> {targets[0]}")
        print(f"wrote blinded adjudication sheet -> {targets[1]}")
        return 0

    completed = merge_adjudication_sheet(
        load_jsonl(args.annotated_master), load_jsonl(args.adjudication_sheet)
    )
    write_new_jsonl(args.output, completed)
    print(f"merged {len(completed)} adjudications -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
