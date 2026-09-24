#!/usr/bin/env python3
"""Merge disjoint fact-extraction shards into the canonical cache."""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--out-root", default="exp")
    args = ap.parse_args()
    if args.num_shards < 1:
        ap.error("--num-shards must be >= 1")

    facts_dir = os.path.join(args.out_root, args.run_id, "facts")
    rows: dict[tuple[str, str], dict] = {}
    for shard in range(args.num_shards):
        path = os.path.join(
            facts_dir, f"{args.model}_shard{shard}-of-{args.num_shards}.jsonl"
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing fact shard: {path}")
        with open(path, encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, 1):
                row = json.loads(line)
                key = (row["item_id"], row["condition"])
                if key in rows:
                    raise ValueError(
                        f"duplicate fact key {key!r} in {path}:{line_number}"
                    )
                rows[key] = row

    output = os.path.join(facts_dir, f"{args.model}.jsonl")
    temporary = f"{output}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as fh:
        for key in sorted(rows):
            fh.write(json.dumps(rows[key], ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    print(f"merged {len(rows)} rows from {args.num_shards} shards -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
