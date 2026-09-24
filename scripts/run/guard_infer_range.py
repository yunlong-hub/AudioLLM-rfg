#!/usr/bin/env python3
"""CLI for stopping a legacy inference worker at an exact coverage boundary."""

from __future__ import annotations

import argparse
from pathlib import Path

from rfg.run.infer_guard import guard_worker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--conditions", default="LISTEN,SPEAK")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--grace", type=int, default=60)
    parser.add_argument("--out-root", type=Path, default=Path("exp"))
    args = parser.parse_args()
    conditions = tuple(part.strip() for part in args.conditions.split(",") if part.strip())
    return guard_worker(
        pid=args.pid,
        run_id=args.run_id,
        model_slug=args.model_slug,
        items_path=args.items,
        offset=args.offset,
        limit=args.limit,
        conditions=conditions,
        interval=args.interval,
        grace=args.grace,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
