"""构造 VoiceBench-BBH 300 条分层子集（4 任务 × 75），供 E4 使用。

输出保持原文件顺序（generator 用 offset/limit 索引，顺序即切片依据）。
"""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/omg_benchmarks/prepared/voicebench_bbh/items_short.jsonl"
DST = ROOT / "data/omg_benchmarks/prepared/voicebench_bbh/items_subset300.jsonl"
PER_TASK = 75
SEED = 20260922

rows = [json.loads(line) for line in SRC.read_text().splitlines() if line.strip()]
groups: dict[str, list[dict]] = collections.defaultdict(list)
for row in rows:
    groups[row.get("difficulty") or row.get("task")].append(row)

rng = random.Random(SEED)
picked: list[dict] = []
for task in sorted(groups):
    pool = groups[task][:]
    rng.shuffle(pool)
    picked.extend(pool[:PER_TASK])

order = {row["id"]: i for i, row in enumerate(rows)}
picked.sort(key=lambda r: order[r["id"]])
DST.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked))
counts = collections.Counter(r["difficulty"] for r in picked)
print(f"wrote {DST.relative_to(ROOT)} n={len(picked)} counts={dict(counts)}")
