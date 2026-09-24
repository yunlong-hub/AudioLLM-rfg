"""E2 决定性诊断：批次组成是否改变同一文本的抽取结果。

三组：
  S1  同一批(16)、同一顺序，跑两次           -> 配置内确定性
  S2  batch=1（逐条）                        -> 无 padding 的参照
  S3  同一条文本分别放在 batch 首/尾          -> 位置效应
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rfg.facts.llm import LlmExtractor  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-Omni-30B-A3B-Instruct"
CATS = ("num", "unit", "name", "neg", "cont")
rng = random.Random(20260921)

by: dict[str, list[dict]] = {c: [] for c in CATS}
with open(_ROOT / f"exp/d0_resample/facts_dual/{MODEL}.jsonl") as fh:
    for line in fh:
        r = json.loads(line)
        k = r.get("key") or ""
        if not k.endswith("|SPEAK#internal"):
            continue
        t = (r.get("text") or "").strip()
        if not t:
            continue
        c = k.split("|")[0].split("_")[0]
        if c in by:
            by[c].append({"key": k, "text": t})
picked: list[dict] = []
for c in CATS:
    rng.shuffle(by[c])
    picked += by[c][:12]
order = sorted(range(len(picked)), key=lambda i: picked[i]["key"])
texts = [picked[i]["text"] for i in order]


def keys(res):
    return [sorted((f.type, f.value, f.polarity) for f in facts) for facts, _ in res]


ex = LlmExtractor("/workspace/yunlong/LLM/pretrain_model/Qwen/Qwen2.5-7B-Instruct", device="cuda:0")

ex.batch_size = 16
a1 = keys(ex.extract_batch(texts[:16]))
a2 = keys(ex.extract_batch(texts[:16]))
same = sum(1 for x, y in zip(a1, a2) if x == y)
print(f"S1 batch16 repeat: identical {same}/16", flush=True)

ex.batch_size = 1
solo = keys(ex.extract_batch(texts[:16]))
match_solo = sum(1 for x, y in zip(a1, solo) if x == y)
print(f"S2 batch1 vs batch16: identical {match_solo}/16", flush=True)

# S3 位置效应：同一条文本放 batch 首/尾
probe = texts[0]
ex.batch_size = 2
first = keys(ex.extract_batch([probe, texts[1]]))[0]
last = keys(ex.extract_batch([texts[1], probe]))[1]
print(f"S3 same text at first vs last: identical {first == last} | first={first} last={last}",
      flush=True)

for i in range(16):
    flag = "same" if a1[i] == solo[i] else "DIFF"
    print(f"  {i:2d} len={len(texts[i]):4d} b16={len(a1[i])} b1={len(solo[i])} {flag} "
          f"| b16={a1[i][:2]} | b1={solo[i][:2]}", flush=True)
