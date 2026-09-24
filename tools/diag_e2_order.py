"""E2 诊断：批次顺序/组成是否改变抽取结果（逐文本对照）。"""
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
shuffled = order[:]
rng.shuffle(shuffled)
texts_shuffled = [picked[i]["text"] for i in shuffled]

ex = LlmExtractor("/workspace/yunlong/LLM/pretrain_model/Qwen/Qwen2.5-7B-Instruct", device="cuda:0")
ex.batch_size = 16
res_a = ex.extract_batch(texts[:16])
res_b = ex.extract_batch(texts_shuffled[:16])

n_diff = n_empty = 0
for i in range(16):
    fa, ea = res_a[i]
    fb, eb = res_b[i]
    sa = sorted((f.type, f.value, f.polarity) for f in fa)
    sb = sorted((f.type, f.value, f.polarity) for f in fb)
    flag = "same" if sa == sb else ("EMPTY_A" if not sa else ("EMPTY_B" if not sb else "DIFF"))
    n_diff += flag != "same"
    n_empty += (not sa) or (not sb)
    print(f"{i:2d} len={len(texts[i]):4d} nA={len(sa):2d} nB={len(sb):2d} {flag:7s} "
          f"| A={sa[:2]} | B={sb[:2]} | errA={str(ea)[:30]} | errB={str(eb)[:30]}", flush=True)
print(f"SUMMARY model={MODEL} diff={n_diff}/16 empty_side={n_empty}/16", flush=True)

# --- 追加：集合级置换检测（批次组成是重排还是真的改变了输出） ---
sa_all = set()
sb_all = set()
for fa, _ in res_a:
    sa_all |= {(f.type, f.value, f.polarity) for f in fa}
for fb, _ in res_b:
    sb_all |= {(f.type, f.value, f.polarity) for f in fb}
print(f"SETSIM unionA={len(sa_all)} unionB={len(sb_all)} "
      f"inter={len(sa_all & sb_all)} jac={len(sa_all & sb_all) / max(1, len(sa_all | sb_all)):.3f}",
      flush=True)
print("only in A:", sorted(sa_all - sb_all)[:8])
print("only in B:", sorted(sb_all - sa_all)[:8])
