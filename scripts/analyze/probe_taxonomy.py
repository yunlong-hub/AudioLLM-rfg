#!/usr/bin/env python3
"""失败解剖：把"丢了多少事实"变成"事实是怎么丢的"。

零 GPU：只用已落盘的内部文本、双 ASR 回读与事实集合。

对每一条**丢失的事实**（在内部文本里、但不在回读里）判定其失败类型：
  * `substitution` 同类替换：回读里出现了**同类型但不同值**的事实（专名被换、单位被换）
  * `number_drift` 数字漂移：回读里有同类型的数值且与原值接近（相对差 < 50%）→ 说错了数
  * `polarity_flip` 极性翻转：同一个值出现，但 polarity 由 + 变 −（或反之）
  * `deletion`      纯删除：回读里既无该值也无同类型候选

同时输出：
  * 各失败类型的占比（按模型 / 按条件 / 按题目类别）
  * 与"内部文本事实总数"的归一化率（即每类失败贡献了多少 pp 的 Δ_render）

用法： python scripts/probe_taxonomy.py --run-id d0_pilot
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict


def load(run_dir: str, mslug: str) -> dict:
    d: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    with open(os.path.join(run_dir, "facts", f"{mslug}.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            d[r["item_id"]][r["condition"]] = r.get("facts", [])
    return d


def key(f: dict) -> tuple:
    return (f["type"], f["value"], f.get("polarity", "+"))


def classify(lost: dict, down: list[dict]) -> str:
    """判定一条丢失事实的失败类型。"""
    same_type = [f for f in down if f["type"] == lost["type"]]
    # 极性翻转：同值不同极性
    if any(f["value"] == lost["value"] and f.get("polarity") != lost.get("polarity")
           for f in down):
        return "polarity_flip"
    if not same_type:
        return "deletion"
    if lost["type"] == "number":
        try:
            a = float(str(lost["value"]).replace(",", ""))
        except ValueError:
            return "substitution"
        best = None
        for f in same_type:
            try:
                b = float(str(f["value"]).replace(",", ""))
            except ValueError:
                continue
            denom = max(abs(a), 1e-9)
            rel = abs(a - b) / denom
            best = rel if best is None else min(best, rel)
        if best is not None and best < 0.5:
            return "number_drift"
    return "substitution"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="d0_pilot")
    ap.add_argument("--out-root", default="exp")
    ap.add_argument("--models", default="Qwen3-Omni-30B-A3B-Instruct,Qwen2.5-Omni-3B")
    ap.add_argument("--conditions", default="SPEAK,ECHO,EF")
    ap.add_argument("--out", default="reports/failure_taxonomy.md")
    args = ap.parse_args()

    run_dir = os.path.join(args.out_root, args.run_id)
    cats = {}
    with open("data/pilot/items.jsonl") as fh:
        for line in fh:
            it = json.loads(line)
            cats[it["id"]] = it["category"]

    lines = ["# 失败解剖：事实是怎么丢的", "",
             f"**运行**: `{args.run_id}` ｜ **生成**: `scripts/probe_taxonomy.py`（零 GPU，只读落盘数据）", ""]
    report = {}
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        p = os.path.join(run_dir, "facts", f"{m}.jsonl")
        if not os.path.exists(p):
            continue
        d = load(run_dir, m)
        lines += [f"## {m}", ""]
        mres = {}
        for cond in args.conditions.split(","):
            counter: Counter = Counter()
            by_cat: dict[str, Counter] = defaultdict(Counter)
            n_up = n_lost = 0
            for iid, cd in d.items():
                up = cd.get(cond + "#internal")
                dn = cd.get(cond)
                if up is None or dn is None:
                    continue
                upk = {key(f) for f in up}
                dnk = {key(f) for f in dn}
                n_up += len(upk)
                for f in up:
                    if key(f) in dnk:
                        continue
                    n_lost += 1
                    t = classify(f, dn)
                    counter[t] += 1
                    by_cat[cats.get(iid, "?")][t] += 1
            if not n_up:
                continue
            mres[cond] = {"n_upstream_facts": n_up, "n_lost": n_lost,
                          "loss_rate": n_lost / n_up, "types": dict(counter),
                          "by_category": {k: dict(v) for k, v in by_cat.items()}}
            lines += [f"### {cond}（上游事实 {n_up}，丢失 {n_lost}，损失率 {n_lost/n_up:.1%}）", "",
                      "| 失败类型 | 条数 | 占丢失 | 贡献 pp |", "|---|---:|---:|---:|"]
            for t, c in counter.most_common():
                lines.append(f"| `{t}` | {c} | {c/n_lost:.1%} | {c/n_up*100:.2f} |")
            lines.append("")
        report[m] = mres

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(run_dir, "metrics", "failure_taxonomy.json"), "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print("\n".join(lines))
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
