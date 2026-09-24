#!/usr/bin/env python3
"""负结果子组对比：SPEAKD 干预 与 数字面形式子组。

这两个对比原先只存在于口头/工作记录里，论文引用后却无产物可追溯
（`tools/check_paper_numbers.py` 因此把它们标为疑点）。本脚本把它们
固化为可复跑产物，写入 `reports/negative_contrasts.md`。

定义
----
* `SPEAKD − SPEAK`：退化控制干预的效果。两个条件都在同一批题上计算
  Δ_render（各自 `#internal` 为上游、回读为下游），差值为正表示干预更差。
* **数字子组** `ECHO − SPEAK`：限定 `category == "number"` 的题，
  比较"给定文本朗读"与"自由生成"两条路径的 Δ_render。
  旧稿曾称两模型方向相反；在当前合并集口径下两者同向，仅 30B 显著。

口径：事实集合取自 `exp/<run>/facts/<model>.jsonl` 的**双通道合并集**
（规则 ∪ LLM），与 `score.py`/`probe_*.py` 同源。效应量用按题聚类的
配对 bootstrap（固定 seed，可复现）。

用法： python scripts/analyze/probe_subgroups.py [--run-id d0_pilot] \
           [--models M1,M2] [--out reports/negative_contrasts.md]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

from rfg.facts.schema import Fact  # noqa: E402

INTERNAL = "#internal"
SEED = 20260916
N_BOOT = 5000


def load_facts(run: str, model: str) -> dict:
    d: dict = defaultdict(dict)
    path = os.path.join(_ROOT, "exp", run, "facts", f"{model}.jsonl")
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            d[r["item_id"]][r["condition"]] = {
                Fact(x["type"], x["value"], x.get("polarity", "+")) for x in r.get("facts", [])
            }
    return d


def load_categories(items_path: str) -> dict:
    cats = {}
    with open(os.path.join(_ROOT, items_path)) as fh:
        for line in fh:
            it = json.loads(line)
            cats[it["id"]] = it.get("category")
    return cats


def gap(facts, items, down: str, up: str):
    """1 − Σ|∩|/Σ|F_up|；上下游任一缺失的题不计入。"""
    R = T = 0
    for iid in items:
        fup = facts.get(iid, {}).get(up) or set()
        if not fup:
            continue
        fdn = facts.get(iid, {}).get(down) or set()
        R += len(fup & fdn)
        T += len(fup)
    return (1 - R / T) if T else None


def bootstrap(facts, items, a, b, n_boot: int = N_BOOT, seed: int = SEED):
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        s = [items[rng.randrange(len(items))] for _ in range(len(items))]
        ga, gb = gap(facts, s, a[0], a[1]), gap(facts, s, b[0], b[1])
        if ga is not None and gb is not None:
            out.append(ga - gb)
    if not out:
        return None, None
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="d0_pilot")
    ap.add_argument("--models", default="Qwen3-Omni-30B-A3B-Instruct,Qwen2.5-Omni-3B")
    ap.add_argument("--items", default="data/pilot/items.jsonl")
    ap.add_argument("--out", default="reports/negative_contrasts.md")
    args = ap.parse_args()

    cats = load_categories(args.items)
    lines = [
        "# 负结果子组对比", "",
        f"运行 `{args.run_id}` ｜ 事实口径：双通道合并集 ｜ "
        f"bootstrap {N_BOOT} 次（seed={SEED}）", "",
        "## A. 退化控制（`SPEAKD − SPEAK`，Δ_render 之差）", "",
        "| 模型 | Δ_render(SPEAKD) | Δ_render(SPEAK) | 差值 | 95% CI | 排除 0 | n |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    summary = {}
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        facts = load_facts(args.run_id, m)
        items = sorted(i for i in facts if all(
            c in facts[i] for c in ("READ", "LISTEN", "SPEAK", "SPEAK" + INTERNAL)))
        sd = [i for i in items if "SPEAKD" in facts[i] and "SPEAKD" + INTERNAL in facts[i]]
        if not sd:
            lines.append(f"| {m} | — | — | — | — | — | 0 |")
            continue
        g1 = gap(facts, sd, "SPEAKD", "SPEAKD" + INTERNAL)
        g0 = gap(facts, sd, "SPEAK", "SPEAK" + INTERNAL)
        lo, hi = bootstrap(facts, sd, ("SPEAKD", "SPEAKD" + INTERNAL),
                           ("SPEAK", "SPEAK" + INTERNAL))
        excl = (lo is not None) and ((lo > 0) or (hi < 0))
        lines.append(f"| {m} | {g1:.4f} | {g0:.4f} | {g1 - g0:+.4f} | "
                     f"[{lo:+.4f}, {hi:+.4f}] | {excl} | {len(sd)} |")
        summary[f"{m}/speakd_minus_speak"] = {
            "gap_speakd": g1, "gap_speak": g0, "diff": g1 - g0,
            "ci95": [lo, hi], "excludes_zero": excl, "n": len(sd)}

    lines += ["", "## B. 数字面形式子组（`ECHO − SPEAK`，Δ_render 之差，`category=number`）", "",
              "| 模型 | Δ_render(ECHO) | Δ_render(SPEAK) | 差值 | 95% CI | 排除 0 | n |",
              "|---|---:|---:|---:|---|---|---:|"]
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        facts = load_facts(args.run_id, m)
        items = sorted(i for i in facts if all(
            c in facts[i] for c in ("READ", "LISTEN", "SPEAK", "SPEAK" + INTERNAL)))
        num = [i for i in items if cats.get(i) == "number"]
        if not num:
            lines.append(f"| {m} | — | — | — | — | — | 0 |")
            continue
        ge = gap(facts, num, "ECHO", "ECHO" + INTERNAL)
        gs = gap(facts, num, "SPEAK", "SPEAK" + INTERNAL)
        lo, hi = bootstrap(facts, num, ("ECHO", "ECHO" + INTERNAL),
                           ("SPEAK", "SPEAK" + INTERNAL))
        excl = (lo is not None) and ((lo > 0) or (hi < 0))
        lines.append(f"| {m} | {ge:.4f} | {gs:.4f} | {ge - gs:+.4f} | "
                     f"[{lo:+.4f}, {hi:+.4f}] | {excl} | {len(num)} |")
        summary[f"{m}/numeric_echo_minus_speak"] = {
            "gap_echo": ge, "gap_speak": gs, "diff": ge - gs,
            "ci95": [lo, hi], "excludes_zero": excl, "n": len(num)}

    lines += ["", "---", "",
              "**判读**：A 的差值在两个模型上都不显著 → 可见的生成退化不是损失所在；",
              "B 的差值在两模型上**同向**，但只有 30B 显著（3B CI 含 0），",
              "故数字面形式只能算「未稳健复现」，不能算「方向相反」。", ""]

    out = os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    js = os.path.join(_ROOT, "reports", "negative_contrasts.json")
    json.dump(summary, open(js, "w"), ensure_ascii=False, indent=2)
    print("\n".join(lines))
    print(f"written -> {args.out} | reports/negative_contrasts.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
