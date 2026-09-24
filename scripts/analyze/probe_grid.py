#!/usr/bin/env python3
"""2×2 机制分析：**输入模态 × 内容来源** 对渲染保真度的影响。

设计（四个格子全部可测，因为每个语音条件都保存了内部文本）

|                | 自生成内容        | 给定内容（要求逐字朗读） |
|----------------|-------------------|--------------------------|
| **文本输入**   | ECHO              | EF（朗读 READ 文本）     |
| **语音输入**   | SPEAK             | EFA（朗读 LISTEN 文本）  |

另外有一对**严格模态对照**：
* `EF`   = 文本问 + 朗读 READ 文本
* `EFB`  = 语音问 + 朗读**同一个** READ 文本
两者目标句完全相同，唯一差别是输入模态 → `Δ_render(EFB) − Δ_render(EF)` 是输入模态对渲染的**因果效应**。

所有 gap 的定义：`Δ_render(c) = 1 − |F(c) ∩ F(c#internal)| / |F(c#internal)|`
（参照系是该条件**自己的内部文本**，因此不受感知误差与上游条件污染）

用法： python scripts/probe_grid.py --run-id d0_pilot
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_ef import INTERNAL, bootstrap_diff, load_facts, pooled  # noqa: E402

def readback_coverage(facts: dict, cond: str, items: list[str]) -> float:
    """该条件有多少题**真的存在回读事实**。

    缺回读会把 Δ_render 假报成 1.0（曾有 200 题全部无回读的情况），因此覆盖率不足时必须报 NA。
    """
    n = sum(1 for i in items if facts.get(i, {}).get(cond + INTERNAL) and facts[i].get(cond))
    return n / len(items) if items else 0.0


CELLS = {"ECHO": ("text", "self"), "SPEAK": ("audio", "self"),
         "EF": ("text", "given"), "EFA": ("audio", "given")}
GIVEN_SOURCE = {"EF": "READ", "EFA": "LISTEN", "EFB": "READ"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="d0_pilot")
    ap.add_argument("--out-root", default="exp")
    ap.add_argument("--models", default="Qwen3-Omni-30B-A3B-Instruct,Qwen2.5-Omni-3B")
    ap.add_argument("--out", default="reports/mechanism.md")
    args = ap.parse_args()

    run_dir = os.path.join(args.out_root, args.run_id)
    facts_dir = os.path.join(run_dir, "facts")

    report: dict = {"models": {}}
    lines = ["# 机制分析：输入模态 × 内容来源对渲染保真度的影响", "",
             f"**运行**: `{args.run_id}` ｜ **生成**: `scripts/probe_grid.py`（只读实测值）", "",
             "`Δ_render(c) = 1 − |F(c) ∩ F(c#internal)| / |F(c#internal)|`，"
             "参照系是该条件自身的内部文本。", ""]

    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        fpath = os.path.join(facts_dir, f"{m}.jsonl")
        if not os.path.exists(fpath):
            continue
        facts = load_facts(fpath)
        items = sorted(i for i in facts if all(
            c in facts[i] for c in ("READ", "LISTEN", "SPEAK", "SPEAK" + INTERNAL)))
        if not items:
            continue
        res: dict = {"n_items": len(items)}

        # ---- 2×2 四格
        res["grid"] = {}
        for cond, (modality, origin) in CELLS.items():
            if cond + INTERNAL not in facts[items[0]]:
                continue
            cov = readback_coverage(facts, cond, items)
            if cov < 0.5:      # 回读缺失/未跑 → 拒绝给出数值
                res.setdefault("invalid_cells", {})[cond] = {
                    "reason": "readback missing", "coverage": round(cov, 3)}
                continue
            res["grid"][cond] = {"input_modality": modality, "content_origin": origin,
                                 "readback_coverage": round(cov, 3),
                                 **{k: v for k, v in pooled(facts, (cond, cond + INTERNAL), items).items()
                                    if k != "per_item"}}

        # ---- 给定内容条件：内部文本是否忠实于被要求朗读的文本（指令遵循）
        res["comply"] = {}
        for cond, src in GIVEN_SOURCE.items():
            if cond + INTERNAL in facts[items[0]] and src in facts[items[0]]:
                res["comply"][cond] = {k: v for k, v in
                                       pooled(facts, (cond + INTERNAL, src), items).items()
                                       if k != "per_item"}

        # ---- 严格模态对照 EF vs EFB（目标句相同）
        if "EFB" + INTERNAL in facts[items[0]]:
            res["modality_control_EF_vs_EFB"] = bootstrap_diff(
                facts, ("EFB", "EFB" + INTERNAL), ("EF", "EF" + INTERNAL), items)

        # ---- 内容来源效应（同输入模态内）
        if "EFA" + INTERNAL in facts[items[0]]:
            res["origin_effect_audio_given_minus_self"] = bootstrap_diff(
                facts, ("EFA", "EFA" + INTERNAL), ("SPEAK", "SPEAK" + INTERNAL), items)
        res["origin_effect_text_given_minus_self"] = bootstrap_diff(
            facts, ("EF", "EF" + INTERNAL), ("ECHO", "ECHO" + INTERNAL), items)

        # ---- 输入模态效应（同内容来源内）
        res["modality_effect_self_text_minus_audio"] = bootstrap_diff(
            facts, ("ECHO", "ECHO" + INTERNAL), ("SPEAK", "SPEAK" + INTERNAL), items)
        if "EFA" + INTERNAL in facts[items[0]]:
            res["modality_effect_given_text_minus_audio"] = bootstrap_diff(
                facts, ("EF", "EF" + INTERNAL), ("EFA", "EFA" + INTERNAL), items)

        report["models"][m] = res

        # ---- 输出
        lines += [f"## {m}（{res['n_items']} 题）", "", "### 渲染保真度 2×2", "",
                  "| 输入模态 | 内容来源 | 条件 | Δ_render | 证据 |", "|---|---|---|---:|---|"]
        for cond in ("ECHO", "EF", "SPEAK", "EFA"):
            b = res["grid"].get(cond)
            if not b:
                continue
            g = b["gap"]
            lines.append(f"| {b['input_modality']} | {b['content_origin']} | `{cond}` | "
                         f"{('%.4f' % g) if g is not None else 'NA'} | {b['retained']}/{b['total']} |")
        lines += ["", "### 朗读指令遵循（内部文本 ← 被要求朗读的文本）", "",
                  "| 条件 | Δ_comply | 证据 |", "|---|---:|---|"]
        for cond, b in res["comply"].items():
            g = b["gap"]
            lines.append(f"| `{cond}` | {('%.4f' % g) if g is not None else 'NA'} | {b['retained']}/{b['total']} |")
        lines += ["", "### 效应量（按题聚类配对 bootstrap）", "",
                  "| 对比 | 差值 | 95% CI | 排除 0 |", "|---|---:|---|---|"]
        for key, label in (
            ("modality_control_EF_vs_EFB", "**输入模态**（EFB语音 − EF文本，目标句相同）"),
            ("modality_effect_self_text_minus_audio", "输入模态（自生成：ECHO − SPEAK）"),
            ("modality_effect_given_text_minus_audio", "输入模态（给定：EF − EFA）"),
            ("origin_effect_text_given_minus_self", "内容来源（文本输入：EF − ECHO）"),
            ("origin_effect_audio_given_minus_self", "内容来源（语音输入：EFA − SPEAK）"),
        ):
            b = res.get(key)
            if not b:
                continue
            d = b.get("diff")
            lines.append(f"| {label} | {('%.4f' % d) if d is not None else 'NA'} | "
                         f"{b.get('ci95')} | {b.get('excludes_zero')} |")
        lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(run_dir, "metrics", "probe_grid.json"), "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print("\n".join(lines))
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
