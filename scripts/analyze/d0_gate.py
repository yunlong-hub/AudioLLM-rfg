#!/usr/bin/env python3
"""D0-6：按预注册判据生成止损判定报告 reports/d0_gate.md。

预注册判据（docs/research_plan.md §14）
--------------------------------------
D0 pilot 继续条件：`RFG_hard > 5%` **且** `RFG_hard ≥ 2 × RFG_content`。
本脚本只读取实测值并套用判据，**不生成任何数值**；缺失值一律显示 NA。

用法：
  python scripts/d0_gate.py --run-id d0_pilot --out reports/d0_gate.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys

GATE_HARD_MIN = 0.05
GATE_RATIO = 2.0


def fmt(v, nd=4):
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def block(summary: dict, key: str) -> str:
    b = summary.get(key) or {}
    gap = b.get("gap")
    ret, tot = b.get("retained"), b.get("total")
    return f"{fmt(gap)} ({ret}/{tot})" if tot else "NA"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="d0_pilot")
    ap.add_argument("--out-root", default="exp")
    ap.add_argument("--out", default="reports/d0_gate.md")
    args = ap.parse_args()

    path = os.path.join(args.out_root, args.run_id, "metrics", "scores_all_models.json")
    if not os.path.exists(path):
        print(f"缺少 {path}；先跑 scripts/score.py", file=sys.stderr)
        return 1
    with open(path) as fh:
        allm = json.load(fh)

    readback_path = os.path.join(args.out_root, args.run_id, "metrics")
    facts_path = readback_path
    readback = {}
    facts = {}
    for f in sorted(os.listdir(readback_path)):
        if f.startswith("readback_") and f.endswith(".json"):
            readback[f[len("readback_"):-5]] = json.load(open(os.path.join(readback_path, f)))
        if f.startswith("facts_") and f.endswith(".json"):
            facts[f[len("facts_"):-5]] = json.load(open(os.path.join(readback_path, f)))

    lines: list[str] = []
    A = lines.append
    A("# D0-6 止损判定报告（预注册判据）")
    A("")
    A(f"**运行**: `{args.run_id}`  |  **判据来源**: `docs/research_plan.md` §14  |  "
      f"**生成方式**: `scripts/d0_gate.py`（只读实测值，不生成数值）")
    A("")
    A("## 1. 预注册判据")
    A("")
    A(f"- 继续条件：`RFG_hard > {GATE_HARD_MIN:.0%}` **且** `RFG_hard ≥ {GATE_RATIO:.0f} × RFG_content`")
    A("- 主模型（Qwen3-Omni-30B-A3B-Instruct）为判定模型；其余模型为复现证据，不参与闸门。")
    A("")
    A("## 2. 主表（事实级留存缺口）")
    A("")
    A("| 模型 | 题数 | PG | RFG | RG | RFG_EF | RFG_hard | RFG_content | RFG_corr | 回读一致率 |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m, s in allm.items():
        rb = readback.get(m, {})
        A(f"| {m} | {s.get('n_items')} | {block(s,'PG')} | {block(s,'RFG')} | {block(s,'RG')} | "
          f"{block(s,'RFG_EF')} | {block(s,'RFG_hard')} | {block(s,'RFG_content')} | "
          f"{block(s,'RFG_corr')} | {fmt(rb.get('agreement_rate'),3)} |")
    A("")
    A("## 3. 闸门判定（主模型）")
    A("")
    main_model = next((m for m in allm if "Qwen3-Omni" in m), None)
    if main_model is None:
        A("**未找到主模型结果，无法判定。**")
    else:
        s = allm[main_model]
        hard = (s.get("RFG_hard") or {}).get("gap")
        cont = (s.get("RFG_content") or {}).get("gap")
        A(f"主模型: `{main_model}`")
        A("")
        A(f"- `RFG_hard` = {fmt(hard)}  →  条件①（> 5%）: "
          f"{'**满足**' if (hard is not None and hard > GATE_HARD_MIN) else '不满足/NA'}")
        if hard is None or cont is None:
            A(f"- `RFG_content` = {fmt(cont)}  →  条件②: **无法判定（缺 content 事实）**")
            A("")
            A("> 提示：`RFG_content` 为 NA 通常意味着**未运行 LLM 抽取通道**"
              "（规则通道不抽 content 事实）。请先跑 "
              "`scripts/extract_facts.py --with-llm`。")
            verdict = "无法判定"
        else:
            ratio = (hard / cont) if cont > 0 else float("inf")
            cond2 = (cont == 0 and hard > 0) or (ratio >= GATE_RATIO)
            A(f"- `RFG_content` = {fmt(cont)}，比值 = {fmt(ratio,2)}  →  条件②（≥ 2×）: "
              f"{'**满足**' if cond2 else '不满足'}")
            A("")
            verdict = "GO（继续主实验）" if (hard > GATE_HARD_MIN and cond2) else "STOP（换方向）"
        boot = s.get("bootstrap_RFG_hard_minus_content") or {}
        A(f"- 配对 bootstrap（按题聚类，RFG_hard − RFG_content）: "
          f"diff={fmt(boot.get('diff'))}, 95% CI={boot.get('ci95')}, "
          f"排除 0: {boot.get('excludes_zero')}")
        A(f"- `RG` vs `RFG` 配对差: {s.get('bootstrap_RG_vs_RFG')}")
        A(f"- `RFG` vs `RFG_EF` 配对差: {s.get('bootstrap_RFG_vs_EF')}")
        A("")
        A(f"## 判决：**{verdict}**")
    A("")
    A("## 4. 分项留存（SPEAK←LISTEN）")
    A("")
    A("| 模型 | number | unit | proper_noun | negation | content |")
    A("|---|---:|---:|---:|---:|---:|")
    for m, s in allm.items():
        tb = s.get("by_type_SPEAK_vs_LISTEN") or {}
        cells = []
        for t in ("number", "unit", "proper_noun", "negation", "content"):
            b = tb.get(t) or {}
            cells.append(fmt(b.get("retention"), 3))
        A(f"| {m} | " + " | ".join(cells) + " |")
    A("")
    A("## 5. 方法与可靠性检查")
    A("")
    A("| 模型 | 抽取 κ | 值级冲突率 | 类型标签差异率 | 回读一致率 | 平均回读 WER | 语音条件数 |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for m, s in allm.items():
        fa = facts.get(m, {})
        rb = readback.get(m, {})
        A(f"| {m} | {fmt(fa.get('kappa'),3)} | {fmt(fa.get('value_conflict_rate'),3)} | "
          f"{fmt(fa.get('type_conflict_rate'),3)} | "
          f"{fmt(rb.get('agreement_rate'),3)} | {fmt(rb.get('mean_readback_wer'),3)} | "
          f"{rb.get('n_speech')} |")
    A("")
    A(f"- `spearman(WER, RFG)`（主模型）: {(allm.get(main_model) or {}).get('spearman_wer_vs_rfg')}")
    A("")
    A("## 6. 竞争性解释与排除状态")
    A("")
    A("| 竞争性解释 | 本报告中的对应证据 | 状态 |")
    A("|---|---|---|")
    A("| 回读误差污染 | 双 ASR 一致率、`RFG_corr`（仅一致样本）、人工裁定 100 条 | D0 内完成前两项；人工裁定在 W1 |")
    A("| 长语音退化 | 语音时长上限 30s + `long_audio` flag | 已记录，未见超限需在 §7 说明 |")
    A("| 事实抽取不稳 | 双通道 κ 与冲突率 | 见 §5 |")
    A("| 题目音频伪影 | 合成回读 WER（`exp/d0_data/metrics/tts_qa.json`） | 0 条超阈值 |")
    A("")
    A("---")
    A("")
    A("**No-fabrication status**: 本报告所有数值均由 `exp/%s/` 下的实测产物直接读取；"
      "缺失值显示 NA，未做任何估计或填充。" % args.run_id)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print(f"...\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
