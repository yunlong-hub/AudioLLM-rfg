#!/usr/bin/env python3
"""渲染损失的长度控制分析：Δ_render 到底由什么决定？

动机
----
2×2 显示同目标句的模态对照（EFB vs EF）不显著，但 ECHO 与 SPEAK 差 +0.082 显著。
两者都是自生成，差别在**要说的内容**（文本问答 vs 语音问答的输出可能长短/复杂度不同）。
若 Δ_render 主要由内部文本长度驱动，则"模态效应"其实是长度效应。

本脚本逐题计算
* `text_len`  : 该条件内部文本的词数
* `audio_dur` : 该条件音频时长（秒）
* `render_gap`: 该题在该条件的 Δ_render（事实级）
并给出
1. 各条件的长度/时长分布（解释 ECHO 与 SPEAK 的差异从何而来）
2. Δ_render 与长度/时长的 Spearman ρ（**长度分层**后效应是否还在）
3. 长度匹配子样本上的 ECHO − SPEAK 配对差（把长度控制掉之后的模态效应）

用法： python scripts/probe_length.py --run-id d0_pilot
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_ef import INTERNAL, load_facts  # noqa: E402
from rfg.score.metrics import spearman_rho  # noqa: E402

CONDS = ("SPEAK", "ECHO", "EF", "EFA", "EFB")


def load_meta(pred_root: str, mslug: str) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    mdir = os.path.join(pred_root, mslug)
    for iid in os.listdir(mdir):
        idir = os.path.join(mdir, iid)
        if not os.path.isdir(idir):
            continue
        for cond in CONDS:
            p = os.path.join(idir, f"{cond}.json")
            if os.path.exists(p):
                r = json.load(open(p))
                out[iid][cond] = {"text_len": len((r.get("text") or "").split()),
                                  "audio_dur": r.get("audio_duration_sec")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="d0_pilot")
    ap.add_argument("--out-root", default="exp")
    ap.add_argument("--models", default="Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--out", default="reports/length_control.md")
    args = ap.parse_args()

    run_dir = os.path.join(args.out_root, args.run_id)
    lines = ["# 长度控制分析：Δ_render 由什么决定", "",
             f"**运行**: `{args.run_id}` ｜ **生成**: `scripts/probe_length.py`", ""]
    report: dict = {}

    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        fpath = os.path.join(run_dir, "facts", f"{m}.jsonl")
        if not os.path.exists(fpath):
            continue
        facts = load_facts(fpath)
        meta = load_meta(os.path.join(run_dir, "predictions"), m)
        items = sorted(i for i in facts if "SPEAK" in facts[i])

        lines += [f"## {m}（{len(items)} 题）", "", "### 1. 各条件的内部文本长度与音频时长", "",
                  "| 条件 | 内部文本词数(均值) | 音频时长s(均值) | Δ_render |", "|---|---:|---:|---:|"]
        res: dict = {}
        for cond in CONDS:
            tl = [meta[i][cond]["text_len"] for i in items if cond in meta.get(i, {})]
            ad = [meta[i][cond]["audio_dur"] for i in items
                  if cond in meta.get(i, {}) and meta[i][cond]["audio_dur"]]
            ret = tot = 0
            for i in items:
                up = facts[i].get(cond + INTERNAL) or set()
                dn = facts[i].get(cond) or set()
                ret += len(up & dn)
                tot += len(up)
            res[cond] = {"mean_text_len": (sum(tl) / len(tl)) if tl else None,
                         "mean_audio_dur": (sum(ad) / len(ad)) if ad else None,
                         "render_gap": (1 - ret / tot) if tot else None}
            b = res[cond]
            lines.append(f"| `{cond}` | {b['mean_text_len']:.1f} | {b['mean_audio_dur']:.2f} | "
                         f"{b['render_gap']:.4f} |")

        # ---- 2. Δ_render 与长度/时长的相关（逐题、按条件）
        lines += ["", "### 2. 逐题 Δ_render 与长度/时长的 Spearman ρ", "",
                  "| 条件 | ρ(gap, 内部文本词数) | ρ(gap, 音频时长) | n |", "|---|---:|---:|---:|"]
        for cond in CONDS:
            xs_len, xs_dur, ys = [], [], []
            for i in items:
                up = facts[i].get(cond + INTERNAL) or set()
                dn = facts[i].get(cond) or set()
                if not up or cond not in meta.get(i, {}):
                    continue
                gap = 1 - len(up & dn) / len(up)
                xs_len.append(meta[i][cond]["text_len"])
                if meta[i][cond]["audio_dur"]:
                    xs_dur.append(meta[i][cond]["audio_dur"])
                else:
                    xs_dur.append(0.0)
                ys.append(gap)
            rho_len = spearman_rho(xs_len, ys) if len(ys) >= 3 else None
            rho_dur = spearman_rho(xs_dur, ys) if len(ys) >= 3 else None
            res.setdefault("spearman", {})[cond] = {"rho_text_len": rho_len, "rho_audio_dur": rho_dur,
                                                    "n": len(ys)}
            lines.append(f"| `{cond}` | {rho_len if rho_len is None else round(rho_len,3)} | "
                         f"{rho_dur if rho_dur is None else round(rho_dur,3)} | {len(ys)} |")

        # ---- 3. 长度匹配子样本上的 ECHO − SPEAK
        pairs = []
        for i in items:
            if "ECHO" not in meta.get(i, {}) or "SPEAK" not in meta.get(i, {}):
                continue
            le, ls = meta[i]["ECHO"]["text_len"], meta[i]["SPEAK"]["text_len"]
            if not le or not ls or abs(le - ls) / max(le, ls) > 0.15:
                continue
            up_e = facts[i].get("ECHO" + INTERNAL) or set()
            up_s = facts[i].get("SPEAK" + INTERNAL) or set()
            if not up_e or not up_s:
                continue
            pairs.append((1 - len(up_e & (facts[i].get("ECHO") or set())) / len(up_e),
                          1 - len(up_s & (facts[i].get("SPEAK") or set())) / len(up_s)))
        if pairs:
            d = [a - b for a, b in pairs]
            d.sort()
            lines += ["", "### 3. 长度匹配后（内部文本词数差 <15%）的 ECHO − SPEAK", "",
                      f"- 匹配题数：{len(pairs)}（全样本 {len(items)}）",
                      f"- Δ_render 均值差：**{sum(d)/len(d):+.4f}**",
                      f"- 逐题差的中位数：{d[len(d)//2]:+.4f}",
                      f"- 差为正（ECHO 更差）的题占比：{sum(1 for x in d if x > 0)/len(d):.1%}"]
            res["length_matched_echo_minus_speak"] = {
                "n_matched": len(pairs), "mean_diff": sum(d) / len(d),
                "median_diff": d[len(d) // 2],
                "frac_echo_worse": sum(1 for x in d if x > 0) / len(d)}
        lines.append("")
        report[m] = res

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(run_dir, "metrics", "probe_length.json"), "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print("\n".join(lines))
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
