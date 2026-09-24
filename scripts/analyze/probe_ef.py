#!/usr/bin/env python3
"""EF 反常探针：判定 `RFG_EF > RFG` 属于三种解释中的哪一种。

分解链（fact 级，全部条件都有"内部文本"，因此每段可独立测量）
--------------------------------------------------------------
    READ ──Δ_perception──▶ LISTEN ──Δ_plan──▶ SPEAK#internal ──Δ_render──▶ SPEAK(回读)

关键量
------
* `Δ_comply`      = 1 − retention(F(EF#internal) ← F(READ))
                    朗读指令是否被**逐字执行**（内容给定，内部文本是否已偏离）
* `Δ_render_EF`   = 1 − retention(F(EF 回读) ← F(EF#internal))
                    内部文本 → 实际语音的**纯渲染**损失（参照系是自己的内部文本，不受感知污染）
* `Δ_render_SPEAK`= 同上，自由生成路径
* `δ_noise`       = **抽取噪声地板**：文本规范化后**完全相同**的 (READ, EF#internal) 对，
                    其事实集合差异。这不是真实损失，是抽取器对表面形式的敏感度。

判定规则
--------
1. 若 `Δ_comply ≲ δ_noise` 且 `Δ_render_EF > Δ_render_SPEAK` → **解释B：渲染给定内容更难**
2. 若 `Δ_comply ≫ δ_noise`                                    → **解释A：朗读时改写了内容**
3. 若 `Δ_render_EF − Δ_render_SPEAK ≲ δ_noise`                 → **解释C：差异是抽取假象**

用法： python scripts/probe_ef.py --run-id d0_pilot
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from rfg.facts.readback import READBACK_MODES, compose_readback
from rfg.facts.schema import Fact
from rfg.score.textnorm import normalize_text

INTERNAL = "#internal"


def load_facts(path: str) -> dict[str, dict[str, set[Fact]]]:
    out: dict[str, dict[str, set[Fact]]] = defaultdict(dict)
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            out[r["item_id"]][r["condition"]] = {
                Fact(d["type"], d["value"], d.get("polarity", "+")) for d in r.get("facts", [])
            }
    return out


def load_texts(pred_root: str, mslug: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)
    mdir = os.path.join(pred_root, mslug)
    for iid in os.listdir(mdir):
        idir = os.path.join(mdir, iid)
        if not os.path.isdir(idir):
            continue
        for f in os.listdir(idir):
            if not f.endswith(".json"):
                continue
            cond = f[: -len(".json")]
            rec = json.load(open(os.path.join(idir, f)))
            if cond in ("SPEAK", "ECHO", "EF"):
                out[iid][cond + INTERNAL] = rec.get("text") or ""
            else:
                out[iid][cond] = rec.get("text") or ""
    return out


def gap(facts: dict, iid: str, down: str, up: str) -> tuple[int, int]:
    fup = facts.get(iid, {}).get(up) or set()
    fdn = facts.get(iid, {}).get(down) or set()
    return len(fup & fdn), len(fup)


def pooled(facts: dict, pairs: list[tuple[str, str]], items: list[str]) -> dict:
    ret = tot = 0
    per_item = {}
    for iid in items:
        r, t = gap(facts, iid, pairs[0], pairs[1])
        ret += r
        tot += t
        if t:
            per_item[iid] = 1 - r / t
    return {"gap": (1 - ret / tot) if tot else None, "retained": ret, "total": tot,
            "per_item": per_item}


def bootstrap_diff(facts: dict, a: tuple[str, str], b: tuple[str, str], items: list[str],
                   n_boot: int = 5000, seed: int = 20260916) -> dict:
    import random

    rng = random.Random(seed)
    pa, pb = pooled(facts, a, items), pooled(facts, b, items)
    if pa["gap"] is None or pb["gap"] is None:
        return {"diff": None, "ci95": None}
    diffs = []
    n = len(items)
    for _ in range(n_boot):
        s = [items[rng.randrange(n)] for _ in range(n)]
        x, y = pooled(facts, a, s)["gap"], pooled(facts, b, s)["gap"]
        if x is not None and y is not None:
            diffs.append(x - y)
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
    return {"diff": pa["gap"] - pb["gap"], "ci95": [lo, hi],
            "excludes_zero": (lo > 0) or (hi < 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="d0_pilot")
    ap.add_argument("--out-root", default="exp")
    ap.add_argument("--models", default="Qwen3-Omni-30B-A3B-Instruct,Qwen2.5-Omni-3B,Qwen2.5-Omni-7B")
    ap.add_argument("--out", default=None)
    ap.add_argument("--readback-mode", choices=READBACK_MODES, default="asr1")
    args = ap.parse_args()

    run_dir = os.path.join(args.out_root, args.run_id)
    facts_dir = os.path.join(run_dir, "facts")
    pred_root = os.path.join(run_dir, "predictions")

    report: dict = {"models": {}}
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        fpath = os.path.join(facts_dir, f"{m}.jsonl")
        if not os.path.exists(fpath):
            continue
        facts = load_facts(fpath)
        facts = {iid: compose_readback(values, args.readback_mode)
                 for iid, values in facts.items()}
        texts = load_texts(pred_root, m)
        items = sorted(i for i in facts if all(
            c in facts[i] for c in ("READ", "LISTEN", "SPEAK", "SPEAK" + INTERNAL)))
        if not items:
            continue

        res: dict = {"n_items": len(items)}
        res["delta_perception"] = pooled(facts, ("LISTEN", "READ"), items)
        res["delta_plan"] = pooled(facts, ("SPEAK" + INTERNAL, "LISTEN"), items)
        res["delta_render_SPEAK"] = pooled(facts, ("SPEAK", "SPEAK" + INTERNAL), items)
        res["delta_total_SPEAK"] = pooled(facts, ("SPEAK", "LISTEN"), items)
        res["delta_comply_EF"] = pooled(facts, ("EF" + INTERNAL, "READ"), items)
        res["delta_comply_ECHO"] = pooled(facts, ("ECHO" + INTERNAL, "READ"), items)
        res["delta_render_EF"] = pooled(facts, ("EF", "EF" + INTERNAL), items)
        res["delta_render_ECHO"] = pooled(facts, ("ECHO", "ECHO" + INTERNAL), items)

        # ---- 抽取噪声地板：文本规范化后完全相同的 (READ, EF#internal) 对
        noise_jac, noise_exact, n_same = [], [], 0
        for iid in items:
            tr = normalize_text(texts.get(iid, {}).get("READ"))
            te = normalize_text(texts.get(iid, {}).get("EF" + INTERNAL))
            if not tr or tr != te:
                continue
            n_same += 1
            a = facts[iid].get("READ") or set()
            b = facts[iid].get("EF" + INTERNAL) or set()
            union = a | b
            if union:
                noise_jac.append(1 - len(a & b) / len(union))
            if a or b:
                noise_exact.append(1 - (len(a & b) / len(a)) if a else 0.0)
        res["noise_floor"] = {
            "n_identical_text_pairs": n_same,
            "mean_false_loss_jaccard": (sum(noise_jac) / len(noise_jac)) if noise_jac else None,
            "mean_false_loss_vs_READ": (sum(noise_exact) / len(noise_exact)) if noise_exact else None,
        }
        res["boot_render_EF_minus_SPEAK"] = bootstrap_diff(
            facts, ("EF", "EF" + INTERNAL), ("SPEAK", "SPEAK" + INTERNAL), items)
        res["boot_comply_EF_minus_noise"] = None  # 噪声地板非配对量，不做配对检验

        # ---- 判定
        d_comply = res["delta_comply_EF"]["gap"]
        d_render_ef = res["delta_render_EF"]["gap"]
        d_render_sp = res["delta_render_SPEAK"]["gap"]
        floor = res["noise_floor"]["mean_false_loss_vs_READ"]
        verdict = "无法判定"
        if None not in (d_comply, d_render_ef, d_render_sp, floor):
            if abs(d_render_ef - d_render_sp) <= max(floor, 0.02):
                verdict = "C：EF 与 SPEAK 的渲染差在抽取噪声地板内（反常是测量假象）"
            elif d_comply <= floor + 0.05:
                verdict = "B：内部文本基本忠实（Δ_comply≈噪声地板），渲染给定内容更难 → 机制线索"
            else:
                verdict = "A：朗读时内部文本已偏离 READ → 指令遵循在计划层失效"
        res["verdict"] = verdict
        report["models"][m] = res

    # ---------------------------------------------------------------- 输出
    lines = ["# EF 反常探针报告", "",
             f"**运行**: `{args.run_id}` ｜ **生成**: `scripts/probe_ef.py`（只读实测值）", "",
             "分解链：`READ --Δ_perception--> LISTEN --Δ_plan--> SPEAK#internal --Δ_render--> SPEAK(回读)`", ""]
    for m, r in report["models"].items():
        lines += [f"## {m}（{r['n_items']} 题）", "",
                  "| 量 | gap | 证据 |", "|---|---:|---|"]
        for k in ("delta_perception", "delta_plan", "delta_render_SPEAK", "delta_total_SPEAK",
                  "delta_comply_EF", "delta_comply_ECHO", "delta_render_EF", "delta_render_ECHO"):
            b = r.get(k) or {}
            g = b.get("gap")
            lines.append(f"| `{k}` | {('%.4f' % g) if g is not None else 'NA'} | {b.get('retained')}/{b.get('total')} |")
        nf = r["noise_floor"]
        lines += ["",
                  f"- **抽取噪声地板 δ**：{nf['n_identical_text_pairs']} 对文本完全相同的样本，"
                  f"平均假损失 = {('%.4f' % nf['mean_false_loss_vs_READ']) if nf['mean_false_loss_vs_READ'] is not None else 'NA'}",
                  f"- 配对 bootstrap `Δ_render_EF − Δ_render_SPEAK`：{r['boot_render_EF_minus_SPEAK']}",
                  f"- **判定：{r['verdict']}**", ""]

    suffix = "" if args.readback_mode == "asr1" else f"_{args.readback_mode}"
    if args.out:
        output = args.out
    elif args.run_id == "d0_pilot":
        output = f"reports/probe_ef{suffix}.md"
    else:
        output = f"reports/probe_ef_{args.run_id}{suffix}.md"
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    report["readback_mode"] = args.readback_mode
    json_path = os.path.join(run_dir, "metrics", f"probe_ef{suffix}.json")
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print("\n".join(lines))
    print(f"written -> {output} | {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
