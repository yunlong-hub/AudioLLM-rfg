#!/usr/bin/env python3
"""定向重渲染的判定：回读 4 个样本，比较原文 vs 改写文本的事实存活。

事实口径
--------
存活判定走**双通道合并集**（规则 ∪ LLM，`extract_dual()`），与随机性/重采样探针同源：
变体文本（ORIG/SPEAK）与 EF 回读文本都现抽，缓存于 `exp/d4_rerender/facts_dual/<model>.jsonl`。
**不能**退回 `extract_rules()`：`content`/`proper_noun` 类目标在纯规则通道里恒不存在，
会被系统性地记成"0 存活"，把"抽取器看不见"伪装成"渲染失败"。

用法： CUDA_VISIBLE_DEVICES=2 python scripts/analyze/judge_rerender.py --model Qwen3-Omni-30B-A3B-Instruct
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

from rfg.facts.extract import DEFAULT_LLM_MODEL, extract_dual  # noqa: E402
from rfg.facts.schema import Fact  # noqa: E402

WHISPER = "/workspace/yunlong/LLM/pretrain_model/Audio/whisper-large-v3"
VARIANTS = ("ORIG", "SPEAK")


def readback_key(iid: str, vname: str, k: int, mode: str) -> str:
    return f"{iid}|{vname}#{k}|{mode}"


def variant_key(iid: str, vname: str) -> str:
    return f"{iid}|{vname}#text"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="d4_rerender")
    ap.add_argument("--model", default="Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--out", default=None)
    ap.add_argument("--readback-mode", choices=("asr1", "asr2", "asr3"), default="asr1",
                    help="用于事实判定的回读；asr1=Whisper，asr2=Seamless，asr3=FunASR")
    ap.add_argument("--cache", default=None,
                    help="双通道抽取缓存；默认 exp/<run-id>/facts_dual/<model>.jsonl")
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    ap.add_argument("--llm-device", default="cuda:0")
    ap.add_argument("--llm-batch-size", type=int, default=16)
    ap.add_argument("--force-llm", action="store_true")
    ap.add_argument("--targets", default=None,
                    help="target manifest; defaults to the historical shared targets.json")
    ap.add_argument("--metrics-out", default=None,
                    help="JSON metric path; defaults to the historical mode-specific path")
    args = ap.parse_args()

    m = args.model
    mode = args.readback_mode
    out_path = args.out or ("reports/rerender.md" if mode == "asr1"
                            else f"reports/rerender_{mode}.md")
    root = os.path.join("exp", args.run_id, m)
    targets_path = args.targets or os.path.join("exp", args.run_id, "metrics", "targets.json")
    targets = json.load(open(targets_path))
    n_targets = sum(len(r.get("facts") or [r.get("fact")]) for r in targets)
    print(f"目标 {n_targets} 个（{len(targets)} 题）", flush=True)
    if not n_targets:
        return 0

    # ---- 1) 回读（脚本只补 Whisper；其他回读由批量回读脚本预先生成）
    todo_asr = []
    for rec in targets:
        iid = rec["item_id"]
        for vname in VARIANTS:
            for wav in sorted(glob.glob(os.path.join(root, iid, f"{vname}_*.wav"))):
                rp = wav[:-4] + ".json"
                r = json.load(open(rp)) if os.path.exists(rp) else {}
                if not r.get(mode):
                    todo_asr.append((rp, wav))
    if todo_asr:
        if mode != "asr1":
            raise RuntimeError(
                f"{mode} 缺少 {len(todo_asr)} 条回读；请先运行对应批量回读脚本"
            )
        from rfg.models.asr import WhisperReadback

        print(f"待回读 {len(todo_asr)} 条，加载 Whisper", flush=True)
        asr = WhisperReadback(WHISPER, device="cuda:0")
        for rp, wav in todo_asr:
            r = json.load(open(rp)) if os.path.exists(rp) else {}
            try:
                r["asr1"] = asr.transcribe(wav).text
            except Exception as e:
                r["asr1_error"] = str(e)
            json.dump(r, open(rp, "w"), ensure_ascii=False)
        del asr

    # ---- 2) 双通道抽取（变体文本 + 全部回读文本）
    texts: dict[str, str] = {}
    for rec in targets:
        iid = rec["item_id"]
        for vname in VARIANTS:
            if rec["variants"].get(vname + "_text"):
                texts[variant_key(iid, vname)] = rec["variants"][vname + "_text"]
            for wav in sorted(glob.glob(os.path.join(root, iid, f"{vname}_*.wav"))):
                rp = wav[:-4] + ".json"
                r = json.load(open(rp)) if os.path.exists(rp) else {}
                k = os.path.basename(wav)[: -len(".wav")].rsplit("_", 1)[-1]
                if r.get(mode):
                    texts[readback_key(iid, vname, int(k), mode)] = r[mode]

    cache_dir = "facts_dual" if mode == "asr1" else f"facts_dual_{mode}"
    cache_path = args.cache or os.path.join("exp", args.run_id, cache_dir, f"{m}.jsonl")
    extracted = extract_dual(texts, cache_path=cache_path, llm_model=args.llm_model,
                             device=args.llm_device, batch_size=args.llm_batch_size,
                             force=args.force_llm)

    # ---- 3) 判定
    results = []
    for rec in targets:
        iid = rec["item_id"]
        fkeys = rec.get("facts") or [rec["fact"]]
        for fkey in fkeys:
            ftype, fval = fkey.split(":", 1)
            fobj = Fact(ftype, fval, "+")
            row = {"item_id": iid, "fact": fkey, "type": ftype, "surv": {}, "n": {}}
            for vname in VARIANTS:
                ks = sorted(k for k in texts if k.startswith(f"{iid}|{vname}#")
                            and not k.endswith("#text"))
                row["surv"][vname] = sum(1 for k in ks if fobj in extracted[k].facts)
                row["n"][vname] = len(ks)
            results.append(row)
            print(f"  {iid:10s} {fkey:28s} ORIG {row['surv']['ORIG']}/{row['n']['ORIG']} "
                  f"-> SPEAK {row['surv']['SPEAK']}/{row['n']['SPEAK']}", flush=True)

    n_items = len(results)
    improved = [r for r in results if r["surv"]["SPEAK"] > r["surv"]["ORIG"]]
    same = [r for r in results if r["surv"]["SPEAK"] == r["surv"]["ORIG"]]
    worse = [r for r in results if r["surv"]["SPEAK"] < r["surv"]["ORIG"]]
    base = sum(r["surv"]["ORIG"] for r in results) / max(sum(r["n"]["ORIG"] for r in results), 1)
    rew = sum(r["surv"]["SPEAK"] for r in results) / max(sum(r["n"]["SPEAK"] for r in results), 1)

    # 按事实类型分解（改写针对数字/单位，内容类不受益）
    by_type: dict[str, dict] = defaultdict(lambda: {"n_facts": 0, "orig": 0, "speak": 0,
                                                    "n_orig": 0, "n_speak": 0, "improved": 0})
    for r in results:
        b = by_type[r["type"]]
        b["n_facts"] += 1
        b["orig"] += r["surv"]["ORIG"]
        b["speak"] += r["surv"]["SPEAK"]
        b["n_orig"] += r["n"]["ORIG"]
        b["n_speak"] += r["n"]["SPEAK"]
        b["improved"] += int(r["surv"]["SPEAK"] > r["surv"]["ORIG"])

    lines = ["# 定向重渲染：能否压掉结构性下限", "",
             f"**模型**: `{m}` ｜ **目标**: {n_items} 个结构性失败事实（4 样本从未说出）｜ "
             f"**每变体样本数**: 4 ｜ **条件**: EF（给定文本朗读）", "",
             "| 变体 | 事实存活率（跨全部样本） |", "|---|---:|",
             f"| `ORIG`（原内部文本） | **{base:.3f}** |",
             f"| `SPEAK`（定向改写） | **{rew:.3f}** |",
             "", f"- 改写后**改善**的目标：{len(improved)}/{n_items}",
             f"- 无变化：{len(same)}/{n_items}",
             f"- 变差：{len(worse)}/{n_items}",
             "", "## 按事实类型分解（改写只作用于数字与单位）", "",
             "| 类型 | 目标数 | ORIG 存活率 | SPEAK 存活率 | 改善目标数 |", "|---|---:|---:|---:|---:|"]
    for t, b in sorted(by_type.items(), key=lambda kv: -kv[1]["n_facts"]):
        o = b["orig"] / b["n_orig"] if b["n_orig"] else float("nan")
        s = b["speak"] / b["n_speak"] if b["n_speak"] else float("nan")
        lines.append(f"| `{t}` | {b['n_facts']} | {o:.3f} | {s:.3f} | {b['improved']}/{b['n_facts']} |")
    lines += ["", "## 明细", "", "| 题目 | 事实 | ORIG 存活 | SPEAK 存活 |", "|---|---|---:|---:|"]
    for r in sorted(results, key=lambda x: (x["surv"]["SPEAK"] - x["surv"]["ORIG"]), reverse=True):
        lines.append(f"| {r['item_id']} | `{r['fact']}` | {r['surv']['ORIG']}/{r['n']['ORIG']} | "
                     f"{r['surv']['SPEAK']}/{r['n']['SPEAK']} |")
    lines += ["", "---", "",
              "**口径**：事实为**双通道合并集**（规则通道 ∪ LLM 通道），变体文本与 EF 回读文本现抽并缓存于 "
              f"`{os.path.relpath(cache_path, _ROOT)}`；目标来自 `probe_rerender.py` 的双通道结构性失败定义。", "",
              "**判读**：若 `SPEAK` 存活率显著高于 `ORIG`，则定向改写是针对**结构性下限**的有效零训练缓解，"
              "与针对随机成分的 FRR 互补；若两者接近，说明这些事实对该渲染器属于不可恢复的硬约束。"]

    os.makedirs("reports", exist_ok=True)
    open(out_path, "w").write("\n".join(lines) + "\n")
    metrics_path = args.metrics_out or os.path.join(
        "exp", args.run_id, "metrics",
        "rerender_judge.json" if mode == "asr1" else f"rerender_judge_{mode}.json",
    )
    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    json.dump({"base": base, "rewritten": rew, "n": n_items, "n_items_with_targets": len(targets),
               "improved": len(improved), "same": len(same), "worse": len(worse),
               "by_type": {t: {**b,
                               "orig_rate": b["orig"] / b["n_orig"] if b["n_orig"] else None,
                               "speak_rate": b["speak"] / b["n_speak"] if b["n_speak"] else None}
                           for t, b in by_type.items()},
               "cache": os.path.relpath(cache_path, _ROOT), "rows": results},
              open(metrics_path, "w"),
              ensure_ascii=False, indent=2)
    print("\n".join(lines[:20]))
    print(f"written -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
