#!/usr/bin/env python3
"""D0-5b：计算 PG / RFG / RG / RFG_EF / 分项 / RFG_corr 与区间估计。

依赖 `scripts/extract_facts.py` 产出的 exp/<run_id>/facts/<model>.jsonl。
逐题落盘 scores/<model>.jsonl（含每题各条件事实与留存），汇总写 metrics/scores_<model>.json。

用法：
  python scripts/score.py --run-id d0_pilot
  python scripts/score.py --run-id d0_pilot --model Qwen3-Omni-30B-A3B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

from rfg.facts.readback import READBACK_MODES, compose_readback
from rfg.facts.schema import Fact
from rfg.score.metrics import ItemFacts, group_by_category, summarize

CONDITIONS = ("READ", "LISTEN", "SPEAK", "ECHO", "EF")
SHARD_CACHE_RE = re.compile(r"_shard\d+-of-\d+\.jsonl$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="d0_pilot")
    ap.add_argument("--out-root", default="exp")
    ap.add_argument("--items", default="data/pilot/items.jsonl")
    ap.add_argument("--model", default=None)
    ap.add_argument("--model-only", action="store_true",
                    help="只重算 --model，并保留 scores_all_models 中其他模型的既有结果")
    ap.add_argument("--readback-mode", choices=READBACK_MODES, default="asr1")
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args()
    if args.model_only and not args.model:
        ap.error("--model-only requires --model")

    run_dir = os.path.join(args.out_root, args.run_id)
    facts_dir = os.path.join(run_dir, "facts")
    if not os.path.isdir(facts_dir):
        print(f"缺少 {facts_dir}；先跑 scripts/extract_facts.py", file=sys.stderr)
        return 1

    category = {}
    with open(args.items) as fh:
        for line in fh:
            it = json.loads(line)
            category[it["id"]] = it["category"]

    # 始终汇总**全部**有事实文件的模型：--model 只影响打印范围，
    # 否则逐模型调用会把 scores_all_models.json 覆盖成单模型，闸门报告随之失真。
    # Parallel extraction leaves auditable intermediate caches beside the canonical
    # model cache. They are partitions, not models, and must never enter scoring.
    all_files = sorted(
        f for f in os.listdir(facts_dir)
        if f.endswith(".jsonl") and not SHARD_CACHE_RE.search(f)
    )
    targets = [f"{args.model}.jsonl"] if args.model else all_files
    files_to_score = targets if args.model_only else all_files
    all_suffix = "" if args.readback_mode == "asr1" else f"_{args.readback_mode}"
    all_path = os.path.join(run_dir, "metrics", f"scores_all_models{all_suffix}.json")
    out_all: dict = {}
    if args.model_only and os.path.isfile(all_path):
        with open(all_path) as fh:
            out_all = json.load(fh)
    for fname in files_to_score:
        path = os.path.join(facts_dir, fname)
        if not os.path.exists(path):
            continue
        mslug = fname[: -len(".jsonl")]
        verbose = fname in targets
        per_item: dict[str, ItemFacts] = {}
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                iid, cond = r["item_id"], r["condition"]
                it = per_item.setdefault(iid, ItemFacts(item_id=iid,
                                                        category=category.get(iid, "unknown")))
                it.facts[cond] = {Fact(d["type"], d["value"], d.get("polarity", "+"))
                                  for d in r.get("facts", [])}
                if cond in ("SPEAK", "ECHO", "EF"):
                    it.asr_agree[cond] = bool(r.get("asr_agree"))
                    it.readback_wer[cond] = r.get("readback_wer")
        items = list(per_item.values())
        for item in items:
            item.facts = compose_readback(item.facts, args.readback_mode)
        if not items:
            continue

        summary = summarize(items, with_bootstrap=not args.no_bootstrap)
        summary["model_dir"] = mslug
        summary["readback_mode"] = args.readback_mode
        summary["by_category"] = {}
        for cat, sub in group_by_category(items).items():
            summary["by_category"][cat] = {
                "n_items": len(sub),
                "RFG": summarize(sub, with_bootstrap=False)["RFG"],
                "RFG_hard": summarize(sub, with_bootstrap=False)["RFG_hard"],
            }

        os.makedirs(os.path.join(run_dir, "scores"), exist_ok=True)
        suffix = "" if args.readback_mode == "asr1" else f"_{args.readback_mode}"
        with open(os.path.join(run_dir, "scores", f"{mslug}{suffix}.jsonl"), "w") as fh:
            for it in items:
                fh.write(json.dumps({
                    "item_id": it.item_id, "category": it.category,
                    "facts": {c: sorted((f.as_dict() for f in v),
                                        key=lambda d: (d["type"], d["value"], d["polarity"]))
                              for c, v in it.facts.items()},
                    "asr_agree": it.asr_agree, "readback_wer": it.readback_wer,
                }, ensure_ascii=False) + "\n")

        out = os.path.join(run_dir, "metrics", f"scores_{mslug}{suffix}.json")
        with open(out, "w") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        out_all[mslug] = summary
        if not verbose:
            continue
        print(f"=== {mslug} ===")
        for k in ("PG", "RFG", "RG", "RFG_EF", "RFG_hard", "RFG_content", "RFG_corr"):
            b = summary.get(k) or {}
            g = b.get("gap")
            print(f"  {k:12s} gap={(f'{g:.4f}' if g is not None else 'NA')} "
                  f"({b.get('retained')}/{b.get('total')})")
        print(f"  RFG_hard - RFG_content: "
              f"{summary.get('bootstrap_RFG_hard_minus_content', '未计算(--no-bootstrap)')}")
        print(f"  spearman(WER, RFG): {summary['spearman_wer_vs_rfg']}")
        print(f"  written -> {out}")

    with open(all_path, "w") as fh:
        json.dump(out_all, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
