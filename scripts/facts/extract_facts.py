#!/usr/bin/env python3
"""D0-5a：对推理产物做双通道事实抽取（规则 + LLM），带磁盘缓存与 κ 统计。

缓存：exp/<run_id>/facts/<model_slug>.jsonl，按 (item_id, condition) 存事实集合与文本哈希；
文本不变时直接复用，不重复跑 LLM。规则通道永远重算（便宜且确定性）。

用法：
  python scripts/extract_facts.py --run-id d0_pilot                        # 仅规则通道
  python scripts/extract_facts.py --run-id d0_pilot --with-llm \
      --llm-model /workspace/yunlong/LLM/pretrain_model/Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from rfg.facts.extract import Extraction, kappa_over_extractions, merge, text_hash
from rfg.facts.readback_state import completed_text
from rfg.facts.rules import extract_rules
from rfg.run.conditions import READBACK_REFERENCE

TEXT_CONDS = ("READ", "LISTEN")
SPEECH_CONDS = tuple(READBACK_REFERENCE)
# 语音条件的**模型内部文本**（"它准备说什么"）单列为抽取目标，条件名加此后缀。
# 这是 plan/render 分解链的关键一环：没有它就无法把"规划损失"与"渲染损失"分开。
INTERNAL_SUFFIX = "#internal"
# 语音条件的**第二路 ASR 回读**（asr2）同样单列。理由：RFG 原先只用 asr1 当测量
# 仪器，于是结果无法区分模型变化与识别器敏感性。把它单列后，指标侧可以报告
# 单 ASR 与多 ASR 并集的完整敏感性梯子（见 docs/correction_plan.md）。
ASR2_SUFFIX = "#asr2"
ASR3_SUFFIX = "#asr3"


def readback_text(rec: dict) -> tuple[str | None, bool]:
    """语音条件的回读文本：双 ASR 一致时用 asr1，否则仍用 asr1 但标记不可靠。"""
    rb = rec.get("readback") or {}
    text = completed_text(rb, "asr1")
    if text is None:
        text = completed_text(rb, "asr2")
    return text, bool(rb.get("asr_agree"))


def readback_text2(rec: dict) -> str | None:
    """第二路 ASR 的回读文本（Seamless）。缺失时返回 None（不猜测）。"""
    return completed_text(rec.get("readback") or {}, "asr2")


def readback_text3(rec: dict) -> str | None:
    """Third ASR readback (Fun-ASR), when available."""
    return completed_text(rec.get("readback") or {}, "asr3")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="d0_pilot")
    ap.add_argument("--out-root", default="exp")
    ap.add_argument("--model", default=None, help="只处理该模型目录；默认全部")
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--force-llm", action="store_true",
                    help="忽略缓存中的 LLM 事实并全部重抽（改提示词/改解析后必须用）")
    ap.add_argument("--llm-model", default="/workspace/yunlong/LLM/pretrain_model/Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--llm-device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="按排序后的 item_id 取模分片；默认不分片")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="当前分片编号，范围为 [0, num_shards)")
    args = ap.parse_args()

    if args.num_shards < 1:
        ap.error("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        ap.error("--shard-index must be in [0, num_shards)")
    artifact_suffix = (
        f"_shard{args.shard_index}-of-{args.num_shards}"
        if args.num_shards > 1 else ""
    )

    run_dir = os.path.join(args.out_root, args.run_id)
    pred_root = os.path.join(run_dir, "predictions")
    facts_dir = os.path.join(run_dir, "facts")
    os.makedirs(facts_dir, exist_ok=True)

    models = [args.model] if args.model else sorted(
        d for d in os.listdir(pred_root) if os.path.isdir(os.path.join(pred_root, d)))

    for mslug in models:
        mdir = os.path.join(pred_root, mslug)
        item_ids = sorted(d for d in os.listdir(mdir) if os.path.isdir(os.path.join(mdir, d)))
        if args.limit:
            item_ids = item_ids[: args.limit]
        if args.num_shards > 1:
            item_ids = [
                iid for index, iid in enumerate(item_ids)
                if index % args.num_shards == args.shard_index
            ]

        # 收集需要抽取的文本
        jobs: list[tuple[str, str, str]] = []  # (item_id, condition, text)
        meta: dict[tuple[str, str], dict] = {}
        for iid in item_ids:
            idir = os.path.join(mdir, iid)
            for cond in TEXT_CONDS + SPEECH_CONDS:
                p = os.path.join(idir, f"{cond}.json")
                if not os.path.exists(p):
                    continue
                with open(p) as fh:
                    rec = json.load(fh)
                if cond in SPEECH_CONDS:
                    text, agree = readback_text(rec)
                    meta[(iid, cond)] = {"asr_agree": agree,
                                         "readback_wer": (rec.get("readback") or {}).get("readback_wer")}
                    internal = rec.get("text")
                    if internal:
                        jobs.append((iid, cond + INTERNAL_SUFFIX, internal))
                        meta[(iid, cond + INTERNAL_SUFFIX)] = {"role": "internal_text"}
                    # 第二路 ASR 的回读文本：单列为 `<cond>#asr2`，供指标侧做并集/交集
                    text2 = readback_text2(rec)
                    if text2 is not None:
                        jobs.append((iid, cond + ASR2_SUFFIX, text2))
                        meta[(iid, cond + ASR2_SUFFIX)] = {"role": "readback_asr2"}
                    text3 = readback_text3(rec)
                    if text3 is not None:
                        jobs.append((iid, cond + ASR3_SUFFIX, text3))
                        meta[(iid, cond + ASR3_SUFFIX)] = {"role": "readback_asr3"}
                else:
                    text = rec.get("text")
                    meta[(iid, cond)] = {}
                if text is not None:
                    jobs.append((iid, cond, text))
        print(f"[{mslug}] 待抽取文本 {len(jobs)} 条（items={len(item_ids)}）", flush=True)

        cache_path = os.path.join(facts_dir, f"{mslug}{artifact_suffix}.jsonl")
        from rfg.facts.llm import PROMPT_SHA256

        cache: dict[tuple[str, str], dict] = {}
        if os.path.exists(cache_path):
            with open(cache_path) as fh:
                for line in fh:
                    r = json.loads(line)
                    if r.get("text_hash") != text_hash(r.get("text")):
                        continue
                    # 提示词变了（或本来就没有 LLM 事实）→ LLM 事实作废；规则事实仍可复用
                    if r.get("prompt_sha256") != PROMPT_SHA256:
                        r["facts_llm"] = []
                        r["llm_error"] = "invalidated: prompt changed"
                    cache[(r["item_id"], r["condition"])] = r
            print(f"[{mslug}] 缓存命中 {len(cache)} 条 (prompt={PROMPT_SHA256[:10]})", flush=True)

        llm_facts: dict[tuple[str, str], tuple[set, str | None]] = {}

        def _needs_llm(key: tuple[str, str]) -> bool:
            """无缓存、缓存的 LLM 事实被作废、或显式 --force-llm → 需要重抽。"""
            if args.force_llm:
                return True
            c = cache.get(key)
            return c is None or str(c.get("llm_error") or "").startswith("invalidated")

        todo = [
            (iid, cond, t) for (iid, cond, t) in jobs
            if args.with_llm and t and _needs_llm((iid, cond))
        ]
        if args.with_llm and todo:
            from rfg.facts.llm import LlmExtractor

            print(f"[{mslug}] 需重抽 {len(todo)}/{len(jobs)} 条；加载 LLM 抽取器 {args.llm_model}", flush=True)
            ex = LlmExtractor(args.llm_model, device=args.llm_device)
            texts = [t for _, _, t in todo]
            parsed = ex.extract_batch(texts)
            for (iid, cond, _), (facts, err) in zip(todo, parsed):
                llm_facts[(iid, cond)] = (facts, err)
            print(f"[{mslug}] LLM 抽取完成 prompt_sha256={PROMPT_SHA256[:12]}", flush=True)

        rows: list[dict] = []
        extractions: list[Extraction] = []
        for iid, cond, text in jobs:
            cached = cache.get((iid, cond))
            # 规则通道总是重算（便宜且确定性）；LLM 事实按缓存/新抽结果取用
            facts_rules = extract_rules(text)
            if not text:
                facts_llm, llm_err = set(), None
            elif (iid, cond) in llm_facts:
                facts_llm, llm_err = llm_facts[(iid, cond)]
            elif cached is not None and not _needs_llm((iid, cond)):
                facts_llm = {_f(d) for d in cached["facts_llm"]}
                llm_err = cached.get("llm_error")
            else:
                facts_llm, llm_err = set(), None if not args.with_llm else "llm disabled"
            ex = merge(text, facts_rules, facts_llm, llm_err)
            extractions.append(ex)
            row = {"item_id": iid, "condition": cond, "text": text, "text_hash": text_hash(text),
                   "prompt_sha256": PROMPT_SHA256 if args.with_llm else None,
                   **meta.get((iid, cond), {}), **ex.as_dict()}
            rows.append(row)

        with open(cache_path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        kappa = kappa_over_extractions(extractions)
        out = os.path.join(run_dir, "metrics", f"facts_{mslug}{artifact_suffix}.json")
        with open(out, "w") as fh:
            json.dump({"model_dir": mslug, "n_texts": len(rows), "with_llm": args.with_llm,
                       "llm_model": args.llm_model if args.with_llm else None,
                       "prompt_sha256": PROMPT_SHA256 if args.with_llm else None, **kappa},
                      fh, indent=2, ensure_ascii=False)
        print(f"[{mslug}] kappa={kappa['kappa']} value_conflict_rate={kappa['value_conflict_rate']} "
              f"type_conflict_rate={kappa['type_conflict_rate']} -> {out}", flush=True)
    return 0


def _f(d: dict):
    from rfg.facts.schema import Fact

    return Fact(d["type"], d["value"], d.get("polarity", "+"))


if __name__ == "__main__":
    sys.exit(main())
