#!/usr/bin/env python3
"""Finalize the E1-a packet: extract facts from the adjudicated transcripts and compare
human recovery against the three recognizers.

Pipeline
--------
1. read `adjudication.labels.jsonl` exported by `adjudication.html` (or the pre-filled
   `adjudication.jsonl`), validate full coverage and non-null `transcript`;
2. run the shared dual-channel extractor on every adjudicated transcript and on the
   three ASR transcripts, with the same prompt and model as the paper;
3. load the internal spoken plan of each item from the resample corpus and compare
   `ret(human <- plan)` with `ret(ASR_i <- plan)` per item, aggregated the same way as
   the paper (fact-micro over items, item-clustered bootstrap for the paired gap);
4. write `reports/supplementary_e1a_human_anchor.{json,md}`.

Everything runs through `rfg.facts.extract.extract_dual`; no second extraction path is
introduced.  Refuses to overwrite an existing report.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_PACKET = ROOT / "data/adjudication/e1a60"
RUN_ROOTS = {
    "Qwen3-Omni-30B-A3B-Instruct": ("d0_resample", "exp/d1_main/facts"),
    "Qwen2.5-Omni-3B": ("d0_resample", "exp/d1_main/facts"),
    "Step-Audio-2-mini": ("d0_resample", "exp/d2_stepaudio/facts"),
}
ASR_NAMES = ("asr1", "asr2", "asr3")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def facts_of(row: dict) -> set[tuple[str, str, str]]:
    return {(f["type"], f["value"], f["polarity"]) for f in (row.get("facts") or [])}


def retention(upstream: set, downstream: set) -> float:
    return len(upstream & downstream) / len(upstream) if upstream else float("nan")


def bootstrap_paired(values: list[float], seed: int, draws: int = 10_000) -> dict:
    rng = random.Random(seed)
    point = statistics.mean(values)
    samples = sorted(statistics.mean(values[rng.randrange(len(values))] for _ in values)
                     for _ in range(draws))
    lo = samples[int(0.025 * draws)]
    hi = samples[int(0.975 * draws) - 1]
    return {"mean": point, "ci95": [lo, hi], "excludes_zero": lo > 0 or hi < 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--labels", type=Path, default=None,
                        help="adjudication.labels.jsonl; defaults to <packet>/adjudication.jsonl")
    parser.add_argument("--out-json", type=Path,
                        default=ROOT / "reports/supplementary_e1a_human_anchor.json")
    parser.add_argument("--out-md", type=Path,
                        default=ROOT / "reports/supplementary_e1a_human_anchor.md")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    if args.out_json.exists() or args.out_md.exists():
        raise SystemExit("refusing to overwrite an existing auxiliary report")

    master_path = args.packet / "e1a60.master.jsonl"
    master = read_jsonl(master_path)
    labels_path = args.labels or (args.packet / "adjudication.jsonl")
    labels = {r["audit_id"]: (r.get("adjudicated") or r.get("annotation") or {})
              for r in read_jsonl(labels_path)}
    ids = {r["audit_id"] for r in master}
    if set(labels) != ids:
        raise SystemExit(f"label file does not cover the packet: {len(labels)} vs {len(ids)}")
    unfinished = [i for i, a in labels.items() if not isinstance(a.get("transcript"), str)]
    if unfinished:
        raise SystemExit(f"{len(unfinished)} adjudications still have no transcript")

    # ---- texts to extract: adjudicated human transcript + the three ASR transcripts
    texts: dict[str, str] = {}
    for row in master:
        audit_id = row["audit_id"]
        texts[f"{audit_id}|human"] = labels[audit_id]["transcript"]
        for name in ASR_NAMES:
            texts[f"{audit_id}|{name}"] = row["asr"][name]["text"]

    cache = args.cache or (args.packet / "facts_dual_e1a.jsonl")
    from rfg.facts.extract import DEFAULT_LLM_MODEL, extract_dual

    extracted = extract_dual(texts, cache_path=str(cache), llm_model=DEFAULT_LLM_MODEL,
                             device=args.device, batch_size=args.batch_size)

    # ---- plan facts per item, from the same extraction caches the paper used
    from rfg.facts.extract import load_extractions
    from rfg.facts.llm import PROMPT_SHA256

    plan_facts: dict[str, set] = {}
    for model, (resample_root, facts_dir) in RUN_ROOTS.items():
        wanted = {r["item_id"] for r in master if r["model"] == model}
        if not wanted:
            continue
        resample_cache = ROOT / resample_root / "facts_dual" / f"{model}.jsonl"
        extracted_resample = (load_extractions(str(resample_cache), prompt_sha256=PROMPT_SHA256)
                              if resample_cache.exists() else {})
        main_path = ROOT / facts_dir / f"{model}.jsonl"
        main_rows: dict[tuple[str, str], dict] = {}
        if main_path.exists():
            for row in read_jsonl(main_path):
                main_rows[(row["item_id"], row["condition"])] = row
        for item_id in wanted:
            prior = extracted_resample.get(f"{item_id}|SPEAK#internal")
            if prior is not None:
                plan_facts[item_id] = set(prior.facts)
                continue
            main_row = main_rows.get((item_id, "SPEAK#internal"))
            if main_row is not None:
                plan_facts[item_id] = {(f["type"], f["value"], f["polarity"])
                                       for f in (main_row.get("facts") or [])}

    # ---- per-item comparison
    rows_out = []
    gaps = defaultdict(list)
    for row in master:
        audit_id = row["audit_id"]
        iid = row["item_id"]
        plan = plan_facts.get(iid)
        if not plan:
            continue
        human = set(extracted[f"{audit_id}|human"].facts)
        record = {
            "audit_id": audit_id, "item_id": iid, "model": row["model"],
            "stratum": row["stratum"], "category": row["category"],
            "n_plan_facts": len(plan),
            "ret_human": retention(plan, human),
        }
        for name in ASR_NAMES:
            frozen = facts_of(row["asr"][name])
            record[f"ret_{name}"] = retention(plan, frozen)
            record[f"ret_{name}_reextract"] = retention(plan, set(extracted[f"{audit_id}|{name}"].facts))
        for name in ASR_NAMES:
            gaps[name].append(record[f"ret_{name}"] - record["ret_human"])
        rows_out.append(record)

    summary = {
        "protocol": ("proportional stratified sample (60 items); human transcript read by the same "
                     "dual-channel extractor as the ASR transcripts and the plan; retention is "
                     "1 - |F(plan) \\ F(x)| / |F(plan)|"),
        "seed": args.seed,
        "n_items_used": len(rows_out),
        "n_items_packet": len(master),
        "fact_micro": {
            "ret_human": statistics.mean(r["ret_human"] for r in rows_out),
            **{f"ret_{n}": statistics.mean(r[f"ret_{n}"] for r in rows_out) for n in ASR_NAMES},
        },
        "gap_asr_minus_human": {
            n: bootstrap_paired(gaps[n], args.seed + i) for i, n in enumerate(ASR_NAMES)
        },
        "by_stratum": {
            stratum: {
                "n": len(group),
                "ret_human": statistics.mean(r["ret_human"] for r in group),
                **{f"ret_{n}": statistics.mean(r[f"ret_{n}"] for r in group) for n in ASR_NAMES},
            }
            for stratum, group in
            ((s, [r for r in rows_out if r["stratum"] == s]) for s in
             sorted({r["stratum"] for r in rows_out}))
        },
        "weighted_population_estimate": None,
        "packet": str(args.packet),
        "cache": str(cache),
    }

    try:
        weights = json.loads((args.packet / "e1a60.stratum_weights.json").read_text())
        summary["weighted_population_estimate"] = {
            "ret_human": sum(weights[r["stratum"]] * r["ret_human"] for r in rows_out) /
                        sum(weights[r["stratum"]] for r in rows_out),
            **{f"ret_{n}": sum(weights[r["stratum"]] * r[f"ret_{n}"] for r in rows_out) /
                             sum(weights[r["stratum"]] for r in rows_out) for n in ASR_NAMES},
            "weights": weights,
        }
    except FileNotFoundError:
        pass

    args.out_json.write_text(json.dumps(
        {"summary": summary, "per_item": rows_out}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    micro = summary["fact_micro"]
    lines = [
        "# E1-a：人工听辨锚定（60 条按总体比例分层抽样）",
        "",
        f"- 使用条目：{summary['n_items_used']} / {summary['n_items_packet']}",
        "- 口径：人工转写与三条 ASR 转写都经同一双通道抽取器；保留率 = |F(plan) ∩ F(x)| / |F(plan)|",
        "",
        "## 事实微平均（保留率，越高越好）",
        "",
        "| 观测 | 保留率 |",
        "| --- | ---: |",
        f"| 人耳转写 | {micro['ret_human']*100:.1f}% |",
        f"| Whisper | {micro['ret_asr1']*100:.1f}% |",
        f"| Seamless | {micro['ret_asr2']*100:.1f}% |",
        f"| Fun-ASR | {micro['ret_asr3']*100:.1f}% |",
        "",
        "## ASR − 人工 的配对差值（按题聚类 bootstrap 95%）",
        "",
        "| 识别器 | 差值 (pp) | 95% CI | 排除 0 |",
        "| --- | ---: | --- | --- |",
    ]
    for name in ASR_NAMES:
        gap = summary["gap_asr_minus_human"][name]
        lines.append(f"| {name} | {gap['mean']*100:.2f} | "
                     f"[{gap['ci95'][0]*100:.2f}, {gap['ci95'][1]*100:.2f}] | "
                     f"{'是' if gap['excludes_zero'] else '否'} |")
    lines += ["", "## 分层明细", "", "| stratum | n | 人耳 | Whisper | Seamless | Fun-ASR |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for stratum, group in summary["by_stratum"].items():
        lines.append(f"| {stratum} | {group['n']} | {group['ret_human']*100:.1f}% | "
                     f"{group['ret_asr1']*100:.1f}% | {group['ret_asr2']*100:.1f}% | "
                     f"{group['ret_asr3']*100:.1f}% |")
    if summary["weighted_population_estimate"]:
        w = summary["weighted_population_estimate"]
        lines += ["", "## 按总体分层权重加权（可写入论文）", "",
                  f"- 人耳：{w['ret_human']*100:.1f}%；Whisper：{w['ret_asr1']*100:.1f}%；"
                  f"Seamless：{w['ret_asr2']*100:.1f}%；Fun-ASR：{w['ret_asr3']*100:.1f}%", "",
                  f"- 权重：{json.dumps(w['weights'], ensure_ascii=False)}"]
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(micro, ensure_ascii=False, indent=2))
    print(f"json: {args.out_json}")
    print(f"md  : {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
