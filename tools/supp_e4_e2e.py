"""E4: 自然任务上的端到端兑现（VoiceBench-BBH 子集）。

流程（只读既有候选、回读与事实缓存；CPU 复算）
--------------------------------
1. 语料：`--resample-root exp/d5_vb_resample` + 基线预测目录（`--pilot-pred`），
   用 `rfg.facts.resample.load_resample_corpus` 装配实际可用候选，报告池大小；
2. 事实：复用 resample-root 的 facts_dual 缓存；缺项时列出键并停止，不调用抽取模型；
3. 选择器（与 E3 同定义）：
   - original = candidate 0
   - random   = 池内均匀随机（fold seed 固定）
   - fact_consistency = 两选择器事实集合两两 Jaccard（去参照）
   - plan_guided = mean Jaccard(U=F(p_S_internal), ·)（论文方法）
   - oracle
4. 终点：
   - 主终点（连续、配对）：被选候选在其**留出评估器回读文本**上的
     `ret(回读事实 <- 原计划事实)`，报告相对 original 的按题配对降幅与 95% CI；
   - 次终点（任务层）：用项目统一的 VoiceBench-BBH 任务级答案抽取与严格标签比较，
     以 item.difficulty 确定任务，报告准确率与配对差。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rfg.facts.extract import load_extractions  # noqa: E402
from rfg.facts.llm import PROMPT_SHA256  # noqa: E402
from rfg.facts.resample import load_resample_corpus  # noqa: E402
from rfg.score.benchmark import answers_equal, extract_bbh_answer  # noqa: E402
from rfg.score.frr import _gap, _jaccard, _bootstrap_reduction, ASR_NAMES  # noqa: E402

SEED = 20260918
N_BOOT = 10_000


def score_answer(text: str, item: dict) -> tuple[str | None, bool]:
    """Use the same task-specific extraction and equality as the main benchmark."""
    task = item["difficulty"]
    answer = extract_bbh_answer(text, task)
    return answer, answers_equal("voicebench_bbh", task, answer, item["expected_answer_text"])


def load_cached_facts(cache: Path, texts: dict[str, str]) -> dict:
    """Require existing, text-aligned extractions without starting inference."""
    extracted = load_extractions(str(cache), prompt_sha256=PROMPT_SHA256)
    missing = sorted(set(texts) - set(extracted))
    mismatched = sorted(key for key, text in texts.items()
                        if key in extracted and extracted[key].text != text)
    if missing or mismatched:
        raise SystemExit(f"[e4] cache {cache}: " + json.dumps({
            "missing_keys": missing, "text_mismatch_keys": mismatched,
        }, ensure_ascii=False))
    return extracted


def readback_coverage(resample_dir: Path) -> dict:
    """Count actual nonempty readbacks by candidate file, before corpus filtering."""
    counts = defaultdict(Counter)
    for path in sorted(resample_dir.glob("*/R*.json")):
        record = json.loads(path.read_text())
        counts[path.stem]["n_files"] += 1
        for asr in ASR_NAMES:
            counts[path.stem][asr] += bool((record.get(asr) or "").strip())
    return {name: dict(values) for name, values in sorted(counts.items())}


def correction_record(previous: dict, folds: dict) -> dict:
    """Retain the original report and check that correcting accuracy kept fact losses."""
    previous = previous.get("correction", {}).get("previous_report", previous)
    differences = []
    for asr, entry in previous["folds"].items():
        for name, values in entry.items():
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                if key.startswith("plan_loss") or key.startswith("loss_reduction"):
                    if folds[asr][name][key] != value:
                        differences.append(f"{asr}.{name}.{key}")
    return {
        "reason": "Replace legacy prefix matching and bare-label omission with the shared task-specific BBH scorer; correct candidate/ASR coverage descriptions.",
        "answer_extractor": "rfg.score.benchmark.extract_bbh_answer(text, item.difficulty)",
        "answer_equality": "rfg.score.benchmark.answers_equal(voicebench_bbh, task, answer, gold)",
        "selection_and_seed_policy": "unchanged",
        "reference_plan_statistics_unchanged": not differences,
        "changed_reference_plan_statistics": differences,
        "previous_report": previous,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-slug", required=True)
    ap.add_argument("--resample-root", default="exp/d5_vb_resample")
    ap.add_argument("--pilot-pred", default="exp/omg_voicebench_short/predictions")
    ap.add_argument("--items", default="data/omg_benchmarks/prepared/voicebench_bbh/items_short.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    mslug = args.model_slug
    corpus, texts = load_resample_corpus(str(_ROOT / args.pilot_pred),
                                         str(_ROOT / args.resample_root), mslug)
    if args.limit:
        corpus = corpus[: args.limit]
        keep = {item["internal_key"] for item in corpus}
        keep |= {k for item in corpus for slot in ("sample_keys", "sample_keys_asr2",
                                                   "sample_keys_asr3")
                 for k in item.get(slot, [])}
        texts = {k: v for k, v in texts.items() if k in keep}
    print(f"[e4] {mslug}: {len(corpus)} items, {len(texts)} texts", flush=True)

    cache = _ROOT / args.resample_root / "facts_dual" / f"{mslug}.jsonl"
    extracted = load_cached_facts(cache, texts)

    items_meta = {}
    with open(_ROOT / args.items) as fh:
        for line in fh:
            rec = json.loads(line)
            items_meta[rec["id"]] = rec

    folds: dict[str, Any] = {}
    rows_all: dict[str, list[dict]] = {}
    # 保留既有 ASR 可用性与候选筛选规则；报告实际三路完整池，区分缓存命中与回读覆盖。
    def coverage(slot: str) -> float:
        keys = [k for item in corpus for k in item.get(slot, [])]
        if not keys:
            return 0.0
        return sum(1 for k in keys if k in extracted) / len(keys)

    available = [name for name, slot in (("asr1", "sample_keys"), ("asr2", "sample_keys_asr2"),
                                         ("asr3", "sample_keys_asr3"))
                 if coverage(slot) >= 0.5]
    if len(available) < 2:
        raise SystemExit(f"[e4] {mslug}: 可用识别器不足（{available}），先跑回读")
    print(f"[e4] {mslug}: available ASRs = {available} "
          f"({len(available)}-way leave-one-out); coverage="
          f"{ {n: round(coverage(s), 3) for n, s in (('asr1','sample_keys'), ('asr2','sample_keys_asr2'), ('asr3','sample_keys_asr3'))} }",
          flush=True)
    fold_names = list(available)

    for fold_index, held_out in enumerate(fold_names):
        sel_rng = random.Random(SEED + fold_index)
        rows: list[dict] = []
        for item in corpus:
            upstream = set(extracted[item["internal_key"]].facts)
            if not upstream:
                continue
            keys = {"asr1": item["sample_keys"], "asr2": item["sample_keys_asr2"],
                    "asr3": item.get("sample_keys_asr3") or []}
            keys = {nm: keys[nm] for nm in fold_names}
            lengths = {len(v) for v in keys.values()}
            if len(lengths) != 1 or next(iter(lengths), 0) < 1:
                continue
            if any(k not in extracted for vs in keys.values() for k in vs):
                continue
            # 回读必须非空：空转写（音频不可识别）不算"零事实的候选"，否则会把不可识别
            # 的候选当成"事实全丢"混进池子，污染选择与损失
            if any(not (texts.get(k) or "").strip() for vs in keys.values() for k in vs):
                continue
            selectors = [nm for nm in fold_names if nm != held_out]
            n = len(keys["asr1"])
            facts = {nm: [set(extracted[keys[nm][i]].facts) for i in range(n)] for nm in fold_names}
            gaps = {nm: [_gap(upstream, facts[nm][i]) for i in range(n)] for nm in fold_names}

            pairs = [(a, b) for a in range(len(selectors)) for b in range(a + 1, len(selectors))]
            scores = {
                "plan_guided": [statistics.mean(_jaccard(upstream, facts[nm][i]) for nm in selectors)
                                for i in range(n)],
                "fact_consistency": [statistics.mean(
                    _jaccard(facts[selectors[a]][i], facts[selectors[b]][i]) for a, b in pairs)
                    for i in range(n)] if pairs else [0.0] * n,
            }
            if not pairs:  # 只有 1 路选择器时无法定义去参照自一致，本折次跳过该选择器
                scores.pop("fact_consistency")
            picks = {"original": 0, "random": sel_rng.randrange(n),
                     "oracle": min(range(n), key=lambda i: gaps[held_out][i])}
            for name, vals in scores.items():
                picks[name] = max(range(n), key=lambda i: (vals[i], -i))

            meta = items_meta[item["item_id"]]
            gold_answer = meta["expected_answer_text"]
            row: dict[str, Any] = {"item_id": item["item_id"], "n_candidates": n,
                                   "gold": gold_answer, "task": meta["difficulty"],
                                   "candidate_keys": keys[held_out]}
            for name, idx in picks.items():
                rb_text = texts[keys[held_out][idx]]
                row[f"{name}_gap"] = gaps[held_out][idx]
                row[f"{name}_pick"] = idx
                row[f"{name}_answer"], row[f"{name}_correct"] = score_answer(rb_text, meta)
            row["readback_facts"] = len(facts[held_out][picks["original"]])
            rows.append(row)
        rows_all[held_out] = rows
        if not rows:
            raise SystemExit(f"[e4] {mslug}: no eligible pools for {held_out}")

        entry: dict[str, Any] = {"evaluator": held_out,
                                 "selectors": [nm for nm in fold_names if nm != held_out],
                                 "n_items": len(rows)}
        names = [n for n in ("original", "random", "fact_consistency", "plan_guided", "oracle")
                 if all(f"{n}_gap" in r for r in rows)]
        for name in names:
            loss = [r[f"{name}_gap"] for r in rows]
            acc = [1.0 if r[f"{name}_correct"] else 0.0 for r in rows]
            entry[name] = {
                "plan_loss_mean": statistics.mean(loss),
                "plan_loss_reduction_ci95": _bootstrap_reduction(
                    [r["original_gap"] - r[f"{name}_gap"] for r in rows],
                    SEED + fold_index, args.n_boot)["ci95"],
                "accuracy": statistics.mean(acc),
                "accuracy_delta_ci95": _bootstrap_reduction(
                    [1.0 * r[f"{name}_correct"] - 1.0 * r["original_correct"] for r in rows],
                    SEED + 50 + fold_index, args.n_boot)["ci95"],
            }
            entry[name]["plan_loss_reduction_mean"] = statistics.mean(
                [r["original_gap"] - r[f"{name}_gap"] for r in rows])
            entry[name]["accuracy_delta_mean"] = statistics.mean(
                [1.0 * r[f"{name}_correct"] - 1.0 * r["original_correct"] for r in rows])
        if "fact_consistency_gap" in rows[0]:
            entry["plan_minus_fact_consistency"] = {
                "loss_reduction_diff": statistics.mean(
                    [r["fact_consistency_gap"] - r["plan_guided_gap"] for r in rows]),
                "loss_reduction_diff_ci95": _bootstrap_reduction(
                    [r["fact_consistency_gap"] - r["plan_guided_gap"] for r in rows],
                    SEED + 90 + fold_index, args.n_boot)["ci95"],
                "accuracy_diff": statistics.mean(
                    [1.0 * r["plan_guided_correct"] - 1.0 * r["fact_consistency_correct"] for r in rows]),
                "accuracy_diff_ci95": _bootstrap_reduction(
                    [1.0 * r["plan_guided_correct"] - 1.0 * r["fact_consistency_correct"] for r in rows],
                    SEED + 95 + fold_index, args.n_boot)["ci95"],
            }
        folds[held_out] = entry
        print(f"[e4] {mslug} {held_out}: n={entry['n_items']} "
              f"orig_loss={100*entry['original']['plan_loss_mean']:.2f} "
              f"plan_loss={100*entry['plan_guided']['plan_loss_mean']:.2f} "
              f"B3_loss={100*entry.get('fact_consistency', {}).get('plan_loss_mean', float('nan')):.2f} | "
              f"orig_acc={100*entry['original']['accuracy']:.1f} "
              f"plan_acc={100*entry['plan_guided']['accuracy']:.1f} "
              f"Δacc={100*entry['plan_guided']['accuracy_delta_mean']:+.2f} "
              f"CI={[round(100*x,2) for x in entry['plan_guided']['accuracy_delta_ci95']]}", flush=True)

    out = Path(args.out) if args.out else _ROOT / f"reports/supplementary_e4_e2e_{mslug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    first_rows = rows_all[fold_names[0]]
    pool_sizes = dict(Counter(row["n_candidates"] for row in first_rows))
    report = {"model": mslug, "protocol": (
                  "VoiceBench-BBH available short-answer subset; actual candidate counts reported in coverage; "
                  "leave-one-ASR-out selection, earliest candidate on ties; "
                  "primary endpoint = reference-plan fact loss on held-out readback; "
                  "secondary = task accuracy by the shared task-specific BBH extractor and strict label equality"),
              "seed": SEED, "n_boot": args.n_boot, "n_items_total": len(corpus),
              "coverage": {
                  "available_asrs": fold_names,
                  "raw_candidate_readbacks": readback_coverage(_ROOT / args.resample_root / mslug),
                  "loaded_candidate_counts": dict(Counter(len(it["sample_keys"]) for it in corpus)),
                  "three_asr_complete_pools": sum(
                      len(it["sample_keys"]) == len(it["sample_keys_asr2"]) == len(it["sample_keys_asr3"])
                      for it in corpus),
                  "empty_reference_pools": sum(not extracted[it["internal_key"]].facts for it in corpus),
                  "evaluated_pools": len(first_rows),
                  "evaluated_candidate_counts": pool_sizes,
                  "loaded_tasks": dict(Counter(items_meta[it["item_id"]]["difficulty"] for it in corpus)),
                  "evaluated_tasks": dict(Counter(row["task"] for row in first_rows)),
                  "cache_required_keys": len(texts), "cache_missing_keys": [],
                  "cache_text_mismatch_keys": [],
              },
              "inputs": {"resample_root": args.resample_root, "pilot_pred": args.pilot_pred,
                         "items": args.items, "cache": str(cache.relative_to(_ROOT))},
              "folds": folds, "per_item": rows_all}
    if out.exists():
        report["correction"] = correction_record(json.loads(out.read_text()), folds)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"[e4] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
