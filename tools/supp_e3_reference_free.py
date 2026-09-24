"""E3: 去参照基线（不使用内部计划的选择器）与计划引导的对照。

候选池与口径完全复用 `exp/d0_resample` 与 `rfg.score.frr.evaluate_loo_folds`：
留一 ASR（两个选择、一个评估）、平手取最早候选、事实集合来自 `facts_dual` 缓存。

选择器
------
B0 original   = candidate 0（主口径 SPEAK）
B1 random     = 同一池内均匀随机（按 fold seed 固定）
B2 transcript = 两选择器转写（词级 1-WER）的均值相似度
B3 fact-consistency = 两选择器事实集合两两 Jaccard 的均值（**不使用 U=F(p_S)**）
B4 B3 + 长度惩罚 = 在 B3 上减去超过池内中位时长的惩罚（音频时长来自 predictions 的 wav 头）
B5 plan-guided = 论文方法（mean Jaccard(U, ·)），即 evaluate_loo_folds 的 selected
B6 oracle     = 评估器自身最优（上界）
B7 plan-text  = 原始计划与两路选择器回读的规范化词级编辑相似度均值

输出：每个选择器逐折的原始/选中损失、配对降幅与按题聚类 bootstrap 95% CI，
以及 B3 是否复制了 B5 的选择（复现率）。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import random
import statistics
import sys
import wave
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rfg.facts.extract import load_extractions  # noqa: E402
from rfg.facts.llm import PROMPT_SHA256  # noqa: E402
from rfg.facts.resample import load_resample_corpus  # noqa: E402
from rfg.score.frr import _gap, _jaccard, _bootstrap_reduction, ASR_NAMES  # noqa: E402
from rfg.score.plan_text import (  # noqa: E402
    NORMALIZATION_PROTOCOL, normalize_plan_tokens, number_conversion_failures, plan_text_scores,
)

MODELS = {
    "Qwen3-Omni-30B-A3B-Instruct": {"pred": "exp/d0_pilot/predictions"},
    "Qwen2.5-Omni-3B": {"pred": "exp/d0_pilot/predictions"},
    "Qwen2.5-Omni-7B": {"pred": "exp/d3_7b_stack/predictions"},
    "MiniCPM-o-4_5": {"pred": "exp/d1_main/predictions"},
    "Step-Audio-2-mini": {"pred": "exp/d2_stepaudio/predictions"},
}
N_BOOT = 10_000
SEED = 20260918
CORRECTED_PREDICTION_ROOTS = {
    name: {"previous": "exp/d1_main/predictions", "current": cfg["pred"]}
    for name, cfg in MODELS.items() if cfg["pred"] == "exp/d0_pilot/predictions"
}


def wav_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as fh:
            frames = fh.getnframes()
            rate = fh.getframerate()
            return frames / rate if rate else None
    except Exception:
        return None


def word_similarity(a: str, b: str) -> float:
    """词级 1 - WER（Levenshtein 距离 / 参考长度），截断到 [0,1]。"""
    wa = (a or "").lower().split()
    wb = (b or "").lower().split()
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    prev = list(range(len(wb) + 1))
    for i, x in enumerate(wa, 1):
        cur = [i] + [0] * len(wb)
        for j, y in enumerate(wb, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    dist = prev[-1]
    return max(0.0, 1 - dist / max(len(wa), len(wb)))


def score_item(item: dict, extracted: dict[str, Any], texts: dict[str, str],
               durations: dict[str, list[float | None]], held_out: str,
               sel_rng: random.Random) -> dict | None:
    upstream = set(extracted[item["internal_key"]].facts)
    if not upstream:
        return None
    keys = {
        "asr1": item["sample_keys"],
        "asr2": item["sample_keys_asr2"],
        "asr3": item.get("sample_keys_asr3") or [],
    }
    lengths = {len(v) for v in keys.values()}
    if len(lengths) != 1 or next(iter(lengths), 0) < 1:
        return None
    if any(k not in extracted for vs in keys.values() for k in vs):
        return None
    n = len(keys["asr1"])
    selectors = [nm for nm in ASR_NAMES if nm != held_out]

    facts = {nm: [set(extracted[keys[nm][i]].facts) for i in range(n)] for nm in ASR_NAMES}
    gaps = {nm: [_gap(upstream, facts[nm][i]) for i in range(n)] for nm in ASR_NAMES}

    scores: dict[str, list[float]] = {}
    scores["plan_guided"] = [
        statistics.mean(_jaccard(upstream, facts[nm][i]) for nm in selectors) for i in range(n)
    ]
    scores["fact_consistency"] = [
        statistics.mean(_jaccard(facts[selectors[a]][i], facts[selectors[b]][i])
                        for a in range(len(selectors)) for b in range(a + 1, len(selectors)))
        for i in range(n)
    ]
    scores["transcript"] = [
        statistics.mean(word_similarity(texts[keys[selectors[a]][i]],
                                        texts[keys[selectors[b]][i]])
                        for a in range(len(selectors)) for b in range(a + 1, len(selectors)))
        for i in range(n)
    ]
    scores["plan_text"] = plan_text_scores(
        extracted[item["internal_key"]].text,
        [[extracted[key].text for key in keys[name]] for name in selectors],
    )
    durs = durations.get(item["item_id"], [None] * n)
    median = None
    known = [d for d in durs if isinstance(d, float)]
    if known:
        median = statistics.median(known)
    penalty = []
    for d in durs:
        if median is None or not isinstance(d, float) or median <= 0:
            penalty.append(0.0)
        else:
            penalty.append(max(0.0, (d - median) / median))
    scores["fact_consistency_length"] = [scores["fact_consistency"][i] - 0.5 * penalty[i]
                                         for i in range(n)]

    row: dict[str, Any] = {"item_id": item["item_id"], "n_candidates": n,
                           "base_gap": gaps[held_out][0],
                           "oracle_gap": min(gaps[held_out]),
                           "random_gap": statistics.mean(gaps[held_out])}
    random_index = sel_rng.randrange(n)
    row["random_pick_gap"] = gaps[held_out][random_index]
    for name, vals in scores.items():
        pick = max(range(n), key=lambda i: (vals[i], -i))
        row[f"{name}_gap"] = gaps[held_out][pick]
        row[f"{name}_pick"] = pick
    row["selector_mean_gap"] = statistics.mean(gaps[held_out])
    row["plan_text_scores"] = scores["plan_text"]
    row["plan_guided_scores"] = scores["plan_guided"]
    row["held_out_candidate_gaps"] = gaps[held_out]
    row["candidate_keys"] = keys[held_out]
    return row


def text_mismatches(texts: dict[str, str], extracted: dict[str, Any]) -> dict:
    mismatches = [key for key, text in texts.items()
                  if key in extracted and text != extracted[key].text]
    return {
        "n_compared": sum(key in extracted for key in texts),
        "n_mismatched": len(mismatches),
        "by_kind": {
            "original_plan": sum(key.endswith("|SPEAK#internal") for key in mismatches),
            "original_readback": sum("|SPEAK" in key and not key.endswith("#internal")
                                     for key in mismatches),
            "resample_readback": sum("|R" in key for key in mismatches),
        },
        "mismatched_keys": mismatches,
    }


def evaluate_model(job: tuple[str, int]) -> tuple[str, dict, list[dict]]:
    model, n_boot = job
    cfg = MODELS[model]
    cache = _ROOT / f"exp/d0_resample/facts_dual/{model}.jsonl"
    extracted = load_extractions(str(cache), prompt_sha256=PROMPT_SHA256)
    corpus, texts = load_resample_corpus(str(_ROOT / cfg["pred"]),
                                       str(_ROOT / "exp/d0_resample"), model)
    current_mismatches = text_mismatches(texts, extracted)
    if current_mismatches["n_mismatched"]:
        raise ValueError(f"{model}: prediction text is not aligned with the fact cache")
    normalization_fallbacks = [
        {"key": key, "failures": failures}
        for key in texts if key in extracted
        if (failures := number_conversion_failures(extracted[key].text))
    ]
    previous_mismatches = None
    if model in CORRECTED_PREDICTION_ROOTS:
        _, previous_texts = load_resample_corpus(
            str(_ROOT / CORRECTED_PREDICTION_ROOTS[model]["previous"]),
            str(_ROOT / "exp/d0_resample"), model)
        previous_mismatches = text_mismatches(previous_texts, extracted)

    durations: dict[str, list[float | None]] = {}
    for item in corpus:
        iid = item["item_id"]
        base = _ROOT / f"exp/d0_resample/{model}/{iid}"
        pred = _ROOT / cfg["pred"] / model / iid / "SPEAK.wav"
        seq: list[float | None] = [wav_seconds(pred) if pred.exists() else None]
        for k in range(7):
            path = base / f"R{k}.wav"
            seq.append(wav_seconds(path) if path.exists() else None)
        durations[iid] = seq

    folds: dict[str, dict] = {}
    rows_all: list[dict] = []
    for fold_index, held_out in enumerate(ASR_NAMES):
        sel_rng = random.Random(SEED + fold_index)
        rows = [r for r in (score_item(item, extracted, texts, durations, held_out, sel_rng)
                            for item in corpus) if r]
        expected_n = 178 if model == "Qwen2.5-Omni-7B" else 200
        if len(rows) != expected_n or {row["n_candidates"] for row in rows} != {8}:
            raise ValueError(f"{model}/{held_out}: expected {expected_n} complete pools of 8")
        rows_all.extend({"model": model, "evaluator": held_out,
                         "selectors": [nm for nm in ASR_NAMES if nm != held_out], **row}
                        for row in rows)
        entry: dict[str, Any] = {"evaluator": held_out,
                                 "selectors": [nm for nm in ASR_NAMES if nm != held_out],
                                 "n_items": len(rows), "n_candidates": 8}
        for name, key in (("original", "base_gap"), ("random", "random_pick_gap"),
                          ("transcript", "transcript_gap"),
                          ("fact_consistency", "fact_consistency_gap"),
                          ("fact_consistency_length", "fact_consistency_length_gap"),
                          ("plan_guided", "plan_guided_gap"), ("oracle", "oracle_gap"),
                          ("plan_text", "plan_text_gap")):
            vals = [r[key] for r in rows]
            reduction = [r["base_gap"] - r[key] for r in rows]
            entry[name] = {
                "loss_mean": statistics.mean(vals),
                "reduction_mean": statistics.mean(reduction),
                "reduction_ci95": _bootstrap_reduction(reduction, SEED + fold_index,
                                                       n_boot)["ci95"],
            }
            entry[name]["ci_excludes_zero"] = (
                entry[name]["reduction_ci95"][0] > 0 or entry[name]["reduction_ci95"][1] < 0)
        for comparison, baseline, seed_offset in (
                ("plan_minus_factconsistency", "fact_consistency", 100),
                ("plan_minus_plantext", "plan_text", 200)):
            paired = [r[f"{baseline}_gap"] - r["plan_guided_gap"] for r in rows]
            interval = _bootstrap_reduction(paired, SEED + seed_offset + fold_index, n_boot)
            entry[comparison] = {"mean": interval["reduction"], "ci95": interval["ci95"],
                                 "ci_excludes_zero": interval["excludes_zero"]}
        entry["same_pick_rate"] = statistics.mean(
            1.0 if r["plan_guided_pick"] == r["fact_consistency_pick"] else 0.0 for r in rows)
        entry["same_pick_rate_plantext"] = statistics.mean(
            1.0 if r["plan_guided_pick"] == r["plan_text_pick"] else 0.0 for r in rows)
        entry["picked_original_rate_plan"] = statistics.mean(
            1.0 if r["plan_guided_pick"] == 0 else 0.0 for r in rows)
        entry["picked_original_rate_factconsistency"] = statistics.mean(
            1.0 if r["fact_consistency_pick"] == 0 else 0.0 for r in rows)
        entry["picked_original_rate_plantext"] = statistics.mean(
            1.0 if r["plan_text_pick"] == 0 else 0.0 for r in rows)
        folds[held_out] = entry
        print(f"[e3] {model}/{held_out}: n={len(rows)} "
              f"text={100*entry['plan_text']['loss_mean']:.2f}% "
              f"fact={100*entry['plan_guided']['loss_mean']:.2f}% "
              f"text-minus-fact={100*entry['plan_minus_plantext']['mean']:+.2f} pp", flush=True)
    result = {"folds": folds, "cache": str(cache.relative_to(_ROOT)),
              "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
              "prediction_root": cfg["pred"],
              "number_normalization_fallbacks": normalization_fallbacks,
              "cache_input_mismatch": {"current": current_mismatches,
                                       "previous_prediction_root": previous_mismatches}}
    return model, result, rows_all


def check_previous_report(previous: dict | None, report: dict) -> dict:
    """Protect established fact-based results; expose corrected legacy controls."""
    if previous is None:
        return {"available": False}
    checked_folds = 0
    changed_fields = []
    invariant_keys = ("n_items", "original", "random", "fact_consistency", "plan_guided",
                      "oracle", "plan_minus_factconsistency", "same_pick_rate",
                      "picked_original_rate_plan", "picked_original_rate_factconsistency")
    for model, current in report["models"].items():
        for asr, fold in current["folds"].items():
            old = previous["models"][model]["folds"][asr]
            for key in invariant_keys:
                if old[key] != fold[key]:
                    raise ValueError(f"unexpected historical result change: {model}/{asr}/{key}")
            for key in ("transcript", "fact_consistency_length"):
                if old[key] != fold[key]:
                    changed_fields.append({"model": model, "fold": asr, "selector": key,
                                           "previous": old[key], "current": fold[key]})
            checked_folds += 1
    return {"available": True, "checked_folds": checked_folds,
            "unchanged_fields": invariant_keys, "all_invariant_fields_identical": True,
            "corrected_source_changed_fields": changed_fields}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out", default="reports/supplementary_e3_reference_free.json")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.workers < 1 or args.n_boot < 1:
        ap.error("workers and n-boot must be positive")
    if normalize_plan_tokens("16") != normalize_plan_tokens("sixteen"):
        raise RuntimeError("number normalization did not use num2words")
    models = [m for m in args.models.split(",") if m]
    out = _ROOT / args.out
    previous_text = out.read_text() if out.exists() else None
    previous = json.loads(previous_text) if previous_text else None
    report: dict = {
        "protocol": ("same pools/protocol as exp/d0_resample LOO (8 candidates, leave-one-ASR-out, "
                     "fact sets from facts_dual cache, ties = earliest candidate); "
                     "B2/B3/B4 are reference-free; plan_text compares the original plan with "
                     "the two selector readbacks after the predeclared text normalization"),
        "seed": SEED, "n_boot": args.n_boot, "models": {},
        "plan_text_protocol": {**NORMALIZATION_PROTOCOL, "texts_source": "facts_dual.text",
                               "paired_gain": "plan_text loss - plan_guided loss; positive favors fact selection",
                               "paired_bootstrap_seed": "20260918 + 200 + fold_index"},
        "corrected_prediction_roots": CORRECTED_PREDICTION_ROOTS,
        "prediction_root_correction": (
            "Qwen-30B and Qwen-3B original candidate/plan texts come from d0_pilot, matching "
            "facts_dual. Same IDs in d1_main were overwritten with different items. "
            "B2 transcript and B4 duration controls are recomputed on the aligned source; "
            "their algorithms and all random/bootstrap seeds are unchanged."),
        "run_metadata": {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                         "python_executable": sys.executable, "python_version": sys.version,
                         "num2words_version": version("num2words"),
                         "command": [sys.executable, *sys.argv], "workers": args.workers,
                         "prompt_sha256": PROMPT_SHA256,
                         "normalizer_sha256": hashlib.sha256(
                             (_ROOT / "src/rfg/score/textnorm.py").read_bytes()).hexdigest(),
                         "previous_report_sha256": hashlib.sha256(previous_text.encode()).hexdigest()
                         if previous_text else None},
    }
    jobs = [(model, args.n_boot) for model in models]
    print(f"[e3] starting {len(jobs)} models with {args.workers} CPU workers; "
          f"num2words={version('num2words')}; bootstrap={args.n_boot}", flush=True)
    if args.workers == 1:
        results = list(map(evaluate_model, jobs))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            results = list(pool.map(evaluate_model, jobs))
    rows_all = []
    for model, result, rows in results:
        report["models"][model] = result
        rows_all.extend(rows)
    report["previous_report_comparison"] = check_previous_report(previous, report)
    rows_path = out.with_suffix(".jsonl")
    report["per_item_file"] = str(rows_path.relative_to(_ROOT))
    report["per_item_rows"] = len(rows_all)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows_all))
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"[e3] wrote {out} and {rows_path}; all established fact-based results unchanged", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
