"""Run the fixed-pool single-/dual-ASR ablation from saved E3 scores and gaps."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from rfg.score.frr import ASR_NAMES, _bootstrap_reduction
from rfg.score.selector_budget import (
    DENOMINATOR_LIMIT, evaluate_selector_budget, restore_single_scores,
)


def analyze_model(model: str, rows: list[dict], cache: Path, n_boot: int, seed: int) -> dict:
    by_item = defaultdict(dict)
    for row in rows:
        by_item[row["item_id"]][row["evaluator"]] = row
    expected_n = 178 if model == "Qwen2.5-Omni-7B" else 200
    if len(by_item) != expected_n:
        raise ValueError(f"{model}: expected {expected_n} E3 pools, got {len(by_item)}")

    per_item = []
    fold_rows = defaultdict(list)
    max_residual = 0.0
    max_denominator = 0
    for item_id, source_folds in sorted(by_item.items()):
        single_scores = restore_single_scores({
            name: source_folds[name]["plan_guided_scores"] for name in ASR_NAMES
        })
        if any(len(values) != 8 for values in single_scores.values()):
            raise ValueError(f"{model}/{item_id}: expected 8 candidates")
        if any(not 0 <= score <= 1 for values in single_scores.values() for score in values):
            raise ValueError(f"{model}/{item_id}: recovered Jaccard outside [0, 1]")
        max_denominator = max(max_denominator, max(
            value.denominator for values in single_scores.values() for value in values))
        item_folds = {}
        for held_out in ASR_NAMES:
            source = source_folds[held_out]
            result = evaluate_selector_budget(single_scores, source["held_out_candidate_gaps"], held_out)
            residual = max(abs(a - b) for a, b in zip(result["dual_scores"], source["plan_guided_scores"]))
            max_residual = max(max_residual, residual)
            if (result["dual"]["picked_index"] != source["plan_guided_pick"]
                    or result["dual"]["loss"] != source["plan_guided_gap"]
                    or residual > 1e-12):
                raise ValueError(f"{model}/{item_id}/{held_out}: dual selector differs from E3")
            del result["dual_scores"]
            item_folds[held_out] = result
            fold_rows[held_out].append(result)
        single_mean = statistics.mean(fold["single_average_loss"] for fold in item_folds.values())
        dual_mean = statistics.mean(fold["dual"]["loss"] for fold in item_folds.values())
        per_item.append({
            "item_id": item_id,
            "single_average_loss": single_mean,
            "dual_loss": dual_mean,
            "single_minus_dual": single_mean - dual_mean,
            "folds": item_folds,
        })

    folds = {}
    for held_out, values in fold_rows.items():
        selectors = values[0]["selectors"]
        folds[held_out] = {
            "n_items": len(values),
            "single_selector_loss": {
                name: statistics.mean(row["single"][name]["loss"] for row in values)
                for name in selectors
            },
            "single_average_loss": statistics.mean(row["single_average_loss"] for row in values),
            "dual_loss": statistics.mean(row["dual"]["loss"] for row in values),
            "single_minus_dual": statistics.mean(row["single_minus_dual"] for row in values),
        }
    gains = [row["single_minus_dual"] for row in per_item]
    interval = _bootstrap_reduction(gains, seed, n_boot)
    return {
        "n_items": len(per_item),
        "n_candidates": 8,
        "single_average_loss": statistics.mean(row["single_average_loss"] for row in per_item),
        "dual_loss": statistics.mean(row["dual_loss"] for row in per_item),
        "single_minus_dual": {
            "mean": interval["reduction"],
            "ci95": interval["ci95"],
            "ci_excludes_zero": interval["excludes_zero"],
        },
        "folds": folds,
        "dual_e3_identical_rows": len(rows),
        "max_recovered_single_score_denominator": max_denominator,
        "max_reconstructed_dual_score_absolute_residual": max_residual,
        "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "per_item": per_item,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("reports/supplementary_e3_reference_free.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("exp/d0_resample/facts_dual"))
    parser.add_argument("--output", type=Path, default=Path("reports/selector_budget.json"))
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    grouped = defaultdict(list)
    for line in args.source.read_text().splitlines():
        row = json.loads(line)
        grouped[row["model"]].append(row)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "protocol": {
            "candidate_pool": "Original SPEAK plus R0-R6; identical E3 item IDs and fact caches",
            "reference": "Original SPEAK plan facts, never candidate-specific plans",
            "score": "Jaccard(original plan facts, selector readback facts)",
            "score_reconstruction": {
                "formula": "s_i = d_j + d_k - d_i, where d_i=(s_j+s_k)/2 is saved E3 plan_guided_scores",
                "rational_conversion": "Fraction(str(saved_score)).limit_denominator(limit)",
                "denominator_limit": DENOMINATOR_LIMIT,
                "dual_score_absolute_tolerance": 1e-12,
                "tie_handling": "Exact rational equality followed by earliest candidate",
                "reason": "Preserve recorded E3 scores without recomputing type-ambiguous channel merges",
            },
            "single": "Each non-held-out ASR separately; equal-weight mean of their held-out losses",
            "dual": "Mean Jaccard from the two non-held-out ASRs",
            "tie": "Earliest candidate wins",
            "aggregation": "Mean of three held-out folds within each item, then mean across items",
            "ci": "Paired percentile bootstrap over items after averaging folds; positive gain favors dual",
            "n_boot": args.n_boot,
            "seed": args.seed,
            "loss_unit": "fraction; multiply by 100 for percent or percentage-point differences",
            "asr_names": {"asr1": "Whisper", "asr2": "Seamless", "asr3": "FunASR"},
            "budget": {
                "single_selector_asr_calls_per_candidate": 1,
                "dual_selector_asr_calls_per_candidate": 2,
                "single_selector_asr_calls_per_pool": 8,
                "dual_selector_asr_calls_per_pool": 16,
                "held_out_evaluation_asr_calls_per_pool": 8,
                "new_asr_calls_in_this_cached_analysis": 0,
                "latency_measured": False,
            },
        },
        "models": {},
    }
    for model, rows in grouped.items():
        result = analyze_model(model, rows, args.cache_dir / f"{model}.jsonl", args.n_boot, args.seed)
        report["models"][model] = result
        gain = result["single_minus_dual"]
        print(f"{model}: n={result['n_items']} single={100*result['single_average_loss']:.4f}% "
              f"dual={100*result['dual_loss']:.4f}% gain={100*gain['mean']:+.4f} pp "
              f"CI=[{100*gain['ci95'][0]:+.4f}, {100*gain['ci95'][1]:+.4f}]", flush=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
