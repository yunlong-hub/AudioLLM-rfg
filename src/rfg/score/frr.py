"""Independent leave-one-ASR-out evaluation for fact-retention reranking."""
from __future__ import annotations

import random
import statistics
from typing import Any


ASR_NAMES = ("asr1", "asr2", "asr3")


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _gap(upstream: set, downstream: set) -> float:
    return 1 - len(upstream & downstream) / len(upstream)


def _bootstrap_reduction(values: list[float], seed: int, n_boot: int) -> dict:
    rng = random.Random(seed)
    point = statistics.mean(values)
    draws = sorted(
        statistics.mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(n_boot)
    )
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws)) - 1]
    return {"reduction": point, "ci95": [lo, hi], "excludes_zero": lo > 0 or hi < 0}


def evaluate_loo_folds(
    corpus: list[dict], extracted: dict[str, Any], *, seed: int = 20260918, n_boot: int = 10_000,
) -> dict:
    """Use two ASRs for selection and hold the third out for evaluation.

    Candidate score is the mean Jaccard similarity between the internal fact set
    and each of the two selector-ASR fact sets.  The held-out ASR is never read by
    the selector.  Ties retain the earliest candidate, making the original sample
    the conservative default.
    """
    folds: dict[str, dict] = {}
    fold_rows: dict[str, list[dict]] = {name: [] for name in ASR_NAMES}

    for item in corpus:
        upstream = set(extracted[item["internal_key"]].facts)
        if not upstream:
            continue
        keys = {
            "asr1": item["sample_keys"],
            "asr2": item["sample_keys_asr2"],
            "asr3": item.get("sample_keys_asr3") or [],
        }
        lengths = {len(value) for value in keys.values()}
        if len(lengths) != 1 or next(iter(lengths), 0) < 1:
            continue
        if any(key not in extracted for values in keys.values() for key in values):
            continue
        n_candidates = len(keys["asr1"])

        for held_out in ASR_NAMES:
            selectors = [name for name in ASR_NAMES if name != held_out]
            candidates = []
            for index in range(n_candidates):
                selector_score = statistics.mean(
                    _jaccard(upstream, set(extracted[keys[name][index]].facts))
                    for name in selectors
                )
                evaluation_gap = _gap(upstream, set(extracted[keys[held_out][index]].facts))
                candidates.append((index, selector_score, evaluation_gap))
            selected = max(candidates, key=lambda value: value[1])
            fold_rows[held_out].append({
                "item_id": item["item_id"],
                "selectors": selectors,
                "held_out_evaluator": held_out,
                "picked_index": selected[0],
                "selector_score": selected[1],
                "base_gap": candidates[0][2],
                "random_gap": statistics.mean(value[2] for value in candidates),
                "selected_gap": selected[2],
                "oracle_gap": min(value[2] for value in candidates),
                "n_candidates": n_candidates,
            })

    for fold_index, held_out in enumerate(ASR_NAMES):
        rows = fold_rows[held_out]
        if not rows:
            raise ValueError(f"no complete candidates for held-out evaluator {held_out}")
        reductions = [row["base_gap"] - row["selected_gap"] for row in rows]
        effect = _bootstrap_reduction(reductions, seed + fold_index, n_boot)
        folds[held_out] = {
            "selectors": [name for name in ASR_NAMES if name != held_out],
            "evaluator": held_out,
            "n_items": len(rows),
            "n_candidates": rows[0]["n_candidates"],
            "delta_render_single": statistics.mean(row["base_gap"] for row in rows),
            "delta_render_random": statistics.mean(row["random_gap"] for row in rows),
            "delta_render_selected": statistics.mean(row["selected_gap"] for row in rows),
            "delta_render_oracle": statistics.mean(row["oracle_gap"] for row in rows),
            "frr_reduction": effect["reduction"],
            "frr_ci95": effect["ci95"],
            "ci_excludes_zero": effect["excludes_zero"],
            "picked_original_rate": sum(row["picked_index"] == 0 for row in rows) / len(rows),
        }

    reductions = [folds[name]["frr_reduction"] for name in ASR_NAMES]
    return {
        "protocol": "leave-one-ASR-out; selector=mean Jaccard from the other two ASRs",
        "folds": folds,
        "macro_frr_reduction": statistics.mean(reductions),
        "min_fold_frr_reduction": min(reductions),
        "all_fold_cis_exclude_zero": all(folds[name]["ci_excludes_zero"] for name in ASR_NAMES),
        "per_item": fold_rows,
    }
