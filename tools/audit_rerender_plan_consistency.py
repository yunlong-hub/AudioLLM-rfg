"""Re-evaluate cached N=8 selection on items with identical text in every candidate.

This is a read-only analysis of generation/selection artifacts, not a new model run.
It retains only items whose R0--R6 texts equal the original SPEAK text after stripping
outer whitespace. Selection was already performed independently of the held-out ASR.
Bootstrap samples questions, not ASR folds or individual facts.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "Qwen3-Omni-30B-A3B-Instruct": "d0_pilot",
    "Qwen2.5-Omni-3B": "d0_pilot",
    "Qwen2.5-Omni-7B": "d3_7b_stack",
    "MiniCPM-o-4_5": "d1_main",
    "Step-Audio-2-mini": "d2_stepaudio",
}


def read(path):
    return json.loads(path.read_text())


def main():
    result = {"protocol": "N=8; exact text equality across original and R0--R6; "
              "outer whitespace stripped; item-mean paired reduction; 10000 bootstrap draws",
              "seed": 20260921, "models": {}}
    for model, run in RUNS.items():
        original = read(ROOT / f"exp/d0_resample/frr_loo_{model}.json")
        rows = original["per_item"]
        ids = {r["item_id"] for r in rows["asr1"]}
        strict = set()
        matched = 0
        for iid in sorted(ids):
            text = read(ROOT / f"exp/{run}/predictions/{model}/{iid}/SPEAK.json")["text"].strip()
            comparisons = [read(ROOT / f"exp/d0_resample/{model}/{iid}/R{k}.json")["text"].strip()
                           == text for k in range(7)]
            matched += sum(comparisons)
            if all(comparisons):
                strict.add(iid)
        item_ids = sorted(strict)
        rng = np.random.default_rng(result["seed"])
        samples = rng.integers(len(item_ids), size=(10000, len(item_ids)))
        folds = {}
        for asr, values in rows.items():
            indexed = {r["item_id"]: r for r in values}
            differences = np.array([indexed[i]["base_gap"] - indexed[i]["selected_gap"]
                                    for i in item_ids])
            draws = differences[samples].mean(axis=1)
            ci = np.quantile(draws, [.025, .975])
            folds[asr] = {"n_items": len(item_ids), "frr_reduction": float(differences.mean()),
                          "frr_ci95": ci.tolist(), "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
        result["models"][model] = {
            "source": f"exp/d0_resample/frr_loo_{model}.json", "base_run": run,
            "n_items_all": len(ids), "n_candidates_compared": 7*len(ids),
            "n_candidate_text_matches": matched, "n_items_text_identical": len(strict),
            "item_ids_text_identical": item_ids, "folds": folds,
            "macro_frr_reduction": float(np.mean([f["frr_reduction"] for f in folds.values()])),
        }
        print(model, len(strict), "/", len(ids), result["models"][model]["macro_frr_reduction"],
              [f["ci_excludes_zero"] for f in folds.values()])
    (ROOT / "reports/rerender_plan_consistency.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
