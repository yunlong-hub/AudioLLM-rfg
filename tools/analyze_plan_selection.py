"""CPU-only post-hoc analysis of the existing E3 candidate selections.

Run from the repository root:
    PYTHONPATH=src python tools/analyze_plan_selection.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from rfg.facts.schema import Fact
from rfg.score.selection_analysis import FACT_TYPES, analyze_model


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=Path("reports/supplementary_e3_reference_free.json"))
    parser.add_argument("--rows", type=Path, default=Path("reports/supplementary_e3_reference_free.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("reports/selection_analysis.json"))
    args = parser.parse_args()
    source = json.loads(args.source_report.read_text())
    rows = defaultdict(list)
    with args.rows.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["model"]].append(row)
    if set(rows) != set(source["models"]):
        raise ValueError("models differ between E3 report and per-question records")
    seed, n_boot = 20260923, 10_000
    report = {
        "protocol": {
            "analysis": "post-hoc descriptive; all five models and all five fact types retained; no multiplicity-adjusted or confirmatory significance claims",
            "selection": "fixed E3 original=0, plan_text_pick, plan_guided_pick; no candidate reselection",
            "facts_source": "read each cache record's persisted facts field directly; do not re-merge facts_rules and facts_llm or call load_extractions; retain the saved type/value/polarity triples",
            "strata": "actual original-plan Fact.type; include a question only when that type is nonempty; strata overlap",
            "loss": "1 - |upstream_type intersect heldout_selected_facts| / |upstream_type|",
            "aggregation": "mean of three heldout-fold losses within each question, then equally weighted question mean",
            "n_upstream_facts": "sum of unique upstream facts per question; count once, not once per ASR fold",
            "gain_direction": "Original minus Plan-Fact or Plan-Text minus Plan-Fact loss; positive favors Plan-Fact",
            "bootstrap": "resample paired question means with replacement; NumPy PCG64 reset to seed for each contrast; linear 2.5/97.5 percentiles",
            "seed": seed,
            "n_boot": n_boot,
            "outcome_zero_tolerance": 1e-12,
            "disagreement": "descriptive only; keep question-folds whose saved Text and Fact picks differ; average those folds per question, then average participating questions; do not reselect",
            "fact_types": list(FACT_TYPES),
            "units": "loss fractions; multiply by 100 for percentages and percentage-point gains",
        },
        "run_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "command": [sys.executable, *sys.argv],
            "implementation_sha256": {
                path: sha256(Path(path)) for path in (
                    "src/rfg/score/selection_analysis.py", "tools/analyze_plan_selection.py")
            },
            "source_report": str(args.source_report),
            "source_report_sha256": sha256(args.source_report),
            "source_rows": str(args.rows),
            "source_rows_sha256": sha256(args.rows),
            "n_question_folds": sum(map(len, rows.values())),
        },
        "models": {},
    }
    for model, source_model in source["models"].items():
        path = Path(source_model["cache"])
        cache_hash = sha256(path)
        if cache_hash != source_model["cache_sha256"]:
            raise ValueError(f"E3 fact cache changed: {path}")
        facts = {}
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("prompt_sha256") == source["run_metadata"]["prompt_sha256"]:
                    facts[record["key"]] = {Fact.from_dict(value) for value in record["facts"]}
        result = analyze_model(rows[model], facts, seed=seed, n_boot=n_boot)
        expected_n = source_model["folds"]["asr1"]["n_items"]
        if result["overall"]["n_questions"] != expected_n:
            raise ValueError(f"E3 question count changed: {model}")
        result["cache"] = str(path)
        result["cache_sha256"] = cache_hash
        report["models"][model] = result
        print(f"{model}: n={expected_n}, Original→Fact "
              f"{result['overall']['original_minus_fact']}", flush=True)
    report["run_metadata"]["n_questions"] = sum(
        model["overall"]["n_questions"] for model in report["models"].values())
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
