"""Post-hoc, question-paired analysis of fixed E3 selections.

No candidate is selected here. The saved Original / Plan-Text / Plan-Fact
indices are evaluated on the saved held-out ASR facts. Fact-type strata are
defined by nonempty types in the original plan, not by item-ID categories.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from rfg.facts.schema import Fact

FACT_TYPES = ("number", "unit", "proper_noun", "negation", "content")
METHODS = ("original", "plan_text", "plan_fact")
FOLDS = ("asr1", "asr2", "asr3")
ZERO_TOLERANCE = 1e-12


def outcome_counts(gains: np.ndarray) -> dict[str, int]:
    """Count positive / numerically zero / negative paired loss reductions."""
    return {
        "improved": int(np.count_nonzero(gains > ZERO_TOLERANCE)),
        "unchanged": int(np.count_nonzero(np.abs(gains) <= ZERO_TOLERANCE)),
        "worsened": int(np.count_nonzero(gains < -ZERO_TOLERANCE)),
    }


def paired_summary(gains: np.ndarray, *, seed: int, n_boot: int) -> dict:
    """Percentile bootstrap of question means, never of individual ASR folds.

    Each contrast starts a PCG64 generator at the declared seed. Percentiles
    use NumPy's linear interpolation. Intervals are descriptive and unadjusted.
    """
    if not len(gains):
        return {"mean": None, "ci95": None, **outcome_counts(gains)}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(gains), size=(n_boot, len(gains)))
    means = gains[indices].mean(axis=1)
    return {
        "mean": float(gains.mean()),
        "ci95": np.quantile(means, [0.025, 0.975], method="linear").tolist(),
        **outcome_counts(gains),
    }


def question_losses(rows: Sequence[dict], facts: Mapping[str, set[Fact]]) -> list[dict]:
    """Evaluate saved indices and retain three-fold loss matrices per question.

    Matrix columns are METHODS; rows are FOLDS. The source loss checks ensure
    that the cache and saved selection records refer to the same experiment.
    """
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        item_id, evaluator = row["item_id"], row["evaluator"]
        if evaluator in grouped[item_id]:
            raise ValueError(f"duplicate fold: {item_id}/{evaluator}")
        grouped[item_id][evaluator] = row

    questions = []
    for item_id in sorted(grouped):
        folds = grouped[item_id]
        if set(folds) != set(FOLDS):
            raise ValueError(f"incomplete three-fold question: {item_id}")
        upstream = facts[f"{item_id}|SPEAK#internal"]
        if not upstream:
            raise ValueError(f"empty original plan: {item_id}")
        unknown = {fact.type for fact in upstream} - set(FACT_TYPES)
        if unknown:
            raise ValueError(f"unknown fact types in {item_id}: {unknown}")
        by_type = {
            fact_type: {fact for fact in upstream if fact.type == fact_type}
            for fact_type in FACT_TYPES
        }
        losses = []
        type_losses = {fact_type: [] for fact_type in FACT_TYPES if by_type[fact_type]}
        different = []
        for evaluator in FOLDS:
            row = folds[evaluator]
            picks = (0, row["plan_text_pick"], row["plan_guided_pick"])
            downstream = [facts[row["candidate_keys"][index]] for index in picks]
            current = [1 - len(upstream & value) / len(upstream) for value in downstream]
            expected = [row["base_gap"], row["plan_text_gap"], row["plan_guided_gap"]]
            if not np.allclose(current, expected, rtol=0, atol=ZERO_TOLERANCE):
                raise ValueError(f"cached/source loss mismatch: {item_id}/{evaluator}")
            losses.append(current)
            different.append(picks[1] != picks[2])
            for fact_type, values in type_losses.items():
                typed = by_type[fact_type]
                values.append([1 - len(typed & value) / len(typed) for value in downstream])
        questions.append({
            "item_id": item_id,
            "n_upstream_facts": len(upstream),
            "losses": np.asarray(losses),
            "different_picks": np.asarray(different),
            "by_type": {
                fact_type: {
                    "n_upstream_facts": len(by_type[fact_type]),
                    "losses": np.asarray(values),
                }
                for fact_type, values in type_losses.items()
            },
        })
    return questions


def summarize_group(questions: Sequence[dict], *, seed: int, n_boot: int) -> dict:
    """Equal question weights after averaging each question's three folds."""
    means = np.asarray([question["losses"].mean(axis=0) for question in questions]).reshape(-1, 3)
    conditional = [
        question["losses"][question["different_picks"]].mean(axis=0)
        for question in questions if question["different_picks"].any()
    ]
    conditional_means = np.asarray(conditional).reshape(-1, 3)
    different_folds = sum(int(question["different_picks"].sum()) for question in questions)
    different_gains = conditional_means[:, 1] - conditional_means[:, 2]

    def loss_means(values: np.ndarray) -> dict[str, float | None]:
        return {
            name: float(values[:, index].mean()) if len(values) else None
            for index, name in enumerate(METHODS)
        }

    return {
        "n_questions": len(questions),
        "n_upstream_facts": sum(question["n_upstream_facts"] for question in questions),
        "loss_mean": loss_means(means),
        "original_minus_fact": paired_summary(means[:, 0] - means[:, 2], seed=seed, n_boot=n_boot),
        "text_minus_fact": paired_summary(means[:, 1] - means[:, 2], seed=seed, n_boot=n_boot),
        "selection_disagreement": {
            "n_questions": len(conditional),
            "n_question_folds": different_folds,
            "question_fold_rate": different_folds / (3 * len(questions)) if questions else None,
            "conditional_loss_mean": loss_means(conditional_means),
            "conditional_text_minus_fact_mean": float(different_gains.mean()) if len(conditional) else None,
            "conditional_question_outcomes": outcome_counts(different_gains),
        },
    }


def analyze_model(rows: Sequence[dict], facts: Mapping[str, set[Fact]], *,
                  seed: int = 20260923, n_boot: int = 10_000) -> dict:
    questions = question_losses(rows, facts)
    by_type = {}
    for fact_type in FACT_TYPES:
        typed = [
            {**question["by_type"][fact_type], "different_picks": question["different_picks"]}
            for question in questions if fact_type in question["by_type"]
        ]
        by_type[fact_type] = summarize_group(typed, seed=seed, n_boot=n_boot)
    return {
        "overall": summarize_group(questions, seed=seed, n_boot=n_boot),
        "by_fact_type": by_type,
        "validation": {"n_question_folds": len(rows), "n_source_losses_matched": 3 * len(rows)},
    }
