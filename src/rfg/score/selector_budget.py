"""Compare one versus two plan-fact ASR selectors on a fixed candidate pool."""
from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from rfg.score.frr import ASR_NAMES

DENOMINATOR_LIMIT = 1_000_000


def restore_single_scores(dual_scores: Mapping[str, Sequence[float]]) -> dict[str, list[Fraction]]:
    """Recover s_i = d_j + d_k - d_i from saved three-fold mean scores.

    d_i is the mean score from the two ASRs other than i. Rational recovery
    removes JSON floating-point roundoff so exact Jaccard ties remain ties.
    """
    means = {
        name: [Fraction(str(value)).limit_denominator(DENOMINATOR_LIMIT) for value in values]
        for name, values in dual_scores.items()
    }
    return {
        name: [
            sum(means[other][index] for other in ASR_NAMES if other != name) - means[name][index]
            for index in range(len(means[name]))
        ]
        for name in ASR_NAMES
    }


def evaluate_selector_budget(
    scores: Mapping[str, Sequence[Fraction]],
    held_out_gaps: Sequence[float],
    held_out: str,
) -> dict[str, Any]:
    """Hold out one evaluator; use either or both other ASRs for selection.

    Inputs are original-plan Jaccard scores and aligned held-out candidate gaps.
    ``single_average_loss`` gives equal weight to both single-ASR alternatives;
    it does not choose the better alternative using the evaluator.
    """
    selectors = [name for name in ASR_NAMES if name != held_out]
    n_candidates = len(held_out_gaps)
    dual_scores = [
        statistics.mean(scores[name][index] for name in selectors)
        for index in range(n_candidates)
    ]

    def pick(values: Sequence[Fraction]) -> int:
        return max(range(n_candidates), key=lambda index: (values[index], -index))

    # The held-out gaps are used only after computing all selection decisions.
    single_picks = {name: pick(scores[name]) for name in selectors}
    dual_pick = pick(dual_scores)
    single = {
        name: {"picked_index": index, "loss": held_out_gaps[index]}
        for name, index in single_picks.items()
    }
    single_average = statistics.mean(entry["loss"] for entry in single.values())
    return {
        "evaluator": held_out,
        "selectors": selectors,
        "n_candidates": n_candidates,
        "original_loss": held_out_gaps[0],
        "single": single,
        "single_average_loss": single_average,
        "dual": {"picked_index": dual_pick, "loss": held_out_gaps[dual_pick]},
        "single_minus_dual": single_average - held_out_gaps[dual_pick],
        "dual_scores": [float(value) for value in dual_scores],
    }
