import copy

import numpy as np
import pytest

from rfg.facts.schema import Fact
from rfg.score.selection_analysis import analyze_model, paired_summary, question_losses, summarize_group


def selection_fixture():
    number = Fact("number", "16")
    content = Fact("content", "count")
    facts = {"example|SPEAK#internal": {number, content}}
    rows = []
    for evaluator in ("asr1", "asr2", "asr3"):
        keys = [f"example|{evaluator}_{index}" for index in range(3)]
        facts.update({keys[0]: {content}, keys[1]: {number, content}, keys[2]: {number}})
        rows.append({
            "item_id": "example", "evaluator": evaluator, "candidate_keys": keys,
            "plan_text_pick": 0, "plan_guided_pick": 2,
            "base_gap": 0.5, "plan_text_gap": 0.5, "plan_guided_gap": 0.5,
        })
    return rows, facts


def test_type_mask_uses_facts_and_preserves_saved_picks():
    rows, facts = selection_fixture()
    original_rows = copy.deepcopy(rows)
    result = analyze_model(rows, facts, n_boot=10)

    assert result["by_fact_type"]["number"]["n_questions"] == 1
    assert result["by_fact_type"]["number"]["n_upstream_facts"] == 1
    assert result["by_fact_type"]["unit"]["n_questions"] == 0
    assert result["by_fact_type"]["unit"]["original_minus_fact"]["ci95"] is None
    assert result["by_fact_type"]["number"]["original_minus_fact"]["mean"] == 1
    # Candidate 1 is better overall, but the saved Fact pick is candidate 2.
    assert result["overall"]["loss_mean"]["plan_fact"] == 0.5
    assert result["by_fact_type"]["content"]["text_minus_fact"]["mean"] == -1
    assert rows == original_rows


def test_question_mean_precedes_bootstrap_and_outcome_counts():
    questions = [
        {"n_upstream_facts": 1, "losses": np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
         "different_picks": np.array([True, False, False])},
        {"n_upstream_facts": 9, "losses": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         "different_picks": np.array([False, False, False])},
    ]
    result = summarize_group(questions, seed=20260923, n_boot=1000)
    assert result["original_minus_fact"]["mean"] == pytest.approx(1 / 6)
    assert result["original_minus_fact"]["ci95"] == pytest.approx([0, 1 / 3])
    assert result["original_minus_fact"]["improved"] == 1
    assert result["original_minus_fact"]["unchanged"] == 1
    assert result["n_upstream_facts"] == 10
    assert result["selection_disagreement"]["n_questions"] == 1
    assert result["selection_disagreement"]["n_question_folds"] == 1
    assert result["selection_disagreement"]["conditional_loss_mean"]["original"] == 1


def test_source_loss_validation_rejects_misaligned_cache():
    rows, facts = selection_fixture()
    facts[rows[0]["candidate_keys"][2]] = set()
    with pytest.raises(ValueError, match="cached/source loss mismatch"):
        question_losses(rows, facts)


def test_bootstrap_is_reproducible_and_retains_negative_gains():
    gains = np.array([-0.5, 0.0, 0.25])
    result = paired_summary(gains, seed=20260923, n_boot=100)
    assert result == paired_summary(gains, seed=20260923, n_boot=100)
    assert result["mean"] == pytest.approx(-1 / 12)
    assert (result["improved"], result["unchanged"], result["worsened"]) == (1, 1, 1)
