import pytest
from fractions import Fraction

from rfg.score.selector_budget import evaluate_selector_budget, restore_single_scores


def test_single_and_dual_ties_choose_earliest_candidate():
    saved_means = {name: [1 / 3, 1 / 3, 0.0] for name in ("asr1", "asr2", "asr3")}
    scores = restore_single_scores(saved_means)
    assert scores["asr1"] == [Fraction(1, 3), Fraction(1, 3), Fraction(0)]
    result = evaluate_selector_budget(scores, [0.5, 0.0, 1.0], "asr3")
    assert result["dual"]["picked_index"] == 0
    assert all(single["picked_index"] == 0 for single in result["single"].values())


def test_held_out_scores_and_gaps_change_losses_not_selection():
    scores = {
        "asr1": [Fraction(0), Fraction(1)],
        "asr2": [Fraction(0), Fraction(1)],
        "asr3": [Fraction(1), Fraction(0)],
    }
    before = evaluate_selector_budget(scores, [0.0, 1.0], "asr3")
    scores["asr3"] = [Fraction(0), Fraction(1)]
    after = evaluate_selector_budget(scores, [1.0, 0.0], "asr3")
    assert before["dual"]["picked_index"] == after["dual"]["picked_index"] == 1
    assert [row["picked_index"] for row in before["single"].values()] == [1, 1]
    assert [row["picked_index"] for row in after["single"].values()] == [1, 1]
    assert before["dual"]["loss"] == 1
    assert after["dual"]["loss"] == 0


def test_single_selector_average_is_not_best_selector():
    scores = restore_single_scores({"asr1": [0.0, 1.0], "asr2": [0.5, 0.5], "asr3": [0.5, 0.5]})
    result = evaluate_selector_budget(scores, [1.0, 0.0], "asr3")
    assert result["single"]["asr1"]["loss"] == 1
    assert result["single"]["asr2"]["loss"] == 0
    assert result["single_average_loss"] == pytest.approx(0.5)
    assert result["dual"]["picked_index"] == 0
    assert result["single_minus_dual"] == pytest.approx(-0.5)
