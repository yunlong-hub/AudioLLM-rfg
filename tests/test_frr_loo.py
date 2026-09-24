from dataclasses import dataclass

from rfg.score.frr import evaluate_loo_folds


@dataclass
class Extracted:
    facts: set[str]


def test_leave_one_asr_out_never_uses_evaluator_for_selection() -> None:
    corpus = [{
        "item_id": "x",
        "internal_key": "up",
        "sample_keys": ["a1_base", "a1_r"],
        "sample_keys_asr2": ["a2_base", "a2_r"],
        "sample_keys_asr3": ["a3_base", "a3_r"],
    }]
    extracted = {
        "up": Extracted({"gold"}),
        "a1_base": Extracted(set()), "a1_r": Extracted({"gold"}),
        "a2_base": Extracted(set()), "a2_r": Extracted({"gold"}),
        "a3_base": Extracted({"gold"}), "a3_r": Extracted(set()),
    }
    result = evaluate_loo_folds(corpus, extracted, n_boot=100)
    # Holding out ASR3: ASR1+ASR2 select the rerender even though ASR3 dislikes it.
    assert result["per_item"]["asr3"][0]["picked_index"] == 1
    assert result["folds"]["asr3"]["frr_reduction"] == -1.0
    # Holding out ASR1: selectors ASR2+ASR3 tie, so conservative tie-break keeps base.
    assert result["per_item"]["asr1"][0]["picked_index"] == 0


def test_single_candidate_is_valid_curve_baseline() -> None:
    corpus = [{
        "item_id": "x",
        "internal_key": "up",
        "sample_keys": ["a1"],
        "sample_keys_asr2": ["a2"],
        "sample_keys_asr3": ["a3"],
    }]
    extracted = {
        "up": Extracted({"gold"}),
        "a1": Extracted({"gold"}),
        "a2": Extracted(set()),
        "a3": Extracted({"gold"}),
    }
    result = evaluate_loo_folds(corpus, extracted, n_boot=100)
    assert all(fold["n_candidates"] == 1 for fold in result["folds"].values())
    assert all(fold["frr_reduction"] == 0 for fold in result["folds"].values())
