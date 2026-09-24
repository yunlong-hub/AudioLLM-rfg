from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import supp_e4_e2e


@pytest.mark.parametrize("text,gold,answer,correct", [
    ("A", "(A)", "(A)", True),
    ("B.", "(B)", "(B)", True),
    ("The answer is B.", "(A)", "(B)", False),
    ("unparseable", "(A)", None, False),
])
def test_e4_uses_shared_option_label_extraction(text, gold, answer, correct):
    item = {"difficulty": "hyperbaton", "expected_answer_text": gold}
    assert supp_e4_e2e.score_answer(text, item) == (answer, correct)


def test_e4_uses_task_specific_yes_no_and_strict_equality():
    item = {"difficulty": "navigate", "expected_answer_text": "no"}
    assert supp_e4_e2e.score_answer("Initially yes, but the final answer is no.", item) == ("no", True)
    assert supp_e4_e2e.score_answer("The answer is nobody.", item) == (None, False)


def test_missing_cache_reports_exact_keys_without_extraction(monkeypatch):
    monkeypatch.setattr(supp_e4_e2e, "load_extractions", lambda *args, **kwargs: {})
    with pytest.raises(SystemExit, match='"missing_keys": \\["item\\|SPEAK"\\]'):
        supp_e4_e2e.load_cached_facts(Path("unused.jsonl"), {"item|SPEAK": "B"})


def test_mismatched_cache_text_is_reported(monkeypatch):
    monkeypatch.setattr(supp_e4_e2e, "load_extractions", lambda *args, **kwargs: {
        "item|SPEAK": SimpleNamespace(text="A"),
    })
    with pytest.raises(SystemExit, match='"text_mismatch_keys": \\["item\\|SPEAK"\\]'):
        supp_e4_e2e.load_cached_facts(Path("unused.jsonl"), {"item|SPEAK": "B"})


def test_correction_preserves_original_report_without_nesting():
    old = {"folds": {"asr1": {"original": {"plan_loss_mean": 0.2, "accuracy": 0.3}}}}
    current = {"asr1": {"original": {"plan_loss_mean": 0.2, "accuracy": 0.7}}}
    first = supp_e4_e2e.correction_record(old, current)
    assert first["reference_plan_statistics_unchanged"]
    second = supp_e4_e2e.correction_record({"correction": first, "folds": current}, current)
    assert second["previous_report"] == old
