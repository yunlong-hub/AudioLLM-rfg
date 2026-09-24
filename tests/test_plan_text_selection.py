from dataclasses import dataclass
import random

import pytest

from rfg.score.plan_text import (
    normalize_plan_tokens,
    normalized_word_similarity,
    number_conversion_failures,
    plan_text_scores,
)
from tools.supp_e3_reference_free import score_item, word_similarity


@dataclass
class Extracted:
    facts: set[str]
    text: str


@pytest.mark.parametrize(
    "text",
    ["The count is sixteen", "THE COUNT IS SIXTEEN!", " the\tcount\nis sixteen. "],
)
def test_normalization_ignores_case_punctuation_and_whitespace(text: str) -> None:
    assert normalize_plan_tokens(text) == ("the", "count", "is", "sixteen")
    assert normalized_word_similarity("The count is sixteen", text) == 1.0


def test_normalization_converts_numbers_and_percent_to_words() -> None:
    assert normalize_plan_tokens("16%") == ("sixteen", "percent")
    assert normalized_word_similarity("The count is 16", "the count is sixteen") == 1.0
    assert normalized_word_similarity("16%", "sixteen percent") == 1.0
    assert normalized_word_similarity("16", "60") == 0.0


def test_word_order_changes_similarity() -> None:
    assert normalized_word_similarity("dog bites man", "dog bites man") == 1.0
    assert normalized_word_similarity("dog bites man", "man bites dog") < 1.0


def test_plan_text_scores_average_the_selector_asrs() -> None:
    scores = plan_text_scores(
        "red green blue",
        [["red green blue", "wrong"], ["wrong", "wrong"]],
    )
    assert scores == pytest.approx([0.5, 0.0])


@pytest.fixture
def selection_inputs() -> tuple[dict, dict[str, Extracted], dict[str, str]]:
    item = {
        "item_id": "example",
        "internal_key": "plan",
        "sample_keys": ["asr1_0", "asr1_1", "asr1_2"],
        "sample_keys_asr2": ["asr2_0", "asr2_1", "asr2_2"],
        "sample_keys_asr3": ["asr3_0", "asr3_1", "asr3_2"],
    }
    texts = {"plan": "The count is sixteen"}
    extracted = {"plan": Extracted({"count sixteen"}, texts["plan"])}
    for asr in ("asr1", "asr2", "asr3"):
        for index, text in enumerate(("wrong words", "the count is sixteen", "The count is 16.")):
            key = f"{asr}_{index}"
            texts[key] = text
            extracted[key] = Extracted(set(), text)
    return item, extracted, texts


def test_score_item_selects_earliest_of_tied_best_candidates(selection_inputs) -> None:
    item, extracted, texts = selection_inputs
    row = score_item(item, extracted, texts, {}, "asr3", random.Random(0))

    assert row is not None
    assert row["plan_text_scores"] == pytest.approx([0.0, 1.0, 1.0])
    assert row["plan_text_pick"] == 1


@pytest.mark.parametrize("held_out", ["asr1", "asr2", "asr3"])
def test_held_out_transcripts_and_facts_do_not_change_selection(
    selection_inputs, held_out: str,
) -> None:
    item, extracted, texts = selection_inputs
    before = score_item(item, extracted, texts, {}, held_out, random.Random(0))

    # Make the evaluator prefer candidate zero, opposing both selectors.
    texts[f"{held_out}_0"] = texts["plan"]
    texts[f"{held_out}_1"] = "wrong words"
    texts[f"{held_out}_2"] = "wrong words"
    for index in range(3):
        key = f"{held_out}_{index}"
        extracted[key].text = texts[key]
    changed_text = score_item(item, extracted, texts, {}, held_out, random.Random(0))

    # Its facts still determine evaluation of the selected candidate.
    extracted[f"{held_out}_1"].facts = {"count sixteen"}
    changed_facts = score_item(item, extracted, texts, {}, held_out, random.Random(0))

    assert before is not None and changed_text is not None and changed_facts is not None
    assert before["plan_text_pick"] == changed_text["plan_text_pick"] == changed_facts["plan_text_pick"] == 1
    assert before["plan_text_scores"] == changed_text["plan_text_scores"] == changed_facts["plan_text_scores"]
    assert before["plan_text_gap"] == changed_text["plan_text_gap"] == 1.0
    assert changed_facts["plan_text_gap"] == 0.0


def test_plan_text_selection_does_not_use_extracted_fact_content(selection_inputs) -> None:
    item, extracted, texts = selection_inputs
    before = score_item(item, extracted, texts, {}, "asr3", random.Random(0))
    changed = {
        key: Extracted({"different fact"}, value.text)
        for key, value in extracted.items()
    }
    after = score_item(item, changed, texts, {}, "asr3", random.Random(0))

    assert before is not None and after is not None
    assert before["plan_text_pick"] == after["plan_text_pick"] == 1
    assert before["plan_text_scores"] == after["plan_text_scores"]


def test_plan_text_selection_uses_fact_cache_texts_when_loader_texts_differ(selection_inputs) -> None:
    item, extracted, texts = selection_inputs
    before = score_item(item, extracted, texts, {}, "asr3", random.Random(0))
    changed_texts = {key: "different loader text" for key in texts}
    after = score_item(item, extracted, changed_texts, {}, "asr3", random.Random(0))

    assert before is not None and after is not None
    assert before["plan_text_pick"] == after["plan_text_pick"] == 1
    assert before["plan_text_scores"] == after["plan_text_scores"]


def test_legacy_word_similarity_keeps_number_and_punctuation_distinctions() -> None:
    assert word_similarity("16", "sixteen") == 0.0
    assert word_similarity("sixteen.", "sixteen") == 0.0
    assert word_similarity("SIXTEEN", "sixteen") == 1.0


def test_out_of_range_number_uses_shared_literal_policy_and_is_reported() -> None:
    large_number = "1" + "0" * 500
    assert normalize_plan_tokens(large_number) == (large_number,)
    failures = number_conversion_failures(large_number)
    assert len(failures) == 1
    assert failures[0]["integer_digits"] == 501
    assert failures[0]["error"] == "OverflowError"
    assert number_conversion_failures("16, 1.05, 2,500") == []
