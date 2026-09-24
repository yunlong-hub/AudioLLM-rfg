import json
from pathlib import Path

from rfg.score.benchmark import (
    extract_bbh_answer,
    extract_numeric_answer,
    score_benchmark,
)


def test_numeric_extractor_prefers_final_cue() -> None:
    assert extract_numeric_answer("We compute 7 + 11 = 18. The final answer is: 18.") == "18"
    assert extract_numeric_answer(r"Intermediate 4, then \\boxed{1,024}") == "1,024"
    assert extract_numeric_answer("Thus the answer is -1 / 2") == "-1/2"


def test_bbh_task_specific_extractors() -> None:
    assert extract_bbh_answer("Reasoning... The answer is: No", "navigate") == "no"
    assert extract_bbh_answer("I choose option (B).", "hyperbaton") == "(B)"
    assert extract_bbh_answer("A is awkward; the answer is B", "hyperbaton") == "(B)"
    assert extract_bbh_answer("A", "hyperbaton") == "(A)"


def test_full_accuracy_requires_complete_non_error_coverage(tmp_path: Path) -> None:
    items = tmp_path / "items.jsonl"
    rows = [
        {"id": "x0", "category": "gsm8k", "difficulty": None, "expected_answer_text": "18"},
        {"id": "x1", "category": "gsm8k", "difficulty": None, "expected_answer_text": "3"},
    ]
    items.write_text("".join(json.dumps(row) + "\n" for row in rows))
    predictions = tmp_path / "predictions"
    (predictions / "x0").mkdir(parents=True)
    (predictions / "x0" / "LISTEN.json").write_text(json.dumps({"text": "Answer: 18"}))

    summary, _ = score_benchmark(items, predictions, ["LISTEN"])
    block = summary["conditions"]["LISTEN"]
    assert block["coverage"] == 0.5
    assert block["accuracy_on_available"] == 1.0
    assert block["accuracy"] is None
    assert not summary["ready"]

    (predictions / "x1").mkdir()
    (predictions / "x1" / "LISTEN.json").write_text(json.dumps({"text": "final answer = 3"}))
    summary, _ = score_benchmark(items, predictions, ["LISTEN"])
    assert summary["conditions"]["LISTEN"]["accuracy"] == 1.0
    assert summary["ready"]


def test_spoken_answer_uses_requested_readback(tmp_path: Path) -> None:
    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps({
        "id": "x0", "category": "gsm8k", "difficulty": None,
        "expected_answer_text": "18",
    }) + "\n")
    predictions = tmp_path / "predictions" / "x0"
    predictions.mkdir(parents=True)
    (predictions / "SPEAK.json").write_text(json.dumps({
        "text": "answer is 19",
        "readback": {"asr1": "answer is 18", "asr1_error": None},
    }))
    summary, _ = score_benchmark(items, predictions.parent, ["SPEAK"], ["asr1"])
    assert summary["conditions"]["SPEAK"]["accuracy"] == 0.0
    assert summary["spoken_answer"]["asr1"]["accuracy"] == 1.0
    assert summary["ready"]


def test_empty_successful_readback_is_complete_and_wrong(tmp_path: Path) -> None:
    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps({
        "id": "x0", "category": "gsm8k", "difficulty": None,
        "expected_answer_text": "18",
    }) + "\n")
    predictions = tmp_path / "predictions" / "x0"
    predictions.mkdir(parents=True)
    (predictions / "SPEAK.json").write_text(json.dumps({
        "text": "answer is 18",
        "readback": {"asr3": None, "asr3_sec": 0.5, "asr3_error": None},
    }))

    summary, details = score_benchmark(
        items, predictions.parent, ["SPEAK"], ["asr3"]
    )
    block = summary["spoken_answer"]["asr3"]
    assert block["coverage"] == 1.0
    assert block["accuracy"] == 0.0
    assert summary["ready"]
    assert details[-1]["status"] == "ok"
    assert details[-1]["text"] == ""
