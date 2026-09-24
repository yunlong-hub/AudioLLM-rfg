import copy

import pytest

from rfg.audit.blinding import (make_adjudication_sheet, make_blinded_sheet,
                                merge_adjudication_sheet,
                                merge_annotator_sheets)


def _master_rows() -> list[dict]:
    rows = []
    for index in range(5):
        rows.append({
            "audit_id": f"three_asr_{index:03d}",
            "audio": f"/tmp/audio_{index}.wav",
            "model": "secret-model",
            "run_id": "secret-run",
            "item_id": f"secret-item-{index}",
            "category": "number",
            "stratum": "all_differ",
            "internal_text": "secret internal text",
            "asr": {"asr1": {"text": "secret ASR"}},
            "pairwise_wer": {"asr1_asr2": 1.0},
            "extractor": {"asr1_llm": []},
            "human": {
                "annotator_1": {"transcript": None, "facts": None, "notes": None},
                "annotator_2": {"transcript": None, "facts": None, "notes": None},
                "adjudicated": {"transcript": None, "facts": None, "notes": None},
                "extractor_facts_on_adjudicated_transcript": None,
            },
        })
    return rows


def _fill(sheet: list[dict], prefix: str) -> list[dict]:
    result = copy.deepcopy(sheet)
    for row in result:
        row["annotation"] = {
            "transcript": f"{prefix} transcript {row['audit_id']}",
            "facts": [{"type": "number", "value": "1", "polarity": "+"}],
            "notes": None,
        }
    return result


def test_blinded_sheets_hide_automatic_evidence_and_use_different_orders() -> None:
    master = _master_rows()
    first = make_blinded_sheet(master, "annotator_1", seed=7)
    second = make_blinded_sheet(master, "annotator_2", seed=7)
    assert {row["audit_id"] for row in first} == {row["audit_id"] for row in second}
    assert [row["audit_id"] for row in first] != [row["audit_id"] for row in second]
    for row in first + second:
        assert set(row) == {"audit_id", "audio", "annotation"}
        assert row["annotation"] == {"transcript": None, "facts": None, "notes": None}
        serialized = str(row)
        assert "secret-model" not in serialized
        assert "secret internal text" not in serialized
        assert "secret ASR" not in serialized


def test_merge_requires_every_annotation_and_preserves_master() -> None:
    master = _master_rows()
    original = copy.deepcopy(master)
    first = _fill(make_blinded_sheet(master, "annotator_1", seed=7), "one")
    second = _fill(make_blinded_sheet(master, "annotator_2", seed=7), "two")
    second[0]["annotation"]["facts"] = None
    with pytest.raises(ValueError, match="facts must be a list"):
        merge_annotator_sheets(master, first, second)
    assert master == original


def test_two_stage_merge_keeps_adjudicator_blind() -> None:
    master = _master_rows()
    first = _fill(make_blinded_sheet(master, "annotator_1", seed=7), "one")
    second = _fill(make_blinded_sheet(master, "annotator_2", seed=7), "two")
    merged = merge_annotator_sheets(master, first, second)
    adjudication = make_adjudication_sheet(merged, seed=11)
    for row in adjudication:
        assert set(row) == {
            "audit_id", "audio", "candidate_a", "candidate_b", "adjudicated"
        }
        serialized = str(row)
        assert "secret-model" not in serialized
        assert "secret internal text" not in serialized
        assert "secret ASR" not in serialized
        row["adjudicated"] = {
            "transcript": "gold transcript",
            "facts": [],
            "notes": "resolved by listening",
        }
    completed = merge_adjudication_sheet(merged, adjudication)
    assert all(row["human"]["adjudicated"]["transcript"] == "gold transcript"
               for row in completed)
    assert all(row["human"]["extractor_facts_on_adjudicated_transcript"] is None
               for row in completed)
    assert all(row["human"]["adjudicated"]["transcript"] is None for row in master)
