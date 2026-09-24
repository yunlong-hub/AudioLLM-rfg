import json

from rfg.audit.score import score_adjudication


def test_unfilled_human_packet_is_not_ready(tmp_path) -> None:
    packet = tmp_path / "packet.jsonl"
    row = {
        "asr": {name: {"text": "one", "facts": [{"type": "number", "value": "1"}]}
                for name in ("asr1", "asr2", "asr3")},
        "human": {
            "annotator_1": {"transcript": None, "facts": None},
            "annotator_2": {"transcript": None, "facts": None},
            "adjudicated": {"transcript": None, "facts": None},
            "extractor_facts_on_adjudicated_transcript": None,
        },
    }
    packet.write_text(json.dumps(row) + "\n")
    result = score_adjudication(packet)
    assert not result["ready"]
    assert result["completion"] == {
        "annotator_1": 0,
        "annotator_2": 0,
        "adjudicated": 0,
        "extractor_on_adjudicated": 0,
    }
    assert result["extractor_only_vs_human"]["n_items"] == 0


def test_completed_packet_scores_asr_and_extractor_separately(tmp_path) -> None:
    fact = {"type": "number", "value": "1", "polarity": "+"}
    wrong = {"type": "number", "value": "2", "polarity": "+"}
    packet = tmp_path / "packet.jsonl"
    row = {
        "asr": {
            "asr1": {"text": "one", "facts": [fact]},
            "asr2": {"text": "two", "facts": [wrong]},
            "asr3": {"text": "one", "facts": [fact]},
        },
        "human": {
            "annotator_1": {"transcript": "one", "facts": [fact]},
            "annotator_2": {"transcript": "one", "facts": [fact]},
            "adjudicated": {"transcript": "one", "facts": [fact]},
            "extractor_facts_on_adjudicated_transcript": [fact],
        },
    }
    packet.write_text(json.dumps(row) + "\n")
    result = score_adjudication(packet)
    assert result["ready"]
    assert result["asr_vs_human"]["asr1"]["fact_metrics"]["f1"] == 1.0
    assert result["asr_vs_human"]["asr2"]["fact_metrics"]["f1"] == 0.0
    assert result["extractor_only_vs_human"]["f1"] == 1.0
