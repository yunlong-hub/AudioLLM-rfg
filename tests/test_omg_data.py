import json

import pytest

from rfg.data.omg import (BBH_INSTRUCTIONS, SPOKEN_MQA_SHARDS, _audio_extension,
                          final_reference_answer, prepare_gsm8k)


def test_audio_extension_recognizes_wave():
    assert _audio_extension(b"RIFF\x00\x00\x00\x00WAVEfmt ") == ".wav"


def test_bbh_conditions_are_distinct_and_explicit():
    assert set(BBH_INSTRUCTIONS) == {"short", "cot"}
    assert BBH_INSTRUCTIONS["short"] != BBH_INSTRUCTIONS["cot"]


def test_spoken_mqa_omg_scope_is_multistep_1402_only():
    assert len(SPOKEN_MQA_SHARDS) == 2
    assert all(name.startswith("multi_step_reasoning-") for name in SPOKEN_MQA_SHARDS)


def test_reference_answer_accepts_direct_and_worked_formats():
    assert final_reference_answer({"text": ["4"]}) == "4"
    assert final_reference_answer({"text": ["work <<2*3=6>>6\n#### 6"]}) == "6"
    assert final_reference_answer("work\n#### 1,509") == "1509"


def test_gsm8k_rejects_partial_dataset(tmp_path):
    source = tmp_path / "test.jsonl"
    source.write_text(json.dumps({"question": "1+1?", "answer": "work #### 2"}) + "\n")
    with pytest.raises(ValueError, match="expected complete set of 1319"):
        prepare_gsm8k(source)
