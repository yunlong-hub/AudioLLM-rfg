import json

import pytest

from rfg.run.resample_generation import candidate_complete, parse_seeds


def test_parse_seeds_requires_enough_fixed_seeds():
    assert parse_seeds("101,202,303", 3) == [101, 202, 303]
    with pytest.raises(ValueError, match="need 4 seeds"):
        parse_seeds("101,202,303", 4)


def test_candidate_complete_requires_json_and_nonempty_wav(tmp_path):
    record = tmp_path / "R0.json"
    wav = tmp_path / "R0.wav"
    record.write_text(json.dumps({"item_id": "x", "seed": 101}))
    wav.write_bytes(b"0" * 45)
    assert candidate_complete(record, wav)
    record.write_text(json.dumps({"item_id": "x", "error": "retry"}))
    assert not candidate_complete(record, wav)
