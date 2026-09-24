from rfg.facts.readback_update import merge_readback_observations


def test_independent_channel_updates_preserve_companion_asrs():
    initial = {
        "asr3": "answer is twelve",
        "asr3_sec": 1.0,
        "asr3_error": None,
        "asr_versions": {"asr3": "funasr"},
    }
    with_asr1 = merge_readback_observations(
        initial,
        {
            "asr1": {
                "text": "answer is 12",
                "error": None,
                "sec": 2.0,
                "model": "whisper",
            }
        },
        reference_condition="LISTEN",
        reference_text="answer is 12",
        self_text="answer is 12",
        agree_threshold=0.2,
        intelligible_wer=0.5,
    )
    assert with_asr1["asr3"] == "answer is twelve"
    assert with_asr1["asr_versions"] == {"asr3": "funasr", "asr1": "whisper"}
    assert not with_asr1["asr_agree"]

    complete = merge_readback_observations(
        with_asr1,
        {
            "asr2": {
                "text": "answer is twelve",
                "error": None,
                "sec": 3.0,
                "model": "seamless",
            }
        },
        reference_condition="LISTEN",
        reference_text="answer is 12",
        self_text="answer is 12",
        agree_threshold=0.2,
        intelligible_wer=0.5,
    )
    assert complete["asr1"] == "answer is 12"
    assert complete["asr2"] == "answer is twelve"
    assert complete["asr3"] == "answer is twelve"
    assert complete["asr_agree"]
    assert complete["audio_intelligible"]
