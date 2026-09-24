from rfg.facts.readback_state import (
    channel_complete,
    channel_current,
    channel_reusable_for_chunking,
    completed_text,
)


def test_successful_empty_channel_is_complete() -> None:
    readback = {"asr3": None, "asr3_sec": 0.25, "asr3_error": None}

    assert channel_complete(readback, "asr3")
    assert completed_text(readback, "asr3") == ""


def test_missing_or_failed_channel_is_incomplete() -> None:
    assert not channel_complete({}, "asr3")
    assert completed_text({}, "asr3") is None
    failed = {"asr3": None, "asr3_sec": 0.25, "asr3_error": "oom"}
    assert not channel_complete(failed, "asr3")
    assert completed_text(failed, "asr3") is None


def test_current_channel_requires_matching_persisted_version() -> None:
    readback = {
        "asr1": "answer",
        "asr1_sec": 0.25,
        "asr1_error": None,
        "asr_versions": {"asr1": "whisper-large-v3-chunk25s"},
    }

    assert channel_current(readback, "asr1", "whisper-large-v3-chunk25s")
    assert not channel_current(readback, "asr1", "whisper-large-v3")
    assert not channel_current({"asr1": "answer"}, "asr1", "whisper-large-v3")


def test_chunked_reuse_requires_version_only_for_long_audio() -> None:
    stale = {"asr1": "answer", "asr1_error": None}
    expected = "whisper-large-v3-chunk25s"

    assert channel_reusable_for_chunking(
        stale,
        "asr1",
        duration_sec=10.0,
        chunk_seconds=25.0,
        expected_version=expected,
    )
    assert not channel_reusable_for_chunking(
        stale,
        "asr1",
        duration_sec=30.0,
        chunk_seconds=25.0,
        expected_version=expected,
    )
    current = {**stale, "asr_versions": {"asr1": expected}}
    assert channel_reusable_for_chunking(
        current,
        "asr1",
        duration_sec=30.0,
        chunk_seconds=25.0,
        expected_version=expected,
    )
