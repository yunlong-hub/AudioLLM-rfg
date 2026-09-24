"""Completion semantics for persisted ASR readback channels.

An ASR may successfully return no recognized text.  That empty observation is
complete and must be scored as an empty answer, rather than retried forever or
excluded from coverage.  A channel is incomplete only when it has an error or
has neither a persisted latency nor a string-valued transcript.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def channel_complete(readback: Mapping[str, Any], channel: str) -> bool:
    """Return whether ``channel`` has a successful persisted observation."""
    if readback.get(f"{channel}_error"):
        return False
    return (
        readback.get(f"{channel}_sec") is not None
        or isinstance(readback.get(channel), str)
    )


def channel_current(
    readback: Mapping[str, Any], channel: str, expected_version: str
) -> bool:
    """Return whether a completed channel was produced by ``expected_version``."""
    versions = readback.get("asr_versions") or {}
    return (
        channel_complete(readback, channel)
        and isinstance(versions, Mapping)
        and versions.get(channel) == expected_version
    )


def channel_reusable_for_chunking(
    readback: Mapping[str, Any],
    channel: str,
    *,
    duration_sec: float | int | None,
    chunk_seconds: float | None,
    expected_version: str,
) -> bool:
    """Return whether persisted output is valid for a chunked readback run.

    Short audio is identical under chunked and unchunked execution, so any
    successful observation remains reusable.  Long audio (and records whose
    duration is unknown) must carry the exact chunked recognizer version.
    """
    requires_chunked_version = bool(chunk_seconds) and (
        duration_sec is None or float(duration_sec) > chunk_seconds
    )
    if requires_chunked_version:
        return channel_current(readback, channel, expected_version)
    return channel_complete(readback, channel)


def completed_text(readback: Mapping[str, Any], channel: str) -> str | None:
    """Return normalized completed text, preserving successful empty output."""
    if not channel_complete(readback, channel):
        return None
    value = readback.get(channel)
    return value if isinstance(value, str) else ""
