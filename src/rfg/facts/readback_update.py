"""Merge independently produced ASR observations into one readback record."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rfg.facts.readback_state import completed_text
from rfg.score.textnorm import agreement_norm, wer_norm


def merge_readback_observations(
    readback: Mapping[str, Any],
    updates: Mapping[str, Mapping[str, Any]],
    *,
    reference_condition: str,
    reference_text: str | None,
    self_text: str | None,
    agree_threshold: float,
    intelligible_wer: float,
) -> dict[str, Any]:
    """Return ``readback`` with only the supplied ASR channels replaced.

    Keeping channel updates independent lets a lightweight Whisper-only or
    Seamless-only worker fill spare GPU compute without deleting an existing
    FunASR observation or the companion ASR channel.
    """
    merged = dict(readback)
    versions = dict(merged.get("asr_versions") or {})
    for channel, observation in updates.items():
        merged[channel] = observation.get("text")
        merged[f"{channel}_error"] = observation.get("error")
        merged[f"{channel}_sec"] = observation.get("sec")
        model = observation.get("model")
        if model:
            versions[channel] = model
    merged["asr_versions"] = versions

    asr1_text = completed_text(merged, "asr1")
    asr2_text = completed_text(merged, "asr2")
    agree, agreement_wer = agreement_norm(asr1_text, asr2_text, agree_threshold)
    readback_wer = wer_norm(reference_text, asr1_text)
    readback_wer_vs_self = wer_norm(self_text, asr1_text)
    merged.update(
        {
            "asr_agree": agree,
            "agreement_wer": agreement_wer,
            "reference_condition": reference_condition,
            "reference_text": reference_text,
            "readback_wer": readback_wer,
            "readback_wer_vs_self": readback_wer_vs_self,
            "audio_intelligible": (
                readback_wer_vs_self is not None
                and readback_wer_vs_self < intelligible_wer
            ),
            "intelligible_wer_threshold": intelligible_wer,
        }
    )
    return merged
