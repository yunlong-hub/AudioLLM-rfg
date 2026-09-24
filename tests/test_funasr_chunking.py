from __future__ import annotations

import numpy as np
import torch

from rfg.models.funasr import FunAsrReadback, funasr_llm_dtype


class FakeFunAsrModel:
    def __init__(self) -> None:
        self.inputs: list[object] = []

    def inference(self, *, data_in, **kwargs):
        self.inputs.extend(data_in)
        return [[{"text": f"chunk{len(self.inputs)}"}]]


def test_chunked_funasr_uses_tensor_chunks_and_joins_text(monkeypatch) -> None:
    recognizer = FunAsrReadback.__new__(FunAsrReadback)
    recognizer.name = FunAsrReadback.version_name(2.0)
    recognizer.model = FakeFunAsrModel()
    recognizer.kwargs = {}
    recognizer.chunk_seconds = 2.0
    wav = np.ones(450, dtype=np.float32)
    monkeypatch.setattr("rfg.models.funasr.load_audio_16k", lambda _: wav)
    monkeypatch.setattr(
        "rfg.models.funasr.split_audio_for_inference",
        lambda audio, seconds: [audio[:200], audio[200:400], audio[400:]],
    )

    result = recognizer.transcribe("long.wav")

    assert result.text == "chunk1 chunk2 chunk3"
    assert result.model == "Fun-ASR-Nano-2512-chunk2s"
    assert len(recognizer.model.inputs) == 3
    assert all(isinstance(value, torch.Tensor) for value in recognizer.model.inputs)


def test_funasr_uses_fp16_on_volta(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (7, 0))

    assert funasr_llm_dtype("cuda:0") == "fp16"


def test_funasr_uses_bf16_on_ampere(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 0))

    assert funasr_llm_dtype("cuda:0") == "bf16"
