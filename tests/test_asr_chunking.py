import numpy as np
import pytest
import torch

from rfg.models import asr
from rfg.models.asr import SeamlessReadback, split_audio_for_inference


def test_short_audio_is_not_split():
    wav = np.ones(100, dtype=np.float32)
    chunks = split_audio_for_inference(wav, 2.0, sample_rate=100)
    assert len(chunks) == 1
    assert chunks[0] is wav


def test_long_audio_is_losslessly_split_under_limit():
    wav = np.ones(650, dtype=np.float32)
    wav[180:190] = 0
    wav[380:390] = 0
    chunks = split_audio_for_inference(wav, 2.0, sample_rate=100)
    assert len(chunks) > 1
    assert max(map(len, chunks)) <= 200
    np.testing.assert_array_equal(np.concatenate(chunks), wav)


def test_chunk_limit_must_be_at_least_one_second():
    with pytest.raises(ValueError):
        split_audio_for_inference(np.ones(200), 0.5, sample_rate=100)


def test_seamless_processor_uses_current_audio_keyword(monkeypatch):
    seen = {}

    class Inputs(dict):
        def to(self, *_args, **_kwargs):
            return self

    class Processor:
        def __call__(self, **kwargs):
            seen.update(kwargs)
            return Inputs(input_features=torch.zeros(1, 1))

        def batch_decode(self, *_args, **_kwargs):
            return ["transcript"]

    class Model:
        dtype = torch.float32

        def generate(self, **_kwargs):
            return torch.ones((1, 1), dtype=torch.long)

    monkeypatch.setattr(asr, "load_audio_16k", lambda _path: np.ones(16000))
    reader = SeamlessReadback.__new__(SeamlessReadback)
    reader.name = "seamless-m4t-v2-large-chunk25s"
    reader.processor = Processor()
    reader.model = Model()
    reader.device = "cpu"
    reader.use_cache = False
    reader.chunk_seconds = 25

    result = reader.transcribe("sample.wav")

    assert result.text == "transcript"
    assert "audio" in seen
    assert "audios" not in seen
