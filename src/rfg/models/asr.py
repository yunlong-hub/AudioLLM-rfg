"""双 ASR 回读通道（同一 `llm` 环境内）。

* ASR-1: whisper-large-v3（transformers）
* ASR-2: seamless-m4t-v2-large（transformers，异族）

两个通道都返回纯文本；WER 与一致 span 计算放在 `src/score/`，此处只负责转写。
所有转写结果与耗时落盘，便于复算与人工审计。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch

WSR = 16000  # 两个 ASR 都按 16k 输入


def inference_dtype(device: str, requested=None):
    """Choose a CUDA dtype supported by the selected accelerator."""
    if requested is not None:
        return requested
    if str(device).startswith("cuda") and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(torch.device(device))
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


@dataclass
class AsrResult:
    text: str | None
    latency_sec: float | None = None
    model: str | None = None


def load_audio_16k(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != WSR:
        # 线性重采样（避免引入额外依赖；主实验可换 librosa/soxr 并冻结）
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=WSR)
    return wav.astype(np.float32)


def split_audio_for_inference(
    wav: np.ndarray,
    max_seconds: float | None,
    *,
    sample_rate: int = WSR,
) -> list[np.ndarray]:
    """Split long audio at the lowest-energy point before each hard limit."""
    if not max_seconds:
        return [wav]
    max_samples = int(max_seconds * sample_rate)
    if max_samples < sample_rate:
        raise ValueError("max_seconds must be at least 1 second")
    if len(wav) <= max_samples:
        return [wav]

    search_samples = min(int(2 * sample_rate), max_samples // 4)
    window_samples = max(1, int(0.05 * sample_rate))
    stride_samples = max(1, int(0.02 * sample_rate))
    chunks: list[np.ndarray] = []
    start = 0
    while len(wav) - start > max_samples:
        hard_end = start + max_samples
        search_start = hard_end - search_samples
        positions = range(search_start, hard_end, stride_samples)
        cut = min(
            positions,
            key=lambda pos: float(
                np.mean(np.abs(wav[pos : min(pos + window_samples, hard_end)]))
            ),
        )
        cut = max(cut, start + 1)
        chunks.append(wav[start:cut])
        start = cut
    chunks.append(wav[start:])
    return chunks


class WhisperReadback:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype=None,
        *,
        chunk_seconds: float | None = None,
    ) -> None:
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        self.name = "whisper-large-v3"
        dtype = inference_dtype(device, dtype)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype)
            .to(device)
            .eval()
        )
        self.device = device
        self.chunk_seconds = chunk_seconds
        if chunk_seconds:
            self.name += f"-chunk{chunk_seconds:g}s"

    def transcribe(self, wav_path: str, language: str = "en") -> AsrResult:
        wav = load_audio_16k(wav_path)
        t0 = time.time()
        texts = []
        for chunk in split_audio_for_inference(wav, self.chunk_seconds):
            inputs = self.processor(chunk, sampling_rate=WSR, return_tensors="pt")
            inputs = inputs.to(self.device, dtype=self.model.dtype)
            with torch.no_grad():
                ids = self.model.generate(**inputs, language=language, task="transcribe")
            texts.append(self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
        dt = time.time() - t0
        text = " ".join(part for part in texts if part).strip()
        return AsrResult(text=text, latency_sec=dt, model=self.name)


class SeamlessReadback:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype=None,
        *,
        low_memory: bool = False,
        chunk_seconds: float | None = None,
    ) -> None:
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        self.name = "seamless-m4t-v2-large"
        dtype = inference_dtype(device, dtype)
        # Seamless ships a SentencePiece tokenizer. Force the native slow
        # tokenizer so Transformers does not route it through the optional
        # tiktoken/blobfile converter, which is less portable on glibc 2.17.
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        model_kwargs = {"torch_dtype": dtype}
        self.model = (
            SeamlessM4Tv2ForSpeechToText.from_pretrained(
                model_path,
                **model_kwargs,
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.use_cache = not low_memory
        self.chunk_seconds = chunk_seconds
        if chunk_seconds:
            self.name += f"-chunk{chunk_seconds:g}s"

    def transcribe(self, wav_path: str, tgt_lang: str = "eng") -> AsrResult:
        wav = load_audio_16k(wav_path)
        # Transformers 5.x removed the deprecated ``audios`` keyword.  The
        # A22/A23/A31 runtime uses the current ``audio`` processor contract.
        t0 = time.time()
        texts = []
        for chunk in split_audio_for_inference(wav, self.chunk_seconds):
            inputs = self.processor(audio=chunk, sampling_rate=WSR, return_tensors="pt")
            inputs = inputs.to(self.device, dtype=self.model.dtype)
            with torch.no_grad():
                ids = self.model.generate(
                    **inputs,
                    tgt_lang=tgt_lang,
                    use_cache=self.use_cache,
                )
            texts.append(
                self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            )
        dt = time.time() - t0
        text = " ".join(part for part in texts if part).strip()
        return AsrResult(text=text, latency_sec=dt, model=self.name)
