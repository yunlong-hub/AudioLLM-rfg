"""Fun-ASR-Nano readback adapter.

This module is imported only by the dedicated FunASR entrypoint.  Keeping the
adapter separate prevents the main ``audio-llm`` runtime from acquiring a hard
dependency on FunASR.
"""
from __future__ import annotations

import time

import torch

from rfg.models.asr import AsrResult, load_audio_16k, split_audio_for_inference


def funasr_llm_dtype(device: str) -> str:
    """Select the FunASR language-model dtype supported by the device."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(torch.device(device))
        return "bf16" if major >= 8 else "fp16"
    return "fp32"


class FunAsrReadback:
    """Transcribe English model speech with Fun-ASR-Nano-2512."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        *,
        chunk_seconds: float | None = None,
    ) -> None:
        from funasr.models.fun_asr_nano.model import FunASRNano

        self.name = self.version_name(chunk_seconds)
        llm_dtype = funasr_llm_dtype(device)
        self.model, self.kwargs = FunASRNano.from_pretrained(
            model=model_path,
            device=device,
            llm_conf={"llm_dtype": llm_dtype},
        )
        self.model.eval()
        self.chunk_seconds = chunk_seconds

    @staticmethod
    def version_name(chunk_seconds: float | None = None) -> str:
        name = "Fun-ASR-Nano-2512"
        if chunk_seconds:
            name += f"-chunk{chunk_seconds:g}s"
        return name

    def transcribe(self, wav_path: str) -> AsrResult:
        t0 = time.time()
        inputs: list[str | torch.Tensor]
        if self.chunk_seconds:
            wav = load_audio_16k(wav_path)
            inputs = [
                torch.from_numpy(chunk)
                for chunk in split_audio_for_inference(wav, self.chunk_seconds)
            ]
        else:
            inputs = [wav_path]

        texts = []
        for input_audio in inputs:
            result = self.model.inference(
                data_in=[input_audio], language="英文", itn=True, **self.kwargs
            )
            texts.append(result[0][0].get("text", "").strip())
        text = " ".join(part for part in texts if part).strip()
        return AsrResult(
            text=text,
            latency_sec=time.time() - t0,
            model=self.name,
        )
