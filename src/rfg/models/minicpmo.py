"""MiniCPM-o-4.5 adapter for the project's text/speech generation contract."""
from __future__ import annotations

import os
import tempfile
import time
import types
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch

from rfg.models.omni import OmniResult

SAMPLE_RATE = 24000


class MiniCPMOModel:
    """Expose MiniCPM-o through the same ``chat`` contract as Qwen Omni."""

    def __init__(
        self,
        model_path: str,
        device_map: str | dict = "auto",
        dtype: str = "float16",
        attn_implementation: str | None = "sdpa",
        verbose: bool = True,
    ) -> None:
        from transformers import AutoModel

        if dtype == "bfloat16" and torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major < 8:
                raise ValueError("MiniCPM-o on pre-Ampere GPUs requires float16, not bfloat16")

        self.model_path = model_path
        self.verbose = verbose
        torch_dtype = getattr(torch, dtype)
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "init_vision": False,
            "init_audio": True,
            "init_tts": True,
            "low_cpu_mem_usage": True,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        # The upstream implementation is validated with a single CUDA device.
        # Keep ``auto`` as the public CLI value, but resolve it explicitly here so
        # Accelerate does not split custom remote-code modules unexpectedly.
        if device_map not in ("auto", "cuda", "cuda:0", None):
            kwargs["device_map"] = device_map
            model = AutoModel.from_pretrained(model_path, **kwargs).eval()
        else:
            model = AutoModel.from_pretrained(model_path, **kwargs).eval().cuda()

        token2wav_dir = Path(model_path) / "assets" / "token2wav"
        if not token2wav_dir.is_dir():
            raise FileNotFoundError(f"MiniCPM token2wav assets missing: {token2wav_dir}")
        # Token2wav 1.0.6 only casts the flow network when float16 is enabled;
        # its HiFT vocoder stays in float32, which yields a half/float Conv1d
        # mismatch. Keep the language/audio model in FP16, but run Token2wav in
        # its supported FP32 mode.
        model.init_tts(model_dir=str(token2wav_dir), enable_float16=False)
        self._install_tts_alignment_guard(model)
        self.model = model

        ref_path = Path(model_path) / "assets" / "HT_ref_audio.wav"
        if not ref_path.is_file():
            raise FileNotFoundError(f"MiniCPM reference voice missing: {ref_path}")
        ref_audio, _ = librosa.load(ref_path, sr=16000, mono=True)
        self.system_message = model.get_sys_prompt(ref_audio=ref_audio, mode="omni", language="en")
        self.arch = "MiniCPMO"
        if verbose:
            print(
                f"[MiniCPMOModel] dtype={dtype} attention={attn_implementation} "
                f"reference_voice={ref_path.name}",
                flush=True,
            )

    @staticmethod
    def _install_tts_alignment_guard(model) -> None:
        """Render capped generations using the token/hidden-state shared prefix."""
        original = model._generate_speech_non_streaming

        def aligned(self, outputs, tts_bound, tts_proj_layer, audio_prompt,
                    output_tts_inputs_embeds_path=None, tts_sampling_params=None):
            start, end = tts_bound
            token_count = outputs["full_sequences"][0][start:end].shape[0]
            states = [state[tts_proj_layer] for state in outputs.hidden_states]
            hidden_count = torch.vstack([state[0] for state in states])[start:end].shape[0]
            if token_count != hidden_count:
                shared = min(token_count, hidden_count)
                end = start + shared
                print("[MiniCPMOModel] TTS boundary aligned "
                      f"tokens={token_count} hidden={hidden_count} shared={shared}",
                      flush=True)
            kwargs = {
                "outputs": outputs,
                "tts_bound": (start, end),
                "tts_proj_layer": tts_proj_layer,
                "audio_prompt": audio_prompt,
                "output_tts_inputs_embeds_path": output_tts_inputs_embeds_path,
            }
            if tts_sampling_params is not None:
                kwargs["tts_sampling_params"] = tts_sampling_params
            return original(**kwargs)

        model._generate_speech_non_streaming = types.MethodType(aligned, model)

    @staticmethod
    def _content_to_native(content: list[dict]) -> list[str | np.ndarray]:
        native: list[str | np.ndarray] = []
        for part in content:
            kind = part.get("type")
            if kind == "text":
                native.append(part["text"])
            elif kind == "audio":
                audio, _ = librosa.load(part["audio"], sr=16000, mono=True)
                native.append(audio)
            else:
                raise ValueError(f"MiniCPM-o unsupported content type: {kind}")
        return native

    def chat(
        self,
        content: list[dict],
        want_audio: bool = False,
        max_new_tokens: int = 512,
        talker_max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = 0,
        speaker: str | None = None,
        extra_gen_kwargs: dict | None = None,
    ) -> OmniResult:
        del talker_max_new_tokens, speaker
        if extra_gen_kwargs:
            raise ValueError(f"MiniCPM-o does not support Qwen talker kwargs: {sorted(extra_gen_kwargs)}")
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        user_message = {"role": "user", "content": self._content_to_native(content)}
        messages = [self.system_message, user_message]
        fd, output_path = tempfile.mkstemp(prefix="minicpmo45-", suffix=".wav")
        os.close(fd)
        os.unlink(output_path)
        generation_kwargs: dict[str, Any] = {
            "msgs": messages,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "enable_thinking": False,
            "use_tts_template": True,
            "generate_audio": want_audio,
            "output_audio_path": output_path if want_audio else None,
            "omni_mode": True,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            if top_p is not None:
                generation_kwargs["top_p"] = top_p

        started = time.time()
        try:
            text = self.model.chat(**generation_kwargs)
            latency = time.time() - started
            audio = None
            if want_audio:
                if not os.path.isfile(output_path) or os.path.getsize(output_path) <= 44:
                    raise RuntimeError("MiniCPM-o TTS returned no valid WAV output")
                audio, sample_rate = sf.read(output_path, dtype="float32", always_2d=False)
                audio = np.asarray(audio)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sample_rate != SAMPLE_RATE:
                    raise RuntimeError(f"MiniCPM-o emitted {sample_rate}Hz audio, expected {SAMPLE_RATE}Hz")
                if audio.size == 0 or not np.isfinite(audio).all():
                    raise RuntimeError("MiniCPM-o emitted empty or non-finite audio")
            return OmniResult(
                text=text.strip() if isinstance(text, str) else text,
                audio=audio,
                sample_rate=SAMPLE_RATE,
                latency_sec=latency,
                output_modality="speech" if audio is not None else "text",
                meta={
                    "arch": self.arch,
                    "raw_text": text,
                    "reference_voice": "HT_ref_audio.wav",
                    "seed": seed,
                    "tts_alignment_guard": "shared_prefix",
                },
            )
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @staticmethod
    def save_wav(path: str, result: OmniResult) -> str:
        if result.audio is None or result.audio.size == 0:
            raise ValueError(f"empty audio, refusing to write: {path}")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        sf.write(path, result.audio, result.sample_rate, subtype="PCM_16")
        return path
