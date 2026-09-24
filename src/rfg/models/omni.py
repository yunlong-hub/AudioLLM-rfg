"""Qwen3-Omni（及同族 Qwen2.5-Omni）封装：同一接口支持 文本/语音 输入与 文本/语音 输出。

设计要点
--------
* 语音输出必须**显式断言**音频非空且时长 > 0，防止静默回退到纯文本（RFC: 工程规约 §6.4）。
* 不假设 generate() 的返回签名：兼容 `(text_ids, audio)` 元组与带 `.audio` 的对象。
* 每次调用记录 latency / output_modality / audio_duration，供 freeze 与效率指标使用。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 24000  # Qwen Omni code2wav 输出采样率

# arch -> generate 是否接受 **kwargs（透传参数时用于校验，避免静默忽略）
_HAS_VAR_KW: dict[str, bool] = {}

# Qwen Omni 的 generate 返回序列包含 prompt（chat template 文本），直接解码会把
# "user ... assistant" 回显混进答案。按最后一个 assistant 标记截断，只留生成内容。
_ASSISTANT_MARKERS = ("<|im_start|>assistant\n", "assistant\n", "assistant")


def strip_prompt_echo(text: str | None) -> str | None:
    """从解码文本中剥掉 prompt 回显，只留模型真正生成的答案。"""
    if not text:
        return text
    for marker in _ASSISTANT_MARKERS:
        if marker in text:
            return text.rsplit(marker, 1)[-1].strip()
    return text.strip()


@dataclass
class OmniResult:
    text: str | None = None
    audio: np.ndarray | None = None
    sample_rate: int = SAMPLE_RATE
    latency_sec: float | None = None
    output_modality: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def duration_sec(self) -> float | None:
        if self.audio is None:
            return None
        return len(self.audio) / float(self.sample_rate)


def _family(arch_name: str | None) -> str | None:
    """把 architectures 类名归一到"家族"：剥掉 Model / ForConditionalGeneration / Thinker 等后缀。

    Qwen2_5OmniModel 与 Qwen2_5OmniForConditionalGeneration 归一后相同，属同一家族；
    Qwen3OmniMoe* 与 Qwen2_5Omni* 归一后不同，属不同家族（跨家族加载必须拒绝）。
    """
    if not arch_name:
        return None
    name = arch_name
    for suffix in ("ForConditionalGeneration", "ThinkerForConditionalGeneration", "Model"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


class OmniModel:
    """薄封装。所有条件共用同一模型实例与同一份解码配置。"""

    def __init__(
        self,
        model_path: str,
        device_map: str | dict = "auto",
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        verbose: bool = True,
    ) -> None:
        from transformers import AutoProcessor, AutoTokenizer

        self.model_path = model_path
        self.verbose = verbose
        torch_dtype = getattr(torch, dtype)

        kwargs: dict[str, Any] = {"torch_dtype": torch_dtype, "device_map": device_map}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        # --- 模型类：必须按 checkpoint 的 architectures 字段选，不能按"类是否存在"试错 ---
        # （按可用性顺序硬试会把 Qwen2.5-Omni 用 Qwen3OmniMoe 类加载，报 shared_expert_intermediate_size 缺失）
        import json as _json

        import transformers as T

        cfg_path = os.path.join(model_path, "config.json")
        arch_name = None
        if os.path.isfile(cfg_path):
            with open(cfg_path) as fh:
                arch_name = (_json.load(fh).get("architectures") or [None])[0]

        fallback_order = ("Qwen3OmniMoeForConditionalGeneration",
                          "Qwen2_5OmniForConditionalGeneration",
                          "Qwen2_5OmniThinkerForConditionalGeneration")
        candidates = ([arch_name] if arch_name else []) + \
                     [n for n in fallback_order if n != arch_name]

        self.arch = None
        last_err: Exception | None = None
        for name in candidates:
            cls = getattr(T, name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(model_path, **kwargs)
                self.arch = name
                break
            except Exception as exc:  # 记录后尝试下一个候选
                last_err = exc
        if self.arch is None:
            raise RuntimeError(f"无法加载 {model_path}（architectures={arch_name}）：{last_err}")
        # 只有**跨家族**误用才拒绝：HF checkpoint 常把 architectures 写成基类名
        # （如 Qwen2.5-Omni-3B 写 Qwen2_5OmniModel，实际应加载 Qwen2_5OmniForConditionalGeneration）
        if arch_name and _family(arch_name) != _family(self.arch):
            raise RuntimeError(
                f"checkpoint 声明的 architectures={arch_name} 与加载类 {self.arch} 属不同家族，拒绝继续")
        if arch_name and arch_name != self.arch and verbose:
            print(f"[OmniModel] architectures={arch_name} -> 实际加载 {self.arch}（同家族）", flush=True)

        # --- processor ---
        proc = None
        for name in ("Qwen3OmniMoeProcessor", "Qwen2_5OmniProcessor"):
            cls = getattr(T, name, None)
            if cls is not None:
                proc = cls.from_pretrained(model_path)
                break
        if proc is None:
            proc = AutoProcessor.from_pretrained(model_path)
        self.processor = proc
        self.tokenizer = getattr(proc, "tokenizer", None) or AutoTokenizer.from_pretrained(model_path)
        self.model = model.eval()

        # 记录可用的 generate 参数（写进冻结清单）
        import inspect

        self.generate_params = sorted(inspect.signature(model.generate).parameters)
        _sig = inspect.signature(model.generate)
        _has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in _sig.parameters.values())
        # 音频输出相关的可用参数。注意 transformers v5 把 return_audio 改名 generation_mode：
        # Qwen3 仍用显式 return_audio；Qwen2.5 无该形参但经 **kwargs 兼容。
        # 判据必须是"显式存在 或 接受 **kwargs"，否则文本条件会静默地联合生成语音。
        self.audio_kwargs_supported = {
            k: (k in self.generate_params)
            for k in ("return_audio", "speaker", "talker_max_new_tokens",
                      "thinker_max_new_tokens", "use_audio_in_video",
                      "talker_repetition_penalty")
        }
        self.accepts_return_audio = ("return_audio" in self.generate_params) or _has_var_kw
        _HAS_VAR_KW[self.arch] = _has_var_kw
        if verbose:
            print(f"[OmniModel] arch={self.arch} audio kwargs={self.audio_kwargs_supported} "
                  f"accepts_return_audio={self.accepts_return_audio}", flush=True)

    # ------------------------------------------------------------------ utils
    def _build_inputs(self, conversation: list[dict]):
        from qwen_omni_utils import process_mm_info

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True,
        )
        return inputs.to(self.model.device).to(self.model.dtype if hasattr(self.model, "dtype") else torch.float32)

    @staticmethod
    def _split_output(out):
        """兼容三种返回形态：
        * `(text_ids, audio)` —— return_audio=True 时的元组
        * **裸 tensor** —— return_audio=False 时 Qwen3-Omni 直接返回文本 id（曾导致文本丢失）
        * 带 `.sequences` / `.audio` 的生成输出对象
        """
        audio = None
        if isinstance(out, (tuple, list)):
            text_ids = out[0] if len(out) > 0 else None
            audio = out[1] if len(out) > 1 else None
            return text_ids, audio
        if torch.is_tensor(out):
            return out, None
        text_ids = getattr(out, "sequences", None)
        audio = getattr(out, "audio", None)
        if audio is None:
            for attr in ("waveform", "speech"):
                audio = getattr(out, attr, None)
                if audio is not None:
                    break
        return text_ids, audio

    # --------------------------------------------------------------- generate
    def chat(
        self,
        content: list[dict],
        want_audio: bool = False,
        max_new_tokens: int = 512,
        talker_max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
        speaker: str | None = None,
        extra_gen_kwargs: dict | None = None,
    ) -> OmniResult:
        """content: [{'type':'text','text':...}, {'type':'audio','audio':path}, ...]"""
        if seed is not None:
            torch.manual_seed(seed)
        conversation = [{"role": "user", "content": content}]
        inputs = self._build_inputs(conversation)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "use_audio_in_video": False,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
        if want_audio:
            if self.audio_kwargs_supported.get("talker_max_new_tokens"):
                gen_kwargs["talker_max_new_tokens"] = talker_max_new_tokens
            if speaker and self.audio_kwargs_supported.get("speaker"):
                gen_kwargs["speaker"] = speaker
        if self.accepts_return_audio:
            # 文本条件必须显式关闭音频生成：否则每次白花数秒~数十秒，且 output_modality 判定失真
            gen_kwargs["return_audio"] = want_audio
        if extra_gen_kwargs:
            # 消融用透传（talker_* 等）；只允许模型 generate 接受的键，避免静默忽略
            unknown = [k for k in extra_gen_kwargs
                       if k not in self.generate_params and not _HAS_VAR_KW[self.arch]]
            if unknown:
                raise ValueError(f"generate() 不接受这些参数: {unknown}")
            gen_kwargs.update(extra_gen_kwargs)

        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        dt = time.time() - t0

        text_ids, audio = self._split_output(out)

        res = OmniResult(latency_sec=dt)
        raw_text = None
        if text_ids is not None:
            try:
                raw_text = self.processor.batch_decode(
                    text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            except Exception:
                raw_text = str(text_ids)
            res.text = strip_prompt_echo(raw_text)

        if audio is not None:
            arr = audio
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().float().numpy()
            arr = np.asarray(arr).reshape(-1)
            res.audio = arr

        res.output_modality = "speech" if (res.audio is not None and res.audio.size > 0) else "text"
        res.meta = {
            "requested_audio": want_audio,
            "gen_kwargs": {k: v for k, v in gen_kwargs.items() if k != "return_audio"},
            "arch": self.arch,
            "raw_text": raw_text,
            # 诊断用：return_audio 开关会改变 generate 的返回形态，必须留痕
            "output_type": type(out).__name__,
            "text_ids_shape": (tuple(text_ids.shape) if torch.is_tensor(text_ids) else None),
        }
        return res

    # ------------------------------------------------------------------ io
    @staticmethod
    def save_wav(path: str, res: OmniResult) -> str:
        assert res.audio is not None and res.audio.size > 0, f"空音频，拒绝落盘: {path}"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        sf.write(path, res.audio, res.sample_rate, subtype="PCM_16")
        return path

    def ask_text(self, question: str, **kw) -> OmniResult:
        return self.chat([{"type": "text", "text": question}], want_audio=False, **kw)

    def ask_audio_text(self, audio_path: str, instruction: str = "", **kw) -> OmniResult:
        content = [{"type": "audio", "audio": audio_path}]
        if instruction:
            content.append({"type": "text", "text": instruction})
        return self.chat(content, want_audio=False, **kw)

    def ask_audio_speech(self, audio_path: str, instruction: str = "", **kw) -> OmniResult:
        content = [{"type": "audio", "audio": audio_path}]
        if instruction:
            content.append({"type": "text", "text": instruction})
        return self.chat(content, want_audio=True, **kw)

    def ask_text_speech(self, question: str, **kw) -> OmniResult:
        return self.chat([{"type": "text", "text": question}], want_audio=True, **kw)

    def ask_forced_readback(self, answer_text: str, **kw) -> OmniResult:
        """EF 条件：内容已给定，只要求逐字朗读（纯渲染）。"""
        prompt = (
            "Read the following sentence aloud exactly as written, word for word. "
            "Do not add, remove, translate, or rephrase anything.\n\n"
            f"Sentence: {answer_text}"
        )
        return self.chat([{"type": "text", "text": prompt}], want_audio=True, **kw)
