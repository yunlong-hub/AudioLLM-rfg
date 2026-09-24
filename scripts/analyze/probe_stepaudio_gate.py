#!/usr/bin/env python3
"""Step-Audio-2-mini 门槛实测：语音条件下内部文本能否稳定取出。

判据（用户指定）：语音输出时能否同时拿到 (a) 内部文本 (b) 音频，且文本正确、音频可懂。
通过 → 该模型可进入我们的五条件框架；不通过 → 不进。
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
REPO = os.path.join(_ROOT, "third_party", "Step-Audio2")
MODEL = "/workspace/yunlong/LLM/AudioLLM-rfg/pretrain_model/Audio/Step-Audio-2-mini"
sys.path.insert(0, REPO)
# torchaudio 2.11 默认走 torchcodec，不支持 BytesIO；用 soundfile 接管写盘
import io as _io
import soundfile as _sf
import torchaudio as _ta

def _sf_save(uri, src, sample_rate=24000, format=None, **kw):
    data = src.detach().cpu().float().numpy()
    if data.ndim == 2:
        data = data.T
    if isinstance(uri, (str, bytes)) or hasattr(uri, "__fspath__"):
        _sf.write(str(uri), data, sample_rate, format="WAV")
    else:
        _sf.write(uri, data, sample_rate, format="WAV")

_ta.save = _sf_save

from stepaudio2 import StepAudio2          # noqa: E402
from token2wav import Token2wav            # noqa: E402

t0 = time.time()
model = StepAudio2(MODEL)
print(f"model loaded {time.time()-t0:.1f}s", flush=True)
t_a = time.time()
t2w = Token2wav(MODEL + "/token2wav")
print(f"token2wav loaded {time.time()-t_a:.1f}s", flush=True)

QS = ["What is seventeen times three?",
      "What is the capital city of Australia?",
      "How many meters are there in two and a half kilometers?"]
os.makedirs("exp/stepaudio_gate", exist_ok=True)
for i, q in enumerate(QS):
    msgs = [{"role": "human", "content": q},
            {"role": "assistant", "content": "<tts_start>", "eot": False}]
    out = model(msgs, max_tokens=2048, temperature=0.7, do_sample=True)
    tokens, text, audio = out
    sp = [x for x in audio if x < 6561]
    wav = f"exp/stepaudio_gate/q{i}.wav"
    data = t2w(sp, os.path.join(REPO, "assets/default_female.wav"))
    with open(wav, "wb") as fh:
        fh.write(data)
    print(f"[{i}] Q: {q[:46]}", flush=True)
    print(f"    内部文本({len(text.split())}词): {text.strip()[:100]!r}", flush=True)
    print(f"    音频tokens={len(sp)} -> {wav}", flush=True)
