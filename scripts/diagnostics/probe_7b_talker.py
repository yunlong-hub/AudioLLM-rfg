"""7B 渲染修复探针：talker 采样配置 × 是否可懂。

目的：Qwen2.5-Omni-7B 的语音输出损坏（回读一致率 0.4%、削顶、噪音）。
若换 talker 采样配置即可恢复，则该模型能作为 2×2 机制的第三个数据点。
"""
import json, os, sys
import numpy as np, soundfile as sf
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from rfg.models.omni import OmniModel
from rfg.models.asr import WhisperReadback
from rfg.score.textnorm import wer_norm

P = "/workspace/yunlong/LLM/pretrain_model"
OUT = _ROOT + "/exp/probe_ef/audio7b"
os.makedirs(OUT, exist_ok=True)
SENT = "The main source of energy for life on Earth is the Sun."
CONFIGS = [
    ("default", {}),
    ("rep1.3", {"talker_repetition_penalty": 1.3}),
    ("temp0.6", {"talker_temperature": 0.6, "talker_top_p": 0.7}),
    ("temp1.0_k60", {"talker_temperature": 1.0, "talker_top_k": 60, "talker_top_p": 0.9}),
    ("cap256_rep1.3", {"talker_repetition_penalty": 1.3, "talker_max_new_tokens": 256}),
]
m = OmniModel(f"{P}/Qwen/Qwen2.5-Omni-7B")
asr = WhisperReadback(f"{P}/ST/whisper-large-v3", device="cuda:0")
res = []
for name, kw in CONFIGS:
    r = m.chat([{"type": "text", "text": f"Read this sentence aloud: {SENT}"}],
               want_audio=True, extra_gen_kwargs=kw)
    wav = f"{OUT}/{name}.wav"
    OmniModel.save_wav(wav, r)
    w, sr = sf.read(wav, dtype="float32")
    rb = asr.transcribe(wav).text
    row = {"config": name, "kwargs": kw, "dur": round(len(w)/sr, 2),
           "peak": round(float(np.abs(w).max()), 3), "readback": rb,
           "wer": wer_norm(SENT, rb)}
    res.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
json.dump(res, open(_ROOT + "/exp/probe_ef/metrics/probe_7b_talker.json", "w"),
          indent=2, ensure_ascii=False)
