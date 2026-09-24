"""Qwen2.5-Omni-7B 复核：语音输出是否真的损坏？"""
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
import numpy as np, soundfile as sf, torch
from rfg.models.omni import OmniModel
from rfg.models.asr import WhisperReadback
from rfg.score.textnorm import wer_norm
from rfg.run.conditions import INSTRUCTION

D = _ROOT + "/pretrain_model/Qwen/Qwen2.5-Omni-7B"
OUT = "exp/probe_ef/smoke7b"; os.makedirs(OUT, exist_ok=True)
QS = ["What is seventeen times three?",
      "What is the capital city of Australia?",
      "How many meters are there in two and a half kilometers?"]
print("loading...", flush=True); t0=time.time()
m = OmniModel(D, dtype="float16")
print(f"loaded {time.time()-t0:.1f}s arch={m.arch}", flush=True)
asr = WhisperReadback("/workspace/yunlong/LLM/pretrain_model/Audio/whisper-large-v3", device="cuda:0")
res = []
for i, q in enumerate(QS):
    r = m.chat([{"type": "text", "text": f"{q} {INSTRUCTION}"}], want_audio=True)
    wav = f"{OUT}/q{i}.wav"
    OmniModel.save_wav(wav, r)
    w, sr = sf.read(wav, dtype="float32"); w = w.mean(1) if w.ndim>1 else w
    txt = r.text or ""
    rb = asr.transcribe(wav).text
    row = {"q": q, "text": txt, "dur": round(len(w)/sr,2), "peak": round(float(np.abs(w).max()),3),
           "rms": round(float(np.sqrt((w**2).mean())),4), "readback": rb,
           "wer_self": wer_norm(txt, rb)}
    res.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
json.dump(res, open(f"{OUT}/result.json","w"), ensure_ascii=False, indent=2)
