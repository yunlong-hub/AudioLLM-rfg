import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
import numpy as np, soundfile as sf
from rfg.models.omni import OmniModel
from rfg.models.asr import WhisperReadback
from rfg.score.textnorm import wer_norm
D=_ROOT + "/pretrain_model/Qwen/Qwen2.5-Omni-7B"
OUT="exp/probe_ef/smoke7b"; os.makedirs(OUT, exist_ok=True)
Q="What is seventeen times three? Answer in one short sentence."
CFG=[("default",{},{}),
     ("greedy_talker",{"extra_gen_kwargs":{"talker_do_sample":False,"talker_temperature":0.0}},{}),
     ("rep1.3_cap256",{"extra_gen_kwargs":{"talker_repetition_penalty":1.3,"talker_max_new_tokens":256}},{}),
     ("speaker_Chelsie",{},{ "speaker":"Chelsie"}),
     ("speaker_Ethan",{},{"speaker":"Ethan"})]
m=OmniModel(D)
a=WhisperReadback("/workspace/yunlong/LLM/pretrain_model/Audio/whisper-large-v3", device="cuda:0")
for name,kw,skw in CFG:
    try:
        r=m.chat([{"type":"text","text":Q}], want_audio=True, **kw, **skw)
        wav=f"{OUT}/k_{name}.wav"; OmniModel.save_wav(wav,r)
        w,sr=sf.read(wav,dtype="float32"); w=w.mean(1) if w.ndim>1 else w
        rb=a.transcribe(wav).text
        print(json.dumps({"cfg":name,"text":(r.text or "")[:44],"dur":round(len(w)/sr,2),
                          "peak":round(float(np.abs(w).max()),3),"readback":rb[:44],
                          "wer":wer_norm(r.text,rb)},ensure_ascii=False), flush=True)
    except Exception as e:
        print(json.dumps({"cfg":name,"error":f"{type(e).__name__}: {e}"[:90]},ensure_ascii=False), flush=True)
