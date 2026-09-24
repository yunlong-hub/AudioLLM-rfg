"""门槛测试：Step-Audio-2-mini 的内部文本能否稳定取出。"""
import os, sys, time, torch
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
D = _ROOT + "/pretrain_model/Audio/Step-Audio-2-mini"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
from transformers import AutoModelForCausalLM, AutoTokenizer
t0 = time.time()
tok = AutoTokenizer.from_pretrained(D, trust_remote_code=True)
print("tokenizer ok", round(time.time()-t0,1), "s", flush=True)
m = AutoModelForCausalLM.from_pretrained(D, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
print("model ok", round(time.time()-t0,1), "s | arch:", type(m).__name__, flush=True)
for q in ["What is seventeen times three? Answer in one short sentence.",
          "How many meters are there in two and a half kilometers? Answer in one short sentence."]:
    msgs = [{"role": "user", "content": q}]
    try:
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception as e:
        prompt = q
        print("  (chat_template 失败，改用裸文本)", e)
    ids = tok(prompt, return_tensors="pt").to(m.device)
    n_in = ids["input_ids"].shape[1]
    out = m.generate(**ids, do_sample=False, max_new_tokens=64)
    txt = tok.decode(out[0][n_in:], skip_special_tokens=True)
    print(f"  Q: {q[:50]}\n    A: {txt.strip()[:90]!r}", flush=True)
print("=== 内部文本可取出性：见上（逐题稳定输出即为通过）===")
