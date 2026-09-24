#!/usr/bin/env python3
"""Step-Audio-2-mini 五条件推理（跨家族复现）。

与 Qwen 路径的差异
------------------
* 驱动代码来自 `third_party/Step-Audio2`（官方仓库），环境为 conda env `stepaudio`
  （transformers 4.49.0，与主环境 5.15.1 冲突，故单列）。
* 语音输出：在对话里插入 `{"role":"assistant","content":"<tts_start>","eot":False}`，
  模型返回 `(tokens, text, audio)` —— **text 即内部文本**，audio 为语音 token，
  经 `Token2wav` 渲染为 24kHz wav。
* torchaudio 2.11 走 torchcodec 不支持 BytesIO，已用 soundfile 接管保存。

产物布局与 `scripts/infer.py` 完全一致，便于复用全部下游分析：
  exp/<run_id>/predictions/<model_slug>/<item_id>/<COND>.{json,wav}

用法：
  CUDA_VISIBLE_DEVICES=3 python scripts/infer_stepaudio.py --run-id d2_stepaudio --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "pyproject.toml")):
    parent = os.path.dirname(_ROOT)
    if parent == _ROOT:
        raise RuntimeError("无法定位项目根目录（缺少 pyproject.toml）")
    _ROOT = parent
REPO = os.path.join(_ROOT, "third_party", "Step-Audio2")
MODEL = os.path.join(_ROOT, "pretrain_model", "Audio", "Step-Audio-2-mini")
sys.path.insert(0, REPO)
sys.path.insert(0, _ROOT)

# ---- 用 soundfile 接管读写（避免 torchaudio 2.x 强制走 torchcodec）
import soundfile as _sf  # noqa: E402
import torch as _torch  # noqa: E402
import torchaudio as _ta  # noqa: E402


def _sf_load(uri, *args, **kwargs):
    """Return a channels-first tensor without importing the torchcodec backend."""
    del args, kwargs
    data, sample_rate = _sf.read(uri, dtype="float32", always_2d=True)
    return _torch.from_numpy(data.T.copy()), sample_rate


def _sf_save(uri, src, sample_rate=24000, format=None, **kw):
    data = src.detach().cpu().float().numpy()
    if data.ndim == 2:
        data = data.T
    _sf.write(uri, data, sample_rate, format="WAV")


_ta.load = _sf_load
_ta.save = _sf_save

from stepaudio2 import StepAudio2  # noqa: E402
from token2wav import Token2wav  # noqa: E402
from rfg.run.conditions import ef_prompt, item_instruction, speakable_format  # noqa: E402

PROMPT_WAV = os.path.join(REPO, "assets", "default_female.wav")
import re as _re

_SPECIAL = _re.compile(r"<\|[^|]*\|>")          # <|BOT|> / <|EOT|> 等特殊标记
_LANGTAG = _re.compile(r"<[^|>]{0,8}>")           # <中文> 之类语言标记


def clean_text(t: str | None) -> str:
    """清洗官方实现输出的特殊标记，得到可与 Qwen 路径对齐的内部文本。"""
    if not t:
        return ""
    t = _SPECIAL.sub(" ", t)
    t = _LANGTAG.sub(" ", t)
    return " ".join(t.split()).strip()
CONDS = ("READ", "LISTEN", "SPEAK", "ECHO", "EF", "EFW")


def call(model, messages, want_audio: bool):
    """统一调用：want_audio=True 时插入 <tts_start> 并取回 (text, audio_tokens)。"""
    msgs = list(messages)
    if want_audio:
        msgs = msgs + [{"role": "assistant", "content": "<tts_start>", "eot": False}]
    out = model(msgs, max_tokens=2048, temperature=0.7, do_sample=True)
    tokens, text, audio = out if len(out) == 3 else (out[0], out[1], [])
    return text, list(audio or [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/pilot/items.jsonl")
    ap.add_argument("--run-id", default="d2_stepaudio")
    ap.add_argument("--out-root", default="exp")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--conditions", default=",".join(CONDS))
    args = ap.parse_args()

    conds = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    mslug = "Step-Audio-2-mini"
    out_dir = os.path.join(args.out_root, args.run_id, "predictions", mslug)
    os.makedirs(out_dir, exist_ok=True)

    items = [json.loads(l) for l in open(args.items)]
    if args.offset:
        items = items[args.offset:]
    if args.limit:
        items = items[: args.limit]
    print(f"[stepaudio] items={len(items)} conds={conds}", flush=True)

    t0 = time.time()
    model = StepAudio2(MODEL)
    t2w = Token2wav(os.path.join(MODEL, "token2wav"))
    print(f"[stepaudio] loaded {time.time()-t0:.1f}s", flush=True)

    n_ok = n_err = 0
    for k, it in enumerate(items, 1):
        iid = it["id"]
        idir = os.path.join(out_dir, iid)
        os.makedirs(idir, exist_ok=True)
        qtext = it["question_text"]
        qaudio = (it.get("question_audio") or {}).get("path")
        instruction = item_instruction(it)
        texts: dict[str, str] = {}

        def run(cond: str, messages, want_audio: bool):
            nonlocal n_ok, n_err
            p = os.path.join(idir, f"{cond}.json")
            if os.path.exists(p):
                prev = json.load(open(p))
                # A failed record is not a completed record: retry it on resume.
                # This matters for transient decoder/runtime failures and keeps a
                # full-dataset rerun from silently reporting errors as successes.
                if not prev.get("error"):
                    texts[cond] = prev.get("text") or ""
                    n_ok += 1
                    return
            rec = {"item_id": iid, "condition": cond, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            try:
                t = time.time()
                text, sp = call(model, messages, want_audio)
                rec.update({"text_raw": text, "text": clean_text(text), "latency_sec": round(time.time() - t, 3),
                            "audio_duration_sec": None, "audio": None,
                            "output_modality": "speech" if want_audio else "text",
                            "input_modality": "audio" if (messages and messages[0].get("content")
                                                          and isinstance(messages[0]["content"], list)) else "text"})
                if want_audio:
                    sp = [x for x in sp if x < 6561]
                    if not sp:
                        raise RuntimeError("无有效语音 token（疑似未产出语音）")
                    wav = os.path.join(idir, f"{cond}.wav")
                    data = t2w(sp, PROMPT_WAV)
                    with open(wav, "wb") as fh:
                        fh.write(data)
                    import soundfile as sf
                    info = sf.info(wav)
                    rec["audio"] = wav
                    rec["audio_duration_sec"] = round(info.frames / info.samplerate, 3)
                    rec["n_speech_tokens"] = len(sp)
                texts[cond] = text or ""
                n_ok += 1
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
                n_err += 1
            json.dump(rec, open(p, "w"), ensure_ascii=False, indent=2)

        if "READ" in conds:
            run("READ", [{"role": "human", "content": f"{qtext} {instruction}"}], False)
        if "LISTEN" in conds and qaudio:
            run("LISTEN", [{"role": "human", "content": [{"type": "audio", "audio": qaudio},
                                                         {"type": "text", "text": instruction}]}], False)
        if "SPEAK" in conds and qaudio:
            run("SPEAK", [{"role": "human", "content": [{"type": "audio", "audio": qaudio},
                                                        {"type": "text", "text": instruction}]}], True)
        if "ECHO" in conds:
            run("ECHO", [{"role": "human", "content": f"{qtext} {instruction}"}], True)
        if "EF" in conds and texts.get("READ"):
            run("EF", [{"role": "human", "content": ef_prompt(texts["READ"])}], True)
        if "EFW" in conds and texts.get("READ"):
            run("EFW", [{"role": "human", "content": f"{qtext} {instruction}",
                         "eot": True},
                        {"role": "assistant", "content": texts["READ"], "eot": True},
                        {"role": "human", "content": ef_prompt(speakable_format(texts["READ"]))}], True)
        if k % 10 == 0 or k == len(items):
            el = time.time() - t0
            print(f"  [{k}/{len(items)}] ok={n_ok} err={n_err} elapsed={el:.0f}s "
                  f"eta={el/k*(len(items)-k):.0f}s", flush=True)

    summary = {"model": mslug, "run_id": args.run_id, "n_items": len(items),
               "n_ok": n_ok, "n_err": n_err, "conds": list(conds),
               "wall_sec": round(time.time() - t0, 1)}
    os.makedirs(os.path.join(args.out_root, args.run_id, "metrics"), exist_ok=True)
    shard_tag = f"_off{args.offset}" if args.offset else "_off0"
    out = os.path.join(args.out_root, args.run_id, "metrics", f"infer_{mslug}{shard_tag}.json")
    summary["shard_offset"] = args.offset
    json.dump(summary, open(out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
