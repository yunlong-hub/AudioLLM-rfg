#!/usr/bin/env python3
"""D0-2 冒烟测试：五条件（READ/LISTEN/SPEAK/ECHO/EF）+ 三项断言。

目的不是产出指标，而是回答三个"能不能做"的问题：
  A1 语音条件真的出声了吗？（防止静默回退到文本）
  A2 模型真的在听吗？（反事实：换问题音频，答案是否跟着换）
  A3 双 ASR 回读可用吗？（一致率与耗时）

输出：
  exp/d0_smoke/api_probe.json          模型/generate 的可用参数（写进冻结清单）
  exp/d0_smoke/<item_id>/<cond>.{json,wav}
  exp/d0_smoke/summary.json            断言结果

用法：
  CUDA_VISIBLE_DEVICES=2,3 python scripts/smoke.py --model <path> [--skip-asr]
  # 只补跑回读（复用已有 wav，不重新加载 30B 模型）
  CUDA_VISIBLE_DEVICES=2,3 python scripts/smoke.py --readback-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")

from rfg.models.omni import OmniModel  # noqa: E402
from rfg.score.textnorm import agreement_norm  # noqa: E402

PRETRAIN = "/workspace/yunlong/LLM/pretrain_model"
DEFAULT_MODEL = f"{PRETRAIN}/Qwen/Qwen3-Omni-30B-A3B-Instruct"
WHISPER = f"{PRETRAIN}/Audio/whisper-large-v3"
SEAMLESS = f"{PRETRAIN}/Audio/seamless-m4t-v2-large"
OUT = "exp/d0_smoke"

INSTRUCTION = "Answer the question in one short sentence."

# A3 的一致率只在这些条件上统计（RFG 相关的语音条件）
READBACK_CORE_CONDS = ("SPEAK", "ECHO", "EF")

# 3 道手写题：覆盖 number / unit / proper_noun
ITEMS = [
    {"id": "num_smoke", "category": "number",
     "question": "What is seventeen times three?",
     "gold": ["51", "fifty-one", "fifty one"]},
    {"id": "unit_smoke", "category": "unit",
     "question": "How many meters are there in two and a half kilometers?",
     "gold": ["2500", "2,500", "two thousand five hundred"]},
    {"id": "name_smoke", "category": "proper_noun",
     "question": "What is the capital city of Australia?",
     "gold": ["Canberra"]},
]


def jdump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def contains_gold(text: str | None, gold: list[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(g.lower() in t for g in gold)


def _safe_transcribe(asr, wav: str) -> tuple[str | None, str | None, float | None]:
    """单通道转写：一个 ASR 失败不影响另一个通道的结果。"""
    try:
        r = asr.transcribe(wav)
        return r.text, None, round(r.latency_sec or 0, 2)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", None


def run_asr_stage(summary: dict, out_dir: str) -> None:
    """对已落盘音频做双 ASR 回读。核心一致率只统计 SPEAK/ECHO/EF。"""
    from rfg.models.asr import SeamlessReadback, WhisperReadback

    print("[smoke] loading ASR-1 whisper-large-v3", flush=True)
    asr1 = WhisperReadback(WHISPER, device="cuda:0")
    print("[smoke] loading ASR-2 seamless-m4t-v2-large", flush=True)
    asr2 = SeamlessReadback(SEAMLESS, device="cuda:0")

    n_speech = n_agree = n_nonempty = 0
    for iid, rec in summary["items"].items():
        for cond, e in rec["runs"].items():
            wav = e.get("audio")
            if not wav or not os.path.exists(wav):
                continue
            t1, err1, sec1 = _safe_transcribe(asr1, wav)
            t2, err2, sec2 = _safe_transcribe(asr2, wav)
            if err1:
                summary["errors"].append(f"{iid}/{cond} asr1: {err1}")
            if err2:
                summary["errors"].append(f"{iid}/{cond} asr2: {err2}")
            if err1 and err2:
                e["readback"] = {"error": f"both ASR failed: {err1} | {err2}"}
                continue
            agree, w = agreement_norm(t1, t2)
            e["readback"] = {"asr1": t1, "asr2": t2, "asr_agree": agree, "agreement_wer": w,
                             "asr1_sec": sec1, "asr2_sec": sec2,
                             "asr1_error": err1, "asr2_error": err2}
            if cond in READBACK_CORE_CONDS:
                n_speech += 1
                n_nonempty += int(bool(t1) and bool(t2))
                n_agree += int(agree)
            print(f"  [readback {iid}/{cond}] agree={agree} wer="
                  f"{(f'{w:.3f}' if w is not None else 'NA')} "
                  f"asr1={(t1 or err1 or '')[:50]!r} asr2={(t2 or err2 or '')[:50]!r}", flush=True)

    summary.setdefault("assertions", {})["A3_readback"] = {
        "n_speech_audio": n_speech,
        "n_both_nonempty": n_nonempty,
        "n_agree_norm_wer_le_0.2": n_agree,
        "agreement_rate": (n_agree / n_speech) if n_speech else None,
        "note": "一致率按规范化 WER<=0.2 判定；正式口径改为事实集合一致（D0-5）",
        "pass": n_speech > 0 and n_nonempty == n_speech,
    }


def finalize_assertions(summary: dict) -> None:
    speech_conds = ("SPEAK", "ECHO", "EF", "Q_AUDIO")
    n_req, n_ok, short = 0, 0, []
    for iid, rec in summary["items"].items():
        for cond in speech_conds:
            e = rec["runs"].get(cond) or {}
            if "error" in e or "audio" not in e:
                continue
            n_req += 1
            if (e.get("audio_duration_sec") or 0) > 0.5:
                n_ok += 1
            else:
                short.append(f"{iid}/{cond}:{e.get('audio_duration_sec')}s")
    summary["assertions"]["A1_speech_emitted"] = {
        "n_speech_conditions": n_req, "n_with_audio_gt_0.5s": n_ok,
        "short_audio": short,
        "pass": n_req > 0 and n_req == n_ok,
    }

    cf = [r.get("counterfactual") for r in summary["items"].values() if r.get("counterfactual")]
    summary["assertions"]["A2_actually_listening"] = {
        "n_checks": len(cf),
        "n_answer_follows_swapped_audio": sum(1 for c in cf if c["answer_matches_swapped"]),
        "n_answer_follows_original_audio": sum(1 for c in cf if c["answer_matches_own"]),
        "pass": bool(cf) and all(c["answer_matches_swapped"] for c in cf),
    }


def run_generation(args, summary: dict) -> None:
    print(f"[smoke] loading {args.model}", flush=True)
    t0 = time.time()
    omni = OmniModel(args.model)
    summary["load_sec"] = round(time.time() - t0, 1)
    jdump(f"{args.out}/api_probe.json", {
        "arch": omni.arch,
        "generate_params": omni.generate_params,
        "audio_kwargs_supported": omni.audio_kwargs_supported,
    })

    for it in ITEMS:
        iid = it["id"]
        rec: dict = {"category": it["category"], "question": it["question"], "runs": {}}
        q_audio = f"{args.out}/{iid}/question_audio.wav"

        def run(cond: str, fn, fname: str | None = None):
            try:
                res = fn()
                entry = {"text": res.text, "output_modality": res.output_modality,
                         "latency_sec": round(res.latency_sec or 0, 2),
                         "audio_duration_sec": (round(res.duration_sec, 2) if res.duration_sec else None),
                         # 诊断：return_audio 会改变 generate 的返回形态，必须留痕
                         "raw_text": res.meta.get("raw_text"),
                         "output_type": res.meta.get("output_type"),
                         "text_ids_shape": res.meta.get("text_ids_shape")}
                if res.audio is not None and res.audio.size > 0:
                    wav = f"{args.out}/{iid}/{fname or cond}.wav"
                    OmniModel.save_wav(wav, res)
                    entry["audio"] = wav
                rec["runs"][cond] = entry
                print(f"  [{iid}/{cond}] {res.output_modality} "
                      f"{entry['audio_duration_sec']}s -> {(res.text or '')[:70]!r}", flush=True)
                return res
            except Exception as exc:
                rec["runs"][cond] = {"error": f"{type(exc).__name__}: {exc}"}
                summary["errors"].append(f"{iid}/{cond}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                return None

        print(f"[smoke] item {iid}", flush=True)
        # 题目音频：冒烟阶段用模型自身的强制朗读合成（D0-3 换成真 TTS）
        run("Q_AUDIO", lambda: omni.ask_forced_readback(it["question"]), fname="question_audio")

        read = run("READ", lambda: omni.ask_text(f"{it['question']} {INSTRUCTION}"))
        run("LISTEN", lambda: omni.ask_audio_text(q_audio, INSTRUCTION))
        run("SPEAK", lambda: omni.ask_audio_speech(q_audio, INSTRUCTION))
        run("ECHO", lambda: omni.ask_text_speech(f"{it['question']} {INSTRUCTION}"))
        if read is not None and read.text:
            run("EF", lambda: omni.ask_forced_readback(read.text))
        else:
            rec["runs"]["EF"] = {"error": "upstream READ gave no text"}

        if not args.skip_counterfactual:
            other = next(x for x in ITEMS if x["id"] != iid)
            other_audio = f"{args.out}/{other['id']}/question_audio.wav"
            if os.path.exists(other_audio):
                sw = run("LISTEN_SWAPPED", lambda: omni.ask_audio_text(other_audio, INSTRUCTION))
                rec["counterfactual"] = {
                    "swapped_question": other["question"],
                    "answer_matches_swapped": contains_gold(sw.text if sw else None, other["gold"]),
                    "answer_matches_own": contains_gold(sw.text if sw else None, it["gold"]),
                }

        summary["items"][iid] = rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--skip-counterfactual", action="store_true")
    ap.add_argument("--readback-only", action="store_true",
                    help="复用已有 summary.json 与 wav，只补跑 ASR 回读与断言（不加载 30B 模型）")
    args = ap.parse_args()

    if args.readback_only:
        with open(f"{args.out}/summary.json") as fh:
            summary = json.load(fh)
        # 丢弃上一轮 ASR 阶段的失败记录，避免误判
        summary["errors"] = [e for e in summary.get("errors", []) if not e.startswith("ASR stage")]
    else:
        summary = {"model": args.model, "assertions": {}, "items": {}, "errors": [],
                   "decode_config": {"temperature": 0.0, "max_new_tokens": 512,
                                     "instruction": INSTRUCTION}}
        run_generation(args, summary)

    if not args.skip_asr:
        try:
            run_asr_stage(summary, args.out)
        except Exception as exc:
            summary["errors"].append(f"ASR stage: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    finalize_assertions(summary)
    jdump(f"{args.out}/summary.json", summary)

    print("\n=== ASSERTIONS ===")
    print(json.dumps(summary["assertions"], indent=2, ensure_ascii=False))
    print(f"errors: {len(summary['errors'])}")
    for e in summary["errors"][:10]:
        print("  -", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
