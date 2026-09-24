#!/usr/bin/env python3
"""定向重渲染：能否把结构性下限（FRR 之后的残余）压下去。

设计
----
对象：在 4 个独立样本中**从未被说出**的事实所在的题目（即 FRR 之后的残余）。
条件：EF（给定文本朗读），两种变体各 4 样本：
  * `ORIG`  = 原内部文本
  * `SPEAK` = 定向改写（数字→词形并显式分段；长单位按词素加空格；专名按音节拆分）
判据：改写后该事实的存活次数是否上升（0/4 → ≥1/4）。

事实口径
--------
"从未被说出"按**双通道合并集**判定（规则 ∪ LLM，`extract_dual()`）：内部文本与单样本回读复用
`exp/d0_pilot/facts/<model>.jsonl`，重采样样本现抽并缓存于 `exp/d0_resample/facts_dual/<model>.jsonl`
——与 `probe_randomness.py` 共用同一份缓存与同一批键，因此"结构性失败"的定义与随机性报告严格一致。
改写效果由 `judge_rerender.py` 回读判定，同样走双通道。

用法： CUDA_VISIBLE_DEVICES=2 python scripts/analyze/probe_rerender.py --model <30B path> --n 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

from rfg.facts.extract import DEFAULT_LLM_MODEL, extract_dual, load_extractions  # noqa: E402
from rfg.facts.llm import PROMPT_SHA256  # noqa: E402
from rfg.facts.resample import load_resample_corpus  # noqa: E402
from rfg.facts.schema import Fact  # noqa: E402
from rfg.run.conditions import ef_prompt  # noqa: E402
from rfg.score.textnorm import _DIGIT_RE  # noqa: E402

MORPHEME_SPLIT = {
    "kilometer": "kilo meter", "centimeter": "centi meter", "millimeter": "milli meter",
    "milliliter": "milli liter", "kilogram": "kilo gram", "megabyte": "mega byte",
    "gigabyte": "giga byte", "percentage": "percent age",
}


def speakable(text: str) -> str:
    """定向改写：数字→词形；长单位按词素加空格；保持其余内容不变。"""
    from num2words import num2words

    def repl(m):
        raw = m.group(0).replace(",", "")
        try:
            if "." in raw:
                w, f = raw.split(".", 1)
                return f"{num2words(int(w))} point {' '.join(num2words(int(c)) for c in f if c.isdigit())}"
            return num2words(int(raw))
        except Exception:
            return raw

    out = _DIGIT_RE.sub(repl, text)
    for k, v in MORPHEME_SPLIT.items():
        out = re.sub(rf"\b{k}s?\b", v, out, flags=re.I)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="生成用 S2S 模型路径（其 basename 即模型目录名）")
    ap.add_argument("--resample-root", default="exp/d0_resample")
    ap.add_argument("--pilot", default="exp/d0_pilot/predictions")
    ap.add_argument("--facts-root", default="exp/d0_pilot/facts")
    ap.add_argument("--items", default="data/pilot/items.jsonl")
    ap.add_argument("--run-id", default="d4_rerender")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--cache", default=None,
                    help="双通道抽取缓存；默认 exp/d0_resample/facts_dual/<model>.jsonl（与随机性探针共用）")
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    ap.add_argument("--llm-device", default="cuda:0")
    ap.add_argument("--llm-batch-size", type=int, default=16)
    ap.add_argument("--force-llm", action="store_true")
    args = ap.parse_args()

    m = os.path.basename(args.model.rstrip("/"))
    out_dir = os.path.join("exp", args.run_id, m)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1) 找出结构性失败（4 样本从未说出），双通道口径
    corpus, texts = load_resample_corpus(args.pilot, args.resample_root, m)
    priors = load_extractions(os.path.join(args.facts_root, f"{m}.jsonl"),
                              prompt_sha256=PROMPT_SHA256)
    cache_path = args.cache or os.path.join(args.resample_root, "facts_dual", f"{m}.jsonl")
    extracted = extract_dual(texts, cache_path=cache_path, llm_model=args.llm_model,
                             device=args.llm_device, batch_size=args.llm_batch_size,
                             force=args.force_llm, priors=priors)

    targets: list[tuple[str, Fact, str]] = []   # (item_id, fact, internal_text)
    n_multi = 0
    for it in corpus:
        iid = it["item_id"]
        if len(it["sample_keys"]) < 4:
            continue
        n_multi += 1
        up = extracted[it["internal_key"]].facts
        seen = [extracted[k].facts for k in it["sample_keys"]]
        for f in sorted(up, key=lambda x: (x.type, x.value)):
            if all(f not in c for c in seen):
                targets.append((iid, f, it["internal_text"]))
    type_hist = Counter(f.type for _, f, _ in targets)
    print(f"[rerender] 结构性失败事实 {len(targets)} 个，覆盖 {len({t[0] for t in targets})} 题"
          f"（可用题 {n_multi}）｜类型 {dict(type_hist)}", flush=True)
    if not targets:
        return 0

    items = {json.loads(l)["id"]: json.loads(l) for l in open(args.items)}
    rows_by_item: dict[str, dict] = {}
    for iid, f, text in targets:
        rows_by_item.setdefault(iid, {"item_id": iid, "text": text, "facts": []})["facts"].append(
            f"{f.type}:{f.value}")

    # 只生成磁盘上确实缺失的样本；全部命中时不加载 30B（重跑判定不占 GPU）
    variants_by_item = {iid: {"ORIG": r["text"], "SPEAK": speakable(r["text"])}
                        for iid, r in rows_by_item.items()}
    missing = [(iid, vname, k)
               for iid, vs in variants_by_item.items()
               for vname in vs
               for k in range(args.n)
               if not os.path.exists(os.path.join(out_dir, iid, f"{vname}_{k}.json"))]
    print(f"[rerender] 待生成样本 {len(missing)} 个（{len(rows_by_item)} 题 × 2 变体 × {args.n}）",
          flush=True)

    if missing:
        from rfg.models.omni import OmniModel

        omni = OmniModel(args.model)
        for idx, (iid, vname, k) in enumerate(missing, 1):
            qa = (items[iid].get("question_audio") or {}).get("path")
            wp = os.path.join(out_dir, iid, f"{vname}_{k}.json")
            os.makedirs(os.path.dirname(wp), exist_ok=True)
            res = omni.chat([{"type": "audio", "audio": qa},
                             {"type": "text", "text": ef_prompt(variants_by_item[iid][vname])}],
                            want_audio=True, seed=1000 + k)
            r = {"text": res.text, "dur": res.duration_sec}
            if res.audio is not None and res.audio.size:
                wav = os.path.join(out_dir, iid, f"{vname}_{k}.wav")
                OmniModel.save_wav(wav, res)
                r["audio"] = wav
            json.dump(r, open(wp, "w"), ensure_ascii=False)
            if idx % 10 == 0:
                print(f"  生成 {idx}/{len(missing)}", flush=True)
        del omni

    # ---- 2) 汇总变体文本（回读与判定由 judge_rerender.py 负责）
    rows = []
    for iid, r in sorted(rows_by_item.items()):
        rec = {"item_id": iid, "facts": r["facts"], "variants": {}}
        for vname, vtext in variants_by_item[iid].items():
            texts_out = []
            for k in range(args.n):
                wp = os.path.join(out_dir, iid, f"{vname}_{k}.json")
                texts_out.append(json.load(open(wp)).get("text") or "" if os.path.exists(wp) else "")
            rec["variants"][vname] = texts_out
            rec["variants"][vname + "_text"] = vtext
        rows.append(rec)

    os.makedirs(os.path.join("exp", args.run_id, "metrics"), exist_ok=True)
    targets_path = os.path.join("exp", args.run_id, "metrics", "targets.json")
    json.dump(rows, open(targets_path, "w"), ensure_ascii=False, indent=2)
    print(f"[rerender] {len(rows)} 题 -> {targets_path}"
          f"（结构性失败事实 {len(targets)} 个，类型 {dict(type_hist)}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
