#!/usr/bin/env python3
"""重采样一致性重排（FRR）：检验渲染损失是否为**随机**的，并测试零训练修复。

动机
----
失败模式刻画显示：损失主体是"正常语速、大体可懂、个别事实被说错"的细微错说（非生成崩溃）。
若这种错说是**随机的渲染噪声**，则同一内容多采样几次应能采到"说对了"的版本，
按"回读与内部文本的事实一致性"挑选即可零训练降低 RFE。

设计
----
对每条题目的 SPEAK 条件再生成 N=3 个样本（不同随机种子），然后：
1. 双 ASR 回读每个样本；
2. 抽取事实（**双通道合并集**，与 `probe_randomness.py` 共用 `exp/d0_resample/facts_dual/` 缓存）；
3. **选择规则（不使用任何 gold）**：取"回读事实集与内部文本事实集 Jaccard 最高"的样本；
4. 比较 `Δ_render(单样本 SPEAK)` vs `Δ_render(FRR 选中样本)`。

同时报告当前评估 ASR 下的 candidate oracle（4 个样本中最好的）以区分
"选择规则不够好"与"候选集合没有更好样本"；它不是真实语音事实的上界。

用法：
  CUDA_VISIBLE_DEVICES=2 python scripts/analyze/probe_resample.py --model <path> --n 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

from rfg.facts.extract import DEFAULT_LLM_MODEL, extract_dual, load_extractions  # noqa: E402
from rfg.facts.llm import PROMPT_SHA256  # noqa: E402
from rfg.facts.resample import load_resample_corpus  # noqa: E402
from rfg.run.conditions import INSTRUCTION  # noqa: E402

P = "/workspace/yunlong/LLM/pretrain_model"
OUT = "exp/d0_resample"


def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def gap(up: set, dn: set) -> float | None:
    return (1 - len(up & dn) / len(up)) if up else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default="data/pilot/items.jsonl")
    ap.add_argument("--n", type=int, default=3, help="额外采样个数（不含原样本）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seeds", default="101,202,303")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pilot-pred", default="exp/d0_pilot/predictions")
    ap.add_argument("--facts-root", default="exp/d0_pilot/facts")
    ap.add_argument("--cache", default=None,
                    help="双通道抽取缓存；默认 <out>/facts_dual/<model>.jsonl（与随机性探针共用）")
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    ap.add_argument("--llm-device", default="cuda:0")
    ap.add_argument("--llm-batch-size", type=int, default=16)
    ap.add_argument("--force-llm", action="store_true")
    ap.add_argument("--selector-asr", choices=("asr1", "asr2", "asr3"), default="asr1")
    ap.add_argument("--evaluator-asr", choices=("asr1", "asr2", "asr3"), default="asr2")
    ap.add_argument("--generate-only", action="store_true",
                    help="只生成缺失的重采样音频，供多卡按 offset/limit 分片")
    args = ap.parse_args()

    mslug = os.path.basename(args.model.rstrip("/"))
    src = os.path.join(args.pilot_pred, mslug)
    odir = os.path.join(args.out, mslug)
    os.makedirs(odir, exist_ok=True)

    items = [json.loads(l) for l in open(args.items)]
    if args.offset:
        items = items[args.offset:]
    if args.limit:
        items = items[: args.limit]
    seeds = [int(s) for s in args.seeds.split(",")][: args.n]

    # ---- 已有资产：内部文本与单样本回读
    base: dict[str, dict] = {}
    for it in items:
        p = os.path.join(src, it["id"], "SPEAK.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        rb = r.get("readback") or {}
        if not r.get("text") or not rb.get("asr1"):
            continue
        base[it["id"]] = {"internal": r["text"], "asr1": rb["asr1"], "asr2": rb.get("asr2"),
                          "wav": os.path.join(src, it["id"], "SPEAK.wav")}
    print(f"可用基线样本 {len(base)} 条", flush=True)

    # ---- 1) 额外采样
    need = [i for i in base if not all(
        os.path.exists(os.path.join(odir, i, f"R{k}.json")) for k in range(len(seeds)))]
    if need:
        from rfg.models.omni import OmniModel

        omni = OmniModel(args.model)
        qaudio = {it["id"]: (it.get("question_audio") or {}).get("path") for it in items}
        for idx, iid in enumerate(sorted(need), 1):
            idir = os.path.join(odir, iid)
            os.makedirs(idir, exist_ok=True)
            for k, sd in enumerate(seeds):
                rp = os.path.join(idir, f"R{k}.json")
                if os.path.exists(rp):
                    continue
                res = omni.chat([{"type": "audio", "audio": qaudio[iid]},
                                 {"type": "text", "text": INSTRUCTION}],
                                want_audio=True, seed=sd)
                if res.audio is None or res.audio.size == 0:
                    raise RuntimeError(f"{iid}/R{k} 未产出音频")
                wav = os.path.join(idir, f"R{k}.wav")
                OmniModel.save_wav(wav, res)
                with open(rp, "w") as fh:
                    json.dump({"item_id": iid, "seed": sd, "text": res.text, "audio": wav,
                               "audio_duration_sec": round(res.duration_sec or 0, 3)}, fh,
                              ensure_ascii=False)
            if idx % 20 == 0:
                print(f"  采样 {idx}/{len(need)}", flush=True)
        del omni

    if args.generate_only:
        print(f"生成阶段完成：{len(base)} 个候选题范围", flush=True)
        return 0

    # ---- 2) 双 ASR 回读（只在确有缺失时才加载 ASR）
    wavs = []
    for iid in base:
        for k in range(len(seeds)):
            w = os.path.join(odir, iid, f"R{k}.wav")
            if os.path.exists(w) and not json.load(open(os.path.join(odir, iid, f"R{k}.json"))).get("asr1"):
                wavs.append((iid, k, w))
    if wavs:
        from rfg.models.asr import SeamlessReadback, WhisperReadback

        a1 = WhisperReadback(f"{P}/Audio/whisper-large-v3", device="cuda:0")
        a2 = SeamlessReadback(f"{P}/Audio/seamless-m4t-v2-large", device="cuda:0")
        for idx, (iid, k, w) in enumerate(wavs, 1):
            rp = os.path.join(odir, iid, f"R{k}.json")
            rec = json.load(open(rp))
            try:
                rec["asr1"] = a1.transcribe(w).text
            except Exception as e:
                rec["asr1_error"] = str(e)
            try:
                rec["asr2"] = a2.transcribe(w).text
            except Exception as e:
                rec["asr2_error"] = str(e)
            json.dump(rec, open(rp, "w"), ensure_ascii=False)
            if idx % 100 == 0:
                print(f"  回读 {idx}/{len(wavs)}", flush=True)
        del a1, a2

    # ---- 3) 事实（双通道合并集）与选择
    corpus, texts = load_resample_corpus(args.pilot_pred, args.out, mslug)
    priors = load_extractions(os.path.join(args.facts_root, f"{mslug}.jsonl"),
                              prompt_sha256=PROMPT_SHA256)
    cache_path = args.cache or os.path.join(args.out, "facts_dual", f"{mslug}.jsonl")
    extracted = extract_dual(texts, cache_path=cache_path, llm_model=args.llm_model,
                             device=args.llm_device, batch_size=args.llm_batch_size,
                             force=args.force_llm, priors=priors)
    order = {it["id"]: i for i, it in enumerate(items)}
    corpus.sort(key=lambda it: order.get(it["item_id"], len(order)))

    rows = []
    for it in corpus:
        iid = it["item_id"]
        up = extracted[it["internal_key"]].facts
        if not up:
            continue
        by_asr = {
            "asr1": it["sample_keys"],
            "asr2": it["sample_keys_asr2"],
            "asr3": it.get("sample_keys_asr3") or [],
        }
        selector_keys = by_asr[args.selector_asr]
        evaluator_keys = by_asr[args.evaluator_asr]
        if len(selector_keys) < 2 or len(selector_keys) != len(evaluator_keys):
            continue
        candidates = []
        for index, (selector_key, evaluator_key) in enumerate(
                zip(selector_keys, evaluator_keys)):
            candidates.append({
                "index": index,
                "name": selector_key.split("|", 1)[1].split("#", 1)[0],
                "selector": jaccard(up, extracted[selector_key].facts),
                "eval_gap": gap(up, extracted[evaluator_key].facts),
            })
        best = max(candidates, key=lambda row: row["selector"])
        valid_eval = [row["eval_gap"] for row in candidates if row["eval_gap"] is not None]
        if not valid_eval or best["eval_gap"] is None:
            continue
        base_gap = candidates[0]["eval_gap"]
        rows.append({"item_id": iid, "base_gap": base_gap,
                     "random_gap": sum(valid_eval) / len(valid_eval),
                     "frr_gap": best["eval_gap"], "oracle_gap": min(valid_eval),
                     "picked": best["name"], "frr_jaccard": best["selector"],
                     "n_candidates": len(candidates)})

    if not rows:
        print("没有可用样本", file=sys.stderr)
        return 1
    import statistics as st
    rng = random.Random(20260916)
    diffs = [r["base_gap"] - r["frr_gap"] for r in rows]
    s = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))) / len(diffs)
               for _ in range(10000))
    res = {
        "model": mslug, "n_items": len(rows), "n_samples": len(seeds) + 1,
        "delta_render_single": st.mean([r["base_gap"] for r in rows]),
        "delta_render_random": st.mean([r["random_gap"] for r in rows]),
        "delta_render_frr": st.mean([r["frr_gap"] for r in rows]),
        "delta_render_oracle": st.mean([r["oracle_gap"] for r in rows]),
        "frr_reduction": st.mean(diffs),
        "frr_ci95": [s[int(0.025 * len(s))], s[int(0.975 * len(s)) - 1]],
        "picked_original_rate": sum(1 for r in rows if r["picked"] == "SPEAK") / len(rows),
        "perfect_rate_single": sum(1 for r in rows if r["base_gap"] <= 0.001) / len(rows),
        "perfect_rate_frr": sum(1 for r in rows if r["frr_gap"] <= 0.001) / len(rows),
        "n_upstream_facts": sum(len(extracted[it["internal_key"]].facts) for it in corpus),
        "n_content_upstream": sum(1 for it in corpus for f in extracted[it["internal_key"]].facts
                                  if f.type == "content"),
        "cache": os.path.relpath(cache_path, _ROOT), "prompt_sha256": PROMPT_SHA256,
        "selector_asr": args.selector_asr, "evaluator_asr": args.evaluator_asr,
    }
    os.makedirs("reports", exist_ok=True)
    os.makedirs(os.path.join("exp", "d0_pilot", "metrics"), exist_ok=True)
    suffix = "" if (args.selector_asr, args.evaluator_asr) == ("asr1", "asr2") else (
        f"_sel-{args.selector_asr}_eval-{args.evaluator_asr}"
    )
    json.dump({"summary": res, "per_item": rows},
              open(os.path.join(args.out, f"resample_{mslug}{suffix}.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
