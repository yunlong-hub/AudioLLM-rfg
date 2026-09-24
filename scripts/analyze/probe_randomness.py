#!/usr/bin/env python3
"""为什么会随机？——用 FRR 已有的每题 4 个独立样本做方差分解。

两个零 GPU 分析
---------------
**A. 每事实的存活谱**：把内部文本里的每个事实放在 4 个独立样本上检验存活次数（0..4）。
   * 若存活谱集中在 0/4 与 4/4 → 失败是**确定性**的（该事实在某些条件下根本说不出来）
   * 若散布在 1/4–3/4 → 失败是**真随机**（同一事实有时说得对、有时说错）
   并给出方差分解：事实间（between-fact）与采样间（within-fact）各占多少。

**B. 位置分析（漂移假说）**：把每个事实在其内部文本中的**相对位置**算出来，
   比较"从不说对"的事实与"总是说对"的事实在位置分布上的差异。
   若失败集中在句子后段 → 支持"生成轨迹累积漂移"；若均匀分布 → 支持"局部采样事故"。

事实口径
--------
内部文本与单样本回读复用主口径产物 `exp/d0_pilot/facts/<model>.jsonl`（规则 ∪ LLM 合并集）；
重采样样本（`exp/d0_resample/<model>/<item_id>/R*.json` 的 `asr1`）不在其中，按同一口径用
`extract_dual()` 现抽并缓存到 `exp/d0_resample/facts_dual/<model>.jsonl`。
**不能**退回 `extract_rules()`：纯规则通道没有 `content` 类型，会造出第二套事实口径。

用法：
  python scripts/analyze/probe_randomness.py --run-id d0_resample --model Qwen3-Omni-30B-A3B-Instruct
  CUDA_VISIBLE_DEVICES=2 python scripts/analyze/probe_randomness.py --llm-device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict

_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

from rfg.facts.extract import DEFAULT_LLM_MODEL, extract_dual, load_extractions  # noqa: E402
from rfg.facts.llm import PROMPT_SHA256  # noqa: E402
from rfg.facts.resample import load_resample_corpus  # noqa: E402


def fact_positions(text: str, facts: set) -> dict:
    """给出每个事实在其文本中的相对位置（0=句首, 1=句末）。用规范值在原文中定位。"""
    low = text.lower()
    n = max(len(low), 1)
    pos = {}
    for f in facts:
        v = str(f.value).lower()
        i = low.find(v)
        pos[f] = (i / n) if i >= 0 else None
    return pos


def permutation_test(a: list[float], b: list[float], n_perm: int = 10000,
                     seed: int = 20260918) -> dict:
    """两组均值的置换检验（双侧）。

    分组（从不说对 / 总是说对）是**按结果事后定义**的，所以这个 p 值不是无偏的因果证据，
    只用于回答"观测到的位置差是否可能由随机分配产生"。报告里必须同时给出，不可只用 p。
    """
    import random as _random

    if len(a) < 2 or len(b) < 2:
        return {"diff": (st.mean(a) - st.mean(b)) if a and b else None,
                "p_value": None, "n_perm": 0, "note": "组样本不足"}
    obs = st.mean(a) - st.mean(b)
    pool = list(a) + list(b)
    na = len(a)
    rng = _random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(st.mean(pool[:na]) - st.mean(pool[na:])) >= abs(obs) - 1e-15:
            hits += 1
    return {"diff": obs, "p_value": (hits + 1) / (n_perm + 1), "n_perm": n_perm,
            "n_a": len(a), "n_b": len(b),
            "note": "双侧置换检验；分组按存活谱事后定义"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="d0_resample")
    ap.add_argument("--model", default="Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--pilot", default="exp/d0_pilot/predictions")
    ap.add_argument("--facts-root", default="exp/d0_pilot/facts",
                    help="主口径双通道产物目录（内部文本与 SPEAK 回读作为 priors 复用）")
    ap.add_argument("--out-root", default="exp")
    ap.add_argument("--out", default="reports/randomness.md")
    ap.add_argument("--metrics-out", default=None,
                    help="指标 JSON 路径；默认保持历史路径 exp/d0_pilot/metrics/randomness.json")
    ap.add_argument("--cache", default=None,
                    help="双通道抽取缓存；默认 exp/<run-id>/facts_dual/<model>.jsonl")
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    ap.add_argument("--llm-device", default="cuda:0")
    ap.add_argument("--llm-batch-size", type=int, default=16)
    ap.add_argument("--force-llm", action="store_true", help="忽略缓存与 priors，全部重抽")
    args = ap.parse_args()

    m = args.model
    run_dir = os.path.join(args.out_root, args.run_id)
    if not os.path.isdir(os.path.join(run_dir, m)):
        print(f"缺少 {os.path.join(run_dir, m)}", file=sys.stderr)
        return 1

    corpus, texts = load_resample_corpus(args.pilot, run_dir, m)
    print(f"可用题目: {len(corpus)}｜待判定文本 {len(texts)} 条", flush=True)
    priors = load_extractions(os.path.join(args.facts_root, f"{m}.jsonl"),
                              prompt_sha256=PROMPT_SHA256)
    cache_path = args.cache or os.path.join(run_dir, "facts_dual", f"{m}.jsonl")
    extracted = extract_dual(texts, cache_path=cache_path, llm_model=args.llm_model,
                             device=args.llm_device, batch_size=args.llm_batch_size,
                             force=args.force_llm, priors=priors)

    per_item: dict[str, dict[str, set]] = {
        it["item_id"]: {k: extracted[k].facts for k in it["sample_keys"]} for it in corpus}
    internals: dict[str, str] = {it["item_id"]: it["internal_text"] for it in corpus}
    upstream: dict[str, set] = {it["item_id"]: extracted[it["internal_key"]].facts for it in corpus}

    items = [i for i, d in per_item.items() if len(d) >= 4]
    print(f"可用题目（≥4 样本）: {len(items)}")

    # ---- A. 存活谱（逐事实留档：item_id / 类型 / 值 / 相对位置 / 每个样本的存活布尔）
    hist = Counter()
    facts_detail: list[dict] = []
    n_facts_up = 0
    for it in corpus:
        iid = it["item_id"]
        if iid not in per_item or len(per_item[iid]) < 4:
            continue
        up = upstream[iid]
        if not up:
            continue
        keys = [k for k in it["sample_keys"] if k in per_item[iid]]
        pos = fact_positions(internals[iid], up)
        n = len(keys)
        n_facts_up += len(up)
        for f in sorted(up, key=lambda x: (x.type, x.value)):
            alive = [f in per_item[iid][k] for k in keys]
            s = sum(alive)
            hist[s] += 1
            facts_detail.append({"item_id": iid, "type": f.type, "value": f.value,
                                 "polarity": f.polarity, "position": pos.get(f),
                                 "survival": s, "n_samples": n, "alive": alive,
                                 "sample_keys": [k.split("|", 1)[1] for k in keys]})

    tot = sum(hist.values())
    n_samp = max(len(per_item[i]) for i in items)
    type_counts = Counter(d["type"] for d in facts_detail)
    lines = ["# 为什么会随机：方差分解与位置分析", "",
             f"**模型**: `{m}` ｜ **题目**: {len(items)} ｜ **每题样本数**: {n_samp} ｜ "
             f"**事实总数**: {tot}", "",
             "## A. 每事实存活谱（在 N 个独立样本中有几次说对）", "",
             "| 存活次数 | 事实数 | 占比 | 解读 |", "|---|---:|---:|---|"]
    for s in range(n_samp + 1):
        c = hist.get(s, 0)
        tag = "确定性失败（从不说对）" if s == 0 else ("确定性成功" if s == n_samp else "**随机**")
        lines.append(f"| {s}/{n_samp} | {c} | {c/tot:.1%} | {tag} |")

    det_fail = hist.get(0, 0) / tot
    det_ok = hist.get(n_samp, 0) / tot
    rnd = 1 - det_fail - det_ok
    # 方差分解：p=存活率，总方差 = p(1-p)；按事实分组后可分解为组内/组间
    # 分组键用 (type, value, polarity) —— 与 Fact 的相等语义一致，跨题的同名事实合并成一组。
    by_fact = defaultdict(list)
    for d in facts_detail:
        by_fact[(d["type"], d["value"], d["polarity"])].append(d["survival"] / d["n_samples"])
    ps = [sum(v) / len(v) for v in by_fact.values()]
    within = st.mean([p * (1 - p) for p in ps]) if ps else 0
    between = st.pvariance(ps) if len(ps) > 1 else 0
    lines += ["", f"- 确定性失败（0/{n_samp}）：**{det_fail:.1%}**",
              f"- 确定性成功（{n_samp}/{n_samp}）：**{det_ok:.1%}**",
              f"- **随机区间（1–{n_samp-1}/{n_samp}）：{rnd:.1%}**",
              f"- 方差分解：事实间方差 {between:.4f}，事实内（采样）方差 {within:.4f} → "
              f"随机成分占 {within/(within+between):.1%}" if (within + between) > 0 else "- 方差退化",
              f"- 上游事实类型分布：" +
              "、".join(f"`{t}` {n}（{n/n_facts_up:.1%}）" for t, n in type_counts.most_common())
              + f"；其中 `content` {type_counts.get('content', 0)} 条"]

    # ---- B. 位置分析（逐事实明细留在 metrics json 里，供独立复核）
    never = [d["position"] for d in facts_detail
             if d["survival"] == 0 and d["position"] is not None]
    always = [d["position"] for d in facts_detail
              if d["survival"] == d["n_samples"] and d["position"] is not None]
    n_unlocatable = sum(1 for d in facts_detail if d["position"] is None)
    pos_test = permutation_test(never, always)
    lines += ["", "## B. 位置分析（相对位置 0=句首，1=句末）", ""]
    if never and always:
        lines += [f"- 从不说对的事实：n={len(never)}，平均相对位置 **{st.mean(never):.3f}**",
                  f"- 总是说对的事实：n={len(always)}，平均相对位置 **{st.mean(always):.3f}**",
                  f"- 差值 **{st.mean(never)-st.mean(always):+.3f}**"
                  f"（正=失败偏后 → 支持累积漂移；≈0 → 支持局部采样事故）",
                  f"- 置换检验（双侧，{pos_test['n_perm']} 次重排）：**p = {pos_test['p_value']:.3f}**"
                  f"（obs={pos_test['diff']:+.3f}）"
                  + ("→ 位置差在 5% 水平上不显著" if pos_test["p_value"] > 0.05
                     else "→ 位置差在 5% 水平上显著"),
                  f"- 注意：两组是**按存活结果事后分组**的（选择效应），且位置只在规范化值能被原文"
                  f"字面定位时才有定义（{n_unlocatable}/{tot} 条无位置），p 值只回答"
                  f"\"这个差能否由随机分组产生\"，不能当作因果证据。"]
    else:
        lines.append(f"- 样本不足（never={len(never)}, always={len(always)}）")
    lines += ["", "---", "",
              "**口径**：事实为**双通道合并集**（规则通道 ∪ LLM 通道，`scripts/facts/extract_facts.py` 同款），"
              "内部文本与单样本回读复用 `exp/d0_pilot/facts/`，重采样样本现抽并缓存于 "
              f"`{os.path.relpath(cache_path, _ROOT)}`；存活判定基于双 ASR 中的 asr1 回读；"
              "样本为同一 SPEAK 提示词的独立重采样。"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    # 分型聚合：论文 §5.5 的分型表直接引用这里，避免再手工重算（曾因此无产物可追溯）
    by_type: dict = {}
    for f in facts_detail:
        t = f["type"]
        d = by_type.setdefault(t, {"n": 0, "never": 0, "always": 0, "survival_sum": 0.0})
        d["n"] += 1
        d["survival_sum"] += f["survival"] / f["n_samples"]
        if f["survival"] == 0:
            d["never"] += 1
        if f["survival"] == f["n_samples"]:
            d["always"] += 1
    for d in by_type.values():
        d["never_rate"] = d["never"] / d["n"]
        d["always_rate"] = d["always"] / d["n"]
        d["mean_survival"] = d["survival_sum"] / d["n"]
        del d["survival_sum"]
    random_share = within / (between + within) if (between + within) else None

    metrics_path = args.metrics_out or os.path.join(
        args.out_root, "d0_pilot", "metrics", "randomness.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    json.dump({"n_items": len(items), "n_facts": tot, "n_samples": n_samp,
               "hist": {str(k): v for k, v in hist.items()},
               "det_fail": det_fail, "det_ok": det_ok, "random_zone": rnd,
               "var_between": between, "var_within": within,
               "random_share": random_share,
               "by_type": by_type,
               "pos_never": st.mean(never) if never else None,
               "pos_always": st.mean(always) if always else None,
               "position_test": pos_test,
               "n_facts_without_position": n_unlocatable,
               "facts_by_type": dict(type_counts),
               "content_facts": type_counts.get("content", 0),
               "facts": facts_detail,
               "cache": os.path.relpath(cache_path, _ROOT),
               "prompt_sha256": PROMPT_SHA256, "llm_model": args.llm_model},
              open(metrics_path, "w"), indent=2, ensure_ascii=False)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
