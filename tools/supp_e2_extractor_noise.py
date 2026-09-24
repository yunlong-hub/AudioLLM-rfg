"""E2: 抽取器噪声下限（不污染既有缓存）。

设计（修订版：只测真正相关且算得动的量）
------------------------------------
样本：`--run-id` 抽取缓存里**计划内部文本**（`SPEAK#internal`）按类别各取
`--per-cat` 条，共 5×per-cat 条。

三个 pass（贪心解码，同一进程）：
  A：batch=--batch 按键排序        —— 基准
  B：batch=--batch 同一排序（重复） —— 同配置重抽一致性（真·噪声）
  C：batch=--batch 随机打乱顺序     —— 批次组成敏感性（README §7 记录的抖动来源）
  D：batch=--batch*4 同一排序      —— 批次大小敏感性

指标：集合级严格一致率、事实级 Jaccard、逐类型一致率、相对既有缓存的漂移率。
缓存行由主流水线以另一种批次组成写成，因此 drift 也可与 C 相互印证。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rfg.facts.extract import merge  # noqa: E402
from rfg.facts.llm import LlmExtractor  # noqa: E402
from rfg.facts.rules import extract_rules  # noqa: E402

MODELS = ("Qwen3-Omni-30B-A3B-Instruct", "Qwen2.5-Omni-3B", "Qwen2.5-Omni-7B",
          "MiniCPM-o-4_5", "Step-Audio-2-mini")
CATS = ("num", "unit", "name", "neg", "cont")
SEED = 20260921


def key_set(extraction) -> set[tuple[str, str, str]]:
    return {(f.type, f.value, f.polarity) for f in extraction.facts}


def key_set_from_dicts(rows) -> set[tuple[str, str, str]]:
    return {(r["type"], r["value"], r["polarity"]) for r in rows}


def sample_texts(model: str, run_id: str, per_cat: int, rng: random.Random) -> list[dict]:
    cache = _ROOT / f"exp/{run_id}/facts_dual/{model}.jsonl"
    by_cat: dict[str, list[dict]] = {c: [] for c in CATS}
    with cache.open() as fh:
        for line in fh:
            rec = json.loads(line)
            key = rec.get("key") or ""
            if not key.endswith("|SPEAK#internal"):
                continue
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            cat = key.split("|")[0].split("_")[0]
            if cat not in by_cat:
                continue
            by_cat[cat].append({"key": key, "text": text,
                                "prior": key_set_from_dicts(rec.get("facts") or [])})
    picked: list[dict] = []
    for cat in CATS:
        pool = by_cat[cat]
        rng.shuffle(pool)
        picked.extend(pool[:per_cat])
    return picked


def run_pass(extractor: LlmExtractor, texts: list[str], batch_size: int) -> list[set]:
    extractor.batch_size = batch_size
    out: list[set] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        parsed = extractor.extract_batch(chunk)
        for text, (facts_llm, _err) in zip(chunk, parsed):
            out.append(key_set(merge(text, extract_rules(text), facts_llm)))
    return out


def compare(base: list[set], other: list[set], label: str) -> dict:
    n = len(base)
    strict = sum(1 for a, b in zip(base, other) if a == b)
    inter = union = sym = 0
    per_type_num: dict[str, int] = defaultdict(int)
    per_type_den: dict[str, int] = defaultdict(int)
    for a, b in zip(base, other):
        inter += len(a & b)
        union += len(a | b)
        sym += len(a ^ b)
        for f in a | b:
            per_type_den[f[0]] += 1
            if f in a and f in b:
                per_type_num[f[0]] += 1
    return {
        "label": label,
        "n_texts": n,
        "set_level_strict_agreement": strict / n,
        "fact_jaccard": inter / union if union else None,
        "facts_union_total": union,
        "symmetric_diff_total": sym,
        "per_type_agreement": {k: (per_type_num[k] / per_type_den[k] if per_type_den[k] else None)
                               for k in sorted(per_type_den)},
        "per_type_support": dict(sorted(per_type_den.items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="d0_resample")
    ap.add_argument("--llm-model",
                    default="/workspace/yunlong/LLM/pretrain_model/Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--per-cat", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="reports/supplementary_e2_extractor_noise.json")
    args = ap.parse_args()

    rng = random.Random(SEED)
    models = [m for m in args.models.split(",") if m]
    print(f"[e2] extractor={args.llm_model} device={args.device} per_cat={args.per_cat} "
          f"batch={args.batch}", flush=True)
    extractor = LlmExtractor(args.llm_model, device=args.device)

    report: dict = {
        "protocol": (f"{args.per_cat} texts per category (internal plans) per model; passes: "
                     f"A=batch{args.batch} sorted, B=batch{args.batch} sorted (repeat), "
                     f"C=batch{args.batch} shuffled, D=batch{args.batch*4} sorted; greedy, one process"),
        "seed": SEED, "run_id": args.run_id, "models": {}}

    for model in models:
        t0 = time.time()
        samples = sample_texts(model, args.run_id, args.per_cat, rng)
        order = sorted(range(len(samples)), key=lambda i: samples[i]["key"])
        texts_sorted = [samples[i]["text"] for i in order]
        shuffled = order[:]
        rng.shuffle(shuffled)
        texts_shuffled = [samples[i]["text"] for i in shuffled]
        prior = [samples[i]["prior"] for i in order]
        print(f"[e2] {model}: {len(samples)} texts", flush=True)

        run_a = run_pass(extractor, texts_sorted, args.batch)
        run_b = run_pass(extractor, texts_sorted, args.batch)
        run_c_shuffled = run_pass(extractor, texts_shuffled, args.batch)
        run_d = run_pass(extractor, texts_sorted, args.batch * 4)
        # 关键：把打乱顺序的结果按 key 映射回排序序，否则 compare() 会拿不同文本互比
        inverse = {shuffled[pos]: pos for pos in range(len(shuffled))}
        run_c = [run_c_shuffled[inverse[i]] for i in range(len(inverse))]
        drift = sum(1 for a, b in zip(prior, run_a) if a != b) / len(prior)

        report["models"][model] = {
            "n_texts": len(samples),
            "A_vs_B_repeat": compare(run_a, run_b, "A vs B (identical config, repeat)"),
            "A_vs_C_batchcomposition": compare(run_a, run_c, "A vs C (shuffled order)"),
            "A_vs_D_batchsize": compare(run_a, run_d, f"A vs D (batch {args.batch*4})"),
            "cache_drift_rate_vs_existing_cache": drift,
            "seconds": round(time.time() - t0, 1),
        }
        m = report["models"][model]
        print(f"[e2] {model}: strict A/B={m['A_vs_B_repeat']['set_level_strict_agreement']:.3f} "
              f"A/C={m['A_vs_C_batchcomposition']['set_level_strict_agreement']:.3f} "
              f"A/D={m['A_vs_D_batchsize']['set_level_strict_agreement']:.3f} "
              f"jac(A,B)={m['A_vs_B_repeat']['fact_jaccard']:.3f} "
              f"cache_drift={drift:.3f} ({m['seconds']}s)", flush=True)

        out = _ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print(f"[e2] wrote {_ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
