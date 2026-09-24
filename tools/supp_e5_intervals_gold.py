"""E5: 表 1 的按题聚类 bootstrap 区间 + 金标事实复核。

只读现有产物：
  exp/d0_resample/frr_loo_<model>.json          选择/评估逐题 gap（8 候选，三折）
  exp/d1_main/metrics/scores_*.json             受控集聚合口径（用于交叉核对）
  data/pilot/items.jsonl                        金标事实
  exp/d0_resample/facts_dual/<model>.jsonl      候选与计划的抽取事实

输出：reports/supplementary_e5_intervals_gold.{json,md}

口径说明
--------
* plan/render loss 的区间：按题聚类 bootstrap（与论文 3.2 节一致），
  不是 scores_*.json 里的 wilson_ci_gap（后者把同一条回答的多个事实当独立样本）。
* gold 复核：对每个候选（原始 SPEAK 与 R1..R7）计算
  ret(候选回读事实 <- 金标事实) 与幻觉率（回读中不属于金标的事实占比）。
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rfg.facts.extract import load_extractions  # noqa: E402
from rfg.facts.llm import PROMPT_SHA256  # noqa: E402
from rfg.facts.resample import load_resample_corpus  # noqa: E402

MODELS = {
    "Qwen3-Omni-30B-A3B-Instruct": {"pred": "exp/d1_main/predictions", "run": "d1_main"},
    "Qwen2.5-Omni-3B": {"pred": "exp/d1_main/predictions", "run": "d1_main"},
    "Qwen2.5-Omni-7B": {"pred": "exp/d3_7b_stack/predictions", "run": "d3_7b_stack"},
    "MiniCPM-o-4_5": {"pred": "exp/d1_main/predictions", "run": "d1_main"},
    "Step-Audio-2-mini": {"pred": "exp/d2_stepaudio/predictions", "run": "d2_stepaudio"},
}
ASR_NAMES = ("asr1", "asr2", "asr3")
N_BOOT = 10_000
SEED = 20260921


def bootstrap_mean(values: list[float], seed: int, n_boot: int = N_BOOT) -> dict:
    rng = random.Random(seed)
    point = statistics.mean(values)
    draws = sorted(statistics.mean(values[rng.randrange(len(values))] for _ in values)
                   for _ in range(n_boot))
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws)) - 1]
    return {"point": point, "ci95": [lo, hi], "excludes_zero": lo > 0 or hi < 0}


def load_gold() -> dict[str, set]:
    gold: dict[str, set] = {}
    with open(_ROOT / "data/pilot/items.jsonl") as fh:
        for line in fh:
            rec = json.loads(line)
            gold[rec["id"]] = {(f["type"], f["value"], f["polarity"])
                               for f in (rec.get("gold_facts") or [])}
    return gold


def gold_metrics(model: str, cfg: dict, extracted: dict) -> dict:
    """对每个候选 ASR 回读算 gold 保留率与幻觉率（按事实微平均）。"""
    gold = load_gold()
    corpus, _ = load_resample_corpus(str(_ROOT / cfg["pred"]),
                                     str(_ROOT / "exp/d0_resample"), model)
    ret_num = ret_den = hall_num = hall_den = 0
    n_items = 0
    for item in corpus:
        iid = item["item_id"]
        g = gold.get(iid)
        if not g:
            continue
        n_items += 1
        for slot in ("sample_keys", "sample_keys_asr2", "sample_keys_asr3"):
            pass
        for index, key in enumerate(item["sample_keys"]):
            if key not in extracted:
                continue
            facts = extracted[key].facts
            values = {(f.type, f.value, f.polarity) for f in facts}
            ret_num += len(values & g)
            ret_den += len(g)
            hall_num += len(values - g)
            hall_den += len(values)
    return {
        "n_items": n_items,
        "gold_retention": ret_num / ret_den if ret_den else None,
        "hallucination_rate": hall_num / hall_den if hall_den else None,
        "gold_facts_total": ret_den,
        "readback_facts_total": hall_den,
    }


def main() -> int:
    out: dict = {"protocol": ("plan/render loss: item-clustered bootstrap over per-item gaps "
                              "(8 candidates, 3 held-out-ASR folds); "
                              "gold: fact-micro retention vs data/pilot gold_facts"),
                 "seed": SEED, "models": {}}
    for model, cfg in MODELS.items():
        loo_path = _ROOT / f"exp/d0_resample/frr_loo_{model}.json"
        if not loo_path.exists():
            print(f"[skip] {model}: no {loo_path}")
            continue
        loo = json.loads(loo_path.read_text())
        rows = loo["per_item"]

        # 逐折：base(original) 与 selected 的按题均值 + 配对降幅区间
        per_fold = {}
        for name in ASR_NAMES:
            fold_rows = rows[name]
            base = [r["base_gap"] for r in fold_rows]
            selected = [r["selected_gap"] for r in fold_rows]
            oracle = [r["oracle_gap"] for r in fold_rows]
            random_gap = [r["random_gap"] for r in fold_rows]
            diffs = [b - s for b, s in zip(base, selected)]
            per_fold[name] = {
                "evaluator": name,
                "n_items": len(fold_rows),
                "n_candidates": fold_rows[0]["n_candidates"],
                "original_loss": bootstrap_mean(base, SEED + hash(name) % 1000),
                "selected_loss": bootstrap_mean(selected, SEED + 7 + hash(name) % 1000),
                "oracle_loss": bootstrap_mean(oracle, SEED + 11 + hash(name) % 1000),
                "random_loss": bootstrap_mean(random_gap, SEED + 13 + hash(name) % 1000),
                "reduction": bootstrap_mean(diffs, SEED + 17 + hash(name) % 1000),
            }
        cache = _ROOT / f"exp/d0_resample/facts_dual/{model}.jsonl"
        gold = {}
        if cache.exists():
            extracted = load_extractions(str(cache), prompt_sha256=PROMPT_SHA256)
            try:
                gold = gold_metrics(model, cfg, extracted)
            except Exception as exc:  # 缺少预测目录时不阻断区间部分
                gold = {"error": f"{type(exc).__name__}: {exc}"}
        out["models"][model] = {"per_fold": per_fold, "gold": gold,
                                "cache": str(cache.relative_to(_ROOT))}
        print(f"[e5] {model} folds={ {k: round(v['reduction']['point'], 4) for k, v in per_fold.items()} }"
              f" gold={gold.get('gold_retention')}", flush=True)

    reports = _ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "supplementary_e5_intervals_gold.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n")

    lines = ["# E5：表 1 区间（按题聚类 bootstrap）与金标复核", "",
             "区间为按题聚类 bootstrap 95%（10,000 次）；`original` 即表 1 的 render loss（evaluator 单识别器口径）。", "",
             "| Model | evaluator | n | original loss [95% CI] | selected loss [95% CI] | reduction [95% CI] |",
             "| --- | --- | ---: | --- | --- | --- |"]
    for model, data in out["models"].items():
        for name, fold in data["per_fold"].items():
            o, s, r = fold["original_loss"], fold["selected_loss"], fold["reduction"]
            lines.append(
                f"| {model} | {name} | {fold['n_items']} | "
                f"{100*o['point']:.2f} [{100*o['ci95'][0]:.2f}, {100*o['ci95'][1]:.2f}] | "
                f"{100*s['point']:.2f} [{100*s['ci95'][0]:.2f}, {100*s['ci95'][1]:.2f}] | "
                f"{100*r['point']:.2f} [{100*r['ci95'][0]:.2f}, {100*r['ci95'][1]:.2f}] |")
    lines += ["", "## 金标复核（事实微平均）", "",
              "| Model | n items | gold retention | hallucination rate | gold facts | readback facts |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for model, data in out["models"].items():
        g = data["gold"]
        if "error" in g:
            lines.append(f"| {model} | - | - | - | - | - |  <!-- {g['error']} -->")
        elif g.get("gold_retention") is not None:
            lines.append(f"| {model} | {g['n_items']} | {100*g['gold_retention']:.2f}% | "
                         f"{100*g['hallucination_rate']:.2f}% | {g['gold_facts_total']} | "
                         f"{g['readback_facts_total']} |")
    (reports / "supplementary_e5_intervals_gold.md").write_text("\n".join(lines) + "\n")
    print("[e5] wrote reports/supplementary_e5_intervals_gold.{json,md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
