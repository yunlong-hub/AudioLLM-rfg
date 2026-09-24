#!/usr/bin/env python3
"""数值审计：论文中每个数值的唯一权威来源。

口径（务必与流水线一致）
-----------------------
所有事实集合一律取自 `exp/<run>/facts/<model>.jsonl` —— 即
`scripts/facts/extract_facts.py` 产出的**双通道合并集**（规则通道 ∪ LLM 通道），
与 `scripts/facts/score.py`、`scripts/analyze/probe_*.py` 完全同源。
上游侧（内部文本）用 `<COND>#internal`，下游侧（语音回读）用 `<COND>`。

> 历史教训：本脚本曾用 `extract_rules()` 从文本**只按规则通道**重算，于是
> 得到一套与全部探针/打分都不一致的数值（例如 30B pilot 感知损失 0.1778 vs
> 探针的 0.2078），并被误当作"权威"去改论文。纯规则通道没有 `content` 类型，
> 而论文的"硬事实 vs 内容事实"拆分依赖该类型，故合并集才是论文口径。

本脚本还会把重算结果与流水线自己落盘的
`exp/<run>/metrics/probe_ef.json` 逐项对照，出现分歧即报警——这正是
"报告与论文数值对不上"这类问题的自动哨兵。

用法： python tools/audit_numbers.py [--out reports/numbers_audit.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from rfg.facts.readback import READBACK_MODES, compose_readback  # noqa: E402
from rfg.facts.schema import Fact  # noqa: E402
from rfg.score.textnorm import normalize_text  # noqa: E402

INTERNAL = "#internal"


# ---------------------------------------------------------------- 载入

def load_facts(run: str, model: str, readback_mode: str) -> dict[str, dict[str, set[Fact]]]:
    """双通道合并事实集：facts[iid][cond] -> {Fact}。"""
    out: dict[str, dict[str, set[Fact]]] = defaultdict(dict)
    path = os.path.join(_ROOT, "exp", run, "facts", f"{model}.jsonl")
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            out[r["item_id"]][r["condition"]] = {
                Fact(d["type"], d["value"], d.get("polarity", "+")) for d in r.get("facts", [])
            }
    return {iid: compose_readback(values, readback_mode) for iid, values in out.items()}


def load_texts(run: str, model: str) -> dict[str, dict[str, str]]:
    """各条件的文本（用于噪声地板的"文本完全相同"判定）。"""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    base = os.path.join(_ROOT, "exp", run, "predictions", model)
    for idir in glob.glob(os.path.join(base, "*/")):
        iid = os.path.basename(idir.rstrip("/"))
        for f in glob.glob(idir + "*.json"):
            try:
                rec = json.load(open(f))
            except Exception:
                continue
            cond = os.path.basename(f)[:-5]
            if cond in ("SPEAK", "ECHO", "EF", "EFA", "EFB", "EFW", "SPEAKD"):
                out[iid][cond] = rec.get("text") or ""
                out[iid][cond + INTERNAL] = rec.get("text") or ""
            else:
                out[iid][cond] = rec.get("text") or ""
    return out


def asr_agree(run: str, model: str) -> dict[str, bool]:
    """SPEAK 的双 ASR 是否一致（RFG_corr 只用一致样本）。"""
    out = {}
    base = os.path.join(_ROOT, "exp", run, "predictions", model)
    for f in glob.glob(os.path.join(base, "*", "SPEAK.json")):
        try:
            rec = json.load(open(f))
        except Exception:
            continue
        iid = os.path.basename(os.path.dirname(f))
        out[iid] = bool((rec.get("readback") or {}).get("asr_agree"))
    return out


# ---------------------------------------------------------------- 计算

def gap(facts, iids, down: str, up: str) -> dict:
    """1 − |F(down) ∩ F(up)| / |F(up)|，按题汇总。"""
    R = T = 0
    n = 0
    for iid in iids:
        fup = facts.get(iid, {}).get(up) or set()
        fdn = facts.get(iid, {}).get(down) or set()
        if not fup:
            continue
        n += 1
        R += len(fup & fdn)
        T += len(fup)
    return {"gap": (1 - R / T) if T else None, "retained": R, "total": T, "n_pairs": n}


def decompose(facts, texts, agree, items) -> dict:
    """与 probe_ef.py 完全一致的样本集与口径：全部量都在同一个 items 上计算。

    items 的定义是"facts 中四键齐全"（READ/LISTEN/SPEAK/SPEAK#internal），
    缺少回读的题会被整体排除，而不是被当作全损。
    """
    out = {
        "perception": gap(facts, items, "LISTEN", "READ"),
        "plan": gap(facts, items, "SPEAK" + INTERNAL, "LISTEN"),
        "render_SPEAK": gap(facts, items, "SPEAK", "SPEAK" + INTERNAL),
        "total_SPEAK": gap(facts, items, "SPEAK", "LISTEN"),
        "comply_EF": gap(facts, items, "EF" + INTERNAL, "READ"),
        "comply_ECHO": gap(facts, items, "ECHO" + INTERNAL, "READ"),
    }
    # 各语音条件的纯渲染损失。注意 total_SPEAK 是"共同样本集上的 SPEAK←LISTEN"，
    # 与 score.py 的主指标 RFG（各题按可用配对计）不是同一个集合，数值可略有差异；
    # 主指标见本文件末尾"主指标"一节，勿把 total_SPEAK 当作 RFG 引用。
    for cond in ("ECHO", "EF", "EFA", "EFB", "EFW"):
        out[f"render_{cond}"] = gap(facts, items, cond, cond + INTERNAL)
    corr = [i for i in items if agree.get(i)]
    out["RFG_corr"] = gap(facts, corr, "SPEAK", "LISTEN")

    # 噪声地板：规范化后文本完全相同的 (READ, EF#internal) 对，事实集仍有多少差异
    # 定义与 probe_ef.py 一致（Jaccard，而非 retention）
    vals, n_same = [], 0
    for iid in items:
        tr = normalize_text(texts.get(iid, {}).get("READ"))
        te = normalize_text(texts.get(iid, {}).get(("EF" + INTERNAL)))
        if not tr or tr != te:
            continue
        n_same += 1
        a = facts.get(iid, {}).get("READ") or set()
        b = facts.get(iid, {}).get("EF" + INTERNAL) or set()
        if a | b:
            vals.append(1 - len(a & b) / len(a | b))
    out["noise_floor"] = {
        "false_loss": sum(vals) / len(vals) if vals else None,
        "n_identical_pairs": n_same,
    }
    return out


def taxonomy(facts, items, cond="SPEAK") -> dict:
    """SPEAK 失败解剖，分类逻辑与 probe_taxonomy.py 一致。"""
    c = Counter()
    n_up = n_lost = 0
    for iid in items:
        fu = facts.get(iid, {}).get(cond + INTERNAL) or set()
        fd = facts.get(iid, {}).get(cond) or set()
        if not fu:
            continue
        n_up += len(fu)
        for f in fu - fd:
            same_t = [x for x in fd if x.type == f.type]
            if any(x.value == f.value and x.polarity != f.polarity for x in fd):
                c["polarity_flip"] += 1
            elif not same_t:
                c["deletion"] += 1
            elif f.type == "number":
                try:
                    a = float(str(f.value).replace(",", ""))
                    best = min(
                        (abs(a - float(str(x.value).replace(",", ""))) / max(abs(a), 1e-9)
                         for x in same_t
                         if str(x.value).replace(",", "").replace(".", "").isdigit()),
                        default=None,
                    )
                except Exception:
                    best = None
                c["number_drift" if (best is not None and best < 0.5) else "substitution"] += 1
            else:
                c["substitution"] += 1
            n_lost += 1
    return {
        "n_upstream": n_up,
        "n_lost": n_lost,
        "types": {k: {"n": v, "share": v / n_lost if n_lost else None} for k, v in c.most_common()},
    }


# ---------------------------------------------------------------- 交叉校验

CHECKED = ("perception", "plan", "render_SPEAK", "total_SPEAK")


def load_headline(run: str, model: str, readback_mode: str) -> dict:
    """主指标：直接取 score.py 落盘的汇总（与分解口径不同，见各表表头）。"""
    suffix = "" if readback_mode == "asr1" else f"_{readback_mode}"
    p = os.path.join(_ROOT, "exp", run, "metrics", f"scores_{model}{suffix}.json")
    if not os.path.exists(p):
        return {}
    try:
        d = json.load(open(p))
    except Exception:
        return {}
    keys = ("PG", "RFG", "RG", "RFG_EF", "RFG_hard", "RFG_content", "RFG_corr")
    out = {k: d[k] for k in keys if isinstance(d.get(k), dict)}
    b = d.get("bootstrap_RFG_hard_minus_content")
    if isinstance(b, dict):
        out["hard_minus_content"] = b
    w = d.get("spearman_wer_vs_rfg")
    if isinstance(w, dict):
        out["rho_wer_rfg"] = w
    return out


def crosscheck(run: str, model: str, dec: dict, readback_mode: str) -> list[str]:
    """把重算值与流水线落盘的 probe_ef.json 对照，返回分歧说明。"""
    suffix = "" if readback_mode == "asr1" else f"_{readback_mode}"
    p = os.path.join(_ROOT, "exp", run, "metrics", f"probe_ef{suffix}.json")
    if not os.path.exists(p):
        return [f"{run}/{model}: 缺少 {os.path.relpath(p, _ROOT)}，无法交叉校验"]
    try:
        ref = json.load(open(p)).get("models", {}).get(model, {})
    except Exception as e:
        return [f"{run}/{model}: probe_ef.json 解析失败 {e}"]
    if not ref:
        return [f"{run}/{model}: probe_ef.json 中无该模型"]
    bad = []
    for k in CHECKED:
        a, b = dec.get(k, {}).get("gap"), (ref.get("delta_" + k) or {}).get("gap")
        if a is None or b is None:
            continue
        if abs(a - b) > 5e-4:
            bad.append(f"{run}/{model}: {k} 审计={a:.4f} vs probe_ef.json={b:.4f}")
    return bad


# ---------------------------------------------------------------- 主流程

MODELS = {
    "d0_pilot": ["Qwen3-Omni-30B-A3B-Instruct", "Qwen2.5-Omni-3B"],
    "d1_main": ["Qwen3-Omni-30B-A3B-Instruct", "Qwen2.5-Omni-3B"],
    "d2_stepaudio": ["Step-Audio-2-mini"],
}

AUX_MODELS = {
    # Independent serving stack; keep outside the default cross-model audit.
    "d3_7b_stack": ["Qwen2.5-Omni-7B"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--readback-mode", choices=READBACK_MODES, default="asr1")
    ap.add_argument("--include-aux", action="store_true")
    args = ap.parse_args()
    suffix = "" if args.readback_mode == "asr1" else f"_{args.readback_mode}"
    output = args.out or f"reports/numbers_audit{suffix}.md"

    lines = [
        f"# 数值审计（回读口径: {args.readback_mode}）", "",
        "本文件是论文所有数值的**唯一权威来源**。",
        "事实集合取自 `exp/<run>/facts/<model>.jsonl`（规则 ∪ LLM 合并集），"
        "与 `score.py`、`probe_*.py` 同源；并与流水线落盘的 `probe_ef.json` 交叉校验。", "",
    ]
    audit, mismatches = {}, []
    selected = dict(MODELS)
    if args.include_aux:
        selected.update(AUX_MODELS)
    for run, models in selected.items():
        for m in models:
            fpath = os.path.join(_ROOT, "exp", run, "facts", f"{m}.jsonl")
            if not os.path.exists(fpath):
                continue
            facts = load_facts(run, m, args.readback_mode)
            texts = load_texts(run, m)
            agree = asr_agree(run, m)
            # 与 probe_ef.py 同一套样本：facts 中四键齐全者
            items = sorted(i for i in facts if all(
                c in facts[i] for c in ("READ", "LISTEN", "SPEAK", "SPEAK" + INTERNAL)))
            if not items:
                # Some auxiliary runs predate the requested ASR channel.  They
                # remain valid for their original mode but do not enter a
                # multi-ASR audit without the corresponding observations.
                continue
            dec = decompose(facts, texts, agree, items)
            tax = taxonomy(facts, items)
            head = load_headline(run, m, args.readback_mode)
            audit[f"{run}/{m}"] = {"decompose": dec, "taxonomy": tax,
                                   "headline": head, "n_items": len(items)}
            mismatches += crosscheck(run, m, dec, args.readback_mode)

            lines += [f"## {run} / {m}", "",
                      f"分解链（共同样本集，四条件齐全，n={len(items)}）：", "",
                      "| 量 | gap | retained/total | n |", "|---|---:|---|---:|"]
            for k, v in dec.items():
                if k == "noise_floor":
                    fl = v["false_loss"]
                    lines.append(f"| noise_floor | {fl:.4f} | n_identical={v['n_identical_pairs']} | — |"
                                 if fl is not None else
                                 f"| noise_floor | NA | n_identical={v['n_identical_pairs']} | — |")
                else:
                    g = v["gap"]
                    shown = f"{g:.4f}" if g is not None else "NA"
                    lines.append(f"| `{k}` | {shown} | {v['retained']}/{v['total']} | {v['n_pairs']} |")

            if head:
                lines += ["", "主指标（各题按可用配对计，来自 `score.py`；**论文引用主指标时用这一节**）：", "",
                          "| 指标 | gap | retained/total |", "|---|---:|---|"]
                for k in ("PG", "RFG", "RG", "RFG_EF", "RFG_hard", "RFG_content", "RFG_corr"):
                    v = head.get(k)
                    if not v:
                        continue
                    g = v.get("gap")
                    shown = f"{g:.4f}" if g is not None else "NA"
                    lines.append(f"| `{k}` | {shown} | {v.get('retained')}/{v.get('total')} |")
                b = head.get("hard_minus_content")
                if b and b.get("diff") is not None:
                    lo, hi = b["ci95"]
                    lines.append(f"| `RFG_hard − RFG_content` | {b['diff']:+.4f} "
                                 f"[{lo:+.4f}, {hi:+.4f}] | 排除 0: {b['excludes_zero']} |")
                w = head.get("rho_wer_rfg")
                if w and w.get("rho") is not None:
                    lines.append(f"| `ρ(WER, RFG)` | {w['rho']:.3f} (n={w.get('n')}) | — |")

            n_up, n_lost = tax["n_upstream"], tax["n_lost"]
            lines += ["", f"失败解剖（SPEAK，丢失 {n_lost}/{n_up} = "
                          f"{(n_lost / n_up if n_up else 0):.1%}）：", "",
                      "| 类型 | 条数 | 占丢失 |", "|---|---:|---:|"]
            for t, v in tax["types"].items():
                lines.append(f"| `{t}` | {v['n']} | {v['share']:.1%} |")
            lines.append("")

    lines += ["## 与流水线落盘结果的一致性", ""]
    if mismatches:
        lines += ["> **发现分歧**（说明报告或产物已过期，需重跑对应探针）：", ""]
        lines += [f"- {x}" for x in mismatches]
    else:
        lines.append("全部核检量与本仓库 `exp/<run>/metrics/probe_ef.json` 一致（容差 5e-4）。")
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    open(output, "w").write("\n".join(lines) + "\n")
    json.dump({"runs": audit, "mismatches": mismatches},
              open(os.path.join(_ROOT, "reports", f"numbers_audit{suffix}.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\n".join(lines))
    print(f"written -> {output} | reports/numbers_audit{suffix}.json")
    if mismatches:
        print(f"\n!! {len(mismatches)} 处分歧", file=sys.stderr)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
