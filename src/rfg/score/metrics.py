"""事实级指标：PG / RFG / RG / RFG_EF / 分项 / RFG_corr，以及区间估计。

口径（与研究方案 §2、§5 一致）
-----------------------------
* `retention(a←b) = |F(a) ∩ F(b)| / |F(b)|`，**分母是上游条件自己的事实集合**（非对称包含）；
* `PG = 1 − retention(LISTEN←READ)`
* `RFG = 1 − retention(SPEAK←LISTEN)`
* `RG = 1 − retention(ECHO←READ)`
* `RFG_EF = 1 − retention(EF←READ)`
* `RFG_hard` / `RFG_content`：按 fact type 分项重算 RFG；
* `RFG_corr`：只在双 ASR 一致的样本上重算 RFG（排除回读误差污染）；
* 统计：Wilson 区间（比例）+ **按题聚类的成对 bootstrap**（差值），避免把同一题的多个事实
  当作独立观测。
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

from rfg.facts.schema import Fact

HARD_TYPES = ("number", "unit", "proper_noun", "negation")


@dataclass
class ItemFacts:
    item_id: str
    category: str
    facts: dict[str, set[Fact]] = field(default_factory=dict)
    readback_wer: dict[str, float | None] = field(default_factory=dict)
    asr_agree: dict[str, bool] = field(default_factory=dict)

    def upstream(self, condition: str) -> set[Fact]:
        return self.facts.get(condition) or set()

    def retained(self, downstream: str, upstream: str,
                 types: tuple[str, ...] | None = None) -> tuple[int, int]:
        """本题在这一对条件上的 (命中数, 上游事实数)。

        只在**上下游都被观测到**时计入。若下游条件缺失（没有该条件的产物或
        回读失败），本题返回 (0, 0) 而不是 (0, |up|) —— 否则会把"未观测"
        当作"事实全丢"，系统性抬高 gap。Step-Audio 缺失条件较多，曾因此把
        PG 从 0.4118 抬到 0.5538。
        """
        if downstream not in self.facts or upstream not in self.facts:
            return 0, 0
        up = self.upstream(upstream)
        down = self.upstream(downstream)
        if types:
            up = {f for f in up if f.type in types}
            down = {f for f in down if f.type in types}
        if not up:
            return 0, 0
        return len(up & down), len(up)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson 区间。注意：事实在同一题内不独立，此区间偏窄；差值请用按题 bootstrap。"""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def pooled_gap(items: list[ItemFacts], downstream: str, upstream: str,
               types: tuple[str, ...] | None = None,
               only_asr_agree_conditions: tuple[str, ...] | None = None
               ) -> tuple[float | None, int, int]:
    """汇总留存缺口 1 − Σ|∩|/Σ|F_up|。返回 (gap, retained, total)。"""
    ret = tot = 0
    for it in items:
        if only_asr_agree_conditions and downstream in only_asr_agree_conditions:
            if not it.asr_agree.get(downstream, False):
                continue
        r, t = it.retained(downstream, upstream, types)
        ret += r
        tot += t
    if tot == 0:
        return None, 0, 0
    return 1 - ret / tot, ret, tot


def per_item_gap(items: list[ItemFacts], downstream: str, upstream: str,
                 types: tuple[str, ...] | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    for it in items:
        r, t = it.retained(downstream, upstream, types)
        if t:
            out[it.item_id] = 1 - r / t
    return out


def paired_bootstrap_ci(items: list[ItemFacts], cond_a: str, cond_b: str,
                        upstream_a: str, upstream_b: str, *, n_boot: int = 10_000,
                        seed: int = 20260916,
                        types: tuple[str, ...] | None = None) -> dict:
    """按题聚类的成对 bootstrap：对题目有放回重采样，重算两个 gap 之差。

    例：RFG_hard − RFG_content 用同一次重采样（配对），因此 CI 反映的是题内配对信息。
    """
    rng = random.Random(seed)
    n = len(items)
    if n == 0:
        return {"n_items": 0, "diff": None, "ci95": None}
    point_a, _, _ = pooled_gap(items, cond_a, upstream_a, types)
    point_b, _, _ = pooled_gap(items, cond_b, upstream_b, types)
    if point_a is None or point_b is None:
        return {"n_items": n, "diff": None, "ci95": None}
    diffs: list[float] = []
    for _ in range(n_boot):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        ga, _, _ = pooled_gap(sample, cond_a, upstream_a, types)
        gb, _, _ = pooled_gap(sample, cond_b, upstream_b, types)
        if ga is None or gb is None:
            continue
        diffs.append(ga - gb)
    if not diffs:
        return {"n_items": n, "diff": point_a - point_b, "ci95": None}
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]
    return {"n_items": n, "diff": point_a - point_b, "ci95": [lo, hi],
            "n_boot_effective": len(diffs),
            "excludes_zero": (lo > 0) or (hi < 0)}


def spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None

    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def type_breakdown(items: list[ItemFacts], downstream: str, upstream: str) -> dict:
    out: dict[str, dict] = {}
    for t in HARD_TYPES + ("content",):
        gap, ret, tot = pooled_gap(items, downstream, upstream, (t,))
        out[t] = {"retention": (ret / tot) if tot else None, "gap": gap,
                  "retained": ret, "total": tot,
                  "wilson_ci": wilson_ci(ret, tot) if tot else None}
    return out


def summarize(items: list[ItemFacts], *, with_bootstrap: bool = True) -> dict:
    """输出一个模型的全部主指标。"""
    res: dict = {"n_items": len(items)}
    res["PG"] = _gap_block(items, "LISTEN", "READ")
    res["RFG"] = _gap_block(items, "SPEAK", "LISTEN")
    res["RG"] = _gap_block(items, "ECHO", "READ")
    res["RFG_EF"] = _gap_block(items, "EF", "READ")
    res["RFG_hard"] = _gap_block(items, "SPEAK", "LISTEN", HARD_TYPES)
    res["RFG_content"] = _gap_block(items, "SPEAK", "LISTEN", ("content",))
    res["RFG_corr"] = _gap_block(items, "SPEAK", "LISTEN",
                                 only_asr_agree_conditions=("SPEAK",))
    res["by_type_SPEAK_vs_LISTEN"] = type_breakdown(items, "SPEAK", "LISTEN")
    res["by_type_EF_vs_READ"] = type_breakdown(items, "EF", "READ")
    if with_bootstrap:
        res["bootstrap_RFG_hard_minus_content"] = _diff_hard_vs_content(items)
        res["bootstrap_RG_vs_RFG"] = paired_bootstrap_ci(
            items, "ECHO", "SPEAK", "READ", "LISTEN")
        res["bootstrap_RFG_vs_EF"] = paired_bootstrap_ci(
            items, "SPEAK", "EF", "LISTEN", "READ")
    # 回读 WER 与 RFG 的相关性（逐题）
    x = [it.readback_wer.get("SPEAK") for it in items]
    y = per_item_gap(items, "SPEAK", "LISTEN")
    pairs = [(a, y[it.item_id]) for it, a in zip(items, x)
             if a is not None and it.item_id in y]
    res["spearman_wer_vs_rfg"] = {
        "n": len(pairs),
        "rho": spearman_rho([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None,
    }
    return res


def _gap_block(items: list[ItemFacts], downstream: str, upstream: str,
               types: tuple[str, ...] | None = None,
               only_asr_agree_conditions: tuple[str, ...] | None = None) -> dict:
    gap, ret, tot = pooled_gap(items, downstream, upstream, types,
                               only_asr_agree_conditions)
    return {"gap": gap, "retention": (ret / tot) if tot else None,
            "retained": ret, "total": tot,
            "wilson_ci_gap": (tuple(1 - h for h in reversed(wilson_ci(ret, tot)))
                              if tot else None)}


def _diff_hard_vs_content(items: list[ItemFacts]) -> dict:
    """RFG_hard − RFG_content：配对 bootstrap（同一重采样下算两个 gap）。"""
    rng = random.Random(20260916)
    n = len(items)
    if n == 0:
        return {"diff": None, "ci95": None}
    gh, _, _ = pooled_gap(items, "SPEAK", "LISTEN", HARD_TYPES)
    gc, _, _ = pooled_gap(items, "SPEAK", "LISTEN", ("content",))
    if gh is None or gc is None:
        return {"diff": None, "ci95": None}
    diffs = []
    for _ in range(10_000):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        a, _, _ = pooled_gap(sample, "SPEAK", "LISTEN", HARD_TYPES)
        b, _, _ = pooled_gap(sample, "SPEAK", "LISTEN", ("content",))
        if a is None or b is None:
            continue
        diffs.append(a - b)
    if not diffs:
        return {"diff": gh - gc, "ci95": None}
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
    return {"diff": gh - gc, "ci95": [lo, hi], "excludes_zero": (lo > 0) or (hi < 0)}


def group_by_category(items: list[ItemFacts]) -> dict[str, list[ItemFacts]]:
    out: dict[str, list[ItemFacts]] = defaultdict(list)
    for it in items:
        out[it.category].append(it)
    return dict(out)
