"""双通道事实抽取的合并与一致性统计（κ）。

合并口径（写进论文方法节）
--------------------------
* number / unit / negation：**以规则通道为准**（确定性、可复算），LLM 通道的同名事实若规范化后
  与规则一致则并入，不一致则记入 `conflict`；
* proper_noun / content：**以 LLM 通道为准**（规则只能覆盖元素符号与句首外大写词）；
* `facts` = 两通道规范化事实的并集；`extractor_conflict` = 对称差。

κ 的计算方式：把所有 (答案, 候选事实) 二元组作为编码单元，规则通道与 LLM 通道各给 0/1，
算 Cohen's κ。这是双通道一致性的直接度量，也是研究方案 §6 要求的报告项。

`extract_dual()` 是"对任意一批新文本做同一口径抽取"的唯一入口
-------------------------------------------------------------
主流水线（`scripts/facts/extract_facts.py`）只覆盖 `exp/<run>/predictions/` 下的条件文本；
重采样/重渲染等探针产生的新文本（`asr1`、改写文本、回读文本）不在其中，必须按**同一口径**
重新抽取：规则通道 ∪ LLM 通道，再 `merge()`。把这段逻辑集中在这里，避免每个探针各写一版
（历史上探针直接调 `extract_rules()`，于是纯规则通道没有 `content` 类型，制造出第二套事实口径）。
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from rfg.facts.rules import extract_rules
from rfg.facts.schema import Fact

RULE_AUTHORITATIVE = ("number", "unit", "negation")

# LLM 通道默认抽取模型：与 scripts/facts/extract_facts.py 的 --llm-model 默认值一致。
DEFAULT_LLM_MODEL = "/workspace/yunlong/LLM/pretrain_model/Qwen/Qwen2.5-7B-Instruct"


@dataclass
class Extraction:
    text: str
    facts_rules: set[Fact] = field(default_factory=set)
    facts_llm: set[Fact] = field(default_factory=set)
    facts: set[Fact] = field(default_factory=set)
    conflict: set[Fact] = field(default_factory=set)          # 值级分歧（真正的不一致）
    type_conflicts: set[Fact] = field(default_factory=set)    # 类型标签差异（非内容分歧）
    llm_error: str | None = None

    def as_dict(self) -> dict:
        return {
            "facts_rules": _sorted_dicts(self.facts_rules),
            "facts_llm": _sorted_dicts(self.facts_llm),
            "facts": _sorted_dicts(self.facts),
            "conflict": _sorted_dicts(self.conflict),
            "type_conflicts": _sorted_dicts(self.type_conflicts),
            "llm_error": self.llm_error,
        }


def _sorted_dicts(facts) -> list[dict]:
    """稳定的落盘顺序（dict 之间不可直接比较，必须给 key）。"""
    return sorted((f.as_dict() for f in facts),
                  key=lambda d: (d["type"], d["value"], d["polarity"]))


def merge(text: str, facts_rules: set[Fact], facts_llm: set[Fact],
          llm_error: str | None = None) -> Extraction:
    """按 **(值, 极性)** 合并两通道（类型标签不同不算内容分歧）。

    为什么按值合并：同一概念常被两通道打上不同类型标签（"sun" 规则通道记 proper_noun、
    LLM 通道记 content）。若按 (type, value) 比较，会把"标签差异"误报成"抽取分歧"——
    D0 首轮实测 κ = −0.42、冲突率 0.60，绝大部分来自这种假分歧。

    为什么必须带上极性：否定事实与它否定的正向事实**共享同一个值**
    （"Ninety-five is not in the category of even numbers." → `number:95:+` 与
    `negation:95:-`）。只按 value 合并会让两者互相覆盖，而覆盖结果取决于 `set[Fact]` 的
    迭代顺序；`Fact` 是 frozen dataclass、哈希由字符串字段决定，迭代顺序随
    PYTHONHASHSEED 变化——于是同一文本在不同进程里会得到不同事实集（实测约 5% 的文本
    受影响，negation 事实被随机吞掉）。带上极性后键唯一，合并结果与进程无关。

    类型归属优先级：规则通道的 number/unit/negation 更可靠；其余用 LLM 通道类型；
    都不满足时退回规则通道类型。
    """
    by_key_rules = {(f.value, f.polarity): f for f in facts_rules}
    by_key_llm = {(f.value, f.polarity): f for f in facts_llm}

    merged: dict[tuple[str, str], Fact] = {}
    type_conflicts: set[Fact] = set()
    for key in set(by_key_rules) | set(by_key_llm):
        fr, fl = by_key_rules.get(key), by_key_llm.get(key)
        if fr and fl and fr.type != fl.type:
            type_conflicts.add(fr)
        if fr and fr.type in RULE_AUTHORITATIVE:
            merged[key] = fr
        elif fl:
            merged[key] = fl
        elif fr:
            merged[key] = fr

    value_conflict = {f for k, f in by_key_rules.items() if k not in by_key_llm} | \
                     {f for k, f in by_key_llm.items() if k not in by_key_rules}
    return Extraction(text=text, facts_rules=set(facts_rules), facts_llm=set(facts_llm),
                      facts=set(merged.values()), conflict=value_conflict,
                      type_conflicts=type_conflicts, llm_error=llm_error)


def cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float | None:
    """标准 Cohen's κ。样本为空或退化（全同类）时返回 None，不猜测。"""
    n = len(labels_a)
    if n == 0 or len(labels_a) != len(labels_b):
        return None
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    pa1 = sum(labels_a) / n
    pb1 = sum(labels_b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if abs(1 - pe) < 1e-12:
        return None
    return (po - pe) / (1 - pe)


def kappa_over_extractions(extractions: list[Extraction],
                           shared_types: tuple[str, ...] = ("number", "unit", "proper_noun"),
                           ) -> dict:
    """一致性统计。

    **κ 只在两通道共同负责的类型上计算**（默认 number/unit/proper_noun）。
    原因：本设计的两个通道是**互补分工**——规则通道负责数字/单位/否定，LLM 通道额外负责
    content。若在并集上算 κ，内容类事实天然只有 LLM 一侧为 1，会造成系统性负相关，
    把"分工"误报成"分歧"（首轮实测 κ=−0.42 即为此假象）。

    同时报告：值级冲突率（含所有类型）、类型标签差异率、以及 content 类覆盖（LLM 专属）。
    """
    a: list[int] = []
    b: list[int] = []
    n_value_conflict = n_type_conflict = n_union = 0
    n_content_rules = n_content_llm = 0
    for ex in extractions:
        vals_r = {f.value for f in ex.facts_rules}
        vals_l = {f.value for f in ex.facts_llm}
        n_union += len(vals_r | vals_l)
        n_value_conflict += len(vals_r ^ vals_l)
        n_type_conflict += len(ex.type_conflicts)
        n_content_rules += sum(1 for f in ex.facts_rules if f.type == "content")
        n_content_llm += sum(1 for f in ex.facts_llm if f.type == "content")
        # κ 的编码单元：仅限共同负责类型
        vals_r_shared = {f.value for f in ex.facts_rules if f.type in shared_types}
        vals_l_shared = {f.value for f in ex.facts_llm if f.type in shared_types}
        for v in vals_r_shared | vals_l_shared:
            a.append(1 if v in vals_r_shared else 0)
            b.append(1 if v in vals_l_shared else 0)
    return {
        "kappa_shared_types": list(shared_types),
        "n_units": len(a),
        "n_candidate_values": n_union,
        "n_value_conflicts": n_value_conflict,
        "n_type_conflicts": n_type_conflict,
        "value_conflict_rate": (n_value_conflict / n_union) if n_union else None,
        "type_conflict_rate": (n_type_conflict / n_union) if n_union else None,
        "content_facts_rules": n_content_rules,
        "content_facts_llm": n_content_llm,
        "kappa": cohen_kappa(a, b),
    }


# ---------------------------------------------------------------- 任意文本的双通道抽取

def extraction_key(item_id: str, condition: str) -> str:
    """缓存键。命名与 `extract_facts.py` 的 (item_id, condition) 对齐，
    因此主口径产物 `exp/<run>/facts/<model>.jsonl` 可以直接当作 priors 复用。"""
    return f"{item_id}|{condition}"


def text_hash(text: str | None) -> str:
    """文本指纹（16 hex），缓存据此判定一行记录是否对应同一段文本。"""
    return hashlib.sha256((text or "").encode()).hexdigest()[:16]


def facts_from_dicts(dicts: Iterable[dict] | None) -> set[Fact]:
    """把落盘的事实字典列表还原成事实集合。"""
    return {Fact(d["type"], d["value"], d.get("polarity", "+")) for d in (dicts or [])}


def _row_key(row: dict) -> str | None:
    """一行缓存记录的主键：`extract_dual` 写 `key`，`extract_facts.py` 写 item_id+condition。"""
    if row.get("key"):
        return str(row["key"])
    if row.get("item_id") and row.get("condition"):
        return extraction_key(str(row["item_id"]), str(row["condition"]))
    return None


def _extraction_from_row(row: dict) -> Extraction:
    """从落盘行重建 `Extraction`。

    `facts` **不**直接取行里的字段，而是用行里的两通道事实重算 `merge()`：行里的 `facts`
    可能是旧版"只按 value 合并"的逻辑写的，那种写法会随机吞掉同值不同极性的事实
    （`number:95:+` / `negation:95:-`）。重算保证读到的合并集与当前口径一致且可复现。
    """
    facts_rules = facts_from_dicts(row.get("facts_rules"))
    facts_llm = facts_from_dicts(row.get("facts_llm"))
    return merge(row.get("text") or "", facts_rules, facts_llm, row.get("llm_error"))


def load_extractions(path: str, *, prompt_sha256: str | None = None,
                     keep_llm: bool = True) -> dict[str, Extraction]:
    """读回 jsonl 抽取产物：键 -> `Extraction`。

    兼容两种行格式：
    * `extract_dual()` 落盘的行（含 `key`）；
    * `scripts/facts/extract_facts.py` 落盘的行（含 `item_id` + `condition`，
      键按 `item_id|condition` 还原）——因此主口径产物可直接作为 `priors` 复用。

    `prompt_sha256` 非空时，指纹不一致的行整体丢弃（返回里没有该键），由调用方重抽；
    这样提示词变更不会被静默沿用。文本完整性（`text_hash`）由调用方按需校验。
    """
    out: dict[str, Extraction] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = _row_key(row)
            if key is None:
                continue
            if prompt_sha256 is not None and row.get("prompt_sha256") != prompt_sha256:
                continue
            ex = _extraction_from_row(row)
            if not keep_llm:
                ex.facts_llm = set()
                ex.facts = ex.facts_rules
            out[key] = ex
    return out


def extract_dual(
    texts: Mapping[str, str] | Sequence[tuple[str, str]],
    *,
    cache_path: str | None = None,
    llm_model: str | None = DEFAULT_LLM_MODEL,
    device: str = "cuda:0",
    batch_size: int = 16,
    force: bool = False,
    priors: Mapping[str, Extraction] | None = None,
    llm=None,
    log=print,
) -> dict[str, Extraction]:
    """对一批**新文本**做双通道抽取（规则 ∪ LLM），返回 键 -> `Extraction`。

    这是重采样/重渲染等探针唯一允许的抽取入口：主流水线产物
    （`exp/<run>/facts/<model>.jsonl`）覆盖不到这些文本，若改用 `extract_rules()` 就会得到
    没有 `content` 类型的纯规则集，与论文口径分裂。

    参数
    ----
    texts : 键 -> 文本。键建议用 `extraction_key(item_id, condition)`，与主口径对齐。
        文本为 None 时按空串处理（事实集为空），不报错。
    cache_path : jsonl 落盘位置（如 `exp/d0_resample/facts_dual/<model>.jsonl`）。
        为 None 则不落盘。缓存行记录 `text_hash` 与 `prompt_sha256`：
        文本变了或提示词变了都会触发重抽；规则通道则**总是**重算（便宜且确定性）。
    llm_model / device / batch_size : LLM 通道的模型、设备与批大小；`llm` 传入已构造的
        `LlmExtractor` 时优先使用它（跨多次调用共享一份权重）。两者都没有而又有待抽文本时报错，
        不静默退化成纯规则集。
    force : 忽略缓存与 priors，全部重抽。
    priors : 键 -> 已可信的 `Extraction`（通常由 `load_extractions()` 读主口径产物得到）。
        键命中且 `text` 逐字相同时直接复用，不跑 GPU。

    返回的字典与 `texts` 键集合、顺序一致。
    """
    pairs: list[tuple[str, str]] = list(texts.items()) if isinstance(texts, Mapping) else list(texts)
    seen: set[str] = set()
    ordered: list[tuple[str, str, str]] = []  # (key, text, origin)
    for key, text in pairs:
        key = str(key)
        if key in seen:
            raise ValueError(f"extract_dual: 重复的键 {key!r}")
        seen.add(key)
        ordered.append((key, "" if text is None else str(text), ""))

    cache: dict[str, dict] = {}
    if cache_path and os.path.exists(cache_path) and not force:
        cache = _load_rows(cache_path)
        log(f"[extract_dual] 缓存命中候选 {len(cache)} 行 <- {cache_path}")

    from rfg.facts.llm import PROMPT_SHA256

    todo: list[tuple[str, str]] = []          # 需要新抽 LLM 事实的 (key, text)
    llm_results: dict[str, Extraction] = {}   # 复用得来的事实（priors / 缓存）
    origins: dict[str, str] = {}

    for key, text, _ in ordered:
        if not force and priors is not None:
            prior = priors.get(key)
            if prior is not None and prior.text == text:
                llm_results[key] = prior
                origins[key] = "prior"
                continue
        row = None if force else cache.get(key)
        if row is not None and row.get("text_hash") == text_hash(text) \
                and row.get("prompt_sha256") == PROMPT_SHA256:
            llm_results[key] = _extraction_from_row(row)
            origins[key] = "cache"
            continue
        todo.append((key, text))

    if todo:
        if llm is None:
            if not llm_model:
                raise ValueError(
                    f"extract_dual: 有 {len(todo)} 条文本需要 LLM 通道，但既未给 llm_model 也未给 llm")
            from rfg.facts.llm import LlmExtractor

            log(f"[extract_dual] 加载 LLM 抽取器 {llm_model}（待抽 {len(todo)} 条，batch={batch_size}）")
            llm = LlmExtractor(llm_model, device=device, batch_size=batch_size)
        log(f"[extract_dual] LLM 抽取 {len(todo)} 条 ...")
        parsed = llm.extract_batch([t for _, t in todo])
        for (key, text), (facts_llm, err) in zip(todo, parsed):
            llm_results[key] = Extraction(text=text, facts_llm=facts_llm, llm_error=err)
            origins[key] = "fresh"
        log(f"[extract_dual] LLM 抽取完成 prompt_sha256={PROMPT_SHA256[:12]}")

    llm_name = getattr(llm, "name", None) if llm is not None else None
    if llm_name is None and llm_model:
        llm_name = os.path.basename(str(llm_model).rstrip("/"))

    out: dict[str, Extraction] = {}
    rows: list[dict] = []
    for key, text, _ in ordered:
        base = llm_results[key]
        ex = merge(text, extract_rules(text), set(base.facts_llm), base.llm_error)
        out[key] = ex
        rows.append({"key": key, "text": text, "text_hash": text_hash(text),
                     "prompt_sha256": PROMPT_SHA256, "fact_origin": origins[key],
                     "llm_model": llm_name if origins[key] == "fresh" else cache.get(key, {}).get("llm_model"),
                     **ex.as_dict()})

    if cache_path:
        # 按键**合并**写回：调用方可能只传缓存的一个子集（例如某次只判定部分题目），
        # 直接覆盖会把其余键从缓存里抹掉（曾把 1000 行截成 0 行）。
        merged: dict[str, dict] = {}
        if os.path.exists(cache_path):
            merged.update(_load_rows(cache_path))
        for r in rows:
            merged[r["key"]] = r
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w") as fh:
            for r in merged.values():
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_fresh = sum(1 for r in rows if r["fact_origin"] == "fresh")
        log(f"[extract_dual] {len(rows)} 条（含 {n_fresh} 条新抽）-> {cache_path}"
            f"（缓存共 {len(merged)} 键）")
    return out


def _load_rows(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = _row_key(row)
            if key is not None:
                out[key] = row
    return out
