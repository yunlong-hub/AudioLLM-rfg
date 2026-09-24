"""规则通道：从答案文本抽取 number / unit / negation 事实。

定位：规则通道只负责**可程序化判定**的部分（数字与单位），并作为冲突时的仲裁者；
专名与 content 交给 LLM 通道（`llm.py`）。这样 gold 与评测都不会被抽取器猜测污染。
"""
from __future__ import annotations

import re

from rfg.facts.schema import (_DIGIT_RE, _NEG_CUE_RE, _NUM_WORD_RE, _UNIT_RE,
                              NEGATION_CUES, Fact, canon_number, canon_proper_noun,
                              UNIT_SYNONYMS, words_to_number)

# 常见专名线索：化学元素符号（白名单，避免把句首单字母 "A" 当成元素符号）
ELEMENT_SYMBOLS = frozenset("""
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu
Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr
""".split())
_CAPITALIZED_RE = re.compile(r"\b([A-Z][a-zA-Z]{2,})\b")


def extract_numbers(text: str) -> set[Fact]:
    facts: set[Fact] = set()
    for m in _DIGIT_RE.finditer(text):
        v = canon_number(m.group(0))
        if v is not None:
            facts.add(Fact("number", v))
    for m in _NUM_WORD_RE.finditer(text):
        val = words_to_number(m.group(0))
        if val is None:
            continue
        v = str(int(val)) if abs(val - round(val)) < 1e-9 else f"{val:g}"
        facts.add(Fact("number", v))
    return facts


def extract_units(text: str) -> set[Fact]:
    return {Fact("unit", UNIT_SYNONYMS[m.group(0).lower()]) for m in _UNIT_RE.finditer(text)}


def extract_negations(text: str) -> set[Fact]:
    """否定事实：取否定线索**之前最近**的数值/专名作为被否定对象，极性记 "-"。

    对应研究方案里的"极性翻转"失败模式；答案句通常是 "<异类> is not in the category ..."。
    """
    cues = list(_NEG_CUE_RE.finditer(text))
    if not cues:
        return set()
    facts: set[Fact] = set()
    for cue in cues:
        prefix = text[: cue.start()]
        cands: list[tuple[int, str, str]] = []  # (位置, 类型, 规范值)
        for m in _DIGIT_RE.finditer(prefix):
            v = canon_number(m.group(0))
            if v is not None:
                cands.append((m.end(), "number", v))
        for m in _NUM_WORD_RE.finditer(prefix):
            val = words_to_number(m.group(0))
            if val is not None:
                v = str(int(val)) if abs(val - round(val)) < 1e-9 else f"{val:g}"
                cands.append((m.end(), "number", v))
        for m in re.finditer(r"\b[A-Z][a-z]?\b", prefix):
            if m.group(0) in ELEMENT_SYMBOLS:
                cands.append((m.end(), "proper_noun", canon_proper_noun(m.group(0))))
        if not cands:
            # 没有数字/符号时，退回到否定词前最后一个实词（如 "The crocodile is not ..."）
            words = re.findall(r"[A-Za-z]{3,}", prefix)
            if words:
                facts.add(Fact("negation", canon_proper_noun(words[-1]), "-"))
            continue
        _, ftype, value = max(cands, key=lambda c: c[0])
        facts.add(Fact("negation", value, "-"))
        # 同时对被否定对象本身保留一个正向事实（数值本身出现在答案里）
        facts.add(Fact(ftype, value, "+"))
    return facts


# 元素符号 <-> 元素名 等价：内部文本写 "Au"、回读说 "gold" 时，必须视为同一事实，
# 否则会把"说对了"误记为"结构性丢失"（实测 20/39 个所谓结构性失败由此而来）。
ELEMENT_NAME = {
    "h": "hydrogen", "he": "helium", "c": "carbon", "n": "nitrogen", "o": "oxygen",
    "ne": "neon", "na": "sodium", "si": "silicon", "s": "sulfur", "k": "potassium",
    "ca": "calcium", "fe": "iron", "ni": "nickel", "cu": "copper", "zn": "zinc",
    "ag": "silver", "sn": "tin", "w": "tungsten", "au": "gold", "hg": "mercury",
    "pb": "lead",
}


NAME_TO_SYMBOL = {v: k for k, v in ELEMENT_NAME.items()}


def extract_element_names(text: str) -> set[Fact]:
    """小写元素名（gold / neon）与其符号视为同一事实，双向登记。

    不做这一步，"内部文本写 Au、回读说 gold" 会被误判为结构性丢失（实测占此类失败的 51%）。
    """
    out: set[Fact] = set()
    for w in re.findall(r"\b([a-z]{3,12})\b", text.lower()):
        if w in NAME_TO_SYMBOL:
            out.add(Fact("proper_noun", w))
            out.add(Fact("proper_noun", NAME_TO_SYMBOL[w]))
    return out


def extract_symbols(text: str) -> set[Fact]:
    """化学元素符号（走白名单）。句首单字母 "A"/"I" 不是符号，必须排除。"""
    out: set[Fact] = set()
    for tok in re.findall(r"\b[A-Z][a-z]?\b", text):
        if tok not in ELEMENT_SYMBOLS:
            continue
        sym = canon_proper_noun(tok)
        out.add(Fact("proper_noun", sym))
        name = ELEMENT_NAME.get(sym)
        if name:                      # 同时登记元素名，令两种写法可互相匹配
            out.add(Fact("proper_noun", name))
    return out


def extract_capitalized(text: str) -> set[Fact]:
    """句首以外的首字母大写词（国家/城市/人名）。句首词不可靠，跳过。"""
    facts: set[Fact] = set()
    seen_sentence_start = True
    for tok in re.finditer(r"[A-Za-z][a-zA-Z]+|[.!?]\s+", text):
        if tok.group(0).strip() in {".", "!", "?"}:
            seen_sentence_start = True
            continue
        if seen_sentence_start:
            seen_sentence_start = False
            continue
        if tok.group(0)[0].isupper():
            facts.add(Fact("proper_noun", canon_proper_noun(tok.group(0))))
    return facts


def extract_rules(text: str, with_names: bool = True) -> set[Fact]:
    """规则通道总入口：数字 + 单位 + 否定（+ 可选专名）。"""
    facts = extract_numbers(text) | extract_units(text) | extract_negations(text)
    if with_names:
        facts |= extract_symbols(text) | extract_element_names(text) | extract_capitalized(text)
    # "%" 不带词边界，_UNIT_RE 匹配不到，单独补一条
    if "%" in text:
        facts.add(Fact("unit", "percent"))
    return facts
