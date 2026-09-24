"""事实元组与规范化：所有跨条件比较都必须经过这里，否则数字/单位的书写差异会伪造出"丢失"。

事实定义（研究方案 §2）：`f = (type, value, polarity)`，`type ∈ {number, unit, proper_noun, negation, content}`。

规范化的必要性（实测教训）
--------------------------
模型写 "2,500"，Whisper 回读成 "2,500"，Seamless 回读成 "two thousand five hundred"。
若不规范化，同一事实会被判成"丢失"。因此：
* number → 规范十进制字符串（去千分位；词形数字解析成数值）；
* unit   → 规范单位名（kilometers/kilometre/km → kilometer）；
* proper_noun → 小写去标点（Hg → hg）；
* negation → 被否定的值 + polarity "-"；
* content → 小写、去标点、去复数（轻量词干）。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

# ------------------------------------------------------------------ 词形数字
_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}

_NUM_WORD_RE = re.compile(
    # 结尾的 \b 至关重要：否则 "seventeen" 会被前缀匹配成 "seven"→7，
    # "fourteen"→4、"sixteen"→6、"nineteen"→9 同理出错。
    r"\b(?:(?:" + "|".join(list(_ONES) + list(_TENS) + list(_SCALES)) + r")[\s-]*)+"
    r"(?:point(?:\s+(?:" + "|".join(_ONES) + r"))+)?"
    # "two and a half" / "three and a quarter" 这类口语分数形式。
    # 注意不能写 \s+and：前面的 [\s-]* 已经把空格吃掉，会导致该可选组永不命中。
    r"(?:and\s+a\s+(?:half|quarter))?\b",
    re.IGNORECASE,
)
_DIGIT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def words_to_number(text: str) -> float | None:
    """把英文词形数字解析成数值；无法解析返回 None。支持 "two thousand five hundred"、
    "twenty-five"、"two point five"、"ninety five"、"two and a half" 等常见口语形式。"""
    text = text.lower().replace("-", " ").strip()
    # 口语分数："two and a half" → 2.5；"a half" → 0.5
    frac_map = {"half": 0.5, "quarter": 0.25}
    m = re.search(r"\band\s+a\s+(half|quarter)\b", text)
    if m:
        base = words_to_number(re.sub(r"\band\s+a\s+(half|quarter)\b", "", text).strip())
        base = 0.0 if base is None else base
        return base + frac_map[m.group(1)]
    if text in ("a half", "half"):
        return 0.5
    if text in ("a quarter", "quarter"):
        return 0.25
    if "point" in text:
        whole, _, frac = text.partition("point")
        w = words_to_number(whole)
        digits = "".join(str(_ONES[t]) for t in frac.split() if t in _ONES)
        if w is None or not digits:
            return None
        return float(f"{int(w)}.{digits}")
    total = 0
    current = 0
    seen = False
    for tok in text.split():
        if tok == "and":
            continue
        if tok in _ONES:
            current += _ONES[tok]
            seen = True
        elif tok in _TENS:
            current += _TENS[tok]
            seen = True
        elif tok in _SCALES:
            scale = _SCALES[tok]
            if scale == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale
                current = 0
            seen = True
        else:
            return None
    return float(total + current) if seen else None


def canon_number(raw: str) -> str | None:
    """数字规范形：去掉千分位与尾随句点，整数值不带小数点。"""
    raw = raw.strip().rstrip(".").replace(",", "")
    try:
        v = float(raw)
    except (ValueError, OverflowError):
        return None
    # 超长数字串会溢出成 inf（Step-Audio 输出里出现过），必须拒绝而不是崩溃
    if not math.isfinite(v):
        return None
    return str(int(v)) if abs(v - round(v)) < 1e-9 else f"{v:g}"


# ------------------------------------------------------------------ 单位同义
UNIT_SYNONYMS: dict[str, str] = {}
for _canon, _alts in {
    "meter": ["meter", "meters", "metre", "metres", "m"],
    "kilometer": ["kilometer", "kilometers", "kilometre", "kilometres", "km"],
    "centimeter": ["centimeter", "centimeters", "centimetre", "cm"],
    "millimeter": ["millimeter", "millimeters", "mm"],
    "kilogram": ["kilogram", "kilograms", "kilo", "kilos", "kg"],
    "gram": ["gram", "grams", "g"],
    "milligram": ["milligram", "milligrams", "mg"],
    "liter": ["liter", "liters", "litre", "litres", "l"],
    "milliliter": ["milliliter", "milliliters", "millilitre", "ml"],
    "hour": ["hour", "hours", "hr", "hrs", "h"],
    "minute": ["minute", "minutes", "min"],
    "second": ["second", "seconds", "sec", "secs", "s"],
    "day": ["day", "days"],
    "week": ["week", "weeks"],
    "month": ["month", "months"],
    "year": ["year", "years"],
    "gigabyte": ["gigabyte", "gigabytes", "gb"],
    "megabyte": ["megabyte", "megabytes", "mb"],
    "kilobyte": ["kilobyte", "kilobytes", "kb"],
    "byte": ["byte", "bytes"],
    "percent": ["percent", "per cent", "percentage", "%"],
    "kilometer per hour": ["kilometer per hour", "kilometers per hour", "km/h", "kph"],
    "meter per second": ["meter per second", "meters per second", "m/s"],
    "degree celsius": ["degree celsius", "degrees celsius", "celsius", "°c"],
    "degree fahrenheit": ["degree fahrenheit", "degrees fahrenheit", "fahrenheit", "°f"],
}.items():
    for _a in _alts:
        UNIT_SYNONYMS[_a] = _canon

_UNIT_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(u) for u in UNIT_SYNONYMS), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# ------------------------------------------------------------------ 否定线索
NEGATION_CUES = ("not", "isn't", "is not", "aren't", "are not", "except", "excluding",
                 "doesn't", "does not", "cannot", "can't", "no", "never", "none")
_NEG_CUE_RE = re.compile(r"\b(" + "|".join(re.escape(c) for c in NEGATION_CUES) + r")\b",
                         re.IGNORECASE)

_STOPWORDS = frozenset("""
a an the is are was were be been being of in on at to for from with without and or but if
it its this that these those as by than then so such not no do does did done have has had
there here what which who whom whose when where why how i you he she they we me my your his
her their our us them answer question following category called known also more most very
""".split())


@dataclass(frozen=True, order=True)
class Fact:
    type: str
    value: str
    polarity: str = "+"

    def as_dict(self) -> dict:
        return {"type": self.type, "value": self.value, "polarity": self.polarity}

    @staticmethod
    def from_dict(d: dict) -> "Fact":
        return Fact(type=str(d["type"]), value=str(d["value"]), polarity=str(d.get("polarity", "+")))


def canon_proper_noun(raw: str) -> str:
    return re.sub(r"[^a-z0-9]", "", raw.lower())


def canon_content(raw: str) -> str:
    t = re.sub(r"[^a-z0-9 ]", " ", raw.lower())
    words = [w for w in t.split() if w and w not in _STOPWORDS]
    # 轻量词干：只处理规则复数，避免过度归并
    out = []
    for w in words:
        if len(w) > 3 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return " ".join(out)


def fact_set_jaccard(a: set[Fact], b: set[Fact]) -> float | None:
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


def retention(downstream: set[Fact], upstream: set[Fact]) -> float | None:
    """非对称包含：|F(down) ∩ F(up)| / |F(up)|。上游为空时返回 None（不猜测）。"""
    if not upstream:
        return None
    return len(downstream & upstream) / len(upstream)
