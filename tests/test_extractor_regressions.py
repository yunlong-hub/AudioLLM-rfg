"""抽取器回归测试：守住历史上真实导致论文数值失真的几类 bug。

背景：论文草稿增量写入期间，抽取器被修过多次（数词边界、单位缩写、
元素符号↔元素名、`inf` 护栏）。每次修复都会让此前写入的数值过期，
最终造成正文与产物不一致。这些用例把当时的修复点固化下来。

运行： pytest tests/ -q      或      python -m pytest tests/ -q
"""
from __future__ import annotations

import pytest

from rfg.facts.rules import extract_rules
from rfg.facts.schema import canon_number, words_to_number
from rfg.score.textnorm import normalize_text


def values(facts, ftype: str) -> set[str]:
    return {f.value for f in facts if f.type == ftype}


# --- 数词解析 ---------------------------------------------------------------

@pytest.mark.parametrize("words, expected", [
    ("seventeen", 17.0),          # 曾因正则缺少尾部 \b 而误判为 7
    ("two and a half", 2.5),      # 曾因不支持 "and a half" 而误判为 2
    ("one hundred", 100.0),
    ("three point five", 3.5),
    ("SEVENTEEN", 17.0),          # 曾因缺少 IGNORECASE 而漏掉大写
])
def test_words_to_number(words, expected):
    assert words_to_number(words) == pytest.approx(expected)


def test_digit_and_word_forms_agree():
    """数字与英文数词必须落到同一事实取值，否则跨条件比对会假性失配。"""
    for digits, words in [("17 apples", "seventeen apples"),
                          ("2.5 kg", "two and a half kg"),
                          ("17 items", "SEVENTEEN items")]:
        assert values(extract_rules(digits), "number") == values(extract_rules(words), "number"), \
            f"{digits!r} 与 {words!r} 未归一到同一取值"


def test_canon_number_non_finite_is_safe():
    """Step-Audio 会吐出超长数字串导致 float() 溢出为 inf，曾是未捕获异常。"""
    assert canon_number("9" * 400) is None


# --- 单位与元素 -------------------------------------------------------------

@pytest.mark.parametrize("abbrev, full", [
    ("The rate rose by 12%", "The rate rose by 12 percent"),
    ("It is 5 km away", "It is 5 kilometers away"),
    ("Wait 2 hr please", "Wait 2 hours please"),
    ("about 30 sec", "about 30 seconds"),
])
def test_unit_abbrev_equals_full_word(abbrev, full):
    """单位缩写与全称必须归一。曾缺 %/km/hr/sec 等价，使结构下界被高估。"""
    assert values(extract_rules(abbrev), "unit") == values(extract_rules(full), "unit")
    assert values(extract_rules(abbrev), "unit"), f"{abbrev!r} 未抽出任何 unit 事实"


def test_element_symbol_and_name_equivalent():
    """Au 与 gold 必须视为同一事实；否则结构下界被高估（曾为 9.1%，实为 5.4%）。"""
    sym = extract_rules("The medal is made of Au")
    name = extract_rules("The medal is made of gold")
    assert values(sym, "proper_noun") & values(name, "proper_noun"), \
        "Au 与 gold 未归一到同一取值"


# --- 归一化 -----------------------------------------------------------------

def test_normalize_text_is_stable_and_lowercases():
    assert normalize_text("Hello, World!") == normalize_text("hello world")
    assert normalize_text(None) == ""


def test_extract_rules_never_raises_on_hostile_input():
    """抽取器跑在模型自由生成的文本上，任何输入都不应抛异常。"""
    for bad in ["", None, "9" * 500, "!!!???", "Ⅻ Ⅷ ½", "\x00\x01", "a" * 10000]:
        extract_rules(bad if bad is not None else "")
