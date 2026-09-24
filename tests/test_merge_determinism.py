"""`merge()` 的确定性与极性保留 —— 守住"同一文本跨进程抽出不同事实集"这个 bug。

背景（2026-09-18 发现）
----------------------
原实现**只按 value** 合并两通道：`{f.value: f for f in facts}`。但 negation 事实与它否定的
正向事实共享同一个 value：

    "Ninety-five is not in the category of even numbers."
      → 规则通道 {number:95:+, negation:95:-}

只按 value 合并时两条互相覆盖，谁活下来取决于 `set[Fact]` 的迭代顺序；`Fact` 是
`frozen=True` dataclass，哈希由字符串字段决定，迭代顺序随 PYTHONHASHSEED 变化。
后果是整个仓库基于合并集的数字（`facts/*.jsonl`、score.py 的 RFG/RFG_hard/RFG_content）
**跨进程不可复现**：实测 randomness 探针连续两次运行给出不同的存活谱。

修复：按 **(value, polarity)** 合并，类型标签差异仍按原优先级裁决。

运行： pytest tests/ -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from rfg.facts.extract import merge
from rfg.facts.rules import extract_rules
from rfg.facts.schema import Fact

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 覆盖三类真实题面：否定句（同值双极性）、纯数字、专名。
TEXTS = (
    "Ninety-five is not in the category of even numbers.",
    "The crocodile is not a mammal.",
    "The medal is made of Au and weighs 12 kilograms.",
    "Water boils at 100 degrees Celsius at sea level.",
)

# 子进程脚本：对固定输入做双通道合并，打印稳定的 JSON，供跨 seed 比对。
_CHILD_CODE = """
import json, sys
sys.path.insert(0, {repo!r})
from rfg.facts.extract import merge
from rfg.facts.rules import extract_rules
from rfg.facts.schema import Fact

TEXTS = {texts!r}
LLM = {{"Ninety-five is not in the category of even numbers.":
        [Fact("content", "even number", "+")],
        "The crocodile is not a mammal.": [Fact("content", "mammal", "+")],
        "The medal is made of Au and weighs 12 kilograms.":
        [Fact("content", "medal", "+"), Fact("proper_noun", "gold", "+")],
        "Water boils at 100 degrees Celsius at sea level.":
        [Fact("content", "sea level", "+")]}}

out = {{}}
for t in TEXTS:
    ex = merge(t, extract_rules(t), set(LLM.get(t, [])))
    out[t] = sorted(f"{{f.type}}:{{f.value}}:{{f.polarity}}" for f in ex.facts)
print(json.dumps(out, sort_keys=True))
"""


def _merge_child(seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    proc = subprocess.run([sys.executable, "-c", _CHILD_CODE.format(repo=REPO, texts=TEXTS)],
                          capture_output=True, text=True, env=env, cwd=REPO)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_merge_is_deterministic_across_hash_seeds():
    """① 同一批文本在多个 PYTHONHASHSEED 下必须抽出完全相同的事实集。"""
    outputs = {_merge_child(seed) for seed in ("0", "1", "2", "3", "4")}
    assert len(outputs) == 1, "merge() 结果随 PYTHONHASHSEED 变化，说明仍存在集合迭代顺序依赖"


def test_negation_and_positive_counterpart_both_kept():
    """② `number:95:+` 与 `negation:95:-` 是同值不同极性的事实，必须同时保留。"""
    text = "Ninety-five is not in the category of even numbers."
    rules = extract_rules(text)
    assert Fact("number", "95", "+") in rules, "规则通道未抽出正向数字事实"
    assert Fact("negation", "95", "-") in rules, "规则通道未抽出否定事实"

    got = merge(text, rules, set()).facts
    assert Fact("number", "95", "+") in got, "正向事实被否定事实吞掉"
    assert Fact("negation", "95", "-") in got, "否定事实被正向事实吞掉"


def test_negation_counterpart_survives_with_llm_channel_present():
    """LLM 通道同时给出 content 事实时，同值双极性仍不被互相覆盖。"""
    text = "Ninety-five is not in the category of even numbers."
    got = merge(text, extract_rules(text), {Fact("content", "even number", "+")}).facts
    assert {Fact("number", "95", "+"), Fact("negation", "95", "-"),
            Fact("content", "even number", "+")} <= got


def test_type_priority_unchanged():
    """③ 类型标签优先级保持原样：number/unit/negation 规则通道权威，其余 LLM 通道优先。"""
    # 规则通道的权威类型胜出，且标签差异被记进 type_conflicts
    m = merge("t", {Fact("number", "5", "+")}, {Fact("content", "5", "+")})
    assert m.facts == {Fact("number", "5", "+")}
    assert m.type_conflicts == {Fact("number", "5", "+")}

    # 规则通道的非权威类型让位给 LLM 通道
    m2 = merge("t", {Fact("proper_noun", "sun", "+")}, {Fact("content", "sun", "+")})
    assert m2.facts == {Fact("content", "sun", "+")}

    # 只有一侧有时取该侧；值级冲突仍被记录
    m3 = merge("t", {Fact("proper_noun", "kotte", "+")}, set())
    assert m3.facts == {Fact("proper_noun", "kotte", "+")}
    assert m3.conflict == {Fact("proper_noun", "kotte", "+")}


def test_merge_is_pure_and_repeatable_in_process():
    """同一进程内重复调用结果一致（防止引入缓存/全局状态类的隐性状态）。"""
    text = "Ninety-five is not in the category of even numbers."
    first = {f for f in merge(text, extract_rules(text), set()).facts}
    for _ in range(5):
        assert {f for f in merge(text, extract_rules(text), set()).facts} == first


@pytest.mark.parametrize("text", TEXTS)
def test_merge_never_loses_rule_facts(text):
    """规则通道抽出的每条事实都必须在合并集里有同 (value, polarity) 的对应项。"""
    rules = extract_rules(text)
    got = merge(text, rules, set()).facts
    keys = {(f.value, f.polarity) for f in got}
    for f in rules:
        assert (f.value, f.polarity) in keys, f"{f} 在合并集中没有同值同极性的对应事实"
