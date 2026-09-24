"""文本规范化与回读一致性判定。

存在的理由
----------
同一段语音，Whisper 倾向输出数字（"2,500 meters"），Seamless 倾向输出词形
（"two thousand five hundred meters"）。直接做字符串精确比较会把**同一个答案判成不一致**，
实测精确一致率仅 16.7%，而这个数字毫无意义。

因此统一把数字规范成词形后再比较；`RFG_corr`（只统计双 ASR 一致的 span）依赖这个判定。
"""
from __future__ import annotations

import re

_DIGIT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def _num_to_words(match: re.Match) -> str:
    raw = match.group(0).replace(",", "")
    try:
        from num2words import num2words

        if "." in raw:
            whole, frac = raw.split(".", 1)
            frac_words = " ".join(_ONES[int(d)] if d.isdigit() else d for d in frac)
            return f"{num2words(int(whole))} point {frac_words}"
        return num2words(int(raw))
    except Exception:
        return raw  # 解析失败就保留原样，不静默丢内容


def normalize_text(text: str | None) -> str:
    """小写化 + 数字转词形 + 去标点 + 折叠空白。"""
    if not text:
        return ""
    t = text.lower().replace("%", " percent ")
    t = _DIGIT_RE.sub(_num_to_words, t)
    t = t.replace("-", " ")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def wer_norm(reference: str | None, hypothesis: str | None) -> float | None:
    """规范化后的 WER；任一侧为空时返回 None（不猜测）。"""
    r, h = normalize_text(reference), normalize_text(hypothesis)
    if not r or not h:
        return None
    import jiwer

    return float(jiwer.wer(r, h))


def agreement_norm(a: str | None, b: str | None, threshold: float = 0.2) -> tuple[bool, float | None]:
    """两个回读是否一致：以规范化 WER ≤ threshold 判定（不是字符串相等）。

    阈值 0.2 是**临时口径**，只用于冒烟；正式管线（D0-5）的 `asr_agree` 定义为
    "两侧回读抽出的事实集合相同"，比字符串 WER 稳健得多（见 docs/execution_plan.md §5）。
    单复数/插入 "city" 这类同义差异（WER≈0.17）应判为一致。
    """
    w = wer_norm(a, b)
    if w is None:
        return False, None
    return w <= threshold, w
