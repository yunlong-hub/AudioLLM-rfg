"""按指定模式合成多路 ASR 的回读事实集。

单路模式用于报告识别器敏感性；并集模式减少单个 ASR 漏识别对事实保留率的影响。
并集仍可能受 ASR 误识别影响，因此不能替代固定文本校准或人工核验。

从磁盘读入 facts 后、计算指标前调用 :func:`compose_readback`。伴随键
``X#asr2`` 和 ``X#asr3`` 会被合入主键 ``X``，且不会出现在返回值中。
"""
from __future__ import annotations

from rfg.facts.schema import Fact

#: Companion readback keys.  The unsuffixed key stores ASR1 (Whisper).
ASR2_SUFFIX = "#asr2"
ASR3_SUFFIX = "#asr3"
#: 模型内部文本的后缀。合成时**原样保留**，它不经 ASR。
INTERNAL_SUFFIX = "#internal"

READBACK_MODES: tuple[str, ...] = (
    "asr1", "asr2", "asr3", "union12", "union13", "union123"
)

MODE_DOC = {
    "asr1": "Whisper only (historical primary readback)",
    "asr2": "仅 asr2——用于观察单路差异",
    "asr3": "Fun-ASR only",
    "union12": "Whisper ∪ Seamless",
    "union13": "Whisper ∪ Fun-ASR",
    "union123": "Whisper ∪ Seamless ∪ Fun-ASR",
}


def compose_readback(facts: dict[str, set[Fact]], mode: str) -> dict[str, set[Fact]]:
    """按 ``mode`` 合成回读事实，并返回新的 facts 字典。

    * `asr1`  —— 原样返回（`X` 即 asr1 的事实集）
    * `asr2`  —— 用 `X#asr2` 覆盖 `X`；**缺失该键的条件整体丢弃**（未观测，不猜测）
    Union modes require all named ASRs for speech conditions.  A condition with
    a missing companion readback is excluded instead of silently falling back
    to a weaker instrument.  Text and internal-text conditions are preserved.

    内部文本键（`#internal`）与文本条件（READ/LISTEN 等无伴随键者）原样保留。
    """
    if mode not in READBACK_MODES:
        raise ValueError(f"未知回读模式 {mode!r}，可选 {READBACK_MODES}")
    out: dict[str, set[Fact]] = {}
    for cond, fs in facts.items():
        if cond.endswith((ASR2_SUFFIX, ASR3_SUFFIX)):
            continue  # 伴随键本身不进结果
        a2 = facts.get(cond + ASR2_SUFFIX)
        a3 = facts.get(cond + ASR3_SUFFIX)
        is_speech = (
            a2 is not None
            or a3 is not None
            or cond in {"SPEAK", "ECHO", "EF", "EFA", "EFB", "EFW", "SPEAKD"}
        )
        if not is_speech:
            out[cond] = fs
        elif mode == "asr1":
            out[cond] = fs
        elif mode == "asr2" and a2 is not None:
            out[cond] = a2
        elif mode == "asr3" and a3 is not None:
            out[cond] = a3
        elif mode == "union12" and a2 is not None:
            out[cond] = fs | a2
        elif mode == "union13" and a3 is not None:
            out[cond] = fs | a3
        elif mode == "union123" and a2 is not None and a3 is not None:
            out[cond] = fs | a2 | a3
    return out
