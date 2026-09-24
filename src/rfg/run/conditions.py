"""条件定义：五条件的输入/输出模态与提示词，集中在一处，避免各脚本口径漂移。

条件语义（论文骨架）
--------------------
| 条件   | 输入            | 输出        | 隔离的环节            |
|--------|-----------------|-------------|-----------------------|
| READ   | 文本问题        | 文本答案    | 文本输入参照          |
| LISTEN | 语音问题        | 文本答案    | 感知 + 推理（=PG 的参照）|
| SPEAK  | 语音问题        | 语音答案    | 端到端（=RFG 的被测项）|
| ECHO   | 文本问题        | 语音答案    | 输出侧，输入为文本     |
| EF     | 给定的文本答案  | 语音答案    | 朗读服从性/渲染控制   |

**关键**：READ/LISTEN/SPEAK/ECHO 四条件使用**完全相同的指令文本**，只有输入/输出模态不同；
EF 的提示词按定义不同（内容已给定，只要求朗读）。
"""
from __future__ import annotations

INSTRUCTION = "Answer the question in one short sentence."


def item_instruction(item: dict) -> str:
    """Return a dataset-provided instruction, falling back to the project default.

    Existing constructed datasets omit ``instruction_text`` and therefore retain
    their historical behavior.  External benchmarks use the field to preserve the
    benchmark's original Short/CoT or step-by-step prompting contract.
    """
    value = item.get("instruction_text")
    return value.strip() if isinstance(value, str) and value.strip() else INSTRUCTION

EF_PROMPT_TEMPLATE = (
    "Read the following sentence aloud exactly as written, word for word. "
    "Do not add, remove, translate, or rephrase anything.\n\n"
    "Sentence: {answer}"
)

CONDITIONS: tuple[str, ...] = ("READ", "LISTEN", "SPEAK", "ECHO", "EF", "EFA", "EFB", "EFW", "SPEAKD")

# 条件 → 是否要求模型输出语音
WANTS_AUDIO: dict[str, bool] = {
    "READ": False,
    "LISTEN": False,
    "SPEAK": True,
    "ECHO": True,
    "EF": True,
    "EFA": True,   # 语音输入 + 给定 LISTEN 文本朗读（2×2 闭合）
    "EFB": True,   # 语音输入 + 给定 READ 文本朗读（与 EF 目标句完全相同，仅输入模态不同）
    "EFW": True,   # 给定 READ 文本的**可朗读改写**（数字→词形）后朗读：零训练缓解
    "SPEAKD": True,  # SPEAK + 退化控制解码（talker 重复惩罚/长度上限）：针对"生成崩溃"型失败
}

# 语音条件的回读 WER 参照哪一段上游文本
READBACK_REFERENCE: dict[str, str] = {
    "SPEAK": "LISTEN",
    "ECHO": "READ",
    "EF": "READ",
    "EFA": "LISTEN",
    "EFB": "READ",
    "EFW": "READ",
    "SPEAKD": "LISTEN",
}

# 2×2 设计：**输入模态 × 内容来源**。EF 朗读的是 READ 文本、EFA 朗读的是 LISTEN 文本，
# 因此每个"给定内容"条件都与同输入模态的"自生成"条件严格配对。
GRID_2X2: dict[tuple[str, str], str] = {
    ("text", "self"): "ECHO",
    ("audio", "self"): "SPEAK",
    ("text", "given"): "EF",
    ("audio", "given"): "EFA",
}

# 带非默认解码参数的语音条件：额外 generate 参数（与默认条件配对比较用）
CONDITION_GEN_KWARGS: dict[str, dict] = {
    # 生成退化（长音频/重复/不可懂）是 3B 的主要失败模式；这里用解码控制对其做干预
    "SPEAKD": {"talker_repetition_penalty": 1.3, "talker_max_new_tokens": 256},
}

# "给定内容朗读"条件：被朗读文本取自哪个上游条件
FORCED_SOURCE: dict[str, str] = {"EF": "READ", "EFA": "LISTEN", "EFB": "READ", "EFW": "READ"}

_DIGIT_RE = __import__("re").compile(r"\d[\d,]*(?:\.\d+)?")


def speakable_format(text: str) -> str:
    """把阿拉伯数字改写成词形（"2,500" → "two thousand, five hundred"）。

    动机：实测 READ 文本平均含 2.70 个阿拉伯数字，而"READ 数字多于 LISTEN"的子组里
    ECHO−SPEAK 的渲染损失差高达 +0.227。若损失源于数字书写形式，改写应能显著降低 Δ_render。
    """
    from num2words import num2words

    def repl(m):
        raw = m.group(0).replace(",", "")
        try:
            if "." in raw:
                whole, frac = raw.split(".", 1)
                from rfg.facts.schema import _ONES
                frac_w = " ".join(_ONES[int(c)] if c.isdigit() else c for c in frac)
                return f"{num2words(int(whole))} point {frac_w}"
            return num2words(int(raw))
        except Exception:
            return raw

    return _DIGIT_RE.sub(repl, text)

# 模态对照对：**目标句完全相同**、只有输入模态不同 → 可直接做因果比较
MODALITY_CONTROLLED_PAIRS: tuple[tuple[str, str], ...] = (("EF", "EFB"),)


def ef_prompt(answer_text: str) -> str:
    return EF_PROMPT_TEMPLATE.format(answer=answer_text.strip())


def forced_readback_prompt(answer_text: str) -> str:
    """所有"给定内容朗读"条件共用同一提示词模板（EF 与 EFA 只差输入模态）。"""
    return ef_prompt(answer_text)
