"""LLM 抽取通道：负责规则通道覆盖不到的 proper_noun 与 content 事实。

与规则通道的关系
----------------
* 数字/单位以**规则通道为准**（确定性、可复算）；
* 专名/content 以 **LLM 通道为准**；
* 两通道都抽到的事实做**规范化后取并集**，不一致处记 `extractor_conflict` 并报 κ。

约束：抽取器与待测 S2S 模型解耦（用 Qwen2.5-7B-Instruct），结果落盘可审计。
"""
from __future__ import annotations

import json
import re

from rfg.facts.schema import Fact, canon_content, canon_number, canon_proper_noun

PROMPT_TEMPLATE = """You extract atomic facts from a short spoken-style answer. Output ONLY JSON, no explanation.

Fact types:
- "number": any numeric value (normalize to digits, no commas, no units)
- "unit": a measurement unit (lowercase, singular, e.g. "meter", "kilogram", "percent")
- "proper_noun": a name, place, country, city, or chemical symbol (lowercase)
- "negation": a value the answer says is NOT in a category; use polarity "-"
- "content": a key non-numeric concept (lowercase, singular)

Rules:
- Only include facts that actually appear in the answer.
- "content" must be a SINGLE atomic concept of at most 3 words (e.g. "nectar", "carbon dioxide").
  Never put a whole clause or sentence into one content fact.
- Prefer several atomic facts over one long fact.
- One fact per distinct value; do not duplicate.
- Do not include the question's facts, only the answer's.
- If the answer has no facts, return {{"facts": []}}.

Schema: {{"facts": [{{"type": "...", "value": "...", "polarity": "+" or "-"}}]}}

Answer: "{answer}"
JSON:"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_TYPE_CANON = {
    "number": lambda v: canon_number(str(v)),
    "unit": lambda v: str(v).strip().lower(),
    "proper_noun": canon_proper_noun,
    "negation": lambda v: str(v).strip().lower(),
    "content": canon_content,
}


def parse_facts_json(raw: str) -> tuple[set[Fact], str | None]:
    """解析 LLM 输出为事实集合；解析失败返回 (空集, 错误信息)。"""
    if not raw:
        return set(), "empty output"
    m = _JSON_RE.search(raw)
    if not m:
        return set(), f"no JSON object in output: {raw[:120]!r}"
    try:
        obj = json.loads(m.group(0))
    except Exception as exc:
        return set(), f"json decode error: {exc}"
    out: set[Fact] = set()
    for item in obj.get("facts", []) or []:
        try:
            ftype = str(item["type"]).strip().lower()
            value = item["value"]
            polarity = str(item.get("polarity", "+")).strip() or "+"
        except Exception:
            continue
        if ftype not in _TYPE_CANON:
            continue
        canon = _TYPE_CANON[ftype](value)
        if not canon:
            continue
        out.add(Fact(ftype, canon, "-" if polarity.startswith("-") else "+"))
    return out, None


class LlmExtractor:
    """批量 LLM 抽取（左填充批推理，贪心解码）。"""

    def __init__(self, model_path: str, device: str = "cuda:0", dtype: str | None = None,
                 batch_size: int = 16, max_new_tokens: int = 256) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.name = model_path.rstrip("/").split("/")[-1]
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if dtype is None:
            if str(device).startswith("cuda") and torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability(torch.device(device))
                model_dtype = torch.bfloat16 if major >= 8 else torch.float16
            else:
                model_dtype = torch.float32
        else:
            model_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=model_dtype).to(device).eval()
        self.prompt_sha256 = _sha256(PROMPT_TEMPLATE)

    def extract_batch(self, answers: list[str]) -> list[tuple[set[Fact], str | None]]:
        import torch

        results: list[tuple[set[Fact], str | None]] = []
        for i in range(0, len(answers), self.batch_size):
            chunk = answers[i: i + self.batch_size]
            prompts = [PROMPT_TEMPLATE.format(answer=a.replace('"', "'")) for a in chunk]
            chat = [self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                for p in prompts]
            inputs = self.tokenizer(chat, return_tensors="pt", padding=True,
                                    truncation=True, max_length=1024).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, do_sample=False,
                                          max_new_tokens=self.max_new_tokens,
                                          pad_token_id=self.tokenizer.pad_token_id)
            gen = out[:, inputs["input_ids"].shape[1]:]
            texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            for t in texts:
                results.append(parse_facts_json(t))
        return results


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


# 提示词指纹：抽取缓存据此失效（改了提示词就必须重抽，不能复用旧结果）
PROMPT_SHA256 = _sha256(PROMPT_TEMPLATE)
