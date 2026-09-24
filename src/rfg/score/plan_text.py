"""Original-plan text matching for the fixed-pool E3 baseline.

Protocol fixed before evaluating results: use the existing ``normalize_text``
(lowercase, English number words, percent expansion, punctuation/whitespace
normalization), then whitespace tokens. Similarity is one minus token-level
Levenshtein distance divided by the larger token count. Both empty gives 1;
only one empty gives 0. No fact extraction or synonym matching is performed.
"""
from __future__ import annotations

import statistics
from collections.abc import Sequence
from functools import lru_cache

from num2words import num2words

from rfg.score.textnorm import _DIGIT_RE, normalize_text


NORMALIZATION_PROTOCOL = {
    "source": "rfg.score.textnorm.normalize_text",
    "operations": "lowercase; digits to English words; percent expansion; punctuation to spaces; collapse whitespace",
    "tokenizer": "split normalized text on whitespace",
    "similarity": "1 - word_Levenshtein_distance / max(reference_tokens, hypothesis_tokens)",
    "empty_policy": "both empty=1.0; one empty=0.0",
    "selection": "mean similarity to the original SPEAK plan over the two non-held-out ASRs; earliest candidate on ties",
    "number_conversion": "num2words required with 16=sixteen preflight; unsupported numeric conversions retain the shared normalizer's literal fallback and are explicitly audited by cache key",
    "fact_normalization": "none; use the existing text normalizer only",
}


@lru_cache(maxsize=65_536)
def normalize_plan_tokens(text: str) -> tuple[str, ...]:
    """Use the shared normalizer; dependency import is mandatory, never optional."""
    return tuple(normalize_text(text).split())


def number_conversion_failures(text: str) -> list[dict]:
    """Expose the shared normalizer's fallback on unsupported numeric strings."""
    failures = []
    for match in _DIGIT_RE.finditer(text):
        raw = match.group(0).replace(",", "")
        whole = raw.split(".", 1)[0]
        try:
            num2words(int(whole))
        except (OverflowError, ValueError, TypeError) as error:
            failures.append({"span": list(match.span()), "integer_digits": len(whole),
                             "token_prefix": raw[:40], "error": type(error).__name__,
                             "policy": "retain literal numeric string, matching normalize_text"})
    return failures


def normalized_word_similarity(reference: str, hypothesis: str) -> float:
    reference_words = normalize_plan_tokens(reference)
    hypothesis_words = normalize_plan_tokens(hypothesis)
    length = max(len(reference_words), len(hypothesis_words))
    if not length:
        return 1.0
    previous = list(range(len(hypothesis_words) + 1))
    for index, word in enumerate(reference_words, start=1):
        current = [index]
        for other_index, other in enumerate(hypothesis_words, start=1):
            current.append(min(previous[other_index] + 1,
                               current[-1] + 1,
                               previous[other_index - 1] + (word != other)))
        previous = current
    return 1 - previous[-1] / length


def plan_text_scores(plan: str, selector_transcripts: Sequence[Sequence[str]]) -> list[float]:
    """Score each candidate using exactly the two supplied selector readbacks."""
    if len(selector_transcripts) != 2:
        raise ValueError("plan_text requires exactly two selector ASRs")
    if len(selector_transcripts[0]) != len(selector_transcripts[1]):
        raise ValueError("selector ASRs must describe the same candidate pool")
    return [statistics.mean(normalized_word_similarity(plan, text) for text in pair)
            for pair in zip(*selector_transcripts)]
