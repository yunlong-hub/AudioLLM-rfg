"""Score completed human adjudication and isolate ASR from extractor error."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from rfg.score.textnorm import wer_norm


def _fact_set(facts: list[dict] | None) -> set[tuple[str, str, str]] | None:
    if facts is None:
        return None
    return {
        (str(fact["type"]), str(fact["value"]), str(fact.get("polarity", "+")))
        for fact in facts
    }


def _counts(predicted: set, gold: set) -> tuple[int, int, int]:
    return len(predicted & gold), len(predicted), len(gold)


def _metrics(counts: Iterable[tuple[int, int, int]]) -> dict:
    tp = predicted = gold = 0
    n = 0
    for row_tp, row_predicted, row_gold in counts:
        tp += row_tp
        predicted += row_predicted
        gold += row_gold
        n += 1
    precision = tp / predicted if predicted else None
    recall = tp / gold if gold else None
    f1 = None
    if precision is not None and recall is not None:
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
    return {
        "n_items": n,
        "true_positive": tp,
        "predicted": predicted,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def score_adjudication(path: str | Path) -> dict:
    rows = [json.loads(line) for line in Path(path).open() if line.strip()]
    asr_fact_counts: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    asr_wers: dict[str, list[float]] = defaultdict(list)
    extractor_counts: list[tuple[int, int, int]] = []
    annotator_fact_counts: list[tuple[int, int, int]] = []
    annotator_wers: list[float] = []
    completion = {
        "annotator_1": 0,
        "annotator_2": 0,
        "adjudicated": 0,
        "extractor_on_adjudicated": 0,
    }

    for row in rows:
        human = row.get("human") or {}
        ann1, ann2 = human.get("annotator_1") or {}, human.get("annotator_2") or {}
        adjudicated = human.get("adjudicated") or {}
        ann1_facts, ann2_facts = _fact_set(ann1.get("facts")), _fact_set(ann2.get("facts"))
        gold = _fact_set(adjudicated.get("facts"))

        if ann1.get("transcript") is not None and ann1_facts is not None:
            completion["annotator_1"] += 1
        if ann2.get("transcript") is not None and ann2_facts is not None:
            completion["annotator_2"] += 1
        if adjudicated.get("transcript") is not None and gold is not None:
            completion["adjudicated"] += 1
        if ann1.get("transcript") is not None and ann2.get("transcript") is not None:
            value = wer_norm(ann1["transcript"], ann2["transcript"])
            if value is not None:
                annotator_wers.append(value)
        if ann1_facts is not None and ann2_facts is not None:
            annotator_fact_counts.append(_counts(ann1_facts, ann2_facts))

        if gold is None or adjudicated.get("transcript") is None:
            continue
        for asr_name in ("asr1", "asr2", "asr3"):
            asr = (row.get("asr") or {}).get(asr_name) or {}
            predicted = _fact_set(asr.get("facts"))
            if predicted is not None:
                asr_fact_counts[asr_name].append(_counts(predicted, gold))
            value = wer_norm(adjudicated["transcript"], asr.get("text"))
            if value is not None:
                asr_wers[asr_name].append(value)
        extractor = _fact_set(human.get("extractor_facts_on_adjudicated_transcript"))
        if extractor is not None:
            completion["extractor_on_adjudicated"] += 1
            extractor_counts.append(_counts(extractor, gold))

    total = len(rows)
    ready = total > 0 and all(completion[key] == total for key in (
        "annotator_1", "annotator_2", "adjudicated", "extractor_on_adjudicated"))
    return {
        "packet": str(path),
        "n_items": total,
        "completion": dict(completion),
        "ready": ready,
        "annotator_agreement": {
            "mean_normalized_wer": (sum(annotator_wers) / len(annotator_wers)
                                    if annotator_wers else None),
            "facts_annotator1_vs_annotator2": _metrics(annotator_fact_counts),
        },
        "asr_vs_human": {
            name: {
                "mean_normalized_wer": (sum(asr_wers[name]) / len(asr_wers[name])
                                        if asr_wers[name] else None),
                "fact_metrics": _metrics(asr_fact_counts[name]),
            }
            for name in ("asr1", "asr2", "asr3")
        },
        "extractor_only_vs_human": _metrics(extractor_counts),
    }
