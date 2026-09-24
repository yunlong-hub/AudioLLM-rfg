"""Answer extraction and coverage-aware scoring for the OMG benchmarks.

The OMG protocol uses one shared numerical extractor for Spoken-MQA/GSM8K and
task-specific yes/no or option-label extraction for VoiceBench-BBH.  This module
keeps extraction deterministic and refuses to expose a final accuracy until the
whole manifest has a non-error prediction for the requested condition.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from rfg.facts.readback_state import completed_text
from rfg.score.metrics import wilson_ci

_NUM = r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?"
_NUM_RE = re.compile(_NUM)
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*(" + _NUM + r")\s*\}", re.I)
_NUM_CUE_RE = re.compile(
    r"(?:final\s+answer|answer|result)\s*(?:is\s*)?(?::|=)?\s*\$?\s*(" + _NUM + r")",
    re.I,
)


def _fraction(value: str | None) -> Fraction | None:
    if not value:
        return None
    raw = value.replace(",", "").replace(" ", "")
    try:
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            return Fraction(numerator) / Fraction(denominator)
        return Fraction(raw)
    except (ValueError, ZeroDivisionError):
        return None


def extract_numeric_answer(text: str | None) -> str | None:
    """Extract the final numerical answer using stable, cue-first rules."""
    if not text:
        return None
    value = text.replace("−", "-").replace("–", "-")
    boxed = _BOXED_RE.findall(value)
    if boxed:
        return boxed[-1].replace(" ", "")
    cued = _NUM_CUE_RE.findall(value)
    if cued:
        return cued[-1].replace(" ", "")
    hashes = re.findall(r"####\s*\$?\s*(" + _NUM + r")", value)
    if hashes:
        return hashes[-1].replace(" ", "")
    numbers = _NUM_RE.findall(value)
    return numbers[-1].replace(" ", "") if numbers else None


def extract_bbh_answer(text: str | None, task: str) -> str | None:
    """Extract VoiceBench-BBH labels for its four supported tasks."""
    if not text:
        return None
    value = text.strip()
    if task == "hyperbaton":
        standalone = re.fullmatch(r"\(?\s*([ABab])\s*\)?[.!]?", value)
        if standalone:
            return f"({standalone.group(1).upper()})"
        parenthesized = re.findall(r"\(([ABab])\)", value)
        if parenthesized:
            return f"({parenthesized[-1].upper()})"
        cued = re.findall(
            r"(?:final\s+answer|answer|option|choice)\s*(?:is\s*)?(?::|=)?\s*([ABab])\b",
            value,
            re.I,
        )
        return f"({cued[-1].upper()})" if cued else None
    labels = re.findall(r"\b(yes|no)\b", value, re.I)
    return labels[-1].lower() if labels else None


def answers_equal(dataset: str, task: str | None, prediction: str | None, gold: str) -> bool:
    if dataset in {"spoken_mqa", "gsm8k"}:
        return _fraction(prediction) is not None and _fraction(prediction) == _fraction(gold)
    if dataset == "voicebench_bbh":
        if task == "hyperbaton":
            return (prediction or "").upper() == gold.strip().upper()
        return (prediction or "").lower() == gold.strip().lower()
    raise ValueError(f"unsupported benchmark dataset: {dataset}")


def extract_answer(dataset: str, task: str | None, text: str | None) -> str | None:
    if dataset in {"spoken_mqa", "gsm8k"}:
        return extract_numeric_answer(text)
    if dataset == "voicebench_bbh" and task:
        return extract_bbh_answer(text, task)
    raise ValueError(f"unsupported benchmark dataset/task: {dataset}/{task}")


@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str
    dataset: str
    task: str | None
    gold: str


def load_benchmark_items(path: str | Path) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            items.append(
                BenchmarkItem(
                    item_id=row["id"],
                    dataset=row["category"],
                    task=row.get("difficulty"),
                    gold=str(row["expected_answer_text"]),
                )
            )
    datasets = {item.dataset for item in items}
    if len(datasets) != 1:
        raise ValueError(f"manifest must contain exactly one benchmark dataset, got {datasets}")
    return items


def score_condition(
    items: Iterable[BenchmarkItem], prediction_root: str | Path, condition: str,
    text_source: str = "internal",
) -> tuple[dict, list[dict]]:
    items = list(items)
    root = Path(prediction_root)
    details: list[dict] = []
    n_present = n_errors = n_extracted = n_correct = 0
    by_task: dict[str, list[bool]] = {}

    for item in items:
        path = root / item.item_id / f"{condition}.json"
        detail = {
            "item_id": item.item_id,
            "condition": condition if text_source == "internal" else f"{condition}@{text_source}",
            "dataset": item.dataset,
            "task": item.task,
            "gold": item.gold,
            "prediction_path": str(path),
        }
        if not path.exists():
            detail["status"] = "missing"
            details.append(detail)
            continue
        n_present += 1
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            n_errors += 1
            detail.update(status="error", error=f"{type(exc).__name__}: {exc}")
            details.append(detail)
            continue
        if record.get("error"):
            n_errors += 1
            detail.update(status="error", error=record["error"])
            details.append(detail)
            continue

        if text_source == "internal":
            text = record.get("text") or record.get("text_raw") or ""
        else:
            readback = record.get("readback") or {}
            text = completed_text(readback, text_source)
            if text is None:
                n_errors += 1
                detail.update(
                    status="error",
                    error=(readback.get(f"{text_source}_error")
                           or f"missing {text_source} readback"),
                )
                details.append(detail)
                continue
        extracted = extract_answer(item.dataset, item.task, text)
        correct = answers_equal(item.dataset, item.task, extracted, item.gold)
        n_extracted += int(extracted is not None)
        n_correct += int(correct)
        by_task.setdefault(item.task or item.dataset, []).append(correct)
        detail.update(status="ok", extracted=extracted, correct=correct, text=text)
        details.append(detail)

    total = len(items)
    n_available = n_present - n_errors
    ready = n_present == total and n_errors == 0
    acc_available = n_correct / n_available if n_available else None
    result = {
        "condition": condition,
        "text_source": text_source,
        "n_items": total,
        "n_present": n_present,
        "n_missing": total - n_present,
        "n_errors": n_errors,
        "n_available": n_available,
        "n_extracted": n_extracted,
        "n_correct": n_correct,
        "coverage": n_available / total if total else None,
        "accuracy_on_available": acc_available,
        "accuracy_full_set_lower_bound": n_correct / total if total else None,
        "accuracy": n_correct / total if ready and total else None,
        "ready": ready,
        "wilson_ci95": wilson_ci(n_correct, total) if ready and total else None,
        "by_task_on_available": {
            task: {"n": len(values), "n_correct": sum(values), "accuracy": sum(values) / len(values)}
            for task, values in sorted(by_task.items())
        },
    }
    return result, details


def score_benchmark(
    items_path: str | Path,
    prediction_root: str | Path,
    conditions: Iterable[str] = ("LISTEN", "SPEAK"),
    speech_asrs: Iterable[str] = (),
) -> tuple[dict, list[dict]]:
    items = load_benchmark_items(items_path)
    blocks: dict[str, dict] = {}
    details: list[dict] = []
    for condition in conditions:
        block, rows = score_condition(items, prediction_root, condition)
        blocks[condition] = block
        details.extend(rows)
    spoken_blocks: dict[str, dict] = {}
    for asr_name in speech_asrs:
        block, rows = score_condition(items, prediction_root, "SPEAK", asr_name)
        spoken_blocks[asr_name] = block
        details.extend(rows)
    listen = blocks.get("LISTEN", {}).get("accuracy")
    speak = blocks.get("SPEAK", {}).get("accuracy")
    gap = listen - speak if listen is not None and speak is not None else None
    if gap is not None and not math.isfinite(gap):
        gap = None
    summary = {
        "dataset": items[0].dataset if items else None,
        "items_path": str(items_path),
        "prediction_root": str(prediction_root),
        "conditions": blocks,
        "spoken_answer": spoken_blocks,
        "output_mode_gap": gap,
        "ready": (bool(blocks)
                  and all(block["ready"] for block in blocks.values())
                  and all(block["ready"] for block in spoken_blocks.values())),
    }
    return summary, details
