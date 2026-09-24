"""Build deterministic, three-ASR-stratified human adjudication packets."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rfg.facts.readback_state import completed_text
from rfg.score.textnorm import wer_norm


@dataclass(frozen=True)
class AuditSource:
    run_id: str
    model_slug: str
    facts_path: Path


def _load_facts(path: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[(row["item_id"], row["condition"])] = row
    return rows


def _fact_signature(row: dict) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted(
        (str(fact["type"]), str(fact["value"]), str(fact.get("polarity", "+")))
        for fact in row.get("facts") or []
    ))


def _agreement_stratum(signatures: list[tuple]) -> str:
    if signatures[0] == signatures[1] == signatures[2]:
        return "all_three_agree"
    if len(set(signatures)) == 2:
        return "two_agree"
    return "all_differ"


def _category(item_id: str) -> str:
    return {
        "cont": "content",
        "num": "number",
        "unit": "unit",
        "neg": "negation",
        "name": "proper_noun",
    }.get(item_id.split("_", 1)[0], "other")


def collect_candidates(sources: Iterable[AuditSource], condition: str = "SPEAK") -> list[dict]:
    candidates: list[dict] = []
    for source in sources:
        facts = _load_facts(source.facts_path)
        prediction_root = Path("exp") / source.run_id / "predictions" / source.model_slug
        for prediction_path in sorted(prediction_root.glob(f"*/{condition}.json")):
            record = json.loads(prediction_path.read_text())
            readback = record.get("readback") or {}
            if record.get("error") or not record.get("audio"):
                continue
            transcripts = [completed_text(readback, f"asr{i}") for i in (1, 2, 3)]
            if any(text is None for text in transcripts):
                continue
            item_id = record["item_id"]
            keys = [
                (item_id, condition),
                (item_id, f"{condition}#asr2"),
                (item_id, f"{condition}#asr3"),
            ]
            if any(key not in facts for key in keys):
                continue
            fact_rows = [facts[key] for key in keys]
            signatures = [_fact_signature(row) for row in fact_rows]
            pairwise_wer = {
                "asr1_asr2": wer_norm(transcripts[0], transcripts[1]),
                "asr1_asr3": wer_norm(transcripts[0], transcripts[2]),
                "asr2_asr3": wer_norm(transcripts[1], transcripts[2]),
            }
            candidates.append({
                "run_id": source.run_id,
                "model": source.model_slug,
                "item_id": item_id,
                "category": _category(item_id),
                "condition": condition,
                "stratum": _agreement_stratum(signatures),
                "audio": str(Path(record["audio"]).resolve()),
                "internal_text": record.get("text") or record.get("text_raw") or "",
                "asr": {
                    "asr1": {"text": transcripts[0], "facts": fact_rows[0].get("facts") or []},
                    "asr2": {"text": transcripts[1], "facts": fact_rows[1].get("facts") or []},
                    "asr3": {"text": transcripts[2], "facts": fact_rows[2].get("facts") or []},
                },
                "pairwise_wer": pairwise_wer,
                "extractor": {
                    "prompt_sha256": fact_rows[0].get("prompt_sha256"),
                    "asr1_rules": fact_rows[0].get("facts_rules") or [],
                    "asr1_llm": fact_rows[0].get("facts_llm") or [],
                    "asr1_conflict": fact_rows[0].get("conflict") or [],
                    "asr2_rules": fact_rows[1].get("facts_rules") or [],
                    "asr2_llm": fact_rows[1].get("facts_llm") or [],
                    "asr2_conflict": fact_rows[1].get("conflict") or [],
                    "asr3_rules": fact_rows[2].get("facts_rules") or [],
                    "asr3_llm": fact_rows[2].get("facts_llm") or [],
                    "asr3_conflict": fact_rows[2].get("conflict") or [],
                },
                "human": {
                    "annotator_1": {"transcript": None, "facts": None, "notes": None},
                    "annotator_2": {"transcript": None, "facts": None, "notes": None},
                    "adjudicated": {"transcript": None, "facts": None, "notes": None},
                    "extractor_facts_on_adjudicated_transcript": None,
                },
            })
    return candidates


def stratified_sample(candidates: list[dict], n: int = 100, seed: int = 20260918) -> list[dict]:
    """Oversample disagreement while balancing model and fact category."""
    if len(candidates) < n:
        raise ValueError(f"requested {n} rows but only {len(candidates)} candidates are complete")
    target = {
        "all_differ": round(n * 0.40),
        "two_agree": round(n * 0.35),
    }
    target["all_three_agree"] = n - sum(target.values())

    grouped: dict[str, dict[tuple[str, str], list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        grouped[row["stratum"]][(row["model"], row["category"])].append(row)
    for cells in grouped.values():
        for key, rows in cells.items():
            rows.sort(key=lambda row: hashlib.sha256(
                f"{seed}:{key}:{row['run_id']}:{row['item_id']}".encode()).hexdigest())

    selected: list[dict] = []
    selected_keys: set[tuple[str, str, str]] = set()

    def take_from_stratum(stratum: str, amount: int) -> None:
        cells = grouped.get(stratum, {})
        keys = sorted(cells)
        while amount > 0 and any(cells[key] for key in keys):
            for key in keys:
                if amount == 0:
                    break
                if not cells[key]:
                    continue
                row = cells[key].pop()
                identity = (row["run_id"], row["model"], row["item_id"])
                if identity in selected_keys:
                    continue
                selected.append(row)
                selected_keys.add(identity)
                amount -= 1

    for stratum in ("all_differ", "two_agree", "all_three_agree"):
        take_from_stratum(stratum, target[stratum])
    if len(selected) < n:
        remaining = [row for row in candidates
                     if (row["run_id"], row["model"], row["item_id"]) not in selected_keys]
        remaining.sort(key=lambda row: hashlib.sha256(
            f"{seed}:fill:{row['run_id']}:{row['model']}:{row['item_id']}".encode()).hexdigest())
        selected.extend(remaining[: n - len(selected)])
    if len(selected) != n:
        raise RuntimeError(f"failed to construct exact audit sample: {len(selected)} != {n}")
    selected.sort(key=lambda row: (row["stratum"], row["model"], row["category"], row["item_id"]))
    for index, row in enumerate(selected, 1):
        row["audit_id"] = f"three_asr_{index:03d}"
    return selected


def summarize_packet(candidates: list[dict], selected: list[dict]) -> dict:
    def counts(rows: Iterable[dict], key: str) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for row in rows:
            out[str(row[key])] += 1
        return dict(sorted(out.items()))

    return {
        "n_candidates": len(candidates),
        "n_selected": len(selected),
        "candidate_strata": counts(candidates, "stratum"),
        "selected_strata": counts(selected, "stratum"),
        "selected_models": counts(selected, "model"),
        "selected_categories": counts(selected, "category"),
        "human_fields_prepopulated": False,
    }
