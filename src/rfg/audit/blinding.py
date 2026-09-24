"""Blind, validate, and merge the two-annotator three-ASR audit packet.

The source packet deliberately contains model identities, internal text, and
all ASR hypotheses for later scoring.  None of those fields may be visible to
the two independent annotators or to the adjudicator.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable


ANNOTATORS = ("annotator_1", "annotator_2")
BLINDED_KEYS = {"audit_id", "audio", "annotation"}
ADJUDICATION_KEYS = {"audit_id", "audio", "candidate_a", "candidate_b", "adjudicated"}


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def jsonl_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Write an artifact once so an existing human label file is never replaced."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _master_by_id(rows: list[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        audit_id = str(row.get("audit_id") or "")
        if not audit_id:
            raise ValueError("every master row must have a non-empty audit_id")
        if audit_id in indexed:
            raise ValueError(f"duplicate audit_id in master packet: {audit_id}")
        if not row.get("audio"):
            raise ValueError(f"master row {audit_id} has no audio path")
        indexed[audit_id] = row
    return indexed


def _blank_annotation() -> dict:
    return {"transcript": None, "facts": None, "notes": None}


def make_blinded_sheet(master_rows: list[dict], annotator: str, seed: int) -> list[dict]:
    """Return a model/ASR-blind sheet in an annotator-specific order."""
    if annotator not in ANNOTATORS:
        raise ValueError(f"unsupported annotator: {annotator}")
    indexed = _master_by_id(master_rows)
    rows = [
        {
            "audit_id": audit_id,
            "audio": master["audio"],
            "annotation": _blank_annotation(),
        }
        for audit_id, master in indexed.items()
    ]
    random.Random(f"{seed}:{annotator}").shuffle(rows)
    return rows


def _validate_fact(fact: object, *, location: str) -> None:
    if not isinstance(fact, dict):
        raise ValueError(f"{location}: every fact must be an object")
    if not isinstance(fact.get("type"), str) or not fact["type"].strip():
        raise ValueError(f"{location}: every fact needs a non-empty string type")
    if not isinstance(fact.get("value"), str) or not fact["value"].strip():
        raise ValueError(f"{location}: every fact needs a non-empty string value")
    polarity = fact.get("polarity", "+")
    if polarity not in {"+", "-"}:
        raise ValueError(f"{location}: polarity must be '+' or '-'")


def _validate_completed_annotation(annotation: object, *, location: str) -> dict:
    if not isinstance(annotation, dict):
        raise ValueError(f"{location}: annotation must be an object")
    transcript = annotation.get("transcript")
    facts = annotation.get("facts")
    notes = annotation.get("notes")
    if not isinstance(transcript, str):
        raise ValueError(f"{location}: transcript must be a string (empty is allowed)")
    if not isinstance(facts, list):
        raise ValueError(f"{location}: facts must be a list (empty is allowed)")
    if notes is not None and not isinstance(notes, str):
        raise ValueError(f"{location}: notes must be null or a string")
    for index, fact in enumerate(facts):
        _validate_fact(fact, location=f"{location}.facts[{index}]")
    return {
        "transcript": transcript,
        "facts": copy.deepcopy(facts),
        "notes": notes,
    }


def _validated_sheet(
    master: dict[str, dict],
    sheet_rows: list[dict],
    *,
    name: str,
) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in sheet_rows:
        if set(row) != BLINDED_KEYS:
            extra = sorted(set(row) - BLINDED_KEYS)
            missing = sorted(BLINDED_KEYS - set(row))
            raise ValueError(f"{name}: invalid blinded fields; extra={extra}, missing={missing}")
        audit_id = str(row.get("audit_id") or "")
        if audit_id not in master:
            raise ValueError(f"{name}: unknown audit_id {audit_id!r}")
        if audit_id in indexed:
            raise ValueError(f"{name}: duplicate audit_id {audit_id}")
        if row.get("audio") != master[audit_id]["audio"]:
            raise ValueError(f"{name}: audio path mismatch for {audit_id}")
        indexed[audit_id] = _validate_completed_annotation(
            row.get("annotation"), location=f"{name}:{audit_id}"
        )
    missing = sorted(set(master) - set(indexed))
    if missing:
        raise ValueError(f"{name}: missing {len(missing)} audit IDs; first={missing[0]}")
    return indexed


def merge_annotator_sheets(
    master_rows: list[dict],
    annotator_1_rows: list[dict],
    annotator_2_rows: list[dict],
) -> list[dict]:
    """Copy two complete blinded sheets into a new master packet."""
    master = _master_by_id(master_rows)
    sheets = {
        "annotator_1": _validated_sheet(master, annotator_1_rows, name="annotator_1"),
        "annotator_2": _validated_sheet(master, annotator_2_rows, name="annotator_2"),
    }
    merged = copy.deepcopy(master_rows)
    for row in merged:
        audit_id = row["audit_id"]
        human = row.setdefault("human", {})
        for annotator in ANNOTATORS:
            current = human.get(annotator) or {}
            if any(current.get(key) is not None for key in ("transcript", "facts", "notes")):
                raise ValueError(f"master already contains {annotator} labels for {audit_id}")
            human[annotator] = copy.deepcopy(sheets[annotator][audit_id])
    return merged


def make_adjudication_sheet(merged_rows: list[dict], seed: int) -> list[dict]:
    """Blind annotator identity and all automatic hypotheses from the adjudicator."""
    _master_by_id(merged_rows)
    rows: list[dict] = []
    for master in merged_rows:
        audit_id = master["audit_id"]
        human = master.get("human") or {}
        first = _validate_completed_annotation(
            human.get("annotator_1"), location=f"master:{audit_id}:annotator_1"
        )
        second = _validate_completed_annotation(
            human.get("annotator_2"), location=f"master:{audit_id}:annotator_2"
        )
        candidates = [first, second]
        random.Random(f"{seed}:adjudication:{audit_id}").shuffle(candidates)
        rows.append({
            "audit_id": audit_id,
            "audio": master["audio"],
            "candidate_a": candidates[0],
            "candidate_b": candidates[1],
            "adjudicated": _blank_annotation(),
        })
    random.Random(f"{seed}:adjudication-order").shuffle(rows)
    return rows


def merge_adjudication_sheet(
    merged_rows: list[dict], adjudication_rows: list[dict]
) -> list[dict]:
    """Copy completed adjudication into a new packet without touching extractor output."""
    master = _master_by_id(merged_rows)
    indexed: dict[str, dict] = {}
    for row in adjudication_rows:
        if set(row) != ADJUDICATION_KEYS:
            extra = sorted(set(row) - ADJUDICATION_KEYS)
            missing = sorted(ADJUDICATION_KEYS - set(row))
            raise ValueError(
                f"adjudication: invalid fields; extra={extra}, missing={missing}"
            )
        audit_id = str(row.get("audit_id") or "")
        if audit_id not in master:
            raise ValueError(f"adjudication: unknown audit_id {audit_id!r}")
        if audit_id in indexed:
            raise ValueError(f"adjudication: duplicate audit_id {audit_id}")
        if row.get("audio") != master[audit_id]["audio"]:
            raise ValueError(f"adjudication: audio path mismatch for {audit_id}")
        indexed[audit_id] = _validate_completed_annotation(
            row.get("adjudicated"), location=f"adjudication:{audit_id}"
        )
    missing = sorted(set(master) - set(indexed))
    if missing:
        raise ValueError(
            f"adjudication: missing {len(missing)} audit IDs; first={missing[0]}"
        )

    completed = copy.deepcopy(merged_rows)
    for row in completed:
        audit_id = row["audit_id"]
        human = row.setdefault("human", {})
        current = human.get("adjudicated") or {}
        if any(current.get(key) is not None for key in ("transcript", "facts", "notes")):
            raise ValueError(f"master already contains adjudicated labels for {audit_id}")
        human["adjudicated"] = copy.deepcopy(indexed[audit_id])
        human.setdefault("extractor_facts_on_adjudicated_transcript", None)
    return completed
