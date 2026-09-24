#!/usr/bin/env python3
"""Import the two E1-a label files, validate them, and build the adjudication sheet.

Input : the `annotator_X.labels.jsonl` files exported by the listening pages
        (or the pre-filled `annotator_X.jsonl` from `build_e1a_listening_packet`).
Output: data/adjudication/e1a60/annotator_1.completed.jsonl
        data/adjudication/e1a60/annotator_2.completed.jsonl
        data/adjudication/e1a60/e1a60.annotators.jsonl      (merged, for the record)
        data/adjudication/e1a60/adjudication.jsonl          (sheet for the third listener)

The adjudication sheet shows only the anonymous audio and the two transcripts;
model identity, internal text, and every ASR hypothesis stay in the master file
and are rejoined later, during scoring.

Reports inter-annotator agreement so the paper can state how reliable the human
transcription is.  Refuses to overwrite an existing artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rfg.score.textnorm import wer_norm  # noqa: E402

DEFAULT_PACKET = ROOT / "data/adjudication/e1a60"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_labels(path: Path, ids: set[str]) -> dict[str, dict]:
    rows = read_jsonl(path)
    indexed: dict[str, dict] = {}
    for row in rows:
        audit_id = str(row.get("audit_id") or "")
        if not audit_id or audit_id in indexed:
            raise ValueError(f"missing or duplicate audit_id in {path}: {audit_id!r}")
        indexed[audit_id] = row.get("annotation") or {}
    if set(indexed) != ids:
        missing = sorted(ids - set(indexed))
        extra = sorted(set(indexed) - ids)
        raise ValueError(f"{path} does not cover the packet; missing={missing[:5]} extra={extra[:5]}")
    unfinished = [i for i, a in indexed.items() if not isinstance(a.get("transcript"), str)]
    if unfinished:
        raise ValueError(f"{path} has {len(unfinished)} unfinished items, e.g. {unfinished[:5]}")
    return indexed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--annotator-1", type=Path, required=True)
    parser.add_argument("--annotator-2", type=Path, required=True)
    args = parser.parse_args()

    master_path = args.packet / "e1a60.master.jsonl"
    master = read_jsonl(master_path)
    ids = {row["audit_id"] for row in master}

    labels = {
        "annotator_1": load_labels(args.annotator_1, ids),
        "annotator_2": load_labels(args.annotator_2, ids),
    }

    outputs = {
        "annotator_1": args.packet / "annotator_1.completed.jsonl",
        "annotator_2": args.packet / "annotator_2.completed.jsonl",
        "merged": args.packet / "e1a60.annotators.jsonl",
        "adjudication": args.packet / "adjudication.jsonl",
    }
    for path in outputs.values():
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")

    for role in ("annotator_1", "annotator_2"):
        with outputs[role].open("x", encoding="utf-8") as handle:
            for row in master:
                handle.write(json.dumps(
                    {"audit_id": row["audit_id"], "audio": f"audio/{row['audit_id']}.wav",
                     "annotation": labels[role][row["audit_id"]]}, ensure_ascii=False) + "\n")

    disagreements = 0
    wer_values: list[float] = []
    with outputs["merged"].open("x", encoding="utf-8") as merged, \
            outputs["adjudication"].open("x", encoding="utf-8") as sheet:
        for row in master:
            audit_id = row["audit_id"]
            a = labels["annotator_1"][audit_id]
            b = labels["annotator_2"][audit_id]
            ta = a.get("transcript") or ""
            tb = b.get("transcript") or ""
            value = wer_norm(ta, tb)
            wer_values.append(value)
            different = ta.strip() != tb.strip()
            disagreements += int(different)
            merged.write(json.dumps({
                "audit_id": audit_id, "stratum": row["stratum"], "category": row["category"],
                "model": row["model"], "run_id": row["run_id"], "item_id": row["item_id"],
                "annotator_1": a, "annotator_2": b,
                "transcript_wer": value, "different": different,
            }, ensure_ascii=False) + "\n")
            sheet.write(json.dumps({
                "audit_id": audit_id,
                "audio": f"audio/{audit_id}.wav",
                "candidate_a": ta, "candidate_b": tb,
                "adjudicated": {"transcript": None, "facts": [], "notes": None},
            }, ensure_ascii=False) + "\n")

    try:
        packet_label = str(args.packet.relative_to(ROOT))
    except ValueError:
        packet_label = str(args.packet)
    summary = {
        "packet": packet_label,
        "n_items": len(master),
        "n_disagreements": disagreements,
        "agreement_rate": 1 - disagreements / len(master),
        "mean_transcript_wer": sum(wer_values) / len(wer_values),
        "annotator_1": str(args.annotator_1),
        "annotator_2": str(args.annotator_2),
        "next": "give adjudication.jsonl to the third listener, then run "
                "scripts/analyze/finalize_e1a_packet.py",
    }
    (args.packet / "e1a60.agreement.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
