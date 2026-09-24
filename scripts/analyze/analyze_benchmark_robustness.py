#!/usr/bin/env python3
"""Produce duration-stratified paired MiniCPM benchmark evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rfg.facts.readback_state import completed_text
from rfg.score.benchmark import answers_equal, extract_answer
from rfg.score.benchmark_robustness import stratified_summary


ASRS = ("asr1", "asr2", "asr3")


def load_rows(items_path: Path, prediction_root: Path) -> list[dict]:
    rows = []
    for line in items_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        path = prediction_root / item["id"] / "SPEAK.json"
        record = json.loads(path.read_text())
        if record.get("error"):
            raise RuntimeError(f"{path}: {record['error']}")
        duration = record.get("audio_duration_sec")
        if duration is None:
            raise RuntimeError(f"{path}: missing audio_duration_sec")
        dataset = item["category"]
        task = item.get("difficulty")
        gold = str(item["expected_answer_text"])
        readback = record.get("readback") or {}
        texts = {"internal": record.get("text") or record.get("text_raw") or ""}
        for source in ASRS:
            text = completed_text(readback, source)
            if text is None:
                raise RuntimeError(f"{path}: missing completed {source}")
            texts[source] = text
        extracted = {source: extract_answer(dataset, task, text)
                     for source, text in texts.items()}
        correct = {source: answers_equal(dataset, task, value, gold)
                   for source, value in extracted.items()}
        rows.append({
            "item_id": item["id"],
            "duration_sec": float(duration),
            "correct": correct,
            "extracted": extracted,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="MiniCPM-o-4_5")
    parser.add_argument("--exp-root", default="exp")
    parser.add_argument("--output", default="reports/minicpmo45_benchmark_robustness.json")
    parser.add_argument("--report", default="reports/minicpmo45_benchmark_robustness.md")
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()

    specs = {
        "Spoken-MQA": (
            Path("data/omg_benchmarks/prepared/spoken_mqa/items.jsonl"),
            Path(args.exp_root) / "omg_spoken_mqa" / "predictions" / args.model,
        ),
        "VoiceBench-BBH Short": (
            Path("data/omg_benchmarks/prepared/voicebench_bbh/items_short.jsonl"),
            Path(args.exp_root) / "omg_voicebench_short" / "predictions" / args.model,
        ),
    }
    result = {"model": args.model, "duration_bins_sec": [15, 30], "datasets": {}}
    for index, (name, (items, predictions)) in enumerate(specs.items()):
        rows = load_rows(items, predictions)
        result["datasets"][name] = stratified_summary(
            rows, seed=20260920 + index * 100, n_boot=args.n_boot,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# MiniCPM-o-4.5 natural-benchmark duration robustness", "",
        "Accuracy and paired internal-minus-ASR drops use the same items in each duration stratum; "
        "confidence intervals are 10,000-draw paired bootstrap intervals.", "",
        "| Dataset | Duration | n | Internal | ASR1 | Drop (95% CI) | ASR2 | Drop (95% CI) | ASR3 | Drop (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"all": "all", "le15": "≤15s", "15to30": "15–30s",
              "le30": "≤30s", "gt30": ">30s"}
    for dataset, groups in result["datasets"].items():
        for group in ("all", "le15", "15to30", "le30", "gt30"):
            block = groups[group]
            if not block.get("n"):
                continue
            acc = block["accuracy"]
            drops = block["paired_drop_from_internal"]
            cells = []
            for source in ASRS:
                ci = drops[source]["ci95"]
                cells.extend([
                    f"{acc[source]:.4f}",
                    f"{drops[source]['mean']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]",
                ])
            lines.append(
                f"| {dataset} | {labels[group]} | {block['n']} | {acc['internal']:.4f} | "
                + " | ".join(cells) + " |"
            )
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"written -> {output} | {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
