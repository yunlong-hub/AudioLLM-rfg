#!/usr/bin/env python3
"""Collect per-ASR and union readback results into one report."""
from __future__ import annotations

import json
import os

MODES = ("asr1", "asr2", "asr3", "union12", "union13", "union123")
RUNS = {
    "d0_pilot": ("Qwen3-Omni-30B-A3B-Instruct", "Qwen2.5-Omni-3B"),
    "d1_main": ("Qwen3-Omni-30B-A3B-Instruct", "Qwen2.5-Omni-3B"),
    "d2_stepaudio": ("Step-Audio-2-mini",),
}


def suffix(mode: str) -> str:
    return "" if mode == "asr1" else f"_{mode}"


def main() -> int:
    rows = []
    for run_id, models in RUNS.items():
        for mode in MODES:
            probe_path = os.path.join("exp", run_id, "metrics", f"probe_ef{suffix(mode)}.json")
            if not os.path.exists(probe_path):
                continue
            probe = json.load(open(probe_path)).get("models", {})
            for model in models:
                block = probe.get(model)
                if not block:
                    continue
                rows.append({
                    "run_id": run_id,
                    "model": model,
                    "mode": mode,
                    "n_items": block["n_items"],
                    "delta_plan": block["delta_plan"]["gap"],
                    "delta_render": block["delta_render_SPEAK"]["gap"],
                    "retained": block["delta_render_SPEAK"]["retained"],
                    "total": block["delta_render_SPEAK"]["total"],
                })

    lines = [
        "# ASR readback sensitivity ladder",
        "",
        "Single-ASR rows quantify instrument sensitivity; union rows report the facts recovered by",
        "at least one named recognizer.  The union is an optimistic sensitivity analysis, not ASR ground truth.",
        "",
        "| run | model | readback | n | $\\Delta_{plan}$ | $\\Delta_{render}$ | retained/total |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_id']} | {row['model']} | `{row['mode']}` | {row['n_items']} | "
            f"{row['delta_plan']:.4f} | {row['delta_render']:.4f} | "
            f"{row['retained']}/{row['total']} |"
        )

    os.makedirs("reports", exist_ok=True)
    with open("reports/asr_readback_ladder.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open("reports/asr_readback_ladder.json", "w") as fh:
        json.dump({"rows": rows}, fh, indent=2, ensure_ascii=False)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
