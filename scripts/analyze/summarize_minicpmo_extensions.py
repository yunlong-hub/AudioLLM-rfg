#!/usr/bin/env python3
"""Summarize completed automated MiniCPM-o extension experiments."""
from __future__ import annotations

import json
from pathlib import Path


MODEL = "MiniCPM-o-4_5"


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> int:
    validation = load(f"exp/d0_resample/validation_{MODEL}.json")
    loo = load(f"exp/d0_resample/frr_loo_{MODEL}.json")
    curve = load(f"exp/d0_resample/frr_curve_{MODEL}.json")
    calibration = load("reports/asr_calibration_d1_main.json")["models"][MODEL]
    robustness = load("reports/minicpmo45_benchmark_robustness.json")["datasets"]
    rerender = {
        mode: load(f"exp/d4_rerender/metrics/rerender_judge_{MODEL}_{mode}.json")
        for mode in ("asr1", "asr2", "asr3")
    }

    lines = [
        "# MiniCPM-o-4.5 automated extension experiments", "",
        "## N=1/2/4/8 fixed-text resampling and independent FRR", "",
        f"Candidate validation: {validation['n_items']} items, "
        f"{validation['n_json']}/{validation['expected']} cumulative R-candidate JSON, "
        f"{validation['n_wav']}/{validation['expected']} WAV, "
        f"{validation['n_complete_three_asr']}/{validation['expected']} complete three-ASR records; "
        f"errors={validation['n_errors']}.", "",
        "| Held-out evaluator | Single gap | Random gap | Selected gap | Oracle gap | FRR reduction | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, fold in loo["folds"].items():
        ci = fold["frr_ci95"]
        lines.append(
            f"| `{mode}` | {fold['delta_render_single']:.4f} | "
            f"{fold['delta_render_random']:.4f} | {fold['delta_render_selected']:.4f} | "
            f"{fold['delta_render_oracle']:.4f} | {fold['frr_reduction']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] |"
        )
    lines += [
        "", f"Macro FRR reduction: **{loo['macro_frr_reduction']:.4f}**; "
        f"all fold CIs exclude zero: **{loo['all_fold_cis_exclude_zero']}**.", "",
        "### Candidate-count curve", "",
        "| Candidates | Macro FRR reduction | Minimum fold reduction |",
        "|---:|---:|---:|",
    ]
    for count in curve["candidate_counts"]:
        block = curve["curve"][str(count)]
        lines.append(f"| {count} | {block['macro_frr_reduction']:.4f} | "
                     f"{block['min_fold_frr_reduction']:.4f} |")

    lines += ["", "## Fixed-text ASR calibration", "",
              "| System | Mean WER | Rule-fact recall | Addition rate |",
              "|---|---:|---:|---:|"]
    for mode in ("asr1", "asr2", "asr3", "union123"):
        block = calibration["systems"][mode]
        wer = "--" if block.get("mean_wer") is None else f"{block['mean_wer']:.4f}"
        lines.append(f"| `{mode}` | {wer} | {block['fact_recall']:.4f} | "
                     f"{block['addition_rate']:.4f} |")

    lines += ["", "## Natural-benchmark duration robustness", "",
              "| Dataset | Stratum | n | Internal | ASR1 drop | ASR2 drop | ASR3 drop |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for dataset, groups in robustness.items():
        for group in ("all", "le30", "gt30"):
            block = groups[group]
            drops = block["paired_drop_from_internal"]
            lines.append(f"| {dataset} | `{group}` | {block['n']} | "
                         f"{block['accuracy']['internal']:.4f} | "
                         f"{drops['asr1']['mean']:.4f} | {drops['asr2']['mean']:.4f} | "
                         f"{drops['asr3']['mean']:.4f} |")

    lines += ["", "## Targeted rerendering of persistent failures", "",
              "| Evaluator | Target facts | ORIG survival | Rewritten survival | Improved |",
              "|---|---:|---:|---:|---:|"]
    for mode, block in rerender.items():
        lines.append(f"| `{mode}` | {block['n']} | {block['base']:.4f} | "
                     f"{block['rewritten']:.4f} | {block['improved']}/{block['n']} |")

    lines += [
        "", "## Human-audit handoff", "",
        "The automated stratified packet contains 100 MiniCPM items and intentionally leaves all "
        "human transcript, fact, and adjudication fields empty: "
        "`data/adjudication/minicpmo_three_asr_sample100.jsonl`.",
    ]
    output = Path("reports/minicpmo45_automated_extensions.md")
    output.write_text("\n".join(lines) + "\n")
    print(f"written -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
