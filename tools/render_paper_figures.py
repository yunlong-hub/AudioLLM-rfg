"""Render manuscript result tables and a trend figure from experiment artifacts.

Run from any directory: python tools/render_paper_figures.py
Outputs: manuscript result tables and figures/candidate_curve.{pdf,svg}
Previews and the exact plotted values are stored in the manuscript build directory.
Numerical comparisons use LaTeX tables; only the ordered candidate-budget
relationship uses a plot with physical-size typography.
"""
from pathlib import Path
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "papers/AudioLLM-rfg"
OUT = PAPER / "figures"
BUILD = PAPER / "build/figures"
MODELS = [
    ("Qwen3-Omni-30B-A3B-Instruct", "Qwen-30B"),
    ("Qwen2.5-Omni-3B", "Qwen2.5-3B"),
    ("Qwen2.5-Omni-7B", "Qwen2.5-7B"),
    ("MiniCPM-o-4_5", "MiniCPM-o"),
    ("Step-Audio-2-mini", "Step-Audio-2"),
]
ASRS = [("asr1", "Whisper", "#0072B2", "o"),
        ("asr2", "Seamless", "#CC79A7", "s"),
        ("asr3", "Fun-ASR", "#009E73", "^")]
MODEL_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#595959"]
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
    "axes.titlesize": 8, "axes.titleweight": "normal",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#A0A0A0", "axes.linewidth": .5,
    "xtick.major.size": 2.5, "ytick.major.size": 0,
    "text.color": "#222222", "axes.labelcolor": "#333333",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def report_block(relative, model):
    """Recover pooled ratios from a checked-in probe report's retained/total."""
    text = (ROOT / relative).read_text()
    section = next(s for s in text.split("## ")[1:] if s.startswith(model))
    rows = re.findall(r"\| `([^`]+)` \| [^|]+ \| (\d+)/(\d+) \|", section)
    result = {key: 1 - int(retained) / int(total) for key, retained, total in rows}
    return result


def controlled_results():
    data = {}
    audit_files = {"asr1": "numbers_audit.json", "asr2": "numbers_audit_asr2.json",
                   "asr3": "numbers_audit_asr3.json", "union": "numbers_audit_union123.json"}
    for asr, filename in audit_files.items():
        audit = read_json("reports/" + filename)["runs"]
        for model, _ in MODELS:
            if model in ("Qwen2.5-Omni-7B", "MiniCPM-o-4_5"):
                suffix = "union123" if asr == "union" else asr
                stem = ("probe_ef_d1_main_qwen25_7b_vllm_" if model.startswith("Qwen")
                        else "probe_ef_d1_main_")
                block = report_block("reports/" + stem + suffix + ".md", model)
                plan, render = block["delta_plan"], block["delta_render_SPEAK"]
            else:
                run = "d2_stepaudio" if model.startswith("Step") else "d1_main"
                block = audit[f"{run}/{model}"]["decompose"]
                plan, render = block["plan"]["gap"], block["render_SPEAK"]["gap"]
            data.setdefault(model, {"plan": plan})[asr] = render
    return data


def save(fig, name):
    for extension in ("pdf", "svg"):
        fig.savefig(OUT / f"{name}.{extension}")
    fig.savefig(BUILD / f"{name}.png", dpi=200)
    plt.close(fig)


def candidate_curve(curves):
    """Use a curve only for the ordered candidate-budget relationship."""
    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    fig.subplots_adjust(left=.13, right=.98, bottom=.22, top=.75)
    for idx, ((model, label), color) in enumerate(zip(MODELS, MODEL_COLORS)):
        values = [100*curves[model]["curve"][str(n)]["macro_frr_reduction"]
                  for n in (1, 2, 4, 8)]
        ax.plot([1, 2, 4, 8], values, label=label, color=color,
                marker=["o", "s", "^", "P", "D"][idx], markersize=3,
                linestyle=["-", "-", "--", "-.", ":"][idx], linewidth=1.0)
    ax.set(xlabel="Number of candidate waveforms", ylabel="Loss reduction (pp)",
           xticks=[1, 2, 4, 8], yticks=[0, 4, 8, 12], ylim=(-.3, 12.5))
    ax.grid(color="#E0E4E8", linewidth=.5)
    ax.legend(loc="lower left", bbox_to_anchor=(-.04, 1.01), ncol=3,
              frameon=False, fontsize=6.7, handlelength=1.5, columnspacing=.7)
    save(fig, "candidate_curve")


def write_result_tables(controlled, natural, curves, strict, reference_free):
    """Generate exact lookup tables from the same artifacts used by the analysis."""
    out = PAPER / "tables"
    out.mkdir(exist_ok=True)
    counts = [(586, 586), (600, 600), (600, 600), (600, 600), (137, 142)]
    lines = [
        r"\begin{tabular}{@{}lc*{5}{S[table-format=2.2]}"
        r"@{\hspace{9pt}}*{4}{S[table-format=2.1]}@{}}",
        r"\toprule",
        r" & & \multicolumn{5}{c}{Controlled fact loss (\%) $\downarrow$}"
        r" & \multicolumn{4}{c}{VoiceBench accuracy (\%) $\uparrow$} \\",
        r"\cmidrule(lr){3-7}\cmidrule(lr){8-11}",
        r"Model & $n_P/n_R$ & {Plan} & {Render W} & {Render S} & {Render F} & {Union}"
        r" & {Internal} & {W} & {S} & {F} \\",
        r"\midrule",
    ]
    for (model, label), (np_, nr) in zip(MODELS, counts):
        d, n = controlled[model], natural[model]
        fields = [label, f"{np_}/{nr}"]
        fields += [f"{100*d[k]:.2f}" for k in ("plan", "asr1", "asr2", "asr3", "union")]
        fields += [f'{100*n["conditions"]["SPEAK"]["accuracy"]:.1f}']
        fields += [f'{100*n["spoken_answer"][k]["accuracy"]:.1f}' for k, *_ in ASRS]
        lines.append(" & ".join(fields) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "path_results.tex").write_text("\n".join(lines) + "\n")

    lines = [
        r"\begin{tabular}{@{}lS[table-format=3.0]"
        r"*{6}{S[table-format=2.1]}@{\hspace{9pt}}S[table-format=3.0]"
        r"*{3}{S[table-format=2.1]@{\,}l}@{}}",
        r"\toprule",
        r" & \multicolumn{7}{c}{Full pool: reference-plan loss (\%)}"
        r" & \multicolumn{7}{c}{Identical-text subset: reduction [95\% CI] (pp)} \\",
        r"\cmidrule(lr){2-8}\cmidrule(lr){9-15}",
        r"Model & {$n$} & {Original} & {Random} & {Cons.} & {Plan-Text} & {Plan-Fact} & {Oracle} & {$n$}"
        r" & \multicolumn{2}{c}{Whisper} & \multicolumn{2}{c}{Seamless}"
        r" & \multicolumn{2}{c}{Fun-ASR} \\",
        r"\midrule",
    ]
    for model, label in MODELS:
        folds = curves[model]["curve"]["8"]["folds"]
        fields = [label, str(folds["asr1"]["n_items"])]
        fields += [f'{100*np.mean([f["delta_render_"+key] for f in folds.values()]):.1f}'
                   for key in ("single", "random")]
        fields += [f'{100*np.mean([f[key]["loss_mean"] for f in reference_free[model]["folds"].values()]):.1f}'
                   for key in ("fact_consistency", "plan_text")]
        fields += [f'{100*np.mean([f["delta_render_"+key] for f in folds.values()]):.1f}'
                   for key in ("selected", "oracle")]
        fields += [str(strict[model]["n_items_text_identical"])]
        for key, *_ in ASRS:
            f = strict[model]["folds"][key]
            value = 100*f["frr_reduction"]
            lo, hi = [100*x for x in f["frr_ci95"]]
            fields.extend([f"{value:.1f}", "$" + f"[{lo:.1f}, {hi:.1f}]" + "$"])
        lines.append(" & ".join(fields) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "selection_results.tex").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    controlled = controlled_results()
    natural = {model: read_json(f"exp/omg_voicebench_short/metrics/benchmark_{model}.json")
               for model, _ in MODELS}
    curves = {model: read_json(f"exp/d0_resample/frr_curve_{model}.json") for model, _ in MODELS}
    strict = read_json("reports/rerender_plan_consistency.json")["models"]
    reference_free = read_json("reports/supplementary_e3_reference_free.json")["models"]
    write_result_tables(controlled, natural, curves, strict, reference_free)
    candidate_curve(curves)
    # Preserve exact displayed values and source identities without duplicating full experiment files.
    snapshot = {"controlled": controlled,
                "natural": {m: {"internal": d["conditions"]["SPEAK"]["accuracy"],
                                   **{k: v["accuracy"] for k, v in d["spoken_answer"].items()}}
                            for m, d in natural.items()},
                "strict_text_subset": strict,
                "plan_free_consistency": {
                    model: {
                        "source": "reports/supplementary_e3_reference_free.json",
                        "loss_mean": float(np.mean([
                            fold["fact_consistency"]["loss_mean"]
                            for fold in result["folds"].values()
                        ])),
                        "paired_reduction_by_fold": {
                            asr: fold["plan_minus_factconsistency"]
                            for asr, fold in result["folds"].items()
                        },
                    } for model, result in reference_free.items()
                },
                "plan_text_selection": {
                    model: {
                        "source": "reports/supplementary_e3_reference_free.json",
                        "loss_mean": float(np.mean([
                            fold["plan_text"]["loss_mean"]
                            for fold in result["folds"].values()
                        ])),
                        "paired_reduction_by_fold": {
                            asr: fold["plan_minus_plantext"]
                            for asr, fold in result["folds"].items()
                        },
                    } for model, result in reference_free.items()
                },
                "rerendering": {m: {n: {"macro": c["macro_frr_reduction"], "folds": c["folds"]}
                                     for n, c in d["curve"].items()} for m, d in curves.items()}}
    (BUILD / "plotted_values.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print("Rendered two detailed LaTeX result tables and the candidate-budget curve.")
    print(json.dumps(controlled, indent=2))


if __name__ == "__main__":
    main()
