"""Render compact, evidence-preserving figures for the project README.

The aggregate values below are transcribed from the manuscript tables
``path_results.tex`` and ``selection_results.tex``.  Raw predictions and
per-example experiment records are intentionally not part of the public
code-only repository.

Run from any directory:

    python tools/render_readme_figures.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
PREVIEW_DIR = os.environ.get("AUDIOLLM_README_PREVIEW_DIR")

MODELS = ["Qwen-30B", "Qwen2.5-3B", "Qwen2.5-7B", "MiniCPM-o", "Step-Audio-2"]

# Full-pool reference-plan loss (%); lower is better.
SELECTION_LOSS = np.array(
    [
        [12.5, 14.6, 11.3, 8.5, 8.2, 3.8],
        [20.8, 21.7, 15.5, 10.2, 9.6, 6.6],
        [20.1, 20.1, 16.9, 13.9, 12.2, 10.3],
        [12.5, 12.6, 12.3, 10.4, 10.0, 5.9],
        [11.4, 24.6, 13.8, 6.9, 6.6, 4.8],
    ]
)
SELECTION_METHODS = ["Original", "Random", "Consistency", "Plan-Text", "Plan-Fact", "Oracle"]

# VoiceBench accuracy (%): internal spoken plan followed by three ASR readbacks.
INTERNAL_ACCURACY = np.array([81.4, 50.3, 59.8, 66.9, 49.7])
READBACK_ACCURACY = np.array(
    [
        [77.3, 75.7, 77.2],
        [46.0, 27.2, 47.3],
        [59.8, 52.3, 59.8],
        [55.5, 52.8, 53.7],
        [46.2, 41.7, 46.4],
    ]
)
ASR_LABELS = ["Whisper", "Seamless", "Fun-ASR"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": "#17202A",
        "axes.labelcolor": "#17202A",
        # Convert labels to paths so GitHub renders identically without local fonts.
        "svg.fonttype": "path",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    }
)


def annotate_heatmap(ax, values, threshold, dark="#17202A", light="white"):
    """Write exact values into a heatmap with contrast-aware text."""
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = light if values[row, column] >= threshold else dark
            ax.text(
                column,
                row,
                f"{values[row, column]:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                fontweight="bold" if column == values.shape[1] - 2 else "normal",
            )


def style_heatmap(ax, xlabels):
    ax.set_xticks(range(len(xlabels)), labels=xlabels)
    ax.set_yticks(range(len(MODELS)), labels=MODELS)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)


def save_figure(fig, name):
    fig.savefig(OUT / f"{name}.svg")
    if PREVIEW_DIR:
        preview_dir = Path(PREVIEW_DIR)
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{name}.png", dpi=180)
    plt.close(fig)


def render_selection_methods():
    cmap = LinearSegmentedColormap.from_list(
        "selection_loss", ["#E8F4F8", "#56B4E9", "#0072B2", "#17365D"]
    )
    fig, ax = plt.subplots(figsize=(9.2, 3.7))
    image = ax.imshow(SELECTION_LOSS, cmap=cmap, vmin=3.5, vmax=25.0, aspect="auto")
    style_heatmap(ax, SELECTION_METHODS)
    annotate_heatmap(ax, SELECTION_LOSS, threshold=14.5)
    ax.set_title(
        "Plan-guided selection lowers reference-plan loss", loc="left", y=1.27, pad=0
    )
    ax.text(
        0,
        1.17,
        "Full candidate pool at N = 8 · values are loss percentages (lower is better)",
        transform=ax.transAxes,
        color="#52616B",
        fontsize=9,
    )
    # Outline the proposed Plan-Fact column without hiding the underlying values.
    ax.add_patch(plt.Rectangle((3.5, -0.5), 1, len(MODELS), fill=False, ec="#E69F00", lw=2.5))
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    colorbar.set_label("Loss (%) ↓", rotation=270, labelpad=16)
    colorbar.outline.set_visible(False)
    save_figure(fig, "selection_methods")


def render_voicebench_drops():
    drops = INTERNAL_ACCURACY[:, None] - READBACK_ACCURACY
    cmap = LinearSegmentedColormap.from_list(
        "accuracy_drop", ["#F4FAF7", "#F0E442", "#E69F00", "#D55E00"]
    )
    fig, ax = plt.subplots(figsize=(7.3, 3.7))
    image = ax.imshow(drops, cmap=cmap, vmin=0.0, vmax=24.0, aspect="auto")
    style_heatmap(ax, ASR_LABELS)
    annotate_heatmap(ax, drops, threshold=12.0)
    ax.set_title(
        "Speech readback can reduce downstream task accuracy", loc="left", y=1.27, pad=0
    )
    ax.text(
        0,
        1.17,
        "VoiceBench drop from internal spoken plan to ASR readback · percentage points",
        transform=ax.transAxes,
        color="#52616B",
        fontsize=9,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.03)
    colorbar.set_label("Drop (pp) ↑", rotation=270, labelpad=16)
    colorbar.outline.set_visible(False)
    save_figure(fig, "voicebench_drops")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    render_selection_methods()
    render_voicebench_drops()
    print(f"Rendered README figures in {OUT}")


if __name__ == "__main__":
    main()
