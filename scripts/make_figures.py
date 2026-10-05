"""Regenerate the figures that can be rebuilt from published results.

Figure 0 (analysis workflow) and Figure 6 (nomogram) are drawn here.
Figures 1-5 require individual-level data and are provided as released.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from painpredict.nomogram import draw_nomogram

FIGURES = Path(__file__).resolve().parents[1] / "figures"
NAVY, BLUE, FILL = "#24364B", "#4C78A8", "#E8EEF3"


def draw_workflow(directory: Path) -> None:
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(8.2, 2.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    labels = [
        "Development\ncohort\n356 patients\n74 events",
        "Descriptive\nassociations\n(not used for\nselection)",
        "Preprocessing\nfitted within\ntraining folds",
        "Repeated\nnested CV\n5 x 5-fold outer\n4-fold inner",
        "Four-variable\nridge logistic\nmodel",
        "Out-of-fold\nmetrics\n2,000 bootstraps\nexplanation",
    ]
    width, height, y = 0.145, 0.62, 0.22
    xs = [0.015, 0.18, 0.345, 0.51, 0.675, 0.84]
    for x, label in zip(xs, labels):
        ax.add_patch(plt.Rectangle((x, y), width, height, facecolor=FILL, edgecolor=NAVY, lw=1.0))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", color=NAVY)
    for left, right in zip(xs, xs[1:]):
        ax.annotate("", xy=(right, y + height / 2), xytext=(left + width, y + height / 2),
                    arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.1})
    ax.set_title("Analysis workflow", loc="left", color=NAVY, fontsize=9)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(directory / f"figure0_workflow.{suffix}", bbox_inches="tight", pad_inches=0.04, dpi=350)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    draw_workflow(FIGURES)
    draw_nomogram(FIGURES)
    print(f"Saved Figure 0 and Figure 6 to {FIGURES}")
