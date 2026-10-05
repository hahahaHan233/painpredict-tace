"""Horizontal points nomogram drawn from the published raw-scale equation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .predict import model_card

LABELS = {
    "Pre_PainScore": "Preprocedural NRS (0-10)",
    "D-dimer": "D-dimer (ng/mL)",
    "AFP": "Alpha-fetoprotein (ng/mL)",
    "Albumin": "Albumin (g/L)",
}
TICKS = {
    "D-dimer": [50.0, 250.0, 500.0, 750.0, 950.0],
    "AFP": [25.0, 75.0, 125.0],
    "Albumin": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
}
RISK_TICKS = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
TITLE = "Nomogram for NRS \u2265 4 at 24 hours after TACE"
FOOTNOTE = (
    "Scales cover complete observations within the development range. When D-dimer, AFP or albumin is missing, "
    "use painpredict.predict_risk, which applies median imputation and missing-indicator terms."
)


def points_table(n_values: int = 201) -> tuple[pd.DataFrame, float, float]:
    """Points per predictor value, the points-per-logit scale and the minimum linear predictor."""

    card = model_card()
    equation = card["equation"]
    spans, rows = {}, []
    terms = {term["variable"]: term for term in equation["terms"]}
    for variable, (low, high) in card["nomogram_ranges"].items():
        values = np.linspace(low, high, n_values)
        effects = values * terms[variable]["coefficient"]
        spans[variable] = (values, effects)
    scale = max(effects.max() - effects.min() for _, effects in spans.values())
    minimum_logit = equation["intercept"] + sum(effects.min() for _, effects in spans.values())
    for variable, (values, effects) in spans.items():
        points = (effects - effects.min()) / scale * 100
        rows.extend({"term": variable, "value": v, "points": p} for v, p in zip(values, points))
    return pd.DataFrame(rows), scale, minimum_logit


def draw_nomogram(directory: str | Path, stem: str = "figure6_nomogram") -> pd.DataFrame:
    """Save the nomogram as PNG, PDF and SVG and return the points table."""

    import matplotlib.pyplot as plt

    table, scale, minimum_logit = points_table()
    variables = list(dict.fromkeys(table.term))
    total_max = table.groupby("term").points.max().sum()
    offset = 43.0

    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    fig.subplots_adjust(left=0.04, right=0.97, top=0.93, bottom=0.12)
    ax.set_xlim(-35, offset + total_max + 8)
    ax.set_ylim(-0.2, len(variables) + 3.2)
    ax.axis("off")
    serif = {"family": "DejaVu Serif"}

    def axis_line(y, ticks, labels, label, minor=None):
        ax.hlines(y, float(np.min(ticks)), float(np.max(ticks)), color="black", linewidth=1.05)
        ax.vlines(ticks, y, y - 0.105, color="black", linewidth=0.9)
        if minor is not None and len(minor):
            ax.vlines(minor, y, y - 0.055, color="black", linewidth=0.6)
        for point, text in zip(ticks, labels):
            ax.text(point, y - 0.23, text, ha="center", va="top", fontsize=8.2, **serif)
        ax.text(-34, y + 0.02, label, ha="left", va="center", fontsize=10.8, **serif)

    top = len(variables) + 2.5
    ticks = np.arange(0, 101, 10, dtype=float)
    axis_line(top, offset + ticks, [str(int(v)) for v in ticks], "Points", minor=offset + np.arange(0, 101, 2, dtype=float))
    for index, variable in enumerate(variables):
        rows = table.loc[table.term.eq(variable)]
        values, points = rows.value.to_numpy(), rows.points.to_numpy()
        tick_values = np.asarray(TICKS.get(variable, np.arange(np.ceil(values.min()), np.floor(values.max()) + 1)))
        tick_values = tick_values[(tick_values >= values.min()) & (tick_values <= values.max())]
        order = np.argsort(values)
        tick_points = offset + np.interp(tick_values, values[order], points[order])
        labels = [f"{v:.0f}" for v in tick_values]
        axis_line(len(variables) + 1.5 - index, tick_points, labels, LABELS[variable])

    total_ticks = np.arange(0, np.ceil(total_max / 50) * 50 + 1, 50, dtype=float)
    total_ticks = total_ticks[total_ticks <= total_max + 1e-9]
    axis_line(1.0, offset + total_ticks, [str(int(v)) for v in total_ticks], "Total Points",
              minor=offset + np.arange(0, total_max + 1, 10, dtype=float))
    risk_points = (np.log(RISK_TICKS / (1 - RISK_TICKS)) - minimum_logit) / scale * 100
    kept = (risk_points >= 0) & (risk_points <= total_max)
    axis_line(0.0, offset + risk_points[kept], [f"{v:.2f}" for v in RISK_TICKS[kept]], "Predicted risk of NRS \u2265 4")
    ax.text(-34, len(variables) + 3.0, TITLE, fontsize=13.5, fontweight="bold", **serif)
    ax.text(-34, -0.55, FOOTNOTE, fontsize=6.6, **serif)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(directory / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.05, dpi=350)
    plt.close(fig)
    return table
