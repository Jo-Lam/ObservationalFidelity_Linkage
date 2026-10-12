"""Figure: synthetic linkage studies mapped by structural/statistical fidelity and Observational Fidelity.

Reads docs/literature_map.csv (also the source of Supplementary Table S2) and
writes outputs/manuscript_figures/fig2_literature_map.png.

Axes are ordinal levels, not measurements:
* x (structural and statistical fidelity) follows the population checks in the
  OF gate: format only -> plausible values -> observed frequencies -> joint
  identifier structure -> joint structure with attributes and change over time;
* y (error-model layer) follows the four-layer OF taxonomy in the Introduction,
  with level 0 for "errors not modelled".
Marker style shows where the error parameters came from, the "observational"
part of OF: specified by the researcher, estimated from labelled data, or
estimated without labels and checked against the observed task.

Usage: python scripts/literature_map.py
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

from ofeval import figures as F

ROOT = Path(__file__).resolve().parents[1]
X_LEVELS = ["Format only", "Plausible values (lookup tables, fields independent)",
            "Observed frequencies and cardinality", "Joint identifier structure (name pairs, shared names, subgroup)",
            "Joint structure with attributes and change over time"]
Y_LEVELS = ["Errors not modelled", "1. Marginal error rates and types", "2. Cross-field co-occurrence",
            "3. Value- or subgroup-dependent errors", "4. Mechanism-specific dependency (type, position, extent)"]
STYLE = {
    "specified": dict(marker="o", s=70, facecolor=F.SURFACE, edgecolor=F.INK_2, linewidth=1.4,
                      label="Error parameters specified by the researcher"),
    "labelled": dict(marker="o", s=90, facecolor=F.SERIES[0], edgecolor=F.SURFACE, linewidth=1.4,
                     label="Estimated from labelled (gold-standard) data"),
    # Colours match the settings in Figures 3-7: labelled = Supervised blue,
    # pseudo-truth = Pseudo-truth orange, this study = Calibrated yellow.
    "pseudotruth": dict(marker="o", s=90, facecolor=F.SERIES[1], edgecolor=F.SURFACE, linewidth=1.4,
                        label="Estimated from unlabelled real data (high-scoring pairs), not checked"),
    "calibrated": dict(marker="D", s=100, facecolor=F.SERIES[3], edgecolor=F.INK_2, linewidth=1.0,
                       label="Estimated without labels, checked against the observed task (this study)"),
}


def main() -> None:
    df = pd.read_csv(ROOT / "docs/literature_map.csv")
    out = ROOT / "outputs/manuscript_figures"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.grid(True, axis="both", color=F.GRID, linewidth=0.6, zorder=0)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)

    # Configurable generators: the range their settings can reach, not a single study.
    ax.add_patch(Rectangle((0.55, 0.55), 1.9, 1.9, facecolor=F.GRID, alpha=0.45, edgecolor="none", zorder=1))
    ax.text(0.62, 2.38, "Configurable generators\n(Febrl, GeCo, GeCkO):\nrange of settings", fontsize=7.5,
            color=F.INK_2, va="top", zorder=2)

    sim = df[df["group"].eq("simulation")]
    rest = df[~df["group"].eq("simulation")].copy()
    rest["label"] = [f"{r['label']}†" if str(r["basis"]).startswith("carried over") else r["label"]
                     for _, r in rest.iterrows()]
    if len(sim):  # the simulation studies share one marker
        group = pd.DataFrame([{"label": f"{len(sim)} simulation\nstudies*", "error_source": "specified",
                               "x_level": sim["x_level"].iloc[0], "y_level": sim["y_level"].iloc[0]}])
        rest = pd.concat([group, rest], ignore_index=True)  # first in its cell: leftmost, label above
    # Spread studies that share a cell; alternate labels above and below.
    for (x, y), cell in rest.groupby(["x_level", "y_level"]):
        n = len(cell)
        for i, (_, r) in enumerate(cell.iterrows()):
            dx = (i - (n - 1) / 2) * 0.34
            ax.scatter([x + dx], [y], zorder=4, **{k: v for k, v in STYLE[r["error_source"]].items() if k != "label"})
            above = i % 2 == 0
            ax.annotate(r["label"], (x + dx, y), xytext=(0, 9 if above else -10), textcoords="offset points",
                        ha="center", va="bottom" if above else "top", fontsize=8, color=F.INK,
                        fontweight="bold" if r["error_source"] == "calibrated" else "normal", zorder=5)
    this = df[df["error_source"].eq("calibrated")].iloc[0]
    ax.add_patch(FancyArrowPatch((this["x_level"], this["y_level"] + 0.12), (this["x_level"], this["y_level"] + 0.82),
                                 arrowstyle="-|>", mutation_scale=10, color=F.INK_2, linewidth=1.2,
                                 linestyle=(0, (3, 2)), zorder=3))
    ax.text(this["x_level"] + 0.06, this["y_level"] + 0.5, "subgroup-dependent\nerrors (future work)", fontsize=7.5,
            color=F.INK_2, va="center")

    ax.set_xticks(range(5))
    ax.set_xticklabels([textwrap.fill(s, 18) for s in X_LEVELS], fontsize=7.5)
    ax.set_yticks(range(5))
    ax.set_yticklabels([textwrap.fill(s, 30) for s in Y_LEVELS], fontsize=7.5)
    ax.set_xlabel("Structural and statistical fidelity of the identifiers")
    ax.set_ylabel("Error model (Observational Fidelity layer)")
    handles = [plt.scatter([], [], **STYLE[k]) for k in ["specified", "labelled", "pseudotruth", "calibrated"]]
    ax.legend(handles=handles, loc="upper left", fontsize=8, title="Source of error parameters",
              title_fontsize=8, frameon=True, facecolor=F.SURFACE, edgecolor=F.GRID)
    sims = "; ".join(sim.sort_values("year")["label"])
    carried = "†Placement carried over from the draft, pending full-text review. " \
        if df["basis"].astype(str).str.startswith("carried over").any() else ""
    fig.text(0.01, 0.0, textwrap.fill(
        f"*Simulation studies varying overall error or missingness rates: {sims}. {carried}Levels are ordinal. "
        "Moving up and to the right brings replicas closer to the real task and raises disclosure risk. Lam 2026 "
        "uses real records with simulated name changes. Details in Supplementary Table S2.", 150), fontsize=7.5,
        color=F.INK_2, va="top")
    fig.tight_layout()  # no in-figure title: the manuscript caption carries it
    fig.savefig(out / "fig2_literature_map.png", bbox_inches="tight")
    plt.close(fig)
    print(out / "fig2_literature_map.png")


if __name__ == "__main__":
    main()
