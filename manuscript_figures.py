"""Manuscript figures 3-7 from the scale-up study and the scale redraw study.

Conventions shared with Figure 2 (scripts/literature_map.py), so the figures read as one set:
* no in-figure titles: the captions in the manuscript carry them;
* one colour per setting in every figure (validated default palette, slots 1-6):
  Supervised blue, Pseudo-truth orange, Pseudo-truth + clerical substitution aqua,
  Calibrated yellow, Calibrated + clerical pink, Clerical sample alone green;
  Observed is neutral grey;
* error types (Figure 3) use slots no setting uses (violet, red, dark grey);
* differences are labelled "Replica minus observed (pp)" (in the redraw study the
  observed task's value is the true value);
* manuscript terms: "errors in both names", "ethno-racial group".
"Pseudo-truth only" is dropped: its replicas are identical to gated pseudo-truth
(the gate judges the generator and does not filter replicas).

Inputs: outputs/study_scale/tables (configs/study_scale.yaml) and
outputs/calibration_sweep_scale/sweep_block_level.csv (scripts/calibration_sweep.py).
Usage: python scripts/manuscript_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ofeval import figures as F
from ofeval.experiment import OBSERVED

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs/study_scale/tables"
OUT = ROOT / "outputs/manuscript_figures"

RENAME = {
    "Supervised": "Supervised",
    "Pseudo-truth + label-free gate": "Pseudo-truth",
    "Partly supervised + label-free gate": "Pseudo-truth + clerical substitution",
    "Calibrated (label-free) + gate": "Calibrated",
    "Calibrated + clerical 200 + gate": "Calibrated + clerical",
}
MODES = list(RENAME.values())
COLOURS = {m: F.SERIES[i] for i, m in enumerate(MODES)}
COLOURS["Clerical sample alone"] = F.SERIES[5]
COLOURS[OBSERVED] = F.NEUTRAL
ERROR_TYPES = [("forename_error_rate", "Forename", "#4a3aa7", "o"),
               ("surname_error_rate", "Surname", "#e34948", "s"),
               ("multi_field_error", "Both names", F.INK_2, "^")]
DIFF = "Replica minus observed (pp)"
REFERENCE = "Non-Hispanic White"


def relabel(df: pd.DataFrame) -> pd.DataFrame:
    keep = df["mode"].isin(list(RENAME) + [OBSERVED])
    return df[keep].assign(mode=df.loc[keep, "mode"].map(lambda m: RENAME.get(m, m)))


def legend_below(fig, ax, ncol: int, labels_map: dict | None = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    labels = [(labels_map or {}).get(lab, lab) for lab in labels]
    fig.tight_layout()  # lay out the axes first, then hang the legend beneath them
    fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 0.0))


def save(fig, name: str) -> None:
    if not fig.legends:
        fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def fig3(sens: pd.DataFrame) -> None:
    """Pseudo-truth learned rates by probability floor; true rates dashed, same colour per error type."""
    scenarios = list(dict.fromkeys(sens["scenario"]))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(9.6, 3.8), sharey=True)
    for ax, sc in zip(axes, scenarios):
        s = sens[sens["scenario"] == sc]
        true = s[s["pseudo_truth_floor"] == "Observed reference"].iloc[0]
        learned = s[s["pseudo_truth_floor"] != "Observed reference"].sort_values("floor_value")
        for col, label, colour, marker in ERROR_TYPES:
            ax.plot(learned["floor_value"], 100 * learned[col], color=colour, linewidth=2, marker=marker,
                    markersize=6, label=f"{label} (learned)", zorder=3)
            ax.axhline(100 * true[col], color=colour, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.axvline(0.70, color=F.AXIS, linewidth=1, zorder=1)
        ax.text(0.705, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] else 1, "floor used", fontsize=7.5,
                color=F.INK_2, va="top")
        ax.set_title(sc)
        ax.set_xlabel("Pseudo-truth probability floor")
        ax.set_xticks(learned["floor_value"])
    axes[0].set_ylabel("Error rate among true pairs (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=F.INK_2, linewidth=1, linestyle=(0, (4, 3))))
    labels.append("True rate (dashed, same colour)")
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig3_floor_sensitivity.png")


def fig4(band: pd.DataFrame) -> None:
    scenarios = list(dict.fromkeys(band["scenario"].astype(str)))
    bands = list(dict.fromkeys(band["band"].astype(str)))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(10.4, 3.8), sharey=True)
    for ax, sc in zip(axes, scenarios):
        F._grouped_bars(ax, bands, MODES, lambda b, m: F._row(band, "mass", scenario=sc, band=b, mode=m), COLOURS)
        ax.set_title(sc)
        ax.set_xlabel("Match-probability band")
    axes[0].set_ylabel(DIFF)
    legend_below(fig, axes[0], ncol=5)
    save(fig, "fig4_band_mass_difference.png")


def fig5(diff: pd.DataFrame) -> None:
    metrics = [("precision", "Precision"), ("missed_match_rate", "Missed-match rate")]
    scenarios = list(dict.fromkeys(diff["scenario"].astype(str)))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, (m, label) in zip(axes, metrics):
        F._grouped_bars(ax, scenarios, MODES, lambda sc, mode: F._row(diff, m, scenario=sc, mode=mode), COLOURS)
        ax.set_title(label)
    axes[0].set_ylabel(DIFF)
    legend_below(fig, axes[0], ncol=3)
    save(fig, "fig5_linkage_quality_difference.png")


def fig6(groups: pd.DataFrame) -> None:
    metrics = [("missed_match_rate", "Missed-match rate"), ("accepted_link_false_match_rate", "False-match rate")]
    names = [g for g in dict.fromkeys(groups["group"].astype(str)) if g != REFERENCE]
    series = [OBSERVED] + MODES
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    for ax, (m, label) in zip(axes, metrics):
        col = f"{m}_difference_vs_reference"
        F._grouped_bars(ax, names, series, lambda g, mode: F._row(groups, col, group=g, mode=mode), COLOURS)
        ax.set_title(label)
        ax.set_xticklabels([g.replace(" / ", "/\n", 1) if len(g) > 20 else g for g in names])
        ax.set_xlabel("Ethno-racial group")
    axes[0].set_ylabel(f"Difference from {REFERENCE} (pp)")
    legend_below(fig, axes[0], ncol=6, labels_map={OBSERVED: "Observed"})
    save(fig, "fig6_group_disparity.png")


def fig7(sweep: pd.DataFrame) -> None:
    """Redraw study: one dot per independent observed task, median as a bar."""
    order = ["Pseudo-truth", "Clerical sample alone", "Calibrated", "Calibrated + clerical", "Supervised"]
    sweep = sweep.assign(setting=np.where(sweep["method"].eq("Calibrated") & sweep["clerical_n"].gt(0),
                                          "Calibrated + clerical", sweep["method"]))
    metrics = [("err_missed_match_rate", "missed-match rate"), ("err_precision", "precision")]
    scenarios = list(dict.fromkeys(sweep["scenario"]))
    fig, axes = plt.subplots(1, len(metrics) * len(scenarios), figsize=(12, 3.4), sharey=True)
    y = np.arange(len(order))[::-1]
    for k, ((col, label), sc) in enumerate([(m, s) for m in metrics for s in scenarios]):
        ax = axes[k]
        ax.grid(axis="x", color=F.GRID, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=F.INK_2, linewidth=0.9, zorder=1)
        for yi, setting in zip(y, order):
            vals = 100 * sweep.loc[(sweep["scenario"] == sc) & (sweep["setting"] == setting), col].to_numpy()
            if not len(vals):
                continue
            jitter = np.linspace(-0.15, 0.15, len(vals))
            ax.scatter(vals, yi + jitter, s=34, color=COLOURS[setting], edgecolor=F.SURFACE, linewidth=1.2, zorder=3)
            ax.plot([np.median(vals)] * 2, [yi - 0.3, yi + 0.3], color=F.INK, linewidth=1.6, zorder=4)
        ax.set_title(f"{sc}: {label}")
        ax.set_xlabel(DIFF)
        if col == "err_precision":
            ax.set_xlim(-3, 3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)
    save(fig, "fig7_redraw_study.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    F.SHORT.clear()
    fig3(pd.read_csv(TABLES / "floor_sensitivity.csv"))
    band = relabel(pd.read_csv(TABLES / "band_mass_difference_from_observed.csv"))
    fig4(band)
    diff = relabel(pd.read_csv(TABLES / "linkage_quality_difference_from_observed.csv"))
    fig5(diff)
    groups = relabel(pd.read_csv(TABLES / "linkage_quality_by_group.csv"))
    groups = groups[groups["scenario"] == "NCVR"]
    fig6(groups)
    sweep = pd.read_csv(ROOT / "outputs/calibration_sweep_scale/sweep_block_level.csv")
    fig7(sweep)
    for name, df in [("fig4", band), ("fig5", diff), ("fig6", groups), ("fig7", sweep)]:
        df.to_csv(OUT / f"{name}_data.csv", index=False)
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
