"""Paper figures (static PNG). Every figure's data is also written as a CSV table."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .experiment import OBSERVED  # noqa: E402

# Categorical slots in fixed order (validated default palette); never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#9a9893"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "font.family": "sans-serif", "savefig.dpi": 220,
})

SHORT = {"Supervised": "Supervised", "Pseudo-truth + label-free gate": "Pseudo-truth + OF gate",
         "Pseudo-truth only": "Pseudo-truth only", "Partly supervised + label-free gate": "Partly supervised + OF gate"}


def _colors(modes: list[str]) -> dict[str, str]:
    return {m: SERIES[i] for i, m in enumerate(modes)}


def _grouped_bars(ax, categories: list[str], modes: list[str], get, colors: dict[str, str], pct: bool = True) -> None:
    """Grouped bars with 95% interval whiskers; ``get(category, mode) -> (mean, lo, hi)``."""
    width = 0.8 / max(len(modes), 1)
    x = np.arange(len(categories))
    scale = 100 if pct else 1
    for i, mode in enumerate(modes):
        vals = [get(c, mode) for c in categories]
        mean = np.array([v[0] for v in vals], dtype=float) * scale
        lo = np.array([v[1] for v in vals], dtype=float) * scale
        hi = np.array([v[2] for v in vals], dtype=float) * scale
        pos = x + (i - (len(modes) - 1) / 2) * width
        ax.bar(pos, mean, width * 0.92, color=colors[mode], edgecolor=SURFACE, linewidth=1.5, label=SHORT.get(mode, mode), zorder=2)
        has = ~np.isnan(lo) & ~np.isnan(hi)
        if has.any():
            ax.errorbar(pos[has], mean[has], yerr=[mean[has] - lo[has], hi[has] - mean[has]], fmt="none",
                        ecolor=INK_2, elinewidth=0.9, capsize=2, zorder=3)
    ax.axhline(0, color=INK_2, linewidth=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)


def _bottom_legend(fig, ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.07))


def _row(df: pd.DataFrame, metric: str, **match) -> tuple[float, float, float]:
    sub = df
    for k, v in match.items():
        sub = sub[sub[k].astype(str) == str(v)]
    if sub.empty:
        return (np.nan, np.nan, np.nan)
    r = sub.iloc[0]
    return (r[f"{metric}_mean"], r[f"{metric}_l95"], r[f"{metric}_u95"])


def fig3_floor_sensitivity(sens: pd.DataFrame, path: Path) -> None:
    metrics = [("forename_error_rate", "Forename error"), ("surname_error_rate", "Surname error"), ("multi_field_error", "Multi-field error")]
    scenarios = list(dict.fromkeys(sens["scenario"]))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.6 * len(scenarios), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, scenarios):
        s = sens[sens["scenario"] == sc]
        obs = s[s["pseudo_truth_floor"] == "Observed reference"].iloc[0]
        learned = s[s["pseudo_truth_floor"] != "Observed reference"].sort_values("floor_value")
        for i, (m, label) in enumerate(metrics):
            ax.plot(learned["floor_value"], learned[m] * 100, color=SERIES[i], linewidth=2, marker="o", markersize=5, label=f"{label} (learned)", zorder=3)
            ax.axhline(obs[m] * 100, color=SERIES[i], linewidth=1, linestyle=(0, (4, 3)), zorder=2)

        ax.set_title(sc)
        ax.set_xlabel("Pseudo-truth probability floor")
        ax.set_xticks(learned["floor_value"])
        ax.set_xlim(learned["floor_value"].min() - 0.04, learned["floor_value"].max() + 0.03)
    axes[0].set_ylabel("Learned error rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK_2, linewidth=1, linestyle=(0, (4, 3))))
    labels.append("Observed (true) rate")
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Learned error rates by pseudo-truth floor", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig4_band_mass(diff: pd.DataFrame, modes: list[str], path: Path) -> None:
    scenarios = list(dict.fromkeys(diff["scenario"].astype(str)))
    bands = list(dict.fromkeys(diff["band"].astype(str)))
    colors = _colors(modes)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, scenarios):
        _grouped_bars(ax, bands, modes, lambda b, m: _row(diff, "mass", scenario=sc, band=b, mode=m), colors)
        ax.set_title(sc)
        ax.set_xlabel("Match-probability band")
    axes[0].set_ylabel("Difference from observed (pp)")
    _bottom_legend(fig, axes[0])
    fig.suptitle("Match-probability band mass: replica minus observed", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig5_linkage_quality(diff: pd.DataFrame, modes: list[str], path: Path) -> None:
    metrics = [("precision", "Precision"), ("accepted_link_false_match_rate", "False-match rate"),
               ("recall", "Recall"), ("missed_match_rate", "Missed-match rate")]
    scenarios = list(dict.fromkeys(diff["scenario"].astype(str)))
    colors = _colors(modes)
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.6), sharey=True)
    for ax, (m, label) in zip(axes, metrics):
        _grouped_bars(ax, scenarios, modes, lambda sc, mode: _row(diff, m, scenario=sc, mode=mode), colors)
        ax.set_title(label)
    axes[0].set_ylabel("Difference from observed (pp)")
    _bottom_legend(fig, axes[0])
    fig.suptitle("Linkage quality at the fixed observed-task threshold: replica minus observed", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig6_group_disparity(groups: pd.DataFrame, modes: list[str], reference_group: str, path: Path) -> None:
    metrics = [("missed_match_rate", "Missed-match rate"), ("accepted_link_false_match_rate", "False-match rate")]
    group_names = [g for g in dict.fromkeys(groups["group"].astype(str)) if g != reference_group]
    series = [OBSERVED] + modes
    colors = {OBSERVED: NEUTRAL, **_colors(modes)}
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 3.8), sharey=True)
    for ax, (m, label) in zip(axes, metrics):
        col = f"{m}_difference_vs_reference"
        _grouped_bars(ax, group_names, series, lambda g, mode: _row(groups, col, group=g, mode=mode), colors)
        ax.set_title(f"{label}: difference vs {reference_group}")
        ax.set_xticklabels([g.replace(" / ", "/\n", 1) if len(g) > 20 else g for g in group_names])
    axes[0].set_ylabel("Difference (pp)")
    handles, labels = axes[0].get_legend_handles_labels()
    labels = ["Observed" if lab == OBSERVED else lab for lab in labels]
    fig.legend(handles, labels, loc="lower center", ncol=len(series), bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("NCVR subgroup disparities in linkage quality", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_gate_stringency(pass_rates: pd.DataFrame, modes: list[str], path: Path) -> None:
    scenarios = list(dict.fromkeys(pass_rates["scenario"].astype(str)))
    policies = list(dict.fromkeys(pass_rates["gate_policy"].astype(str)))
    colors = _colors(modes)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.2 * len(scenarios), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, scenarios):
        def get(policy, mode):
            r = pass_rates[(pass_rates["scenario"].astype(str) == sc) & (pass_rates["gate_policy"] == policy) & (pass_rates["mode"].astype(str) == mode)]
            v = float(r["pass_rate"].iloc[0]) if len(r) else np.nan
            return (v, np.nan, np.nan)
        _grouped_bars(ax, policies, modes, get, colors)
        ax.set_xticklabels([p.replace("_", "\n") for p in policies])
        ax.set_title(sc)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Replicas accepted (%)")
    _bottom_legend(fig, axes[0])
    fig.suptitle("Gate pass rate by gate policy", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_generator_gate(gen: pd.DataFrame, policy: str, modes: list[str], path: Path) -> None:
    """Generator bias per diagnostic in units of its margin, with the replica noise band.

    A diagnostic fails when the dot lies beyond both the margin line (1) and the
    grey noise band. The x axis is logarithmic.
    """
    gen = gen[gen["gate_policy"] == policy]
    scenarios = list(dict.fromkeys(gen["scenario"].astype(str)))
    diags = list(dict.fromkeys(gen["diagnostic"]))
    colors = _colors(modes)
    floor = 0.01
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.2 * len(scenarios), 0.42 * len(diags) * len(modes) / 2 + 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    step = 0.8 / max(len(modes), 1)
    for ax, sc in zip(axes, scenarios):
        for i, mode in enumerate(modes):
            sub = gen[(gen["scenario"].astype(str) == sc) & (gen["mode"].astype(str) == mode)].set_index("diagnostic")
            for j, dname in enumerate(diags):
                if dname not in sub.index:
                    continue
                r = sub.loc[dname]
                if r["classification"] == "not_applicable":
                    continue
                y = j + (i - (len(modes) - 1) / 2) * step
                noise = max(float(r["noise"]) / float(r["margin"]), floor)
                ax.plot([floor, noise], [y, y], color=GRID, linewidth=5, solid_capstyle="butt", zorder=1)
                eff = max(float(r["effect"]) / float(r["margin"]), floor)
                failed = not bool(r["passed"])
                ax.scatter([eff], [y], s=42, color=colors[mode], edgecolor=INK if failed else SURFACE,
                           linewidth=1.4 if failed else 1.0, zorder=3, label=SHORT.get(mode, mode) if j == 0 else None)
        ax.axvline(1, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(floor * 0.8, None)
        ax.set_yticks(range(len(diags)))
        ax.set_yticklabels([d.replace("_", " ") for d in diags])
        ax.invert_yaxis()
        ax.grid(axis="x", which="major")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Generator bias / margin (log)")
        ax.set_title(sc)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=GRID, linewidth=5), plt.Line2D([], [], color=INK_2, linewidth=1, linestyle=(0, (4, 3)))]
    labels += ["Replica noise band", "Margin"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Generator-level OF gate ({policy} policy): outlined dots fail", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_figures(tables: dict[str, pd.DataFrame], modes: list[str], reference_group: str | None, fig_dir: Path,
                 gated_modes: list[str] | None = None, gate_policies: list[str] | None = None) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig3_floor_sensitivity(tables["floor_sensitivity"], fig_dir / "fig3_floor_sensitivity.png")
    fig4_band_mass(tables["band_mass_difference_from_observed"], modes, fig_dir / "fig4_band_mass_difference.png")
    fig5_linkage_quality(tables["linkage_quality_difference_from_observed"], modes, fig_dir / "fig5_linkage_quality_difference.png")
    if "linkage_quality_by_group" in tables and reference_group:
        fig6_group_disparity(tables["linkage_quality_by_group"], modes, reference_group, fig_dir / "fig6_group_disparity.png")
    if "gate_stringency_pass_rates" in tables:
        fig_gate_stringency(tables["gate_stringency_pass_rates"], modes, fig_dir / "gate_pass_rate_by_policy.png")
    if "generator_gate_all_policies" in tables:
        for policy in gate_policies or []:
            fig_generator_gate(tables["generator_gate_all_policies"], policy, gated_modes or modes,
                               fig_dir / f"generator_gate_{policy}.png")
