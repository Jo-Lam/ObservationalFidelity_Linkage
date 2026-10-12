"""Turn long per-replica tables into summary tables.

Replica rows are summarised with the mean and a 95% Monte Carlo interval
(2.5th and 97.5th percentiles across replicas).

* Main tables (``build_tables``) use **all** replicas. Each mode carries the
  generator-gate verdict for its gate policy; replicas are never discarded.
* Legacy tables (``build_tables(..., legacy=True)``) reproduce the paper: gated
  modes keep only replicas that passed the per-replica ``legacy_gate``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .diagnostics import names
from .experiment import OBSERVED

LINK_METRICS = ["precision", "accepted_link_false_match_rate", "recall", "missed_match_rate", "f1"]
PROVENANCE_METRICS = ["full_name_overlap", "exact_tuple_overlap", "rare_singleton_overlap"]


def _ordered(df: pd.DataFrame, modes: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["mode"] = pd.Categorical(df["mode"], [OBSERVED] + modes, ordered=True)
    sort = [c for c in ["gate_policy", "scenario", "group", "mode", "layer", "band", "floor_value"] if c in df.columns]
    return df.sort_values(sort, kind="stable").reset_index(drop=True)


def interval_summary(df: pd.DataFrame, by: list[str], metrics: list[str], digits: int = 3) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(by, dropna=False, sort=False, observed=True):
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        row["n_replicas"] = int(g["seed"].nunique())
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy()
            mean = float(vals.mean()) if len(vals) else np.nan
            lo, hi = (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if len(vals) > 1 else (np.nan, np.nan)
            row[f"{m}_mean"], row[f"{m}_l95"], row[f"{m}_u95"] = mean, lo, hi
            row[m] = (f"{mean:.{digits}f}" if np.isnan(lo) else f"{mean:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})") if not np.isnan(mean) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def build_tables(long: dict[str, pd.DataFrame], study: dict[str, Any], legacy: bool = False) -> dict[str, pd.DataFrame]:
    modes = [m["label"] for m in study["modes"]]
    keep = (lambda df: df[df["legacy_accepted"].astype(bool)]) if legacy else (lambda df: df)
    out = _gate_tables_legacy(long, study, modes) if legacy else _gate_tables(long, study, modes)

    verdict = None if legacy else out["table3_gate_results"][["scenario", "mode", "verdict"]].astype({"mode": str})

    def with_verdict(df: pd.DataFrame) -> pd.DataFrame:
        if verdict is None:
            return df
        return df.astype({"mode": str}).merge(verdict, on=["scenario", "mode"], how="left").fillna({"verdict": "reference"})

    # Table 3 (bottom): pre-linkage fidelity of the error structure.
    out["table3_prelinkage_fidelity"] = _ordered(with_verdict(
        interval_summary(keep(long["tasks"]), ["scenario", "mode"], names("mechanism", observable=False))), modes)
    # Supplementary Table 1: provenance audit.
    out["supp_table1_provenance"] = _ordered(
        interval_summary(keep(long["provenance"]), ["scenario", "mode"], PROVENANCE_METRICS), modes)
    # Figure 5: overall linkage quality.
    link = long["linkage"]
    overall = keep(link[link["group"] == "Overall"])
    out["linkage_quality_overall"] = _ordered(with_verdict(
        interval_summary(overall, ["scenario", "mode"], LINK_METRICS)), modes)
    out["linkage_quality_difference_from_observed"] = _ordered(
        _difference_from_observed(overall, ["scenario"], LINK_METRICS), modes)
    # Figure 6: subgroup linkage quality and disparities vs the reference group.
    groups = keep(link[link["group"] != "Overall"])
    if len(groups):
        diff_cols = [c for c in groups.columns if c.endswith("_difference_vs_reference")]
        out["linkage_quality_by_group"] = _ordered(
            interval_summary(groups, ["scenario", "group", "mode"], LINK_METRICS + diff_cols), modes)
    # Figure 4: match-probability band mass and its difference from observed.
    bands = keep(long["band_mass"])
    out["band_mass"] = _ordered(interval_summary(bands, ["scenario", "mode", "band"], ["mass"]), modes)
    out["band_mass_difference_from_observed"] = _ordered(
        _difference_from_observed(bands, ["scenario", "band"], ["mass"]), modes)
    # Figure 3: learned error rates by pseudo-truth floor.
    out["floor_sensitivity"] = long["floor_sensitivity"]
    # Continuous OF distance vs linkage replication error.
    out["of_distance_vs_replication_error"] = _ordered(_of_distance_table(keep(long["tasks"]), link), modes)
    out["observed_thresholds"] = long["thresholds"]
    # Synthesiser invariants (hold by construction; any violation is a bug, not a gate failure).
    inv = long["tasks"][long["tasks"]["mode"] != OBSERVED]
    out["invariant_violations"] = (inv.assign(violations=inv["invariant_violations"].fillna(""))
                                   .groupby(["scenario", "mode"], sort=False)["violations"]
                                   .agg(n_replicas="size", n_violating=lambda s: int((s != "").sum())).reset_index())
    return out


def _gate_tables(long: dict[str, pd.DataFrame], study: dict[str, Any], modes: list[str]) -> dict[str, pd.DataFrame]:
    """Generator-level gate: one verdict per (scenario, mode, policy) from all replicas."""
    gate_of = {m["label"]: m.get("gate") for m in study["modes"]}
    main = study["main_gate_policy"]
    ver = long["generator_verdicts"].copy()
    ver["verdict"] = np.where(ver["passed"].astype(bool), "accepted", "rejected")
    # Table 3 (top): each mode under its own gate; ungated modes show the main policy as a diagnostic.
    reported = ver[ver["gate_policy"] == ver["mode"].map(lambda m: gate_of.get(m) or main)].copy()
    reported["gate_applied"] = reported["mode"].map(lambda m: gate_of.get(m) is not None)
    reported.loc[~reported["gate_applied"], "verdict"] = "ungated (" + reported["verdict"] + " as diagnostic)"
    cols = ["scenario", "mode", "gate_policy", "gate_applied", "n_replicas", "verdict", "failed_diagnostics",
            "population_verdict", "mechanism_verdict", "landscape_verdict"]
    gen = long["generator_gate"]
    return {
        "table3_gate_results": _ordered(reported[cols], modes),
        "gate_policy_verdicts": _ordered(ver[[c for c in cols if c != "gate_applied"]], modes),
        "generator_gate_diagnostics": _ordered(gen[gen["is_mode_gate"].astype(bool)].drop(columns=["is_mode_gate"]), modes),
        "generator_gate_all_policies": _ordered(gen.drop(columns=["is_mode_gate"]), modes),
    }


def _gate_tables_legacy(long: dict[str, pd.DataFrame], study: dict[str, Any], modes: list[str]) -> dict[str, pd.DataFrame]:
    """The paper's per-replica gate: accepted-replica counts and stringency across policies."""
    gate_of = {m["label"]: m.get("legacy_gate") for m in study["modes"]}
    main = study["legacy_main_gate_policy"]
    dec = long["legacy_gate_decisions"].copy()
    dec["reported_policy"] = dec["mode"].map(lambda m: gate_of.get(m) or main)
    dec = dec[dec["gate_policy"] == dec["reported_policy"]]
    rows = []
    for (scenario, mode), g in dec.groupby(["scenario", "mode"], sort=False):
        failed = pd.Series([m for s in g["failed_metrics"].fillna("").astype(str) for m in s.split(";") if m])
        rows.append({"scenario": scenario, "mode": mode, "gate_policy": g["reported_policy"].iloc[0],
                     "gate_applied": gate_of.get(mode) is not None, "n_replicas": int(len(g)),
                     "n_passed": int(g["passed"].sum()), "pass_rate": float(g["passed"].mean()),
                     "common_failed_metrics": ";".join(failed.value_counts().head(3).index) or "none"})
    allp = long["legacy_gate_decisions"].groupby(["gate_policy", "scenario", "mode"], sort=False).agg(
        n_replicas=("passed", "size"), n_passed=("passed", "sum"), pass_rate=("passed", "mean")).reset_index()
    lay = long["legacy_gate_layers"]
    lrows = []
    for (policy, scenario, mode, layer), g in lay.groupby(["gate_policy", "scenario", "mode", "layer"], sort=False):
        assessed = g["verdict"] != "not_assessed"
        lrows.append({"gate_policy": policy, "scenario": scenario, "mode": mode, "layer": layer,
                      "n_checks": int(g["n_checks"].iloc[0]),
                      "layer_pass_rate": float((g["verdict"] == "pass").mean()) if assessed.any() else "not_assessed"})
    return {
        "table3_gate_results": _ordered(pd.DataFrame(rows), modes),
        "gate_stringency_pass_rates": _ordered(allp, modes),
        "gate_layer_verdicts": _ordered(pd.DataFrame(lrows), modes),
    }


def _difference_from_observed(df: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    obs = df[df["mode"] == OBSERVED][keys + metrics].rename(columns={m: f"{m}__obs" for m in metrics})
    rep = df[df["mode"] != OBSERVED].merge(obs, on=keys, how="left")
    for m in metrics:
        rep[m] = rep[m] - rep[f"{m}__obs"]
    return interval_summary(rep, keys[:1] + ["mode"] + keys[1:], metrics, digits=4)


def _of_distance_table(tasks: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    overall = link[link["group"] == "Overall"]
    obs = overall[overall["mode"] == OBSERVED].set_index("scenario")[LINK_METRICS]
    rep = tasks[tasks["mode"] != OBSERVED][["scenario", "mode", "seed", "of_distance_mean", "of_distance_max"]].merge(
        overall[["scenario", "mode", "seed"] + LINK_METRICS], on=["scenario", "mode", "seed"])
    for m in LINK_METRICS:
        rep[f"{m}_abs_error"] = (rep[m] - obs.loc[rep["scenario"], m].to_numpy()).abs()
    return interval_summary(rep, ["scenario", "mode"], ["of_distance_mean", "of_distance_max"] + [f"{m}_abs_error" for m in LINK_METRICS], digits=4)
