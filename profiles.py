"""Learn name-error (corruption) profiles from linked pairs.

A profile records how often names disagree between linked records, and how:
forename and surname error rates, their co-occurrence (multi-field error), full
surname replacement and edit-distance distributions. Profiles can be learned

* supervised, from known true pairs;
* label-free, from "pseudo-truth" candidate pairs whose match probability is at
  or above a floor, weighted by match probability; or
* partly supervised, from a pseudo-truth profile whose link-breaking mechanisms
  are calibrated against a small clerically reviewed sample of true pairs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .names import jw_similarity, lev_distance, normalise_name

# Profile quantities that drive replica corruption (see synthesis.corrupt_replica).
CORRUPTION_RATES = ["forename_error_rate", "surname_error_rate", "multi_field_error", "full_surname_replacement"]


def true_pairs(reference: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    left = reference[["truth_id", "forename_norm", "surname_norm"]].rename(
        columns={"forename_norm": "forename_ref", "surname_norm": "surname_ref"})
    flags = [c for c in ["forename_was_corrupted", "surname_was_corrupted", "full_surname_replacement_flag"]
             if c in comparison.columns]
    right = comparison[["truth_id", "forename_norm", "surname_norm"] + flags].rename(
        columns={"forename_norm": "forename_cmp", "surname_norm": "surname_cmp"})
    return left.merge(right, on="truth_id", how="inner")


def pseudo_truth_pairs(scored: pd.DataFrame, reference: pd.DataFrame, comparison: pd.DataFrame,
                       floor: float) -> pd.DataFrame:
    """Candidate pairs with ``match_probability >= floor``; the probability is the pair weight."""
    pairs = scored[scored["match_probability"] >= floor].copy()
    left = reference[["record_id", "forename_norm", "surname_norm"]].rename(
        columns={"record_id": "id_l", "forename_norm": "forename_ref", "surname_norm": "surname_ref"})
    right = comparison[["record_id", "forename_norm", "surname_norm"]].rename(
        columns={"record_id": "id_r", "forename_norm": "forename_cmp", "surname_norm": "surname_cmp"})
    out = pairs.merge(left, on="id_l", how="left").merge(right, on="id_r", how="left")
    out["profile_weight"] = out["match_probability"]
    return out


def pair_error_features(pairs: pd.DataFrame, weight_col: str | None = None) -> pd.DataFrame:
    """Per-pair disagreement indicators and edit distances.

    Full surname replacement uses the injected flag when it is known (true pairs)
    and otherwise a Jaro-Winkler < 0.20 surname disagreement (pseudo-truth).
    """
    df = pairs.copy()
    for c in ["forename_ref", "surname_ref", "forename_cmp", "surname_cmp"]:
        df[c] = df[c].map(normalise_name)
    df["forename_error"] = (df["forename_ref"] != df["forename_cmp"]).astype(int)
    df["surname_error"] = (df["surname_ref"] != df["surname_cmp"]).astype(int)
    df["any_error"] = ((df["forename_error"] == 1) | (df["surname_error"] == 1)).astype(int)
    df["multi_field_error"] = ((df["forename_error"] == 1) & (df["surname_error"] == 1)).astype(int)
    df["forename_edit_distance"] = [lev_distance(a, b) for a, b in zip(df["forename_ref"], df["forename_cmp"])]
    df["surname_edit_distance"] = [lev_distance(a, b) for a, b in zip(df["surname_ref"], df["surname_cmp"])]
    df["surname_jw"] = [jw_similarity(a, b) for a, b in zip(df["surname_ref"], df["surname_cmp"])]
    if "full_surname_replacement_flag" in df.columns:
        df["full_surname_replacement"] = df["full_surname_replacement_flag"].fillna(False).astype(int)
    else:
        df["full_surname_replacement"] = ((df["surname_error"] == 1) & (df["surname_jw"] < 0.20)).astype(int)
    df["profile_weight"] = df[weight_col].astype(float) if weight_col and weight_col in df.columns else 1.0
    return df


def _weighted_mean(df: pd.DataFrame, col: str) -> float:
    w = df["profile_weight"].astype(float)
    return float(np.average(df[col].astype(float), weights=w)) if w.sum() else float("nan")


def _weighted_distribution(df: pd.DataFrame, col: str) -> dict[str, float]:
    s = df.groupby(col, dropna=False)["profile_weight"].sum()
    s = s / s.sum() if s.sum() else s
    return {str(k): float(v) for k, v in s.items()}


def summarise_profile(errors: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_pairs_used": int(len(errors)),
        "error_exposure": _weighted_mean(errors, "any_error"),
        "forename_error_rate": _weighted_mean(errors, "forename_error"),
        "surname_error_rate": _weighted_mean(errors, "surname_error"),
        "multi_field_error": _weighted_mean(errors, "multi_field_error"),
        "full_surname_replacement": _weighted_mean(errors, "full_surname_replacement"),
        "forename_edit_distance_distribution": _weighted_distribution(errors, "forename_edit_distance"),
        "surname_edit_distance_distribution": _weighted_distribution(errors, "surname_edit_distance"),
    }


def task_error_profile(reference: pd.DataFrame, comparison: pd.DataFrame) -> dict[str, Any]:
    """Error profile of a task measured against its known truth (observed task or replica)."""
    return summarise_profile(pair_error_features(true_pairs(reference, comparison)))


def learn_supervised(reference: pd.DataFrame, comparison: pd.DataFrame) -> dict[str, Any]:
    profile = task_error_profile(reference, comparison)
    profile["learning"] = {"mode": "supervised", "source": "all known true pairs"}
    return profile


def learn_pseudo_truth(scored: pd.DataFrame, reference: pd.DataFrame, comparison: pd.DataFrame,
                       floor: float) -> dict[str, Any]:
    errors = pair_error_features(pseudo_truth_pairs(scored, reference, comparison, floor), weight_col="profile_weight")
    profile = summarise_profile(errors)
    if len(errors):
        combo = ((errors["full_surname_replacement"] == 1)
                 & ((errors["forename_error"] == 1) | (errors["multi_field_error"] == 1))).astype(float)
        profile["surname_replacement_plus_other_error"] = float(np.average(combo, weights=errors["profile_weight"].astype(float)))
    else:
        profile["surname_replacement_plus_other_error"] = float("nan")
    profile["learning"] = {"mode": "pseudo_truth", "floor": float(floor), "weight": "match_probability"}
    return profile


def perturb_profile(profile: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """A deliberately wrong profile, for testing the gate's power.

    ``spec`` keys (applied in this order):
    * ``set``: {rate: value} replaces rates;
    * ``scale``: {rate: factor} multiplies rates;
    * ``add``: {rate: delta} adds to rates (clipped to [0, 1]);
    * ``independent_fields: true`` sets multi-field error to forename x surname rate,
      removing forename-surname error co-occurrence.
    """
    out = dict(profile)
    for key, value in (spec.get("set") or {}).items():
        out[key] = float(value)
    for key, factor in (spec.get("scale") or {}).items():
        out[key] = float(out[key]) * float(factor)
    for key, delta in (spec.get("add") or {}).items():
        out[key] = min(1.0, max(0.0, float(out[key]) + float(delta)))
    if spec.get("independent_fields"):
        out["multi_field_error"] = float(out["forename_error_rate"]) * float(out["surname_error_rate"])
    out["full_surname_replacement"] = min(float(out["full_surname_replacement"]), float(out["surname_error_rate"]))
    out["learning"] = {"mode": "perturbed", "base": profile.get("learning"), "perturbation": spec}
    return out


def learn_partly_supervised(pseudo_profile: dict[str, Any], reference: pd.DataFrame, comparison: pd.DataFrame,
                            sample_size: int, seed: int, calibrate: list[str]) -> dict[str, Any]:
    """Calibrate a pseudo-truth profile with a clerically reviewed random sample of true pairs.

    Pseudo-truth pairs above a probability floor under-represent link-breaking
    errors (full surname replacement, multi-field disagreement). A small random
    clerical sample is unbiased for those mechanisms, so the rates listed in
    ``calibrate`` are taken from the sample; the remaining rates keep their
    pseudo-truth estimates.
    """
    clerical = clerical_sample(reference, comparison, sample_size, seed)
    profile = dict(pseudo_profile)
    for key in calibrate:
        profile[key] = clerical[key]
    profile["learning"] = {
        "mode": "partly_supervised",
        "base": pseudo_profile.get("learning"),
        "clerical_sample_size": clerical["n"],
        "clerical_sample_seed": int(seed),
        "calibrated_from_clerical_sample": list(calibrate),
        "clerical_sample_estimates": {k: clerical[k] for k in CORRUPTION_RATES + ["error_exposure"]},
    }
    return profile


def clerical_sample(reference: pd.DataFrame, comparison: pd.DataFrame, sample_size: int, seed: int) -> dict[str, Any]:
    """Simulated clerical review: error rates measured on a random sample of true pairs.

    In practice these labels come from reviewing a random sample of records and
    finding each one's true match. Returns the profile summary plus ``n``.
    """
    pairs = true_pairs(reference, comparison)
    sample = pairs.sample(n=min(sample_size, len(pairs)), random_state=seed)
    return {"n": int(len(sample)), **summarise_profile(pair_error_features(sample))}
