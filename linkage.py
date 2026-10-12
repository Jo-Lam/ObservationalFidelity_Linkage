"""Name-only probabilistic linkage model, threshold selection and quality metrics.

The model is deliberately restricted to forename and surname so that linkage
difficulty reflects name structure and name error only.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

PROBABILITY_BANDS = ["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"]
CANDIDATE_SIZE_BINS = ["1", "2", "3-5", "6-10", "11-20", "21-50", "51+"]


def _logistic(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)


def name_similarity(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Cheap prefix-and-length similarity in [0, 1]; exact agreement scores 1."""
    a = a.fillna("").astype(str)
    b = b.fillna("").astype(str)
    exact = (a == b).to_numpy(dtype=float)
    len_a = a.str.len().replace(0, np.nan)
    len_b = b.str.len().replace(0, np.nan)
    len_sim = (1 - (len_a - len_b).abs() / np.maximum(len_a, len_b)).fillna(0).clip(0, 1).to_numpy(dtype=float)
    sim = (0.15 * (a.str[:1] == b.str[:1]).to_numpy(dtype=float)
           + 0.25 * (a.str[:2] == b.str[:2]).to_numpy(dtype=float)
           + 0.25 * (a.str[:3] == b.str[:3]).to_numpy(dtype=float)
           + 0.35 * len_sim)
    return np.where(exact == 1, 1.0, sim)


def build_candidates(left: pd.DataFrame, right: pd.DataFrame, blocking_rules: list[list[str]]) -> pd.DataFrame:
    """Union of exact-key blocking joins on ``record_id``."""
    parts = []
    for i, rule in enumerate(blocking_rules):
        lft = left[["record_id"] + rule].rename(columns={"record_id": "id_l"})
        rgt = right[["record_id"] + rule].rename(columns={"record_id": "id_r"})
        m = lft.merge(rgt, on=rule, how="inner")
        if len(m):
            parts.append(m[["id_l", "id_r"]].assign(blocking_rule_id=i))
    if not parts:
        return pd.DataFrame(columns=["id_l", "id_r", "blocking_rule_id"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(["id_l", "id_r"])


def score_pairs(pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, scoring: dict,
                subgroup_col: str | None = None) -> pd.DataFrame:
    """Attach truth labels, comparison features and a logistic match probability."""
    if pairs.empty:
        return pd.DataFrame(columns=["id_l", "id_r", "match_probability", "is_true_match", "truth_id_l", "truth_id_r"])
    left_cols = ["record_id", "truth_id", "forename_norm", "surname_norm"]
    if subgroup_col and subgroup_col in left.columns:
        left_cols.append(subgroup_col)
    lft = left[left_cols].rename(columns={"record_id": "id_l"})
    rgt = right[["record_id", "truth_id", "forename_norm", "surname_norm"]].rename(columns={"record_id": "id_r"})
    df = pairs.merge(lft, on="id_l", how="left").merge(rgt, on="id_r", how="left", suffixes=("_l", "_r"))
    df["forename_exact"] = (df["forename_norm_l"] == df["forename_norm_r"]).astype(int)
    df["surname_exact"] = (df["surname_norm_l"] == df["surname_norm_r"]).astype(int)
    df["forename_similarity"] = name_similarity(df["forename_norm_l"], df["forename_norm_r"])
    df["surname_similarity"] = name_similarity(df["surname_norm_l"], df["surname_norm_r"])
    logit = (scoring.get("intercept", -7.0)
             + scoring.get("forename_weight", 3.0) * df["forename_similarity"]
             + scoring.get("surname_weight", 5.0) * df["surname_similarity"]
             + scoring.get("exact_forename_bonus", 1.0) * df["forename_exact"]
             + scoring.get("exact_surname_bonus", 1.5) * df["surname_exact"])
    df["match_probability"] = [_logistic(float(x)) for x in logit]
    df["is_true_match"] = (df["truth_id_l"].astype(str) == df["truth_id_r"].astype(str)).astype(int)
    if subgroup_col and subgroup_col in df.columns:
        df = df.rename(columns={subgroup_col: "group_l"})
    return df


def link(reference: pd.DataFrame, comparison: pd.DataFrame, linkage_cfg: dict,
         subgroup_col: str | None = None) -> pd.DataFrame:
    pairs = build_candidates(reference, comparison, linkage_cfg["blocking_rules"])
    return score_pairs(pairs, reference, comparison, linkage_cfg["scoring"], subgroup_col)


def select_threshold_for_precision(scored: pd.DataFrame, target_precision: float,
                                   true_links: dict[str, int] | None = None, tie_aware: bool = False) -> dict[str, float]:
    """Lowest score threshold that meets the target pair-level precision.

    ``tie_aware=False`` (the original analysis): walk pairs in score order and take
    the score of the last pair at which cumulative precision meets the target. A
    threshold accepts *all* pairs tied at that score, so when scores are discrete
    the achieved precision can fall below the target.

    ``tie_aware=True``: evaluate precision at each distinct score (all ties
    included) and take the lowest score whose achieved precision meets the target.
    If no score does, both rules fall back to the highest score.
    """
    if tie_aware:
        by_score = (scored.groupby("match_probability")["is_true_match"].agg(["sum", "size"])
                    .sort_index(ascending=False))
        precision_at = by_score["sum"].cumsum() / by_score["size"].cumsum()
        eligible = precision_at[precision_at >= target_precision]
        threshold = float(eligible.index.min() if len(eligible) else by_score.index.max())
        overall = linkage_metrics(scored, threshold, true_links=true_links).iloc[0]
        return {"threshold": threshold, "target_precision": float(target_precision),
                "achieved_precision": float(overall["precision"]), "recall_at_threshold": float(overall["recall"]),
                "accepted_pairs": int(overall["n_accepted_pairs"])}
    df = scored.sort_values("match_probability", ascending=False).copy()
    precision_cum = df["is_true_match"].cumsum() / np.arange(1, len(df) + 1)
    eligible = df[precision_cum.to_numpy() >= target_precision]
    chosen = eligible.iloc[[-1]] if not eligible.empty else df.iloc[[0]]
    threshold = float(chosen["match_probability"].iloc[0])
    overall = linkage_metrics(scored, threshold, true_links=true_links).iloc[0]
    return {
        "threshold": threshold,
        "target_precision": float(target_precision),
        "achieved_precision": float(overall["precision"]),
        "recall_at_threshold": float(overall["recall"]),
        "accepted_pairs": int(overall["n_accepted_pairs"]),
    }


def linkage_metrics(scored: pd.DataFrame, threshold: float, group_col: str | None = None,
                    reference_group: str | None = None, true_links: dict[str, int] | None = None) -> pd.DataFrame:
    """Pair-level precision/recall overall and by subgroup.

    By default recall uses the number of distinct reference records that reach
    the candidate set as its denominator, matching the analysis reported in the
    paper. Records whose true pair is never blocked together then do not count
    as missed matches. Pass ``true_links`` ({"Overall": n, group: n, ...}, the
    number of true matches) to use all true matches instead; this is required
    when the incoming file is a subset of the reference.
    """
    df = scored.copy()
    df["accepted"] = df["match_probability"] >= threshold
    true_links_given = true_links

    def summarise(sub: pd.DataFrame, label: str) -> dict:
        tp = int((sub["accepted"] & (sub["is_true_match"] == 1)).sum())
        fp = int((sub["accepted"] & (sub["is_true_match"] == 0)).sum())
        true_links = int(sub["truth_id_l"].nunique()) if true_links_given is None else int(true_links_given.get(label, 0))
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / true_links if true_links else np.nan
        f1 = (2 * precision * recall / (precision + recall)
              if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0 else np.nan)
        return {"group": label, "n_candidate_pairs": int(len(sub)), "n_accepted_pairs": int(sub["accepted"].sum()),
                "true_links": true_links, "true_positives": tp, "false_positives": fp,
                "precision": precision, "accepted_link_false_match_rate": 1 - precision if not np.isnan(precision) else np.nan,
                "recall": recall, "missed_match_rate": 1 - recall if not np.isnan(recall) else np.nan, "f1": f1}

    rows = [summarise(df, "Overall")]
    if group_col and group_col in df.columns:
        for g, sub in df.groupby(group_col, dropna=False):
            rows.append(summarise(sub, str(g) if pd.notna(g) else "Unknown"))
    out = pd.DataFrame(rows)
    if reference_group and reference_group in set(out["group"]):
        ref = out[out["group"] == reference_group].iloc[0]
        for m in ["precision", "accepted_link_false_match_rate", "recall", "missed_match_rate", "f1"]:
            out[f"{m}_difference_vs_reference"] = out[m] - ref[m]
    return out


def probability_band_mass(scored: pd.DataFrame, bins: Iterable[float] = (0, 0.25, 0.5, 0.75, 1.0000001)) -> pd.Series:
    """Share of candidate pairs in each match-probability band (the score landscape)."""
    b = pd.cut(scored["match_probability"], bins=list(bins), labels=PROBABILITY_BANDS, include_lowest=True, right=False)
    return b.value_counts(normalize=True).reindex(PROBABILITY_BANDS, fill_value=0).sort_index()


def candidate_size_distribution(scored: pd.DataFrame) -> pd.Series:
    """Distribution of candidate-set sizes per reference record (blocking/ambiguity burden)."""
    counts = scored.groupby("id_l")["id_r"].size()
    b = pd.cut(counts, bins=[0, 1, 2, 5, 10, 20, 50, 1_000_000], labels=CANDIDATE_SIZE_BINS,
               include_lowest=True, right=True)
    return b.value_counts(normalize=True).reindex(CANDIDATE_SIZE_BINS, fill_value=0).sort_index()


def total_variation(p: pd.Series, q: pd.Series) -> float:
    idx = p.index.union(q.index)
    return float(0.5 * np.abs(p.reindex(idx, fill_value=0).astype(float) - q.reindex(idx, fill_value=0).astype(float)).sum())
