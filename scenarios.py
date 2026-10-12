"""Load a source dataset and build the observed (reference, corrupted comparison) linkage task."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .names import add_name_features, classify_ncvr_ethnicity, corrupt_names


def read_source(path: Path, sheet: str | None = None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet or 0)
    return pd.read_csv(path)


def build_reference(source: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Clean reference file with one record per person, plus an optional subgroup column."""
    cols = cfg["columns"]
    max_rows = cfg.get("max_rows")
    if max_rows is not None and len(source) > int(max_rows):
        source = source.head(int(max_rows)).copy()
    ref = pd.DataFrame()
    ref["truth_id"] = source[cols["id"]].astype(str)
    ref["record_id"] = "ref_" + ref["truth_id"]
    ref["forename_norm"] = source[cols["forename"]]
    ref["surname_norm"] = source[cols["surname"]]
    subgroup = cfg.get("subgroup")
    if subgroup:
        raw = source[subgroup["column"]]
        ref[subgroup["name"]] = raw.map(classify_ncvr_ethnicity) if subgroup.get("coding") == "ncvr_collapsed" else raw
    return exclude_records(add_name_features(ref), cfg.get("exclude"))


def exclude_records(ref: pd.DataFrame, exclude: dict[str, Any] | None) -> pd.DataFrame:
    """Optionally drop non-person records (e.g. test-harness entries in synthetic extracts).

    ``exclude`` may set ``name_pattern`` (regex matched against normalised forename
    or surname) and/or ``max_full_name_cluster`` (drop full names occurring at
    least this many times). The number dropped is kept in ``ref.attrs``.
    """
    if not exclude:
        return ref
    drop = pd.Series(False, index=ref.index)
    if exclude.get("name_pattern"):
        pattern = exclude["name_pattern"]
        drop |= ref["forename_norm"].str.contains(pattern, regex=True) | ref["surname_norm"].str.contains(pattern, regex=True)
    if exclude.get("max_full_name_cluster"):
        full = ref["forename_norm"] + "|" + ref["surname_norm"]
        drop |= full.map(full.value_counts()) >= int(exclude["max_full_name_cluster"])
    out = ref[~drop].copy()
    out.attrs["excluded_records"] = int(drop.sum())
    return out


def build_observed_task(cfg: dict[str, Any], data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observed task = clean reference (spine) + a comparison (incoming) file with injected, recorded name errors.

    The injected errors are the "true" error mechanism that the learning modes
    try to recover. By default the comparison file is a corrupted copy of the
    whole reference. With ``incoming: {fraction, seed}`` it is a corrupted copy
    of a random subset (the paper's spine/incoming design), so every incoming
    record has exactly one true match and most spine records have none.
    """
    source = read_source(data_dir / cfg["source"]["file"], cfg["source"].get("sheet"))
    reference = build_reference(source, cfg)
    base = incoming_subset(reference, cfg.get("incoming"))
    comparison = corrupt_names(base, reference, **cfg["observed_corruption"])
    comparison["record_id"] = "cmp_" + comparison["truth_id"].astype(str)
    return reference, add_name_features(comparison)


def incoming_subset(reference: pd.DataFrame, incoming: dict[str, Any] | None, seed: int | None = None) -> pd.DataFrame:
    """Records that receive an incoming copy: all of them, or a random ``fraction`` (spine order kept).

    ``seed`` overrides ``incoming['seed']`` (replicas draw their own subset).
    """
    if not incoming or float(incoming.get("fraction", 1.0)) >= 1.0:
        return reference
    n = int(round(len(reference) * float(incoming["fraction"])))
    picked = reference.sample(n=n, random_state=int(seed if seed is not None else incoming["seed"]))
    return reference.loc[picked.index.sort_values()]
