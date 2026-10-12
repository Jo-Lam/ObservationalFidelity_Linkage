"""Provenance (non-copying) audit: is high fidelity being achieved by copying observed records?

This is a diagnostic, not a gate, and is not a formal privacy guarantee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def provenance_audit(observed_reference: pd.DataFrame, observed_comparison: pd.DataFrame,
                     replica_comparison: pd.DataFrame) -> dict[str, float]:
    """Share of replica full names (forename, surname) that also occur in the observed data.

    * ``full_name_overlap``: in the observed (corrupted) comparison file;
    * ``exact_tuple_overlap``: in the observed clean reference file;
    * ``rare_singleton_overlap``: among full names that occur exactly once in the observed comparison file.

    Note: the original analysis counted singletons over the *set* of observed
    names, so every name was a "singleton" and this column equalled
    ``full_name_overlap``. Counts are now taken over records.
    """
    obs_cmp_names = list(zip(observed_comparison["forename_norm"], observed_comparison["surname_norm"]))
    obs_cmp = set(obs_cmp_names)
    obs_ref = set(zip(observed_reference["forename_norm"], observed_reference["surname_norm"]))
    counts = pd.Series(obs_cmp_names).value_counts()
    rare = set(counts[counts == 1].index)
    rep = list(zip(replica_comparison["forename_norm"], replica_comparison["surname_norm"]))
    if not rep:
        return {"full_name_overlap": 0.0, "exact_tuple_overlap": 0.0, "rare_singleton_overlap": 0.0}
    return {
        "full_name_overlap": float(np.mean([x in obs_cmp for x in rep])),
        "exact_tuple_overlap": float(np.mean([x in obs_ref for x in rep])),
        "rare_singleton_overlap": float(np.mean([x in rare for x in rep])),
    }
