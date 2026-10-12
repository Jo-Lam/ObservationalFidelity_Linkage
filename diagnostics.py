"""Registry of Observational Fidelity diagnostics.

A diagnostic is one statistic computed by the same code on the observed task and
on a replica, plus a distance between the two values. Each diagnostic declares

* ``layer``: what it validates.
  - ``population``: the clean identifiers (cardinality, concentration).
  - ``mechanism``: the error process (taxonomy layers 1-4 in the paper).
  - ``landscape``: what the linkage model sees (score, candidate-set and
    best-match distributions, ambiguity).
* ``observable``: whether it can be computed on the *observed* task without true
  match labels. Only observable diagnostics may appear in a label-free gate.
  Mechanism diagnostics come in two forms: truth-based rates (benchmark only)
  and ``learned_*`` rates, which apply the label-free pseudo-truth learner to
  the task itself. A learner's bias is the same on the observed task and on a
  correct replica, so comparing learned rates checks the mechanism without
  labels (indirect inference).
* ``by_construction``: whether the synthesiser guarantees it. Such diagnostics are
  invariants. They are checked and reported, but they cannot discriminate between
  replicas and so are never gate checks.
* ``subgroup``: whether it is computed per subgroup. Subgroup diagnostics take the
  worst subgroup, and are not applicable (``None``) when the task has no subgroup.

Distance kinds: ``absolute`` and ``relative`` (scalars, signed difference),
``distribution`` (total variation), ``grouped_absolute`` (largest absolute
difference across subgroups) and ``grouped_distribution`` (largest within-subgroup
total variation).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable

import numpy as np
import pandas as pd

from .linkage import candidate_size_distribution, probability_band_mass, total_variation
from .profiles import learn_pseudo_truth, task_error_profile

LAYERS = ("population", "mechanism", "landscape")
SCALAR_KINDS = ("absolute", "relative")
BEST_MATCH_BANDS = ["no candidate", "<0.25", "0.25-0.50", "0.50-0.75", "0.75-0.90", ">=0.90"]


class TaskBundle:
    """A linkage task and the context needed to describe it.

    ``reference`` and ``comparison`` carry known truth; ``scored`` is the
    candidate-pair table. Optional context: the subgroup column, the fixed
    operating threshold and the pseudo-truth floor. Diagnostics that need missing
    context are not applicable. Shared derived quantities are computed once.

    ``record_side`` sets the unit for per-record diagnostics (best-match bands,
    ambiguity, subgroup grouping): ``reference`` (default) or ``comparison``. Use
    ``comparison`` when the incoming file is a subset of the spine, since there
    only incoming records are being linked.
    """

    def __init__(self, reference: pd.DataFrame, comparison: pd.DataFrame, scored: pd.DataFrame,
                 subgroup: str | None = None, threshold: float | None = None,
                 pseudo_truth_floor: float | None = None, record_side: str = "reference") -> None:
        self.reference = reference
        self.comparison = comparison
        self.scored = scored
        self.subgroup = subgroup if subgroup and subgroup in reference.columns else None
        self.threshold = threshold
        self.pseudo_truth_floor = pseudo_truth_floor
        if record_side not in {"reference", "comparison"}:
            raise ValueError(f"record_side must be 'reference' or 'comparison', not {record_side!r}")
        self.records = reference if record_side == "reference" else comparison
        self.record_key = "id_l" if record_side == "reference" else "id_r"
        if self.subgroup and self.subgroup not in self.records.columns:
            self.subgroup = None

    @cached_property
    def error_profile(self) -> dict[str, Any]:
        return task_error_profile(self.reference, self.comparison)

    @cached_property
    def learned_profile(self) -> dict[str, Any] | None:
        if self.pseudo_truth_floor is None:
            return None
        return learn_pseudo_truth(self.scored, self.reference, self.comparison, self.pseudo_truth_floor)

    @cached_property
    def band_mass(self) -> pd.Series:
        return probability_band_mass(self.scored)

    @cached_property
    def candidate_sizes(self) -> pd.Series:
        return candidate_size_distribution(self.scored)

    @cached_property
    def best_match_band(self) -> pd.Series:
        """Per record (see ``record_side``): band of its best candidate score (``no candidate`` if none)."""
        best = self.scored.groupby(self.record_key)["match_probability"].max()
        best = best.reindex(self.records["record_id"].to_numpy()).fillna(-0.5)
        bands = pd.cut(best, bins=[-1, 0, 0.25, 0.5, 0.75, 0.9, 1.0000001], labels=BEST_MATCH_BANDS, right=False)
        return pd.Series(bands.to_numpy(), index=self.records.index)

    @cached_property
    def ambiguous(self) -> pd.Series | None:
        """Per record: two or more candidates at or above the operating threshold."""
        if self.threshold is None:
            return None
        above = self.scored[self.scored["match_probability"] >= self.threshold]
        n_above = above.groupby(self.record_key).size().reindex(self.records["record_id"].to_numpy()).fillna(0)
        return pd.Series((n_above >= 2).to_numpy(), index=self.records.index)

    def groups(self) -> pd.Series:
        return self.records[self.subgroup].astype(str)


@dataclass(frozen=True)
class Diagnostic:
    name: str
    layer: str
    observable: bool
    kind: str
    statistic: Callable[[TaskBundle], Any]
    description: str
    taxonomy_layer: int | None = None
    by_construction: bool = False
    subgroup: bool = False

    @property
    def scalar(self) -> bool:
        return self.kind in SCALAR_KINDS

    def distance(self, observed: Any, replica: Any) -> float:
        """Replica-versus-observed distance; signed for scalar kinds. NaN when not applicable."""
        if observed is None or replica is None:
            return float("nan")
        if self.kind == "distribution":
            return total_variation(observed, replica)
        if self.kind == "grouped_absolute":
            idx = observed.index.union(replica.index)
            return float((replica.reindex(idx, fill_value=0) - observed.reindex(idx, fill_value=0)).abs().max())
        if self.kind == "grouped_distribution":
            groups = observed.index.get_level_values(0).union(replica.index.get_level_values(0)).unique()
            return float(max(total_variation(_level(observed, g), _level(replica, g)) for g in groups))
        diff = float(replica) - float(observed)
        if self.kind == "relative":
            return diff / (abs(float(observed)) if observed != 0 else 1.0)
        return diff


def _level(s: pd.Series, group: str) -> pd.Series:
    return s.xs(group, level=0) if group in s.index.get_level_values(0) else pd.Series(dtype=float)


def _top_share(s: pd.Series, k: int = 10) -> float:
    vc = s.value_counts(normalize=True, dropna=False)
    return float(vc.head(k).sum()) if len(vc) else 0.0


def _error(key: str) -> Callable[[TaskBundle], float]:
    return lambda t: float(t.error_profile[key])


def _learned(key: str) -> Callable[[TaskBundle], float | None]:
    return lambda t: None if t.learned_profile is None else float(t.learned_profile[key])


def _best_match_distribution(t: TaskBundle) -> pd.Series:
    return t.best_match_band.value_counts(normalize=True).reindex(BEST_MATCH_BANDS, fill_value=0.0)


def _subgroup_best_match(t: TaskBundle) -> pd.Series | None:
    if t.subgroup is None:
        return None
    df = pd.DataFrame({"group": t.groups(), "band": t.best_match_band})
    out = df.groupby("group", observed=True)["band"].value_counts(normalize=True)
    full = pd.MultiIndex.from_product([sorted(df["group"].unique()), BEST_MATCH_BANDS], names=["group", "band"])
    return out.reindex(full, fill_value=0.0).astype(float)


def _full_name_cluster_size(reference: pd.DataFrame) -> pd.Series:
    full = reference["forename_norm"].astype(str) + "|" + reference["surname_norm"].astype(str)
    return full.map(full.value_counts())


def _full_name_shared(reference: pd.DataFrame) -> pd.Series:
    return _full_name_cluster_size(reference) > 1


def _full_name_duplication_rate(t: TaskBundle) -> float:
    """Share of spine (reference) records whose full name is shared with another spine record."""
    return float(_full_name_shared(t.reference).mean())


def _subgroup_full_name_duplication(t: TaskBundle) -> pd.Series | None:
    if not t.subgroup or t.subgroup not in t.reference.columns:
        return None
    return _full_name_shared(t.reference).astype(float).groupby(t.reference[t.subgroup].astype(str)).mean()


def _ambiguity_rate(t: TaskBundle) -> float | None:
    return None if t.ambiguous is None else float(t.ambiguous.mean())


def _subgroup_ambiguity(t: TaskBundle) -> pd.Series | None:
    if t.subgroup is None or t.ambiguous is None:
        return None
    return t.ambiguous.astype(float).groupby(t.groups()).mean()


_DIAGNOSTICS = [
    Diagnostic("unique_forename_count", "population", True, "relative",
               lambda t: float(t.reference["forename_norm"].nunique()),
               "Distinct forenames in the reference file.", by_construction=True),
    Diagnostic("unique_surname_count", "population", True, "relative",
               lambda t: float(t.reference["surname_norm"].nunique()),
               "Distinct surnames in the reference file.", by_construction=True),
    Diagnostic("top_forename_share", "population", True, "absolute",
               lambda t: _top_share(t.reference["forename_norm"]),
               "Share of records with one of the 10 most common forenames."),
    Diagnostic("top_surname_share", "population", True, "absolute",
               lambda t: _top_share(t.reference["surname_norm"]),
               "Share of records with one of the 10 most common surnames."),
    Diagnostic("error_exposure", "mechanism", False, "absolute", _error("error_exposure"),
               "Share of true pairs with any name disagreement.", taxonomy_layer=1),
    Diagnostic("forename_error_rate", "mechanism", False, "absolute", _error("forename_error_rate"),
               "Share of true pairs whose forenames disagree.", taxonomy_layer=1),
    Diagnostic("surname_error_rate", "mechanism", False, "absolute", _error("surname_error_rate"),
               "Share of true pairs whose surnames disagree.", taxonomy_layer=1),
    Diagnostic("full_surname_replacement", "mechanism", False, "absolute", _error("full_surname_replacement"),
               "Share of true pairs whose surname was wholly replaced.", taxonomy_layer=4),
    Diagnostic("multi_field_error", "mechanism", False, "absolute", _error("multi_field_error"),
               "Share of true pairs where forename and surname both disagree.", taxonomy_layer=2),
    Diagnostic("match_probability_band_mass_tv", "landscape", True, "distribution", lambda t: t.band_mass,
               "Candidate-pair mass in match-probability bands 0-.25-.5-.75-1."),
    Diagnostic("candidate_size_distribution_tv", "landscape", True, "distribution", lambda t: t.candidate_sizes,
               "Distribution of candidate-set sizes per reference record."),
    # --- M4: label-free footprints of the error mechanism and of subgroup structure
    Diagnostic("best_match_band_tv", "landscape", True, "distribution", _best_match_distribution,
               "Per reference record, band of its best candidate score; link-breaking errors leave records "
               "without a good candidate."),
    Diagnostic("learned_error_exposure", "mechanism", True, "absolute", _learned("error_exposure"),
               "Pseudo-truth learner's error exposure, computed on the task itself.", taxonomy_layer=1),
    Diagnostic("learned_forename_error_rate", "mechanism", True, "absolute", _learned("forename_error_rate"),
               "Pseudo-truth learner's forename error rate, computed on the task itself.", taxonomy_layer=1),
    Diagnostic("learned_surname_error_rate", "mechanism", True, "absolute", _learned("surname_error_rate"),
               "Pseudo-truth learner's surname error rate, computed on the task itself.", taxonomy_layer=1),
    Diagnostic("learned_multi_field_error", "mechanism", True, "absolute", _learned("multi_field_error"),
               "Pseudo-truth learner's multi-field error rate, computed on the task itself.", taxonomy_layer=2),
    Diagnostic("ambiguity_rate", "landscape", True, "absolute", _ambiguity_rate,
               "Share of reference records with two or more candidates at or above the operating threshold "
               "(a false-match footprint)."),
    Diagnostic("subgroup_ambiguity_rate", "landscape", True, "grouped_absolute", _subgroup_ambiguity,
               "Ambiguity rate per subgroup; distance is the largest subgroup difference.",
               taxonomy_layer=3, subgroup=True),
    Diagnostic("subgroup_best_match_band_tv", "landscape", True, "grouped_distribution", _subgroup_best_match,
               "Best-match bands per subgroup; distance is the largest within-subgroup TV.",
               taxonomy_layer=3, subgroup=True),
    # --- spine full-name duplication: different people sharing an identical full name drive
    # false matches; swapping or resampling names changes it (scripts/name_duplication_check.py)
    Diagnostic("full_name_duplication_rate", "population", True, "absolute", _full_name_duplication_rate,
               "Share of spine records whose full name (forename + surname) is shared with another spine record."),
    Diagnostic("subgroup_full_name_duplication_rate", "population", True, "grouped_absolute",
               _subgroup_full_name_duplication,
               "Full-name duplication rate per subgroup; distance is the largest subgroup difference.",
               taxonomy_layer=3, subgroup=True),
    # Cluster size matters, not just sharing: an incoming record whose name is held by m spine
    # records has m - 1 exact-name false candidates (scripts/name_duplication_check.py).
    Diagnostic("full_name_collisions_per_record", "population", True, "relative",
               lambda t: float((_full_name_cluster_size(t.reference) - 1).mean()),
               "Mean number of other spine records sharing a record's full name (exact-name false "
               "candidates per record); relative distance."),
]

REGISTRY: dict[str, Diagnostic] = {d.name: d for d in _DIAGNOSTICS}


def names(layer: str | None = None, *, observable: bool | None = None, scalar_only: bool = False) -> list[str]:
    return [d.name for d in REGISTRY.values()
            if (layer is None or d.layer == layer)
            and (observable is None or d.observable == observable)
            and not (scalar_only and not d.scalar)]


def compute(task: TaskBundle) -> dict[str, Any]:
    """Every registered statistic for one task (``None`` where not applicable)."""
    return {name: d.statistic(task) for name, d in REGISTRY.items()}


def scalars(stats: dict[str, Any]) -> dict[str, float]:
    return {n: (float(v) if v is not None else np.nan) for n, v in stats.items() if REGISTRY[n].scalar}


def compare(observed: dict[str, Any], replica: dict[str, Any]) -> pd.DataFrame:
    """One row per diagnostic: metadata, observed and replica values (scalars only) and the distance."""
    rows = []
    for name, d in REGISTRY.items():
        o, r = observed[name], replica[name]
        rows.append({
            "diagnostic": name, "layer": d.layer, "taxonomy_layer": d.taxonomy_layer, "observable": d.observable,
            "kind": d.kind, "by_construction": d.by_construction, "subgroup": d.subgroup,
            "observed": float(o) if d.scalar and o is not None else None,
            "replica": float(r) if d.scalar and r is not None else None,
            "distance": d.distance(o, r),
        })
    return pd.DataFrame(rows)


def of_distance(comparison: pd.DataFrame, unit: float = 0.05) -> dict[str, float]:
    """Continuous OF distance over the truth-based mechanism diagnostics, in units of ``unit`` (5 points)."""
    mech = comparison["layer"].eq("mechanism") & ~comparison["observable"].astype(bool)
    d = comparison.loc[mech, "distance"].abs() / unit
    return {"of_distance_mean": float(d.mean()), "of_distance_max": float(d.max())}
