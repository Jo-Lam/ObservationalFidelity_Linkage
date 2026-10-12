"""Generate a synthetic replica of a linkage task: clean names, then profile-driven corruption."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .names import add_name_features, corrupt_names
from .scenarios import incoming_subset


def _cardinality_preserving_sample(values: pd.Series, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` names that include every observed value once (if n allows), then fill by frequency.

    This keeps the replica's name cardinality and long tail close to the
    observed data, which governs blocking volume and ambiguity.

    Names are ordered by frequency with ties broken by first occurrence. A
    plain ``value_counts()`` breaks ties with an unstable, CPU-dependent sort,
    which made replicas differ across machines.
    """
    s = values.dropna().astype(str)
    vc = s.groupby(s, sort=False).size().sort_values(ascending=False, kind="stable")
    if vc.empty:
        return np.array([""] * n, dtype=object)
    unique = vc.index.to_numpy()
    if len(unique) >= n:
        return rng.choice(unique, size=n, replace=False)
    fill = rng.choice(unique, size=n - len(unique), replace=True, p=(vc / vc.sum()).to_numpy())
    out = np.concatenate([unique, fill])
    rng.shuffle(out)
    return out


def synthesise_clean_reference(source: pd.DataFrame, n: int, seed: int, subgroup_col: str | None = None) -> pd.DataFrame:
    """Synthetic clean reference file.

    Forenames and surnames are sampled independently (within subgroup when
    ``subgroup_col`` is given), so observed full-name combinations are not
    copied as units.
    """
    rng = np.random.default_rng(seed)
    ref = pd.DataFrame({"truth_id": [f"syn_{seed}_{i}" for i in range(n)]})
    ref["record_id"] = ref["truth_id"]
    if subgroup_col and subgroup_col in source.columns:
        group_probs = source[subgroup_col].fillna("Unknown").value_counts(normalize=True)
        groups = rng.choice(group_probs.index.to_numpy(), size=n, p=group_probs.to_numpy())
        ref[subgroup_col] = groups
        forenames = np.empty(n, dtype=object)
        surnames = np.empty(n, dtype=object)
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            pool = source[source[subgroup_col].fillna("Unknown") == g]
            if len(pool) == 0:
                pool = source
            forenames[idx] = _cardinality_preserving_sample(pool["forename_norm"], len(idx), rng)
            surnames[idx] = _cardinality_preserving_sample(pool["surname_norm"], len(idx), rng)
        ref["forename_norm"] = forenames
        ref["surname_norm"] = surnames
    else:
        ref["forename_norm"] = _cardinality_preserving_sample(source["forename_norm"], n, rng)
        ref["surname_norm"] = _cardinality_preserving_sample(source["surname_norm"], n, rng)
    return ref


def _assign_strata(source: pd.DataFrame, levels: list[list[str]], min_size: int) -> np.ndarray:
    """Assign each record to the finest stratum level whose stratum still has >= ``min_size`` unassigned records.

    Levels run from finest to coarsest. Whatever is left after the last level is
    grouped by that level's keys regardless of size.
    """
    n = len(source)
    stratum = np.full(n, -1, dtype=int)
    next_id = 0
    for i, keys in enumerate(levels):
        open_idx = np.where(stratum == -1)[0]
        if not len(open_idx):
            break
        sub = source.iloc[open_idx]
        labels = sub[keys].astype(str).agg("|".join, axis=1) if keys else pd.Series("all", index=sub.index)
        sizes = labels.map(labels.value_counts()).to_numpy()
        selected = (sizes >= min_size) | (i == len(levels) - 1)
        # Stratum ids in order of first appearance (as pd.unique), assigned in one pass.
        codes, uniques = pd.factorize(labels[selected])
        stratum[open_idx[selected]] = next_id + codes
        next_id += len(uniques)
    return stratum


def _derange(idx: np.ndarray, values: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A random cyclic derangement within ``idx``: returns (positions, values to place there).

    Position ``perm[j]`` receives the value of ``perm[j - 1]``, so nobody keeps
    their own value (unless values repeat).
    """
    perm = rng.permutation(idx)
    return perm, values[np.roll(perm, 1)]


def _groups(stratum: np.ndarray) -> list[np.ndarray]:
    """Member indices (ascending) of each stratum, in ascending stratum id order."""
    order = np.argsort(stratum, kind="stable")
    _, starts = np.unique(stratum[order], return_index=True)
    return np.split(order, starts[1:])


def _swap_clusters(cluster: np.ndarray, csize: np.ndarray, stratum: np.ndarray, group: np.ndarray,
                   forenames: np.ndarray, surnames: np.ndarray, swapped: np.ndarray, rng: np.random.Generator,
                   rounds: int) -> int:
    """Derange surnames among same-size full-name clusters (size >= 2), in place.

    Clusters are matched within ``stratum`` (that of their first member) when it
    holds at least two clusters of their size, otherwise within ``group``.
    Returns the number of records left with their own surname for lack of a partner.
    """
    rows = pd.Series(np.arange(len(cluster)))
    members = rows.groupby(cluster).apply(np.asarray)
    first = rows.groupby(cluster).first().to_numpy()
    size = csize[first]
    multi = size >= 2
    units = pd.DataFrame({"cluster": np.flatnonzero(multi), "first": first[multi], "size": size[multi],
                          "stratum": stratum[first[multi]], "group": group[first[multi]]})

    n_in = units.groupby(["stratum", "size"])["cluster"].transform("size")
    rest = units[n_in < 2]
    n_rest = rest.groupby(["group", "size"])["cluster"].transform("size")
    kept = rest[n_rest < 2]
    # Observed shared names are reserved for their own cluster (which may keep it), so a
    # swap never merges into another cluster; names created by swaps are added as they go.
    used = {(forenames[f], surnames[f]) for f in units["first"]}

    def assign(sub: pd.DataFrame) -> None:
        """Greedy: each cluster takes a donor surname (another cluster's) that does not form a
        full name already in use, so clusters do not merge; failing that it keeps its own."""
        firsts = sub["first"].to_numpy()
        pool = pd.Series(surnames[firsts]).value_counts()
        # Most constrained first: clusters with the fewest donor surnames that are neither
        # their own nor already in use with their forename.
        options = np.array([len(firsts) - pool.get(surnames[f], 0)
                            - sum(pool.get(s, 0) for s in pool.index if s != surnames[f] and (forenames[f], s) in used)
                            for f in firsts]) if len(firsts) <= 200 else np.zeros(len(firsts))
        best, best_bad = None, None
        for attempt in range(rounds + 1):  # retry orders; keep the draw with the fewest merges
            donors = list(surnames[firsts[rng.permutation(len(sub))]])
            order = np.argsort(options + rng.random(len(sub)) * (0.5 + attempt), kind="stable")
            taken, plan, bad = set(), [], 0
            for c, f in zip(sub["cluster"].to_numpy()[order], sub["first"].to_numpy()[order]):
                fore, own = forenames[f], surnames[f]
                pick = next((i for i, s in enumerate(donors)
                             if s != own and (fore, s) not in used and (fore, s) not in taken), None)
                if pick is None:  # keep its own (shared, so not unique) name if its surname is still free
                    pick = next((i for i, s in enumerate(donors) if s == own), None)
                if pick is None:
                    bad += 1
                    pick = next((i for i, s in enumerate(donors) if s != own), 0)
                s = donors.pop(pick)
                taken.add((fore, s))
                plan.append((c, s))
            if best is None or bad < best_bad:
                best, best_bad = (plan, taken), bad
            if bad == 0:
                break
        used.update(best[1])
        for c, s in best[0]:
            swapped[members[c]] = s

    for _, sub in units[n_in >= 2].groupby(["stratum", "size"], sort=True):
        assign(sub)
    for _, sub in rest[n_rest >= 2].groupby(["group", "size"], sort=True):
        assign(sub)
    return int(kept["size"].sum())


def _resolve_collisions(firsts: np.ndarray, members: list[np.ndarray], size: np.ndarray, keys: list,
                        forenames: np.ndarray, swapped: np.ndarray, rng: np.random.Generator, rounds: int) -> None:
    """Undo chance full-name coincidences created by swapping, in place.

    Units are records or same-name clusters (``members``; ``firsts`` is one
    member of each). A unit collides when its new full name is held by more
    records than the unit itself. Each colliding unit swaps surnames with a random
    non-colliding unit sharing its key, trying ``keys`` from finest to coarsest
    (for clusters the keys include the size, so sizes and surname frequencies are
    unchanged). Repeats until no unit collides or ``rounds`` is reached.
    """
    key_groups = [pd.Series(np.arange(len(firsts))).groupby(np.asarray(k)).indices for k in keys]
    for _ in range(rounds):
        full = pd.Series(forenames) + "\x1f" + pd.Series(swapped)
        hit = full.map(full.value_counts()).to_numpy()[firsts] > size
        if not hit.any():
            return
        todo = hit.copy()
        for groups in key_groups:
            for idx in groups.values():
                h = idx[todo[idx]]
                if not len(h):
                    continue
                pool = idx[~hit[idx]]
                h = h[rng.permutation(len(h))][:len(pool)]
                for a, b in zip(h, rng.choice(pool, len(h), replace=False)):
                    sa, sb = swapped[firsts[a]], swapped[firsts[b]]
                    swapped[members[a]], swapped[members[b]] = sb, sa
                todo[h] = False


def stratified_swap_reference(source: pd.DataFrame, seed: int, subgroup_col: str | None, strata: list[list[str]],
                              min_stratum_size: int = 5, rare_pair_max_count: int = 1,
                              max_reshuffles: int = 20,
                              preserve_name_clusters: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Synthetic clean reference by swapping surnames within blocking-relevant strata.

    Each record keeps its forename (and subgroup) and receives the surname of
    another record in the same stratum. Strata are the finest level in
    ``strata`` (subgroup prepended) holding at least ``min_stratum_size``
    records, so every swap is among at least that many candidates where possible.

    By construction the forename and surname frequency distributions, subgroup
    composition and the joint distribution of the stratum keys (for example the
    forename/surname prefix pairs used in blocking) equal the observed ones,
    while individual forename-surname pairings are broken.

    Rare-pair guard: a swap can recreate an observed full name. Strata whose
    result recreates an observed name occurring at most ``rare_pair_max_count``
    times are reshuffled, up to ``max_reshuffles`` times, keeping the best draw.
    The guard cannot help when a stratum's surnames are identical, which is common
    in fine strata; ``share_recreating_unique_observed_name`` reports the outcome.

    Choice of strata (``scripts/synthesis_comparison.py``): swapping within
    subgroup only (``strata: []``) reproduces the observed score landscape almost
    to the Monte Carlo null with disclosure close to independent sampling. Fine
    blocking strata reach the null exactly, but recreate several times more
    unique observed names. Avoid two-level designs that leave a large pooled
    remainder: the remainder is a biased subset and distorts the joint structure.

    ``preserve_name_clusters``: also keep the full-name cluster-size distribution
    (how many records share each full name). Records sharing a full name move as
    one unit and receive the surname of another cluster with the same size and
    subgroup make-up; observed shared names are reserved for their own cluster;
    chance coincidences created by swapping unshared records are undone. Cluster
    size, not just sharing, drives exact-name false matches, and plain swaps
    break large clusters up (PDS 100k: precision overstated by 2-4 pp; see
    scripts/name_duplication_check.py). A cluster may keep its own name when no
    swap is possible; such names are shared by two or more records, so they do
    not count as recreated unique names.
    """
    rng = np.random.default_rng(seed)
    src = source.reset_index(drop=True)
    prefix = [subgroup_col] if subgroup_col and subgroup_col in src.columns else []
    levels = [prefix + list(level) for level in strata] + [prefix]
    stratum = _assign_strata(src, levels, min_stratum_size)

    forenames = src["forename_norm"].to_numpy(dtype=object)
    surnames = src["surname_norm"].to_numpy(dtype=object)
    pair_counts = pd.Series(list(zip(forenames, surnames))).value_counts()
    rare = set(pair_counts[pair_counts <= rare_pair_max_count].index)  # what the guard avoids
    singletons = set(pair_counts[pair_counts == 1].index)  # what is reported, independent of the guard

    swapped = surnames.copy()
    unswappable = 0
    swap_stratum = stratum
    if preserve_name_clusters:
        # Records sharing a full name move as one unit: the cluster receives the
        # surname of another cluster with the same size and subgroup make-up, so the
        # full-name cluster-size distribution is kept along with the name frequencies
        # (overall and within subgroup).
        group_key = src[prefix[0]].astype(str) if prefix else pd.Series("all", index=src.index)
        cluster = pd.factorize(pd.Series(forenames) + "\x1f" + pd.Series(surnames))[0]
        csize = np.bincount(cluster)[cluster]
        multi = csize >= 2
        makeup = pd.Series("", index=src.index)
        makeup[multi] = (group_key[multi].groupby(cluster[multi])
                         .transform(lambda s: ";".join(f"{k}:{v}" for k, v in sorted(s.value_counts().items()))))
        unswappable += _swap_clusters(cluster, csize, (pd.Series(stratum).astype(str) + "|" + makeup).to_numpy(),
                                      makeup.to_numpy(), forenames, surnames, swapped, rng, max_reshuffles)
        swap_stratum = np.where(csize == 1, stratum, -1)
    for idx in _groups(swap_stratum):
        if preserve_name_clusters and swap_stratum[idx[0]] == -1:
            continue
        if len(idx) < 2:
            unswappable += len(idx)
            continue
        best, best_bad = None, None
        for _ in range(max_reshuffles + 1):
            positions, values = _derange(idx, surnames, rng)
            bad = sum((forenames[p], v) in rare for p, v in zip(positions, values))
            if best is None or bad < best_bad:
                best, best_bad = (positions, values), bad
            if bad == 0:
                break
        swapped[best[0]] = best[1]
    if preserve_name_clusters:
        # Swapping singletons creates chance coincidences on top of the preserved clusters.
        singles = np.flatnonzero(csize == 1)
        _resolve_collisions(singles, [np.array([i]) for i in singles], np.ones(len(singles), dtype=int),
                            [stratum[singles], group_key.to_numpy()[singles]], forenames, swapped, rng,
                            max_reshuffles)

    ref = pd.DataFrame({"truth_id": [f"syn_{seed}_{i}" for i in range(len(src))]})
    ref["record_id"] = ref["truth_id"]
    if prefix:
        ref[subgroup_col] = src[subgroup_col].to_numpy()
    ref["forename_norm"] = forenames
    ref["surname_norm"] = swapped
    pairs = list(zip(forenames, swapped))
    sizes = pd.Series(stratum).value_counts()
    info = {
        "n_strata": int(len(sizes)),
        "median_stratum_size": float(sizes.median()),
        "share_records_in_strata_below_min": float(sizes[sizes < min_stratum_size].sum() / len(src)),
        "share_unswappable": float(unswappable / len(src)),
        "share_kept_own_surname": float(np.mean(swapped == surnames)),
        "share_recreating_unique_observed_name": float(np.mean([p in singletons for p in pairs])),
    }
    return ref, info


def synthesise(source: pd.DataFrame, seed: int, subgroup_col: str | None,
               synthesis: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Dispatch to the configured clean-name synthesiser; returns (clean reference, synthesis info)."""
    synthesis = synthesis or {"method": "independent"}
    method = synthesis.get("method", "independent")
    if method == "independent":
        return synthesise_clean_reference(source, len(source), seed, subgroup_col), {}
    if method == "identity":
        # Calibration only: the observed names themselves (100% disclosure). With the
        # supervised profile this is a correct generator by construction, used to
        # measure the gate's size (scripts/gate_operating_characteristics.py).
        cols = ["forename_norm", "surname_norm"] + ([subgroup_col] if subgroup_col and subgroup_col in source.columns else [])
        ref = source[cols].reset_index(drop=True).copy()
        ref.insert(0, "truth_id", [f"syn_{seed}_{i}" for i in range(len(ref))])
        ref.insert(1, "record_id", ref["truth_id"])
        return ref, {"share_recreating_unique_observed_name": 1.0}
    if method == "stratified_swap":
        return stratified_swap_reference(
            source, seed, subgroup_col, strata=synthesis["strata"],
            min_stratum_size=int(synthesis.get("min_stratum_size", 5)),
            rare_pair_max_count=int(synthesis.get("rare_pair_max_count", 1)),
            max_reshuffles=int(synthesis.get("max_reshuffles", 20)),
            preserve_name_clusters=bool(synthesis.get("preserve_name_clusters", False)))
    raise ValueError(f"Unknown synthesis method: {method!r}")


def corrupt_replica(clean: pd.DataFrame, observed_reference: pd.DataFrame, profile: dict[str, Any], seed: int) -> pd.DataFrame:
    """Corrupt a clean replica using a learned profile's error rates and co-occurrence."""
    cmp = clean.copy().reset_index(drop=True)
    cmp["record_id"] = [f"cmp_{seed}_{i}" for i in range(len(cmp))]
    return corrupt_names(
        cmp, observed_reference, seed=seed,
        forename_error_rate=float(profile.get("forename_error_rate", 0.0) or 0.0),
        surname_error_rate=float(profile.get("surname_error_rate", 0.0) or 0.0),
        multi_field_error_rate=float(profile.get("multi_field_error", 0.0) or 0.0),
        full_surname_replacement_rate=float(profile.get("full_surname_replacement", 0.0) or 0.0),
    )


def generate_replica(observed_reference: pd.DataFrame, profile: dict[str, Any], replicate_seed: int,
                     subgroup_col: str | None = None, synthesis: dict[str, Any] | None = None,
                     incoming: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return (clean reference, corrupted comparison, synthesis info) for one replicate seed.

    The same ``replicate_seed`` gives the same clean reference (and incoming
    subset) in every learning mode, so modes differ only in the corruption
    profile (common random numbers). ``incoming`` mirrors the observed task's
    spine/incoming design; the subset is drawn afresh per replica.
    """
    generation_seed = replicate_seed * 1000 + 1
    clean, info = synthesise(observed_reference, generation_seed, subgroup_col, synthesis)
    base = incoming_subset(clean, incoming, seed=generation_seed + 555)
    corrupted = corrupt_replica(base, observed_reference, profile, seed=generation_seed + 777)
    return add_name_features(clean), add_name_features(corrupted), info
