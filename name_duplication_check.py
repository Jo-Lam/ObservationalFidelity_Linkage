"""Spine full-name duplication: observed vs synthesised, per synthesiser.

Different people sharing an identical full name drive false matches, so a
synthesiser that changes how often full names are shared changes precision.
This reports, for each scenario and synthesiser, the share of spine records
whose full name is shared (observed, replica mean, bias, replica SD).

Read with scripts/scale_synthesis_check.py (replica precision error per
synthesiser): at 50k-100k, synthesisers that shift duplication by more than
about 1 pp mis-state precision by about 3 pp or more (NCVR blocking strata
+1.1 to +1.3 pp -> precision -3.2 to -3.8 pp; PDS k=10 -1.1 pp -> +3.6 pp),
which motivates the 0.01 margin in configs/study*.yaml. The share alone is not
enough: PDS k=5 matches it (+0.1 pp) yet overstates precision by ~2 pp, because
it has too few 3-5 record clusters. Collisions per record (mean number of other
records sharing the full name, i.e. exact-name false candidates) tracks the
precision error in every case (~0.11 pp per 1%); "swap + keep name clusters"
reproduces it. Near-identical (not identical) names were checked too (spine
records with a neighbour scoring above the threshold): every synthesiser
reproduces their density within ~1 pp, so they do not explain the gap.

Usage: python scripts/name_duplication_check.py [--seeds 8]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ofeval.scenarios import build_observed_task
from ofeval.synthesis import synthesise

ROOT = Path(__file__).resolve().parents[1]
BLOCKING = [["forename_prefix2", "surname_prefix2", "surname_len_bin"], ["forename_prefix2", "surname_prefix2"],
            ["forename_initial", "surname_prefix2"], ["surname_prefix2"]]
SYNTHESISERS = {
    "subgroup swap": {"method": "stratified_swap", "strata": [], "min_stratum_size": 2},
    "blocking k=10": {"method": "stratified_swap", "strata": BLOCKING, "min_stratum_size": 10},
    "blocking k=5": {"method": "stratified_swap", "strata": BLOCKING, "min_stratum_size": 5},
    "blocking k=2": {"method": "stratified_swap", "strata": BLOCKING, "min_stratum_size": 2},
    "swap + keep name clusters": {"method": "stratified_swap", "strata": [], "min_stratum_size": 2,
                                  "preserve_name_clusters": True},
    "independent (paper)": {"method": "independent"},
}


def cluster_stats(df: pd.DataFrame) -> tuple[float, float]:
    """(share of records whose full name is shared, mean number of other records sharing it)."""
    full = df["forename_norm"].astype(str) + "|" + df["surname_norm"].astype(str)
    size = full.map(full.value_counts())
    return float((size > 1).mean()), float((size - 1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    rows = []
    for scn in ["ncvr", "pds", "ncvr_50k", "pds_100k"]:
        cfg = yaml.safe_load((ROOT / f"configs/scenarios/{scn}.yaml").read_text())
        sub = (cfg.get("subgroup") or {}).get("name")
        ref, _ = build_observed_task(cfg, ROOT / "data/raw")
        obs, obs_coll = cluster_stats(ref)
        for label, synth in SYNTHESISERS.items():
            out = [synthesise(ref, s * 1000 + 1, sub, synth) for s in range(1, args.seeds + 1)]
            stats = np.array([cluster_stats(o[0]) for o in out])
            rates, coll = stats[:, 0], stats[:, 1]
            rows.append({"scenario": scn, "synthesiser": label, "observed": obs, "replica_mean": rates.mean(),
                         "bias_pp": 100 * (rates.mean() - obs), "replica_sd_pp": 100 * rates.std(ddof=1),
                         "collisions_observed": obs_coll, "collisions_replica": coll.mean(),
                         "collisions_rel_bias": coll.mean() / obs_coll - 1 if obs_coll else np.nan,
                         "unique_names_recreated": np.mean([o[1].get("share_recreating_unique_observed_name", np.nan)
                                                            for o in out])})
        print(f"[{scn}] done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "outputs/name_duplication_check"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
