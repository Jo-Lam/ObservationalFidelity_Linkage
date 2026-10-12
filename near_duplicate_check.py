"""Near-identical spine names and false matches: observed vs synthesised.

Two questions about the residual PDS precision gap at scale:

1. Near-duplicate density: the share of spine records with another spine record
   whose *different* name scores at or above the observed link threshold. Every
   synthesiser reproduces it within ~1 pp, so it does not explain the gap.
2. False-match decomposition (observed task vs supervised replicas): accepted
   false pairs per 100 incoming records, split by whether the names agree
   exactly and whether the incoming record is uncorrupted. In PDS the whole gap
   is in exact-name false matches against uncorrupted records, i.e. people who
   share a full name: a cluster-size effect (scripts/name_duplication_check.py).

Usage: python scripts/near_duplicate_check.py [--seeds 4]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ofeval.linkage import build_candidates, link, name_similarity
from ofeval.names import add_name_features
from ofeval.profiles import learn_supervised
from ofeval.scenarios import build_observed_task
from ofeval.synthesis import generate_replica, synthesise

ROOT = Path(__file__).resolve().parents[1]
BLOCKING = [["forename_prefix2", "surname_prefix2", "surname_len_bin"], ["forename_prefix2", "surname_prefix2"],
            ["forename_initial", "surname_prefix2"], ["surname_prefix2"]]
SWAP = {"method": "stratified_swap", "strata": [], "min_stratum_size": 2}
SYNTHESISERS = {
    "subgroup swap": SWAP,
    "blocking k=10": {**SWAP, "strata": BLOCKING, "min_stratum_size": 10},
    "blocking k=5": {**SWAP, "strata": BLOCKING, "min_stratum_size": 5},
    "blocking k=2": {**SWAP, "strata": BLOCKING, "min_stratum_size": 2},
    "swap + keep name clusters": {**SWAP, "preserve_name_clusters": True},
    "blocking k=5 + keep name clusters": {**SWAP, "strata": BLOCKING, "min_stratum_size": 5,
                                          "preserve_name_clusters": True},
}
SCENARIOS = [("pds_100k", "PDS", "blocking k=5"), ("ncvr_50k", "NCVR", "subgroup swap")]


def near_duplicate_rate(ref: pd.DataFrame, linkage_cfg: dict, threshold: float) -> float:
    """Share of spine records with a differently named spine record scoring >= threshold (on distinct names)."""
    rules = linkage_cfg["blocking_rules"]
    ref = add_name_features(ref)
    cols = sorted({c for r in rules for c in r} | {"forename_norm", "surname_norm"})
    u = ref[cols].drop_duplicates(["forename_norm", "surname_norm"]).reset_index(drop=True)
    u["record_id"] = np.arange(len(u))
    p = build_candidates(u, u, rules)
    p = p[p["id_l"] < p["id_r"]]
    lft, rgt = u.loc[p["id_l"].to_numpy()].reset_index(drop=True), u.loc[p["id_r"].to_numpy()].reset_index(drop=True)
    sc = linkage_cfg["scoring"]
    logit = (sc.get("intercept", -7.0)
             + sc.get("forename_weight", 3.0) * name_similarity(lft["forename_norm"], rgt["forename_norm"])
             + sc.get("surname_weight", 5.0) * name_similarity(lft["surname_norm"], rgt["surname_norm"])
             + sc.get("exact_forename_bonus", 1.0) * (lft["forename_norm"] == rgt["forename_norm"]).to_numpy(float)
             + sc.get("exact_surname_bonus", 1.5) * (lft["surname_norm"] == rgt["surname_norm"]).to_numpy(float))
    hit = 1 / (1 + np.exp(-logit)) >= threshold
    flagged = np.zeros(len(u), bool)
    flagged[p["id_l"].to_numpy()[hit]] = True
    flagged[p["id_r"].to_numpy()[hit]] = True
    key = lambda d: d["forename_norm"].astype(str) + "|" + d["surname_norm"].astype(str)
    return float(key(ref).map(pd.Series(flagged, index=key(u))).mean())


def false_match_decomposition(ref: pd.DataFrame, comp: pd.DataFrame, scored: pd.DataFrame, thr: float) -> dict:
    acc = scored[scored["match_probability"] >= thr]
    fp = acc[acc["is_true_match"] == 0]
    truth = ref.set_index("truth_id")[["forename_norm", "surname_norm"]].reindex(fp["truth_id_r"].astype(ref["truth_id"].dtype).to_numpy())
    clean = ((truth["forename_norm"].to_numpy() == fp["forename_norm_r"].to_numpy())
             & (truth["surname_norm"].to_numpy() == fp["surname_norm_r"].to_numpy()))
    exact = ((fp["forename_exact"] == 1) & (fp["surname_exact"] == 1)).to_numpy()
    per100 = 100 / len(comp)
    return {"precision": 1 - len(fp) / len(acc), "false_per_100": len(fp) * per100,
            "exact_name_uncorrupted": (exact & clean).sum() * per100,
            "exact_name_corrupted": (exact & ~clean).sum() * per100,
            "near_name_uncorrupted": (~exact & clean).sum() * per100,
            "near_name_corrupted": (~exact & ~clean).sum() * per100}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    out = ROOT / "outputs/near_duplicate_check"
    out.mkdir(parents=True, exist_ok=True)
    density, decomp = [], []
    for scn, run, current in SCENARIOS:
        cfg = yaml.safe_load((ROOT / f"configs/scenarios/{scn}.yaml").read_text())
        thr_file = ROOT / f"outputs/study_scale/scenarios/{run}/observed_threshold.json"
        thr = json.loads(thr_file.read_text())["threshold"]  # run configs/study_scale.yaml first
        sub = (cfg.get("subgroup") or {}).get("name")
        ref, comp = build_observed_task(cfg, ROOT / "data/raw")
        obs = near_duplicate_rate(ref, cfg["linkage"], thr)
        for label, synth in SYNTHESISERS.items():
            v = np.array([near_duplicate_rate(synthesise(ref, s * 1000 + 1, sub, synth)[0], cfg["linkage"], thr)
                          for s in range(1, args.seeds + 1)])
            density.append({"scenario": run, "synthesiser": label, "observed": obs, "replica_mean": v.mean(),
                            "bias_pp": 100 * (v.mean() - obs), "replica_sd_pp": 100 * v.std(ddof=1)})
        decomp.append({"scenario": run, "task": "observed",
                       **false_match_decomposition(ref, comp, link(ref, comp, cfg["linkage"], sub), thr)})
        prof = learn_supervised(ref, comp)
        for label in [current, "swap + keep name clusters", "blocking k=5 + keep name clusters"]:
            for s in range(1, args.seeds + 1):
                cl, co, _ = generate_replica(ref, prof, s, sub, synthesis=SYNTHESISERS[label],
                                             incoming=cfg.get("incoming"))
                decomp.append({"scenario": run, "task": f"replica: {label}", "seed": s,
                               **false_match_decomposition(cl, co, link(cl, co, cfg["linkage"], sub), thr)})
        print(f"[{run}] done", flush=True)
    density, decomp = pd.DataFrame(density), pd.DataFrame(decomp)
    density.to_csv(out / "near_duplicate_density.csv", index=False)
    decomp.to_csv(out / "false_match_decomposition.csv", index=False)
    pd.set_option("display.width", 250)
    print(density.round(4).to_string(index=False))
    print(decomp.drop(columns="seed").groupby(["scenario", "task"], sort=False).mean().round(3).to_string())


if __name__ == "__main__":
    main()
