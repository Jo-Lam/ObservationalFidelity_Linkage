"""M3: which clean-name synthesiser reproduces the observed population and score landscape?

Every variant is corrupted with the TRUE observed mechanism, so differences from
the observed task reflect name synthesis only. The target is the null band:
observed names re-corrupted with fresh seeds. Disclosure cost is reported as
the share of synthetic full names that recreate a name occurring once
(``rare``) or at all (``exact``) in the observed reference.

Usage: python scripts/synthesis_comparison.py [--seeds 10]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from ofeval import diagnostics
from ofeval.linkage import link
from ofeval.names import add_name_features, corrupt_names
from ofeval.scenarios import build_observed_task
from ofeval.synthesis import synthesise

ROOT = Path(__file__).resolve().parents[1]
BLOCKING_STRATA = [
    ["forename_prefix2", "surname_prefix2", "surname_len_bin"],
    ["forename_prefix2", "surname_prefix2"],
    ["forename_initial", "surname_prefix2"],
    ["surname_prefix2"],
]
VARIANTS = {
    "null (observed names)": None,
    "independent (paper)": {"method": "independent"},
    "swap within subgroup": {"method": "stratified_swap", "strata": [], "min_stratum_size": 2},
    "swap initials k=20": {"method": "stratified_swap", "strata": [["forename_initial", "surname_initial"]], "min_stratum_size": 20},
    "swap initials k=50": {"method": "stratified_swap", "strata": [["forename_initial", "surname_initial"]], "min_stratum_size": 50},
    "swap surname prefix2 k=20": {"method": "stratified_swap", "strata": [["surname_prefix2"]], "min_stratum_size": 20},
    "swap blocking strata k=2": {"method": "stratified_swap", "strata": BLOCKING_STRATA, "min_stratum_size": 2},
    "swap blocking strata k=5": {"method": "stratified_swap", "strata": BLOCKING_STRATA, "min_stratum_size": 5},
    "swap blocking strata k=10": {"method": "stratified_swap", "strata": BLOCKING_STRATA, "min_stratum_size": 10},
}
CHECKS = ["top_forename_share", "top_surname_share", "match_probability_band_mass_tv", "candidate_size_distribution_tv"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    rows = []
    for name in ["NCVR", "PDS"]:
        cfg = yaml.safe_load((ROOT / f"configs/scenarios/{name.lower()}.yaml").read_text())
        sub = (cfg.get("subgroup") or {}).get("name")
        ref, cmp = build_observed_task(cfg, ROOT / "data/raw")
        obs = diagnostics.compute(diagnostics.TaskBundle(ref, cmp, link(ref, cmp, cfg["linkage"], sub)))
        oc = cfg["observed_corruption"]
        rates = {k: v for k, v in oc.items() if k != "seed"}
        obs_pairs = pd.Series(list(zip(ref["forename_norm"], ref["surname_norm"]))).value_counts()
        singletons = set(obs_pairs[obs_pairs == 1].index)
        for variant, synth in VARIANTS.items():
            for s in range(1, args.seeds + 1):
                if synth is None:
                    clean, info = ref.copy(), {}
                    seed = oc["seed"] + 1000 * s
                else:
                    clean, info = synthesise(ref, s * 1000 + 1, sub, synth)
                    clean = add_name_features(clean)
                    seed = s * 1000 + 778
                c = corrupt_names(clean, ref, seed=seed, **rates)
                c["record_id"] = "cmp_" + c["truth_id"].astype(str)
                c = add_name_features(c)
                stats = diagnostics.compute(diagnostics.TaskBundle(clean, c, link(clean, c, cfg["linkage"], sub)))
                comp = diagnostics.compare(obs, stats).set_index("diagnostic")["distance"]
                pairs = list(zip(clean["forename_norm"], clean["surname_norm"]))
                rows.append({"scenario": name, "variant": variant, "seed": s,
                             **{c_: abs(comp[c_]) for c_ in CHECKS},
                             "exact_name_overlap": float(pd.Series([p in obs_pairs.index for p in pairs]).mean()),
                             "rare_name_overlap": float(pd.Series([p in singletons for p in pairs]).mean()),
                             **{f"info_{k}": v for k, v in info.items()}})
        print(f"[{name}] done", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "outputs/synthesis_comparison"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "synthesis_comparison_seed_level.csv", index=False)
    cols = CHECKS + ["exact_name_overlap", "rare_name_overlap"]
    summary = df.groupby(["scenario", "variant"], sort=False)[cols].median().round(4)
    summary.to_csv(out / "synthesis_comparison_median.csv")
    pd.set_option("display.width", 250)
    print("\nMedian over seeds (|distance| from observed; overlap = share of synthetic clean names):")
    print(summary.to_string())
    info_cols = [c for c in df.columns if c.startswith("info_")]
    if info_cols:
        print("\nSwap diagnostics (median):")
        print(df.dropna(subset=info_cols[:1]).groupby(["scenario", "variant"], sort=False)[info_cols].median().round(4).to_string())


if __name__ == "__main__":
    main()
