"""Which swap strata does the gate accept at scale, and at what disclosure cost?

Runs the supervised mode of configs/study_scale.yaml (10 replicas) for several
synthesisers per scenario and reports replica-vs-observed precision and
missed-match error, the live gate verdict, the share of synthetic names that
recreate a unique observed name, and (NCVR) the Non-Hispanic Black false-match
disparity. Rule used in study_scale.yaml: the lowest-disclosure synthesiser the
gate accepts.

Usage: python scripts/scale_synthesis_check.py [--workers 4]
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import yaml

from ofeval.experiment import OBSERVED, run_scenario

ROOT = Path(__file__).resolve().parents[1]
BLOCKING = [["forename_prefix2", "surname_prefix2", "surname_len_bin"], ["forename_prefix2", "surname_prefix2"],
            ["forename_initial", "surname_prefix2"], ["surname_prefix2"]]
SWAP = {"method": "stratified_swap", "strata": [], "min_stratum_size": 2}
KEEP = {"preserve_name_clusters": True}
BLOCK = lambda k: {**SWAP, "strata": BLOCKING, "min_stratum_size": k}
CANDIDATES = [
    ("pds_100k", "PDS", [("swap + keep name clusters", {**SWAP, **KEEP}),
                         ("blocking k=10 + keep name clusters", {**BLOCK(10), **KEEP}),
                         ("blocking k=5 + keep name clusters", {**BLOCK(5), **KEEP}),
                         ("blocking k=10", BLOCK(10)), ("blocking k=5", BLOCK(5)), ("blocking k=2", BLOCK(2))]),
    ("ncvr_50k", "NCVR", [("swap within subgroup", SWAP), ("swap + keep name clusters", {**SWAP, **KEEP}),
                          ("blocking k=5", BLOCK(5)), ("blocking k=2", BLOCK(2))]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="run only these synthesiser labels (merged into summary.csv)")
    args = ap.parse_args()
    base = yaml.safe_load((ROOT / "configs/study_scale.yaml").read_text())
    out = ROOT / "outputs/scale_synthesis_check"
    rows = []
    for scn, name, variants in CANDIDATES:
        cfg = yaml.safe_load((ROOT / f"configs/scenarios/{scn}.yaml").read_text())
        for label, synth in variants:
            if args.only and label not in args.only:
                continue
            study = copy.deepcopy(base)
            study.pop("synthesis_by_scenario", None)
            study["synthesis"] = synth
            study["evaluation"].update({"seeds": 10, "first_seed": 1})
            study["modes"] = [{"label": "Supervised", "profile": "supervised", "gate": "live", "legacy_gate": None}]
            res = run_scenario(cfg, study, ROOT / "data/raw", out / "runs" / f"{name}_{label}", workers=args.workers,
                               log=lambda *_: None)
            link = res["linkage"]
            ov = link[link["group"] == "Overall"]
            obs, rep = ov[ov["mode"] == OBSERVED].iloc[0], ov[ov["mode"] == "Supervised"]
            live = res["generator_verdicts"].query("gate_policy == 'live'").iloc[0]
            row = {"scenario": name, "synthesiser": label,
                   "precision_err": rep["precision"].mean() - obs["precision"],
                   "missed_match_err": rep["missed_match_rate"].mean() - obs["missed_match_rate"],
                   "live_gate": "accepted" if live["passed"] else "rejected", "failed": live["failed_diagnostics"],
                   "unique_names_recreated": res["tasks"]["synthesis_share_recreating_unique_observed_name"].median()}
            grp = link[link["group"] == "Non-Hispanic Black"]
            if len(grp):
                col = "accepted_link_false_match_rate_difference_vs_reference"
                row["nhb_fmr_disparity_observed"] = grp[grp["mode"] == OBSERVED][col].iloc[0]
                row["nhb_fmr_disparity_replica_median"] = grp[grp["mode"] == "Supervised"][col].median()
            rows.append(row)
            print(f"[{name}] {label} done", flush=True)
    df = pd.DataFrame(rows)
    if args.only and (out / "summary.csv").exists():
        old = pd.read_csv(out / "summary.csv")
        keep = ~(old["scenario"] + "|" + old["synthesiser"]).isin(df["scenario"] + "|" + df["synthesiser"])
        df = pd.concat([old[keep], df], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
