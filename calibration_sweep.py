"""Calibration with and without a small clerical sample: rate recovery and proxy validity.

For several redraws of the observed task (new seed for the true mechanism) and
clerical sample sizes n (0 = label-free), this compares four ways to obtain an
error profile:

* pseudo-truth (label-free, uncorrected);
* clerical sample alone (the n reviewed pairs' rates);
* calibrated: indirect inference on label-free moments, plus the clerical sample
  as precision-weighted evidence when n > 0;
* supervised (all true pairs), the reference.

For each it reports (1) error of the recovered rates against the truth,
(2) linkage quality of its replicas at the observed task's fixed threshold
against the observed task's own linkage quality (is the replica a valid proxy?)
and (3) the label-free `live` gate verdict.

At scale (configs/study_scale.yaml) a redraw also draws a new incoming subset, so
each block is a new observed task: new true error mechanism and new incoming file.

Usage: python scripts/calibration_sweep.py [--blocks 5] [--sizes 0,25,50,100,200] [--seeds 10] [--workers 4]
       python scripts/calibration_sweep.py --study configs/study_scale.yaml --out outputs/calibration_sweep_scale \
           --blocks 5 --sizes 0,50,200
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from ofeval.experiment import OBSERVED, run_scenario

ROOT = Path(__file__).resolve().parents[1]
RATES = ["forename_error_rate", "surname_error_rate", "multi_field_error", "full_surname_replacement"]
LINK = ["recall", "missed_match_rate", "precision", "accepted_link_false_match_rate"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--sizes", default="0,25,50,100,200")
    ap.add_argument("--seeds", type=int, default=10, help="evaluation replicas per generator")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--study", default="configs/study.yaml")
    ap.add_argument("--out", default="outputs/calibration_sweep")
    ap.add_argument("--scenarios", default="NCVR,PDS",
                    help="scenarios to (re)run; rows for other scenarios in an existing output are kept")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    study_path = ROOT / args.study
    base = yaml.safe_load(study_path.read_text())
    out = ROOT / args.out
    rows = []
    t0 = time.time()
    scenarios = [s.strip() for s in args.scenarios.split(",")]
    for name in scenarios:
        scn = yaml.safe_load((study_path.parent / base["scenarios"][name]).read_text())
        for block in range(1, args.blocks + 1):
            cfg = copy.deepcopy(scn)
            cfg["observed_corruption"]["seed"] = scn["observed_corruption"]["seed"] + 104729 * block
            if cfg.get("incoming"):
                cfg["incoming"]["seed"] = scn["incoming"]["seed"] + 104729 * block
            for n in sizes:
                study = copy.deepcopy(base)
                study["evaluation"].update({"seeds": args.seeds, "first_seed": 1})
                study["learning"]["calibration"].update({"clerical_sample_size": max(n, 1), "clerical_seed": 20260508 + block,
                                                         "workers": args.workers})
                study["learning"]["partly_supervised"] = {"clerical_sample_size": max(n, 1), "seed": 20260508 + block,
                                                          "calibrate": RATES}
                modes = [{"label": "Calibrated", "profile": "calibrated_clerical" if n else "calibrated"}]
                if n:
                    modes.append({"label": "Clerical sample alone", "profile": "partly_supervised"})
                if n == sizes[0]:
                    modes += [{"label": "Pseudo-truth", "profile": "pseudo_truth"}, {"label": "Supervised", "profile": "supervised"}]
                study["modes"] = [{**m, "gate": "live", "legacy_gate": None} for m in modes]
                run_dir = out / "runs" / f"{name}_b{block}_n{n}"
                res = run_scenario(cfg, study, ROOT / "data/raw", run_dir, workers=args.workers, log=lambda *_: None)
                profiles = {p.stem: json.loads(p.read_text()) for p in (run_dir / "scenarios" / name / "profiles").glob("*.json")}
                truth = profiles["supervised"]
                link = res["linkage"][res["linkage"]["group"] == "Overall"]
                obs_link = link[link["mode"] == OBSERVED].iloc[0]
                ver = res["generator_verdicts"]
                for m in modes:
                    prof = profiles[m["profile"]]
                    rep = link[link["mode"] == m["label"]][LINK].mean()
                    live = ver[(ver["mode"] == m["label"]) & (ver["gate_policy"] == "live")].iloc[0]
                    rows.append({
                        "scenario": name, "block": block, "clerical_n": n if m["label"] not in {"Pseudo-truth", "Supervised"} else 0,
                        "method": m["label"],
                        **{f"{r}": prof[r] for r in RATES}, **{f"true_{r}": truth[r] for r in RATES},
                        **{f"abs_err_{r}": abs(prof[r] - truth[r]) for r in RATES},
                        "max_abs_rate_error": max(abs(prof[r] - truth[r]) for r in RATES),
                        **{f"replica_{k}": rep[k] for k in LINK}, **{f"observed_{k}": obs_link[k] for k in LINK},
                        **{f"err_{k}": rep[k] - obs_link[k] for k in LINK},
                        "live_gate": "accepted" if live["passed"] else "rejected",
                    })
                print(f"[{name}] block {block} n={n} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    existing = out / "sweep_block_level.csv"
    if existing.exists():
        old = pd.read_csv(existing)
        df = pd.concat([old[~old["scenario"].isin(scenarios)], df], ignore_index=True)
    df.to_csv(out / "sweep_block_level.csv", index=False)
    df["accepted"] = df["live_gate"].eq("accepted")
    agg = df.groupby(["scenario", "method", "clerical_n"], sort=False).agg(
        n_blocks=("block", "nunique"),
        median_max_abs_rate_error=("max_abs_rate_error", "median"),
        **{f"median_abs_err_{r}": (f"abs_err_{r}", "median") for r in RATES},
        median_abs_err_missed_match=("err_missed_match_rate", lambda s: s.abs().median()),
        median_abs_err_precision=("err_precision", lambda s: s.abs().median()),
        live_gate_acceptance=("accepted", "mean"),
    ).reset_index()
    agg.to_csv(out / "sweep_summary.csv", index=False)
    pd.set_option("display.width", 260)
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
