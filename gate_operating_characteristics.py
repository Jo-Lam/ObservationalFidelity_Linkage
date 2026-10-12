"""M4: operating characteristics of the generator-level OF gate (validate the validator).

Each block redraws the observed task (new corruption seed for the true mechanism),
relearns profiles and generates R replicas for:

* the CORRECT generator: identity synthesis (observed names) + supervised
  profile. Its rejection rate is the gate's size (false-alarm rate);
* WRONG mechanisms (identity synthesis): pseudo-truth, no full surname
  replacement, independent field errors, forename +5 pts, surname -5 pts. Their
  rejection rates are the gate's power;
* WRONG populations (supervised profile): the paper's sampler (frequency
  distortion) and the subgroup swap (loses joint forename-surname structure).

Verdicts are recomputed from the raw statistics under every policy and every
gate setting (noise estimate x multiplicity correction), so settings are
compared on identical replicas.

Usage: python scripts/gate_operating_characteristics.py [--blocks 10] [--workers 4]
"""
from __future__ import annotations

import argparse
import copy
import itertools
import time
from pathlib import Path

import pandas as pd
import yaml

from ofeval.experiment import run_scenario
from ofeval.gate import GateSettings, load_generator_policies

ROOT = Path(__file__).resolve().parents[1]
CORRECT = "Correct (identity + supervised)"
MECHANISM_GENERATORS = [
    {"label": CORRECT, "profile": "supervised"},
    {"label": "Pseudo-truth", "profile": "pseudo_truth"},
    {"label": "No full replacement", "profile": "supervised", "perturb": {"set": {"full_surname_replacement": 0.0}}},
    {"label": "Independent field errors", "profile": "supervised", "perturb": {"independent_fields": True}},
    {"label": "Forename +5 pts", "profile": "supervised", "perturb": {"add": {"forename_error_rate": 0.05}}},
    {"label": "Surname -5 pts", "profile": "supervised", "perturb": {"add": {"surname_error_rate": -0.05}}},
]
POPULATION_GENERATORS = {
    "Paper sampler (frequency distortion)": {"method": "independent"},
    "Subgroup swap (loses joint structure)": {"method": "stratified_swap", "strata": [], "min_stratum_size": 2},
}
SETTINGS = {f"{n}/{m}": GateSettings(alpha=0.05, noise=n, multiplicity=m)
            for n, m in itertools.product(["empirical", "half_normal"], ["none", "bonferroni"])}


def _modes(generators: list[dict]) -> list[dict]:
    return [{**g, "gate": "benchmark", "legacy_gate": None} for g in generators]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    base = yaml.safe_load((ROOT / "configs/study.yaml").read_text())
    base["learning"].pop("partly_supervised", None)
    policies = load_generator_policies(base["gate"])
    out = ROOT / "outputs/gate_operating_characteristics"
    rows, diag_rows = [], []
    t0 = time.time()
    for name in ["NCVR", "PDS"]:
        scn = yaml.safe_load((ROOT / f"configs/scenarios/{name.lower()}.yaml").read_text())
        for block in range(1, args.blocks + 1):
            cfg = copy.deepcopy(scn)
            cfg["observed_corruption"]["seed"] = scn["observed_corruption"]["seed"] + 104729 * block
            runs = [({"method": "identity"}, _modes(MECHANISM_GENERATORS))]
            runs += [(synth, _modes([{"label": label, "profile": "supervised"}]))
                     for label, synth in POPULATION_GENERATORS.items()]
            for synth, modes in runs:
                study = copy.deepcopy(base)
                study["synthesis"], study["modes"] = synth, modes
                study.pop("synthesis_by_scenario", None)  # each run tests exactly this synthesiser
                study["evaluation"]["first_seed"] = 1 + 20 * (block - 1)
                res = run_scenario(cfg, study, ROOT / "data/raw", out / "runs" / f"{name}_{block}",
                                   workers=args.workers, log=lambda *_: None, return_stats=True)
                observed, replicas = res["stats"]["observed"], res["stats"]["replicas"]
                for mode in modes:
                    key = f"{mode['profile']}~{mode['label']}" if mode.get("perturb") else mode["profile"]
                    for (sname, settings), (pname, policy) in itertools.product(SETTINGS.items(), policies.items()):
                        rep = policy.assess(observed, replicas[key], settings)
                        rows.append({"scenario": name, "block": block, "generator": mode["label"],
                                     "correct": mode["label"] == CORRECT, "setting": sname, "policy": pname,
                                     "rejected": not rep.passed, "failed": ";".join(rep.failed)})
                        if sname == "half_normal/bonferroni":
                            diag_rows.extend({"scenario": name, "block": block, "generator": mode["label"],
                                              "policy": pname, "diagnostic": d["diagnostic"],
                                              "failed": not d["passed"], "effect": d["effect"], "noise": d["noise"]}
                                             for d in rep.diagnostics.to_dict("records")
                                             if d["classification"] != "not_applicable")
            print(f"[{name}] block {block}/{args.blocks} done ({time.time() - t0:.0f}s)", flush=True)

    df, dd = pd.DataFrame(rows), pd.DataFrame(diag_rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "verdicts_block_level.csv", index=False)
    dd.to_csv(out / "diagnostics_block_level.csv", index=False)
    rates = df.groupby(["scenario", "generator", "policy", "setting"], sort=False)["rejected"].mean().unstack("setting")
    rates.to_csv(out / "rejection_rates.csv")
    per_diag = dd.groupby(["scenario", "generator", "policy", "diagnostic"], sort=False)["failed"].mean().unstack("generator")
    per_diag.to_csv(out / "rejection_rates_by_diagnostic.csv")
    pd.set_option("display.width", 250)
    print("\nRejection rate (correct generator = size; others = power):")
    print(rates.round(2).to_string())


if __name__ == "__main__":
    main()
