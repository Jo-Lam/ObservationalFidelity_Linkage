"""Run one scenario end to end: observed task, error-profile learning, replicas, OF gate, linkage evaluation."""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from . import diagnostics
from .gate import GateSettings, ReplicaGateReport, load_generator_policies, load_replica_policies
from .linkage import link, linkage_metrics, probability_band_mass, select_threshold_for_precision
from .calibration import CLERICAL_RATES, MOMENT_NAMES as CAL_MOMENTS, ReplicaSimulator, calibrate
from .calibration import moments as calibration_moments
from .profiles import clerical_sample, learn_partly_supervised, learn_pseudo_truth, learn_supervised, perturb_profile
from .provenance import provenance_audit
from .scenarios import build_observed_task
from .synthesis import generate_replica

OBSERVED = "Observed corrupted"
MECHANISM = diagnostics.names("mechanism", observable=False)  # truth-based rates


def profile_key(mode: dict[str, Any]) -> str:
    """Replicas are shared by modes with the same key; a perturbed mode gets its own."""
    return f"{mode['profile']}~{mode['label']}" if mode.get("perturb") else mode["profile"]


class ScenarioContext:
    """Observed task plus everything replicas are compared against."""

    def __init__(self, cfg: dict[str, Any], study: dict[str, Any], data_dir: Path) -> None:
        self.cfg = cfg
        self.study = study
        self.name = cfg["name"]
        self.subgroup = (cfg.get("subgroup") or {}).get("name")
        self.reference_group = (cfg.get("subgroup") or {}).get("reference_group")
        # Per-scenario synthesis override (study.yaml: synthesis_by_scenario), else the study default.
        self.synthesis = (study.get("synthesis_by_scenario") or {}).get(self.name, study.get("synthesis"))
        # Spine/incoming design (scenario `incoming`), per-record unit and recall denominator (study `evaluation`).
        self.incoming = cfg.get("incoming")
        self.record_side = study["evaluation"].get("record_side", "reference")
        self.recall_denominator = study["evaluation"].get("recall_denominator", "candidate_reference")
        if self.recall_denominator not in {"candidate_reference", "all_true_matches"}:
            raise ValueError(f"Unknown recall_denominator {self.recall_denominator!r}")
        self.reference, self.comparison = build_observed_task(cfg, data_dir)
        self.scored = link(self.reference, self.comparison, cfg["linkage"], self.subgroup)
        recall_all = study["evaluation"].get("recall_denominator") == "all_true_matches"
        rule = study["evaluation"].get("threshold_rule", "cumulative")
        if rule not in {"cumulative", "tie_aware"}:
            raise ValueError(f"Unknown threshold_rule {rule!r}")
        self.threshold = select_threshold_for_precision(self.scored, study["evaluation"]["target_precision"],
                                                        {"Overall": len(self.comparison)} if recall_all else None,
                                                        tie_aware=rule == "tie_aware")
        self.stats = diagnostics.compute(self.bundle(self.reference, self.comparison, self.scored))
        self.replica_policies = load_replica_policies(study["legacy_gate_policies"])
        self.generator_policies = load_generator_policies(study["gate"])
        self.profiles = self._learn_profiles()

    def _learn_profiles(self) -> dict[str, dict[str, Any]]:
        learning = self.study["learning"]
        floor = learning["pseudo_truth_floor"]
        profiles = {
            "supervised": learn_supervised(self.reference, self.comparison),
            "pseudo_truth": learn_pseudo_truth(self.scored, self.reference, self.comparison, floor),
        }
        if any(m["profile"] == "partly_supervised" for m in self.study["modes"]):
            clerical = learning["partly_supervised"]
            profiles["partly_supervised"] = learn_partly_supervised(
                profiles["pseudo_truth"], self.reference, self.comparison,
                sample_size=clerical["clerical_sample_size"], seed=clerical["seed"], calibrate=clerical["calibrate"])
        used = {m["profile"] for m in self.study["modes"]}
        for key in ["calibrated", "calibrated_clerical"]:
            if key in used:
                profiles[key] = self._calibrate(profiles["pseudo_truth"], clerical=key == "calibrated_clerical")
        for mode in self.study["modes"]:
            if mode.get("perturb"):
                profiles[profile_key(mode)] = perturb_profile(profiles[mode["profile"]], mode["perturb"])
        return profiles

    def _calibrate(self, pseudo_profile: dict[str, Any], clerical: bool) -> dict[str, Any]:
        """Indirect-inference calibration (src/ofeval/calibration.py), optionally with a clerical sample."""
        c = self.study["learning"]["calibration"]
        sample = (clerical_sample(self.reference, self.comparison, int(c["clerical_sample_size"]), int(c["clerical_seed"]))
                  if clerical else None)
        start = {**pseudo_profile, **{k: sample[k] for k in CLERICAL_RATES}} if sample else pseudo_profile
        observed = calibration_moments(self.bundle(self.reference, self.comparison, self.scored))
        seeds = list(range(int(c["first_seed"]), int(c["first_seed"]) + int(c["seeds"])))
        with ReplicaSimulator(self.reference, self.cfg["linkage"], self.subgroup, self.threshold["threshold"],
                              self.study["learning"]["pseudo_truth_floor"], self.synthesis,
                              seeds, workers=int(c.get("workers", 1)), incoming=self.incoming,
                              record_side=self.record_side) as sim:
            res = calibrate(sim, observed, start, clerical=sample, max_iter=int(c.get("max_iter", 12)))
        profile = dict(res.profile)
        profile["learning"] = {
            "mode": "calibrated_clerical" if clerical else "calibrated",
            "method": "indirect inference: label-free moments" + (" + clerical sample" if clerical else ""),
            "calibration_seeds": seeds, "converged": res.converged,
            "objective_start": res.start_objective, "objective": res.objective,
            "moments": [{"moment": m, "observed": float(o), "fitted": float(f), "sd": float(s)}
                        for m, o, f, s in zip(CAL_MOMENTS, res.observed_moments, res.fitted_moments, res.moment_sd)],
            "trace": res.trace,
            "clerical_sample": ({k: sample[k] for k in ["n"] + CLERICAL_RATES} if sample else None),
        }
        return profile

    def bundle(self, reference: pd.DataFrame, comparison: pd.DataFrame, scored: pd.DataFrame) -> diagnostics.TaskBundle:
        """Every task (observed or replica) is described with the same context: subgroup,
        the threshold fixed on the observed task, and the pseudo-truth floor."""
        return diagnostics.TaskBundle(reference, comparison, scored, subgroup=self.subgroup,
                                      threshold=self.threshold["threshold"],
                                      pseudo_truth_floor=self.study["learning"]["pseudo_truth_floor"],
                                      record_side=self.record_side)

    def task_metrics(self, scored: pd.DataFrame, comparison: pd.DataFrame) -> dict[str, Any]:
        """Linkage quality at the fixed observed-task threshold, overall and by subgroup.

        With ``recall_denominator: all_true_matches`` recall is over every incoming
        record (each has one true match), by its subgroup.
        """
        group_col = "group_l" if self.subgroup else None
        true_links = None
        if self.recall_denominator == "all_true_matches":
            true_links = {"Overall": int(len(comparison))}
            if self.subgroup and self.subgroup in comparison.columns:
                true_links.update({str(g): int(n) for g, n in comparison[self.subgroup].value_counts().items()})
        lm = linkage_metrics(scored, self.threshold["threshold"], group_col, self.reference_group, true_links)
        return {"linkage": lm, "band_mass": probability_band_mass(scored)}


_CTX: ScenarioContext | None = None


def _init_worker(ctx: ScenarioContext) -> None:
    global _CTX
    _CTX = ctx


def _run_replica(profile_key: str, seed: int) -> dict[str, Any]:
    """Generate, link and measure one replica. Runs in a worker process."""
    ctx = _CTX
    assert ctx is not None
    t0 = time.time()
    ref, cmp, synthesis_info = generate_replica(ctx.reference, ctx.profiles[profile_key], seed, ctx.subgroup,
                                                ctx.synthesis, ctx.incoming)
    scored = link(ref, cmp, ctx.cfg["linkage"], ctx.subgroup)
    stats = diagnostics.compute(ctx.bundle(ref, cmp, scored))
    comparison = diagnostics.compare(ctx.stats, stats)
    metrics = ctx.task_metrics(scored, cmp)
    return {
        "profile": profile_key, "seed": seed, "stats": stats, "scalars": diagnostics.scalars(stats), "comparison": comparison,
        "synthesis_info": synthesis_info,
        "of_distance": diagnostics.of_distance(comparison), "linkage": metrics["linkage"],
        "band_mass": metrics["band_mass"], "provenance": provenance_audit(ctx.reference, ctx.comparison, cmp),
        "seconds": time.time() - t0,
    }


def _floor_sensitivity(ctx: ScenarioContext) -> pd.DataFrame:
    rows = [{"scenario": ctx.name, "pseudo_truth_floor": "Observed reference", "floor_value": None,
             **{m: ctx.stats[m] for m in MECHANISM}, "n_pairs_used": None}]
    for floor in ctx.study["learning"]["sensitivity_floors"]:
        prof = learn_pseudo_truth(ctx.scored, ctx.reference, ctx.comparison, floor)
        rows.append({"scenario": ctx.name, "pseudo_truth_floor": f"p >= {floor:.2f}", "floor_value": floor,
                     **{m: prof[m] for m in MECHANISM}, "n_pairs_used": prof["n_pairs_used"],
                     "surname_replacement_plus_other_error": prof["surname_replacement_plus_other_error"]})
    return pd.DataFrame(rows)


def run_scenario(cfg: dict[str, Any], study: dict[str, Any], data_dir: Path, out_dir: Path,
                 workers: int = 1, log=print, return_stats: bool = False) -> dict[str, Any]:
    t0 = time.time()
    ctx = ScenarioContext(cfg, study, data_dir)
    scenario_dir = out_dir / "scenarios" / ctx.name
    (scenario_dir / "profiles").mkdir(parents=True, exist_ok=True)
    for key, profile in ctx.profiles.items():
        (scenario_dir / "profiles" / f"{key}.json").write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
    (scenario_dir / "observed_threshold.json").write_text(json.dumps(ctx.threshold, indent=2), encoding="utf-8")
    (scenario_dir / "synthesis.json").write_text(json.dumps(ctx.synthesis, indent=2), encoding="utf-8")
    # Unreachable target: no threshold meets it, so selection falls back to the highest score.
    # (Ties at a reachable threshold can leave achieved precision slightly below target; not flagged.)
    target = ctx.threshold["target_precision"]
    at_max = ctx.threshold["threshold"] >= float(ctx.scored["match_probability"].max()) - 1e-12
    if at_max and ctx.threshold["achieved_precision"] < target - 1e-9:
        log(f"[{ctx.name}] WARNING: target precision {target:.2f} is not reachable; the threshold falls back to the "
            f"highest score ({ctx.threshold['threshold']:.4f}), with precision {ctx.threshold['achieved_precision']:.3f}")
    log(f"[{ctx.name}] observed task: {len(ctx.reference):,} records, {len(ctx.scored):,} candidate pairs, "
        f"threshold={ctx.threshold['threshold']:.4f} ({time.time() - t0:.0f}s)")

    first = int(study["evaluation"].get("first_seed", 1))
    seeds = list(range(first, first + int(study["evaluation"]["seeds"])))
    jobs = sorted({(profile_key(m), s) for m in study["modes"] for s in seeds})
    replicas: dict[tuple[str, int], dict[str, Any]] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(ctx,)) as pool:
            futures = {pool.submit(_run_replica, p, s): (p, s) for p, s in jobs}
            for fut in futures:
                r = fut.result()
                replicas[(r["profile"], r["seed"])] = r
                log(f"[{ctx.name}] {r['profile']} seed {r['seed']:>2} done ({r['seconds']:.0f}s)")
    else:
        _init_worker(ctx)
        for p, s in jobs:
            r = _run_replica(p, s)
            replicas[(p, s)] = r
            log(f"[{ctx.name}] {p} seed {s:>2} done ({r['seconds']:.0f}s)")

    tables = _assemble(ctx, replicas, seeds)
    violations = tables["tasks"]["invariant_violations"].fillna("")
    broken = sorted({v for cell in violations if cell for v in cell.split(";")})
    if broken:
        log(f"[{ctx.name}] WARNING: synthesiser invariants violated in {int((violations != '').sum())} replicas: {broken}")
    tables["floor_sensitivity"] = _floor_sensitivity(ctx)
    tables["thresholds"] = pd.DataFrame([{"scenario": ctx.name, **ctx.threshold}])
    if return_stats:  # raw statistics, to re-evaluate gate settings without regenerating replicas
        tables["stats"] = {"observed": ctx.stats,
                           "replicas": {profile_key(m): [replicas[(profile_key(m), s)]["stats"] for s in seeds]
                                        for m in study["modes"]}}
    log(f"[{ctx.name}] finished in {time.time() - t0:.0f}s")
    return tables


def _assemble(ctx: ScenarioContext, replicas: dict[tuple[str, int], dict[str, Any]], seeds: list[int]) -> dict[str, pd.DataFrame]:
    """Long tables.

    Per-replica rows carry ``legacy_accepted`` (the paper's per-replica gate for
    the mode's ``legacy_gate``). Generator verdicts are computed once per
    (profile, policy) from all replicas and attached to every mode using that profile.
    """
    key = {"scenario": ctx.name}
    obs_link = ctx.task_metrics(ctx.scored, ctx.comparison)
    obs = {**key, "mode": OBSERVED, "seed": 0, "legacy_accepted": True}
    tasks = [{**obs, **diagnostics.scalars(ctx.stats)}]
    linkage = [{**obs, **r} for r in obs_link["linkage"].to_dict("records")]
    bands = [{**obs, "band": b, "mass": float(v)} for b, v in obs_link["band_mass"].items()]
    diags, decisions, layers, provenance = [], [], [], []
    gen_diags, gen_verdicts = [], []
    settings = GateSettings.from_config(ctx.study["gate"])

    for mode in ctx.study["modes"]:
        label, gate, legacy_gate = mode["label"], mode.get("gate"), mode.get("legacy_gate")
        mode_key = {**key, "mode": label}
        replica_stats = [replicas[(profile_key(mode), s)]["stats"] for s in seeds]
        for name, policy in ctx.generator_policies.items():
            report = policy.assess(ctx.stats, replica_stats, settings)
            flags = {**mode_key, "gate_policy": name, "is_mode_gate": name == gate}
            gen_verdicts.append({**flags, "n_replicas": len(seeds), **report.summary()})
            gen_diags.extend({**flags, **d} for d in report.diagnostics.to_dict("records"))

        for seed in seeds:
            r = replicas[(profile_key(mode), seed)]
            report = ReplicaGateReport.build(r["comparison"], ctx.replica_policies)
            accepted = report.passed(legacy_gate)
            row = {**mode_key, "seed": seed}
            for name, (passed, failed) in report.verdicts.items():
                decisions.append({**row, "gate_policy": name, "is_mode_gate": name == legacy_gate,
                                  "passed": passed, "failed_metrics": ";".join(failed)})
                layers.extend({**row, "gate_policy": name, "is_mode_gate": name == legacy_gate, **lv}
                              for lv in report.layer_verdicts[name])
            diags.extend({**row, **d} for d in report.comparison.to_dict("records"))
            tasks.append({**row, "legacy_accepted": accepted, **r["scalars"], **r["of_distance"],
                          "invariant_violations": ";".join(report.invariant_violations),
                          **{f"synthesis_{k}": v for k, v in r["synthesis_info"].items()}})
            linkage.extend({**row, "legacy_accepted": accepted, **x} for x in r["linkage"].to_dict("records"))
            bands.extend({**row, "legacy_accepted": accepted, "band": b, "mass": float(v)} for b, v in r["band_mass"].items())
            provenance.append({**row, "legacy_accepted": accepted, **r["provenance"]})

    return {
        "tasks": pd.DataFrame(tasks),
        "linkage": pd.DataFrame(linkage),
        "band_mass": pd.DataFrame(bands),
        "diagnostics": pd.DataFrame(diags),
        "generator_gate": pd.DataFrame(gen_diags),
        "generator_verdicts": pd.DataFrame(gen_verdicts),
        "legacy_gate_decisions": pd.DataFrame(decisions),
        "legacy_gate_layers": pd.DataFrame(layers),
        "provenance": pd.DataFrame(provenance),
    }
