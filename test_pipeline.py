from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ofeval import diagnostics
from ofeval.gate import (GeneratorPolicy, ReplicaGatePolicy, ReplicaGateReport, assess_diagnostic,
                         load_generator_policies, load_replica_policies)
from ofeval.experiment import run_scenario
from ofeval.names import add_name_features, assign_error_flags, corrupt_string
from ofeval.profiles import learn_partly_supervised, learn_pseudo_truth, learn_supervised
from ofeval.provenance import provenance_audit
from ofeval.scenarios import build_observed_task
from ofeval.synthesis import _cardinality_preserving_sample, generate_replica, stratified_swap_reference
from ofeval.linkage import link

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A 600-record synthetic source file and matching scenario/study configs."""
    rng = np.random.default_rng(0)
    forenames = [f"{a}{b}" for a in ["ANNA", "JOHN", "MARIA", "DAVID", "SARAH", "AHMED", "WEI", "OLU"] for b in ["", "N", "E", "LY"]]
    surnames = [f"{a}{b}" for a in ["SMITH", "JONES", "PATEL", "NGUYEN", "OKAFOR", "BROWN", "KHAN", "GARCIA", "TAYLOR"] for b in ["", "SON", "ER"]]
    df = pd.DataFrame({
        "id": np.arange(600),
        "first": rng.choice(forenames, 600),
        "last": rng.choice(surnames, 600),
        "eth": rng.choice(["w", "b", "a"], 600, p=[0.6, 0.25, 0.15]),
    })
    data = tmp_path_factory.mktemp("data")
    df.to_csv(data / "source.csv", index=False)
    scenario = {
        "name": "TINY", "source": {"file": "source.csv"}, "max_rows": 600,
        "columns": {"id": "id", "forename": "first", "surname": "last"},
        "subgroup": {"name": "ethnic_group", "column": "eth", "coding": "ncvr_collapsed", "reference_group": "Non-Hispanic White"},
        "observed_corruption": {"seed": 1, "forename_error_rate": 0.2, "surname_error_rate": 0.2,
                                "multi_field_error_rate": 0.08, "full_surname_replacement_rate": 0.1},
        "linkage": yaml.safe_load((ROOT / "configs/scenarios/ncvr.yaml").read_text())["linkage"],
    }
    study = yaml.safe_load((ROOT / "configs/study.yaml").read_text())
    study["evaluation"]["seeds"] = 2
    study["learning"]["partly_supervised"]["clerical_sample_size"] = 100
    study["learning"]["calibration"].update({"seeds": 2, "workers": 1, "max_iter": 2, "clerical_sample_size": 100})
    return scenario, study, data


def test_error_flags_hit_requested_rates():
    fe, se, fsr = assign_error_flags(200_000, 0.2, 0.25, 0.08, 0.1, seed=3)
    assert fe.mean() == pytest.approx(0.2, abs=0.005)
    assert se.mean() == pytest.approx(0.25, abs=0.005)
    assert (fe & se).mean() == pytest.approx(0.08, abs=0.005)
    assert fsr.mean() == pytest.approx(0.1, abs=0.005)
    assert not (fsr & ~se).any()


def test_corrupt_string_always_changes_a_nonempty_name():
    rng = random.Random(0)
    for name in ["A", "AA", "ANNA", "O'NEIL", "SMITH-JONES"] * 50:
        assert corrupt_string(name, rng) != name


def test_name_sampling_is_deterministic_and_keeps_cardinality():
    values = pd.Series(["B", "A", "C", "A", "B"] * 40 + [f"N{i}" for i in range(300)])
    a = _cardinality_preserving_sample(values, 1000, np.random.default_rng(5))
    b = _cardinality_preserving_sample(values, 1000, np.random.default_rng(5))
    assert (a == b).all()
    assert set(a) == set(values)


def test_policy_validation_against_registry():
    with pytest.raises(ValueError, match="truth-based"):
        ReplicaGatePolicy.from_config("bad", {"label_free": True, "checks": {"surname_error_rate": 0.05}})
    with pytest.raises(ValueError, match="unknown"):
        ReplicaGatePolicy.from_config("bad", {"checks": {"not_a_diagnostic": 0.05}})
    with pytest.raises(ValueError, match="by construction"):
        ReplicaGatePolicy.from_config("bad", {"checks": {"unique_surname_count": 0.05}})
    study = yaml.safe_load((ROOT / "configs/study.yaml").read_text())
    load_replica_policies(study["legacy_gate_policies"])
    load_generator_policies(study["gate"])
    with pytest.raises(ValueError, match="truth-based"):
        GeneratorPolicy.from_config("bad", {"label_free": True, "margins": {"multi_field_error": 0.02}})


def _comparison(distances: dict[str, float]) -> pd.DataFrame:
    rows = []
    for name, d in diagnostics.REGISTRY.items():
        rows.append({"diagnostic": name, "layer": d.layer, "observable": d.observable,
                     "by_construction": d.by_construction, "distance": distances.get(name, 0.0)})
    return pd.DataFrame(rows)


def test_policy_evaluation_and_layer_verdicts():
    policy = ReplicaGatePolicy("p", {"top_surname_share": 0.05, "match_probability_band_mass_tv": 0.05}, label_free=True)
    assert policy.evaluate(_comparison({"top_surname_share": -0.04})) == (True, [])
    assert policy.evaluate(_comparison({"match_probability_band_mass_tv": 0.06})) == (False, ["match_probability_band_mass_tv"])
    layers = {r["layer"]: r["verdict"] for r in policy.layer_verdicts(_comparison({"match_probability_band_mass_tv": 0.06}))}
    assert layers == {"population": "pass", "mechanism": "not_assessed", "landscape": "fail"}


def test_invariants_are_reported_not_gated():
    report = ReplicaGateReport.build(_comparison({"unique_forename_count": 0.01}),
                              {"p": ReplicaGatePolicy("p", {"top_surname_share": 0.05}, label_free=True)})
    assert report.invariant_violations == ["unique_forename_count"]
    assert report.passed("p") and report.passed(None)


def test_registry_statistics_compare_identical_tasks_at_zero(tiny):
    scenario, _, data = tiny
    ref, cmp = build_observed_task(scenario, data)
    scored = link(ref, cmp, scenario["linkage"], "ethnic_group")
    full = diagnostics.TaskBundle(ref, cmp, scored, subgroup="ethnic_group", threshold=0.8, pseudo_truth_floor=0.7)
    stats = diagnostics.compute(full)
    cmp_df = diagnostics.compare(stats, stats)
    assert set(cmp_df["diagnostic"]) == set(diagnostics.REGISTRY)
    assert (cmp_df["distance"].abs() < 1e-12).all()
    # without context, context-dependent diagnostics are not applicable (NaN), never zero or a failure
    bare = diagnostics.compare(stats, diagnostics.compute(diagnostics.TaskBundle(ref, cmp, scored)))
    na = set(bare.loc[bare["distance"].isna(), "diagnostic"])
    assert na == {"learned_error_exposure", "learned_forename_error_rate", "learned_surname_error_rate",
                  "learned_multi_field_error", "ambiguity_rate", "subgroup_ambiguity_rate", "subgroup_best_match_band_tv",
                  "subgroup_full_name_duplication_rate"}
    assert diagnostics.names("mechanism", observable=False) == ["error_exposure", "forename_error_rate", "surname_error_rate",
                                              "full_surname_replacement", "multi_field_error"]


def test_provenance_rare_singletons_counted_over_records():
    obs = add_name_features(pd.DataFrame({"forename_norm": ["A", "A", "B"], "surname_norm": ["X", "X", "Y"]}))
    rep = add_name_features(pd.DataFrame({"forename_norm": ["A", "B", "C"], "surname_norm": ["X", "Y", "Z"]}))
    out = provenance_audit(obs, obs, rep)
    assert out["full_name_overlap"] == pytest.approx(2 / 3)
    assert out["rare_singleton_overlap"] == pytest.approx(1 / 3)  # only (B, Y) is a singleton


def test_supervised_profile_recovers_injected_errors(tiny):
    scenario, _, data = tiny
    ref, cmp = build_observed_task(scenario, data)
    prof = learn_supervised(ref, cmp)
    assert prof["forename_error_rate"] == pytest.approx(0.2, abs=0.05)
    assert prof["surname_error_rate"] == pytest.approx(0.2, abs=0.05)
    assert prof["full_surname_replacement"] == pytest.approx(0.1, abs=0.04)


def test_pseudo_truth_misses_link_breaking_errors_and_clerical_sample_restores_them(tiny):
    scenario, study, data = tiny
    ref, cmp = build_observed_task(scenario, data)
    scored = link(ref, cmp, scenario["linkage"], "ethnic_group")
    pseudo = learn_pseudo_truth(scored, ref, cmp, floor=0.7)
    assert pseudo["full_surname_replacement"] < 0.02
    partly = learn_partly_supervised(pseudo, ref, cmp, sample_size=300, seed=1,
                                     calibrate=study["learning"]["partly_supervised"]["calibrate"])
    assert partly["full_surname_replacement"] > 0.05
    assert partly["forename_error_rate"] == pseudo["forename_error_rate"]


def test_end_to_end_run(tiny, tmp_path):
    scenario, study, data = tiny
    tables = run_scenario(scenario, study, data, tmp_path)
    modes = {m["label"] for m in study["modes"]}
    assert set(tables["tasks"]["mode"]) == modes | {"Observed corrupted"}
    assert len(tables["tasks"]) == 1 + 2 * len(modes)
    # legacy: ungated modes keep every replica; decisions exist for every legacy policy
    assert tables["tasks"].loc[tables["tasks"]["mode"] == "Pseudo-truth only", "legacy_accepted"].all()
    assert len(tables["legacy_gate_decisions"]) == 2 * len(modes) * len(study["legacy_gate_policies"])
    assert len(tables["legacy_gate_layers"]) == 3 * len(tables["legacy_gate_decisions"])
    # generator gate: one verdict per (mode, policy); modes sharing a profile get identical verdicts
    ver = tables["generator_verdicts"]
    assert len(ver) == len(modes) * len(study["gate"]["policies"])
    pt = ver[ver["mode"].str.startswith("Pseudo-truth")].groupby("gate_policy")["passed"].nunique()
    assert (pt == 1).all()
    assert (tables["tasks"]["invariant_violations"].fillna("") == "").all()
    assert (tmp_path / "scenarios" / "TINY" / "profiles" / "partly_supervised.json").exists()


def test_generator_gate_noise_never_fails_but_material_bias_does():
    rng = np.random.default_rng(0)
    # Observed task and replicas exchangeable (same generator): consistent, never a failure.
    draws = list(rng.normal(0.20, 0.01, 21))
    ok = assess_diagnostic("surname_error_rate", draws[0], draws[1:], margin=0.001, alpha=0.05)
    assert ok["passed"] and ok["noise"] > ok["margin"]
    # A detectable bias that stays within the margin is "equivalent".
    reps = list(rng.normal(0.21, 0.001, 20))
    eq = assess_diagnostic("surname_error_rate", 0.20, reps, margin=0.02, alpha=0.05)
    assert eq["classification"] == "equivalent" and eq["passed"] and eq["p_value"] < 0.1
    # A bias beyond both the margin and the noise fails.
    bad = assess_diagnostic("surname_error_rate", 0.20, list(rng.normal(0.08, 0.005, 20)), margin=0.02, alpha=0.05)
    assert bad["classification"] == "not_equivalent" and not bad["passed"]
    assert bad["signed_effect"] == pytest.approx(-0.12, abs=0.01)


def test_generator_gate_distribution_diagnostics_use_centroid():
    bands = ["a", "b"]
    obs = pd.Series([0.5, 0.5], index=bands)
    reps = [pd.Series([0.6, 0.4], index=bands), pd.Series([0.62, 0.38], index=bands), pd.Series([0.58, 0.42], index=bands)]
    r = assess_diagnostic("match_probability_band_mass_tv", obs, reps, margin=0.02, alpha=0.05)
    assert r["effect"] == pytest.approx(0.10)  # TV between observed and the replica centroid
    assert not r["passed"]
    policy = GeneratorPolicy("live", {"match_probability_band_mass_tv": 0.02}, label_free=True)
    report = policy.assess({"match_probability_band_mass_tv": obs}, [{"match_probability_band_mass_tv": x} for x in reps], 0.05)
    assert not report.passed and report.failed == ["match_probability_band_mass_tv"]
    assert {lv["layer"]: lv["verdict"] for lv in report.layers}["mechanism"] == "not_assessed"


@pytest.mark.parametrize("config", ["paper.yaml", "study.yaml"])
def test_configs_load_and_validate(config):
    from ofeval.cli import load_study
    study = load_study(ROOT / "configs" / config)
    assert study["synthesis"]["method"] in {"independent", "stratified_swap"}


def test_stratified_swap_keeps_marginals_and_strata_breaks_pairs(tiny):
    scenario, _, data = tiny
    ref, _ = build_observed_task(scenario, data)
    strata = [["forename_initial", "surname_initial"]]
    clean, info = stratified_swap_reference(ref, seed=3, subgroup_col="ethnic_group", strata=strata, min_stratum_size=5)
    src = ref.reset_index(drop=True)
    # exact marginals and subgroup composition: records keep forename and subgroup, surnames are permuted
    assert (clean["forename_norm"].to_numpy() == src["forename_norm"].to_numpy()).all()
    assert (clean["ethnic_group"].to_numpy() == src["ethnic_group"].to_numpy()).all()
    assert sorted(clean["surname_norm"]) == sorted(src["surname_norm"])
    # most records receive another record's surname
    assert info["share_kept_own_surname"] < 0.5
    # surnames move only within the stratum: (subgroup, surname initial) counts are unchanged
    key = lambda d: (d["ethnic_group"] + "|" + d["surname_norm"].str[:1]).value_counts().sort_index()
    assert key(clean).equals(key(src))


def test_stratified_swap_can_keep_full_name_clusters():
    rng = np.random.default_rng(1)
    n = 3000
    src = pd.DataFrame({"forename_norm": rng.choice([f"F{i}" for i in range(60)], n),
                        "surname_norm": rng.choice([f"S{i}" for i in range(2000)], n),
                        "ethnic_group": rng.choice(["w", "b"], n, p=[0.7, 0.3])})
    src.loc[:59, ["forename_norm", "surname_norm"]] = ["F0", "S0"]  # one large cluster
    for k in range(20):  # and some clusters of three
        src.loc[60 + 3 * k: 62 + 3 * k, ["forename_norm", "surname_norm"]] = [f"F{k}", f"S{100 + k}"]
    clean, _ = stratified_swap_reference(src, seed=3, subgroup_col="ethnic_group", strata=[], min_stratum_size=2,
                                         preserve_name_clusters=True)
    # marginals as before, within subgroup
    assert (clean["forename_norm"].to_numpy() == src["forename_norm"].to_numpy()).all()
    key = lambda d: (d["ethnic_group"] + "|" + d["surname_norm"]).value_counts().sort_index()
    assert key(clean).equals(key(src))
    # the full-name cluster-size distribution is kept
    full = lambda d: d["forename_norm"] + "|" + d["surname_norm"]
    sizes = lambda d: full(d).value_counts().value_counts().sort_index()
    assert sizes(clean).equals(sizes(src))
    # whereas a plain swap breaks the planted clusters up
    plain, _ = stratified_swap_reference(src, seed=3, subgroup_col="ethnic_group", strata=[], min_stratum_size=2)
    assert full(plain).value_counts().max() < 60
    # and pairings are still broken for most records
    assert (clean["surname_norm"].to_numpy() != src["surname_norm"].to_numpy()).mean() > 0.5


def test_generate_replica_uses_configured_synthesiser(tiny):
    scenario, study, data = tiny
    ref, cmp = build_observed_task(scenario, data)
    prof = learn_supervised(ref, cmp)
    clean, _, info = generate_replica(ref, prof, 1, "ethnic_group", study["synthesis"])
    assert sorted(clean["surname_norm"]) == sorted(ref["surname_norm"])
    assert "share_recreating_unique_observed_name" in info
    _, _, none = generate_replica(ref, prof, 1, "ethnic_group", {"method": "independent"})
    assert none == {}


def test_perturb_profile():
    from ofeval.profiles import perturb_profile
    base = {"forename_error_rate": 0.2, "surname_error_rate": 0.2, "multi_field_error": 0.08,
            "full_surname_replacement": 0.1, "error_exposure": 0.32}
    assert perturb_profile(base, {"set": {"full_surname_replacement": 0}})["full_surname_replacement"] == 0
    assert perturb_profile(base, {"independent_fields": True})["multi_field_error"] == pytest.approx(0.04)
    assert perturb_profile(base, {"add": {"forename_error_rate": 0.05}})["forename_error_rate"] == pytest.approx(0.25)
    halved = perturb_profile(base, {"scale": {"surname_error_rate": 0.25}})
    assert halved["full_surname_replacement"] == pytest.approx(0.05)  # capped at the surname rate
    assert base["multi_field_error"] == 0.08  # original untouched


def test_grouped_distances_take_worst_subgroup():
    amb = diagnostics.REGISTRY["subgroup_ambiguity_rate"]
    obs = pd.Series({"a": 0.05, "b": 0.08})
    assert amb.distance(obs, pd.Series({"a": 0.06, "b": 0.05})) == pytest.approx(0.03)
    bm = diagnostics.REGISTRY["subgroup_best_match_band_tv"]
    idx = pd.MultiIndex.from_product([["a", "b"], ["x", "y"]])
    o = pd.Series([0.5, 0.5, 0.5, 0.5], index=idx)
    r = pd.Series([0.5, 0.5, 0.9, 0.1], index=idx)
    assert bm.distance(o, r) == pytest.approx(0.4)
    assert np.isnan(amb.distance(None, obs))
    na = assess_diagnostic("subgroup_ambiguity_rate", None, [None, None], margin=0.01, alpha=0.05)
    assert na["classification"] == "not_applicable" and na["passed"]


def test_calibration_reduces_objective_and_keeps_profile_coherent(tiny):
    from ofeval.calibration import ReplicaSimulator, calibrate, clerical_moments, moments, profile_from_theta
    from ofeval.profiles import clerical_sample
    scenario, study, data = tiny
    ref, cmp = build_observed_task(scenario, data)
    scored = link(ref, cmp, scenario["linkage"], "ethnic_group")
    obs = moments(diagnostics.TaskBundle(ref, cmp, scored, subgroup="ethnic_group", threshold=0.8, pseudo_truth_floor=0.7))
    pt = learn_pseudo_truth(scored, ref, cmp, 0.7)
    sim = ReplicaSimulator(ref, scenario["linkage"], "ethnic_group", 0.8, 0.7, study["synthesis"], seeds=[1, 2])
    res = calibrate(sim, obs, pt, max_iter=3)
    assert res.objective <= res.start_objective
    p = res.profile
    assert p["multi_field_error"] <= min(p["forename_error_rate"], p["surname_error_rate"]) + 1e-12
    assert p["full_surname_replacement"] <= p["surname_error_rate"] + 1e-12
    # parameterisation is coherent everywhere, and clerical standard errors stay positive at zero rates
    q = profile_from_theta(np.array([0.3, 0.1, 1.0, 1.0]))
    assert q["multi_field_error"] == pytest.approx(0.1) and q["full_surname_replacement"] == pytest.approx(0.1)
    cler = clerical_sample(ref, cmp, 50, seed=1)
    _, se = clerical_moments({**cler, "full_surname_replacement": 0.0})
    assert (se > 0).all()


def test_synthesis_override_per_scenario(tiny, tmp_path):
    import copy
    import json
    scenario, study, data = tiny
    s = copy.deepcopy(study)
    s["modes"] = [m for m in s["modes"] if m["profile"] == "supervised"]
    s["synthesis_by_scenario"] = {"TINY": {"method": "identity"}}
    tables = run_scenario(scenario, s, data, tmp_path)
    assert json.loads((tmp_path / "scenarios" / "TINY" / "synthesis.json").read_text())["method"] == "identity"
    assert (tables["tasks"]["synthesis_share_recreating_unique_observed_name"].dropna() == 1.0).all()


def test_incoming_subset_design(tiny, tmp_path):
    import copy
    scenario, study, data = tiny
    scn = copy.deepcopy(scenario)
    scn["incoming"] = {"fraction": 0.3, "seed": 5}
    ref, cmp = build_observed_task(scn, data)
    assert len(cmp) == round(0.3 * len(ref))
    assert set(cmp["truth_id"]) <= set(ref["truth_id"])  # every incoming record has a true match in the spine
    s = copy.deepcopy(study)
    s["modes"] = [m for m in s["modes"] if m["profile"] == "supervised"]
    s["evaluation"].update({"recall_denominator": "all_true_matches", "record_side": "comparison"})
    tables = run_scenario(scn, s, data, tmp_path)
    overall = tables["linkage"][tables["linkage"]["group"] == "Overall"]
    assert (overall["true_links"] == len(cmp)).all()  # recall over all incoming records, observed and replicas
    groups = tables["linkage"][(tables["linkage"]["group"] != "Overall") & (tables["linkage"]["mode"] == "Observed corrupted")]
    assert groups["true_links"].sum() == len(cmp)


def test_exclude_records():
    from ofeval.scenarios import exclude_records
    ref = add_name_features(pd.DataFrame({
        "forename_norm": ["ANNA", "OSSIE", "JOHN", "JOHN", "JOHN", "MARY"],
        "surname_norm": ["SMITH", "IQZ-APR-FIFTY-A", "LEE", "LEE", "LEE", "VERYVERYLONG"]}))
    assert len(exclude_records(ref, None)) == 6
    out = exclude_records(ref, {"name_pattern": r"\d|-.*-|VERYVERY", "max_full_name_cluster": 3})
    assert list(out["forename_norm"]) == ["ANNA"] and out.attrs["excluded_records"] == 5


def test_tie_aware_threshold_meets_target_including_ties():
    from ofeval.linkage import select_threshold_for_precision
    # score 0.9: 10 pairs, 9 true; score 0.8: a tie block of 10 pairs, 5 true
    scored = pd.DataFrame({
        "match_probability": [0.9] * 10 + [0.8] * 10,
        "is_true_match": [1] * 9 + [0] + [1] * 5 + [0] * 5,
        "truth_id_l": [f"t{i}" for i in range(20)]})
    legacy = select_threshold_for_precision(scored, 0.8)
    aware = select_threshold_for_precision(scored, 0.8, tie_aware=True)
    assert legacy["threshold"] == 0.8 and legacy["achieved_precision"] == pytest.approx(0.7)  # overshoots
    assert aware["threshold"] == 0.9 and aware["achieved_precision"] == pytest.approx(0.9)


def test_full_name_duplication_diagnostics():
    ref = add_name_features(pd.DataFrame({
        "forename_norm": ["ANNA", "ANNA", "JOHN", "JOHN", "MARY", "OLU"],
        "surname_norm": ["SMITH", "SMITH", "LEE", "KHAN", "LEE", "OKAFOR"],
        "g": ["a", "a", "a", "b", "b", "b"]}))
    ref["record_id"] = [f"r{i}" for i in range(6)]
    empty = pd.DataFrame(columns=["id_l", "id_r", "match_probability"])
    task = diagnostics.TaskBundle(ref, ref, empty, subgroup="g")
    rate = diagnostics.REGISTRY["full_name_duplication_rate"].statistic(task)
    by_group = diagnostics.REGISTRY["subgroup_full_name_duplication_rate"].statistic(task)
    assert rate == pytest.approx(2 / 6)  # only ANNA SMITH is shared
    assert by_group.to_dict() == pytest.approx({"a": 2 / 3, "b": 0.0})
