"""Label-free calibration of an error profile by indirect inference.

Pseudo-truth learning is biased: pairs above a probability floor miss
link-breaking errors (full surname replacement, multi-field error) and include
some false pairs. But the bias of the learner is a property of the learner and
the task, so it is reproduced on a replica whose error mechanism is right. This
module therefore searches for the profile ``theta`` whose replicas *look like the
observed task through the same label-free lens*:

* moments ``g(task)``: the pseudo-truth learner's forename, surname and
  multi-field rates on the task, and the shares of records in each best-match
  band (the footprint of link-breaking errors). None needs true matches;
* objective: ``sum_k ((mean_r g_k(replica_r(theta)) - g_k(observed)) / sd_k)^2``,
  with ``sd_k`` the replica-to-replica standard deviation of moment k at the
  starting profile;
* parameters ``theta = (f, s, u, v)``: forename rate f, surname rate s, multi-field
  rate ``u * min(f, s)`` and full surname replacement ``v * s``. The fractions u and v
  keep every candidate profile coherent;
* solver: Levenberg-Marquardt (damped Gauss-Newton) with forward-difference
  Jacobians, within bounds;
* common random numbers: the same calibration seeds for every ``theta``, with the
  clean synthetic names built once per seed. These seeds are distinct from the
  evaluation seeds, so the gate never judges the replicas the profile was fitted on.

No truth labels are used anywhere. The calibrated profile is then judged by the
ordinary generator gate like any other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .diagnostics import BEST_MATCH_BANDS, TaskBundle
from .linkage import link
from .names import add_name_features
from .scenarios import incoming_subset
from .synthesis import corrupt_replica, synthesise

RATES = ["forename_error_rate", "surname_error_rate", "multi_field_error"]
MOMENT_NAMES = [f"learned_{r}" for r in RATES] + [f"best_match_share[{b}]" for b in BEST_MATCH_BANDS[:-1]]
PARAM_NAMES = ["forename_error_rate", "surname_error_rate", "multi_field_fraction", "full_replacement_fraction"]
LOWER = np.array([0.0, 0.0, 0.0, 0.0])
UPPER = np.array([0.6, 0.6, 1.0, 1.0])


def moments(task: TaskBundle) -> np.ndarray:
    """Label-free moment vector of a task (learned rates, then best-match band shares minus the last)."""
    learned = task.learned_profile
    shares = task.best_match_band.value_counts(normalize=True).reindex(BEST_MATCH_BANDS, fill_value=0.0)
    return np.concatenate([[float(learned[r]) for r in RATES], shares.to_numpy(dtype=float)[:-1]])


def theta_from_profile(profile: dict[str, Any]) -> np.ndarray:
    f, s = float(profile["forename_error_rate"]), float(profile["surname_error_rate"])
    m, r = float(profile["multi_field_error"]), float(profile["full_surname_replacement"])
    u = m / min(f, s) if min(f, s) > 0 else 0.0
    v = r / s if s > 0 else 0.0
    return np.clip(np.array([f, s, u, v]), LOWER, UPPER)


def profile_from_theta(theta: np.ndarray) -> dict[str, float]:
    f, s, u, v = (float(x) for x in np.clip(theta, LOWER, UPPER))
    return {"forename_error_rate": f, "surname_error_rate": s,
            "multi_field_error": u * min(f, s), "full_surname_replacement": v * s}


_SIM: "ReplicaSimulator | None" = None


def _init_sim(sim: "ReplicaSimulator") -> None:
    global _SIM
    _SIM = sim


def _seed_moments(seed: int, profile: dict[str, float]) -> np.ndarray:
    assert _SIM is not None
    return _SIM.moments_for_seed(seed, profile)


class ReplicaSimulator:
    """Generates replicas for a candidate profile with common random numbers and returns their moments.

    With ``workers > 1`` seeds are evaluated in a persistent process pool; use it
    as a context manager (``with ReplicaSimulator(...) as sim:``) so the pool is closed.
    """

    def __init__(self, observed_reference: pd.DataFrame, linkage_cfg: dict[str, Any], subgroup: str | None,
                 threshold: float, pseudo_truth_floor: float, synthesis: dict[str, Any] | None, seeds: list[int],
                 workers: int = 1, incoming: dict[str, Any] | None = None, record_side: str = "reference") -> None:
        self.reference = observed_reference
        self.linkage_cfg = linkage_cfg
        self.subgroup = subgroup
        self.threshold = threshold
        self.floor = pseudo_truth_floor
        self.seeds = list(seeds)
        self.incoming = incoming
        self.record_side = record_side
        # Clean synthetic references do not depend on the profile: build them once per seed.
        self.clean = {s: synthesise(observed_reference, s * 1000 + 1, subgroup, synthesis)[0] for s in self.seeds}
        self.n_evaluations = 0
        self._pool = None
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            self._pool = ProcessPoolExecutor(max_workers=min(workers, len(self.seeds)),
                                             initializer=_init_sim, initargs=(self,))

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_pool"] = None  # the pool itself is not sent to workers
        return state

    def __enter__(self) -> "ReplicaSimulator":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def moments_for_seed(self, seed: int, profile: dict[str, float]) -> np.ndarray:
        clean = self.clean[seed]
        base = incoming_subset(clean, self.incoming, seed=seed * 1000 + 1 + 555)  # as generate_replica
        corrupted = add_name_features(corrupt_replica(base, self.reference, profile, seed=seed * 1000 + 778))
        ref = add_name_features(clean)
        scored = link(ref, corrupted, self.linkage_cfg, self.subgroup)
        return moments(TaskBundle(ref, corrupted, scored, subgroup=self.subgroup,
                                  threshold=self.threshold, pseudo_truth_floor=self.floor,
                                  record_side=self.record_side))

    def per_seed(self, profile: dict[str, float]) -> np.ndarray:
        self.n_evaluations += 1
        if self._pool is not None:
            return np.vstack(list(self._pool.map(_seed_moments, self.seeds, [profile] * len(self.seeds))))
        return np.vstack([self.moments_for_seed(s, profile) for s in self.seeds])

    def mean(self, profile: dict[str, float]) -> np.ndarray:
        return self.per_seed(profile).mean(axis=0)


@dataclass
class CalibrationResult:
    profile: dict[str, Any]
    theta: np.ndarray
    objective: float
    start_objective: float
    converged: bool
    trace: list[dict[str, Any]] = field(default_factory=list)
    observed_moments: np.ndarray | None = None
    fitted_moments: np.ndarray | None = None
    moment_sd: np.ndarray | None = None


CLERICAL_RATES = ["forename_error_rate", "surname_error_rate", "multi_field_error", "full_surname_replacement"]


def clerical_moments(clerical: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Clerical estimates of the four rates and their binomial standard errors.

    ``clerical`` holds ``n`` (pairs reviewed) and the four rates. A continuity
    correction ((x + 0.5) / (n + 1)) keeps the standard error positive when a rate is 0.
    """
    n = int(clerical["n"])
    p = np.array([float(clerical[k]) for k in CLERICAL_RATES])
    p_adj = (p * n + 0.5) / (n + 1)
    return p, np.sqrt(p_adj * (1 - p_adj) / n)


def calibrate(simulator: ReplicaSimulator, observed_moments: np.ndarray, start_profile: dict[str, Any],
              clerical: dict[str, Any] | None = None, max_iter: int = 12,
              step: tuple[float, ...] = (0.01, 0.01, 0.05, 0.05), tol: float = 1e-3,
              lam: float = 1e-2, log=lambda *_: None) -> CalibrationResult:
    """Fit ``theta`` so that replica moments match the observed moments (see module docstring).

    Partly supervised: with ``clerical`` (a reviewed sample of true pairs), its four
    rate estimates enter the objective as extra residuals scaled by their binomial
    standard errors. Label-free moments and labelled evidence are combined in one
    fit, each weighted by its precision. A small sample mainly resolves what the
    label-free moments cannot separate (e.g. forename vs multi-field error).
    """
    theta = theta_from_profile(start_profile)
    start_rows = simulator.per_seed(profile_from_theta(theta))
    sd = np.maximum(start_rows.std(axis=0, ddof=1), 1e-3)
    g_obs = np.asarray(observed_moments, dtype=float)
    p_cler, se_cler = clerical_moments(clerical) if clerical else (None, None)

    def residual_at(t: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        g = (rows if rows is not None else simulator.per_seed(profile_from_theta(t))).mean(axis=0)
        res = (g - g_obs) / sd
        if p_cler is not None:
            prof = profile_from_theta(t)
            res = np.concatenate([res, (np.array([prof[k] for k in CLERICAL_RATES]) - p_cler) / se_cler])
        return res

    r = residual_at(theta, start_rows)
    obj = float(r @ r)
    start_obj = obj
    trace = [{"iteration": 0, **dict(zip(PARAM_NAMES, theta.tolist())), "objective": obj, "lambda": lam}]
    log(f"  calibration start: objective {obj:.2f}, theta {np.round(theta, 4).tolist()}")
    converged = False
    for it in range(1, max_iter + 1):
        jac = np.empty((len(r), len(theta)))
        for j in range(len(theta)):
            t = theta.copy()
            h = step[j] if theta[j] + step[j] <= UPPER[j] else -step[j]
            t[j] += h
            jac[:, j] = (residual_at(t) - r) / h
        jtj, jtr = jac.T @ jac, jac.T @ r
        improved = False
        for _ in range(6):
            delta = -np.linalg.solve(jtj + lam * np.diag(np.diag(jtj) + 1e-9), jtr)
            cand = np.clip(theta + delta, LOWER, UPPER)
            r_new = residual_at(cand)
            obj_new = float(r_new @ r_new)
            if obj_new < obj:
                theta, r, lam, improved = cand, r_new, max(lam / 3, 1e-6), True
                break
            lam *= 10
        rel = (obj - obj_new) / max(obj, 1e-12) if improved else 0.0
        obj = float(r @ r)
        trace.append({"iteration": it, **dict(zip(PARAM_NAMES, theta.tolist())), "objective": obj, "lambda": lam})
        log(f"  iteration {it}: objective {obj:.2f}, theta {np.round(theta, 4).tolist()}")
        if not improved or rel < tol:
            converged = True
            break
    profile = dict(start_profile)
    profile.update(profile_from_theta(theta))
    profile["error_exposure"] = (profile["forename_error_rate"] + profile["surname_error_rate"]
                                 - profile["multi_field_error"])
    k = len(g_obs)
    return CalibrationResult(profile=profile, theta=theta, objective=obj, start_objective=start_obj,
                             converged=converged, trace=trace, observed_moments=g_obs,
                             fitted_moments=g_obs + r[:k] * sd, moment_sd=sd)
