"""Observational Fidelity gates.

Two gate levels share one set of validation rules (see ``_validate``):

* every named check must be a registered diagnostic;
* diagnostics that hold by construction are invariants, not checks;
* a ``label_free`` policy may only use diagnostics observable without truth labels.

**Generator gate** (``GeneratorPolicy``; the principled gate). It judges a
replica *generator* (profile + synthesiser) from all of its R replicas. For
each diagnostic, with observed statistic T0 and replicas T1..TR:

* ``effect``: distance from T0 to the replica centroid (mean value, or mean
  distribution). This is the generator's bias, with seed noise averaged out.
* ``noise``: the (1 - alpha) quantile of leave-one-out distances from each Ti to
  the centroid of the other replicas. This is how far an exchangeable draw
  typically lands.
* ``p_value``: (1 + #{loo >= effect}) / (R + 1), a predictive check of whether the
  observed task looks like one more replica.
* the check fails only if ``effect > max(margin, noise)``. Pure seed noise can
  never fail, and a detectable difference below the stated margin is reported as
  ``equivalent`` rather than failed.

Replicas are never discarded; the verdict applies to the generator.

**Replica gate** (``ReplicaGatePolicy``; legacy, as in the paper). Each replica
passes when every ``|distance| <= tolerance``. Gated modes keep only passing
replicas. It is retained to reproduce the paper.

Invariants are evaluated on every replica with zero tolerance. A violation means
the synthesiser no longer does what it claims; it is reported, not gated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from .diagnostics import LAYERS, REGISTRY


def _validate(policy: str, checks: list[str], label_free: bool) -> None:
    unknown = sorted(set(checks) - set(REGISTRY))
    if unknown:
        raise ValueError(f"Gate policy {policy!r} uses unknown diagnostics: {unknown}")
    invariants = [c for c in checks if REGISTRY[c].by_construction]
    if invariants:
        raise ValueError(f"Gate policy {policy!r} checks {invariants}, which hold by construction; "
                         "they are reported as invariants and cannot be gate checks")
    if label_free:
        needs_truth = [c for c in checks if not REGISTRY[c].observable]
        if needs_truth:
            raise ValueError(f"Label-free gate policy {policy!r} uses truth-based diagnostics: {needs_truth}")


def _layer_rollup(checks: list[str], failed: list[str]) -> list[dict[str, Any]]:
    """Per layer: number of checks and failures; a layer with no checks is ``not_assessed``."""
    rows = []
    for layer in LAYERS:
        in_layer = [c for c in checks if REGISTRY[c].layer == layer]
        n_failed = sum(c in failed for c in in_layer)
        verdict = "not_assessed" if not in_layer else ("fail" if n_failed else "pass")
        rows.append({"layer": layer, "n_checks": len(in_layer), "n_failed": n_failed, "verdict": verdict})
    return rows


# --------------------------------------------------------------------------- generator gate

@dataclass(frozen=True)
class GateSettings:
    """How the noise band is estimated and how many checks it is corrected for.

    * ``noise="empirical"``: the (1 - alpha) quantile of the R leave-one-out distances.
      With R = 20 this cannot resolve small alpha.
    * ``noise="half_normal"``: ``z * RMS(leave-one-out distances)`` with
      ``P(|Z| > z) = alpha``. This extrapolates to small alpha; it is exact for
      normally distributed scalar statistics and approximate for distances.
    * ``multiplicity="bonferroni"``: alpha is divided by the number of applicable
      checks in the policy, so a correct generator is rejected by the whole
      policy with probability about alpha (when noise, not the margin, binds).
    """

    alpha: float = 0.05
    noise: str = "empirical"
    multiplicity: str = "none"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "GateSettings":
        s = cls(alpha=float(cfg.get("alpha", 0.05)), noise=cfg.get("noise", "empirical"),
                multiplicity=cfg.get("multiplicity", "none"))
        if s.noise not in {"empirical", "half_normal"} or s.multiplicity not in {"none", "bonferroni"}:
            raise ValueError(f"Unknown gate settings: noise={s.noise!r}, multiplicity={s.multiplicity!r}")
        return s


def _centroid(values: list[Any]) -> Any:
    if isinstance(values[0], pd.Series):
        return pd.concat(values, axis=1).fillna(0.0).mean(axis=1)
    return float(np.mean([float(v) for v in values]))


def assess_diagnostic(name: str, observed: Any, replicas: list[Any], margin: float, alpha: float,
                      noise_method: str = "empirical") -> dict[str, Any]:
    """Generator-level assessment of one diagnostic (see module docstring)."""
    d = REGISTRY[name]
    r = len(replicas)
    if observed is None or any(v is None for v in replicas):
        return {"diagnostic": name, "layer": d.layer, "taxonomy_layer": d.taxonomy_layer, "observable": d.observable,
                "kind": d.kind, "n_replicas": r, "observed": None, "replica_mean": None, "signed_effect": None,
                "effect": float("nan"), "noise": float("nan"), "margin": margin, "tolerance": margin, "alpha": alpha,
                "p_value": float("nan"), "classification": "not_applicable", "passed": True}
    centre = _centroid(replicas)
    effect = abs(d.distance(observed, centre))
    if r >= 2:
        loo = np.array([abs(d.distance(_centroid(replicas[:i] + replicas[i + 1:]), replicas[i])) for i in range(r)])
        if noise_method == "half_normal":
            rms = float(np.sqrt(np.mean(loo ** 2)))
            noise = NormalDist().inv_cdf(1 - alpha / 2) * rms
            p_value = float(2 * (1 - NormalDist().cdf(effect / rms))) if rms > 0 else float(effect == 0)
        else:
            noise = float(np.quantile(loo, 1 - alpha))
            p_value = float((1 + (loo >= effect).sum()) / (r + 1))
    else:
        noise, p_value = float("nan"), float("nan")
    tolerance = max(margin, noise) if not np.isnan(noise) else margin
    if not np.isnan(noise) and effect <= noise:
        classification = "consistent"
    elif effect <= margin:
        classification = "equivalent"
    else:
        classification = "not_equivalent"
    scalar = d.scalar
    return {
        "diagnostic": name, "layer": d.layer, "taxonomy_layer": d.taxonomy_layer, "observable": d.observable,
        "kind": d.kind, "n_replicas": r,
        "observed": float(observed) if scalar else None,
        "replica_mean": float(centre) if scalar else None,
        "signed_effect": d.distance(observed, centre) if scalar else None,
        "effect": effect, "noise": noise, "margin": margin, "tolerance": tolerance, "alpha": alpha, "p_value": p_value,
        "classification": classification, "passed": classification != "not_equivalent",
    }


@dataclass(frozen=True)
class GeneratorPolicy:
    name: str
    margins: dict[str, float]
    label_free: bool = False
    description: str = ""

    @classmethod
    def from_config(cls, name: str, cfg: dict[str, Any]) -> "GeneratorPolicy":
        policy = cls(name=name, margins={k: float(v) for k, v in cfg["margins"].items()},
                     label_free=bool(cfg.get("label_free", False)), description=cfg.get("description", ""))
        _validate(name, list(policy.margins), policy.label_free)
        return policy

    def assess(self, observed: dict[str, Any], replicas: list[dict[str, Any]],
               settings: GateSettings | float = GateSettings()) -> "GeneratorReport":
        if not isinstance(settings, GateSettings):
            settings = GateSettings(alpha=float(settings))
        applicable = [n for n in self.margins if observed[n] is not None]
        alpha = settings.alpha / max(len(applicable), 1) if settings.multiplicity == "bonferroni" else settings.alpha
        rows = [assess_diagnostic(n, observed[n], [rep[n] for rep in replicas], m, alpha, settings.noise)
                for n, m in self.margins.items()]
        diagnostics = pd.DataFrame(rows)
        failed = diagnostics.loc[~diagnostics["passed"], "diagnostic"].tolist()
        applicable = diagnostics.loc[diagnostics["classification"] != "not_applicable", "diagnostic"].tolist()
        return GeneratorReport(self.name, diagnostics, not failed, failed, _layer_rollup(applicable, failed))


@dataclass
class GeneratorReport:
    policy: str
    diagnostics: pd.DataFrame
    passed: bool
    failed: list[str]
    layers: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        return {"gate_policy": self.policy, "passed": self.passed, "failed_diagnostics": ";".join(self.failed),
                **{f"{lv['layer']}_verdict": lv["verdict"] for lv in self.layers}}


def load_generator_policies(config: dict[str, Any]) -> dict[str, GeneratorPolicy]:
    return {name: GeneratorPolicy.from_config(name, cfg) for name, cfg in config["policies"].items()}


# --------------------------------------------------------------------------- replica gate (legacy)

@dataclass(frozen=True)
class ReplicaGatePolicy:
    name: str
    checks: dict[str, float]
    label_free: bool = False
    description: str = ""

    @classmethod
    def from_config(cls, name: str, cfg: dict[str, Any]) -> "ReplicaGatePolicy":
        policy = cls(name=name, checks={k: float(v) for k, v in cfg["checks"].items()},
                     label_free=bool(cfg.get("label_free", False)), description=cfg.get("description", ""))
        _validate(name, list(policy.checks), policy.label_free)
        return policy

    def evaluate(self, comparison: pd.DataFrame) -> tuple[bool, list[str]]:
        """Return (passed, failed checks). A missing or NaN distance fails."""
        distance = comparison.set_index("diagnostic")["distance"]
        failed = [c for c, tol in self.checks.items()
                  if c not in distance.index or pd.isna(distance[c]) or abs(float(distance[c])) > tol]
        return not failed, failed

    def layer_verdicts(self, comparison: pd.DataFrame) -> list[dict[str, Any]]:
        return _layer_rollup(list(self.checks), self.evaluate(comparison)[1])


def load_replica_policies(config: dict[str, dict[str, Any]]) -> dict[str, ReplicaGatePolicy]:
    return {name: ReplicaGatePolicy.from_config(name, cfg) for name, cfg in config.items()}


def invariant_violations(comparison: pd.DataFrame) -> list[str]:
    inv = comparison[comparison["by_construction"].astype(bool)]
    return inv.loc[inv["distance"].abs() > 0, "diagnostic"].tolist()


@dataclass
class ReplicaGateReport:
    """Everything the legacy replica gate concluded about one replica."""

    comparison: pd.DataFrame
    verdicts: dict[str, tuple[bool, list[str]]]
    layer_verdicts: dict[str, list[dict[str, Any]]]
    invariant_violations: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, comparison: pd.DataFrame, policies: dict[str, ReplicaGatePolicy]) -> "ReplicaGateReport":
        return cls(
            comparison=comparison,
            verdicts={n: p.evaluate(comparison) for n, p in policies.items()},
            layer_verdicts={n: p.layer_verdicts(comparison) for n, p in policies.items()},
            invariant_violations=invariant_violations(comparison),
        )

    def passed(self, policy: str | None) -> bool:
        """Whether the replica passes ``policy``; an ungated mode (None) always passes."""
        return True if policy is None else self.verdicts[policy][0]
