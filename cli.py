"""Command line: ``python -m ofeval run`` (full study) and ``python -m ofeval summarise`` (tables/figures only)."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import pandas as pd
import yaml

from .experiment import run_scenario
from .gate import GateSettings, load_generator_policies, load_replica_policies
from .figures import make_figures
from .summarise import build_tables

LONG_TABLES = ["tasks", "linkage", "band_mass", "diagnostics", "generator_gate", "generator_verdicts",
               "legacy_gate_decisions", "legacy_gate_layers", "provenance", "floor_sensitivity", "thresholds"]


def load_study(path: Path) -> dict:
    study = yaml.safe_load(path.read_text(encoding="utf-8"))
    study["_path"] = path.resolve()
    # Both gate levels are validated against the diagnostic registry.
    generator = load_generator_policies(study["gate"])
    GateSettings.from_config(study["gate"])
    replica = load_replica_policies(study["legacy_gate_policies"])
    for key, defined in [("main_gate_policy", generator), ("legacy_main_gate_policy", replica)]:
        if study[key] not in defined:
            raise ValueError(f"{key} {study[key]!r} is not defined")
    for mode in study["modes"]:
        for key, defined in [("gate", generator), ("legacy_gate", replica)]:
            if mode.get(key) and mode[key] not in defined:
                raise ValueError(f"Mode {mode['label']!r} uses undefined {key} policy {mode[key]!r}")
    return study


def _resolve(study: dict, key: str) -> Path:
    p = Path(study[key])
    return p if p.is_absolute() else (study["_path"].parent / p).resolve()


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ["numpy", "pandas", "rapidfuzz", "matplotlib", "pyyaml"]:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def cmd_run(args: argparse.Namespace) -> None:
    study = load_study(Path(args.study))
    data_dir, out_dir = _resolve(study, "data_dir"), Path(args.out) if args.out else _resolve(study, "output_dir")
    if args.seeds:
        study["evaluation"]["seeds"] = args.seeds
    if args.first_seed:
        study["evaluation"]["first_seed"] = args.first_seed
    long_dir = out_dir / "long"
    long_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    names = args.scenario or [s for s in study["scenarios"]]
    parts: dict[str, list[pd.DataFrame]] = {k: [] for k in LONG_TABLES}
    for name in names:
        scn_cfg = yaml.safe_load((study["_path"].parent / study["scenarios"][name]).read_text(encoding="utf-8"))
        tables = run_scenario(scn_cfg, study, data_dir, out_dir, workers=args.workers)
        for k in LONG_TABLES:
            parts[k].append(tables[k])
    for k in LONG_TABLES:
        pd.concat(parts[k], ignore_index=True).to_csv(long_dir / f"{k}.csv", index=False)
    meta = {"study": str(study["_path"]), "scenarios": names, "seeds": study["evaluation"]["seeds"],
            "first_seed": study["evaluation"].get("first_seed", 1),
            "workers": args.workers, "runtime_seconds": round(time.time() - started, 1), "versions": _versions()}
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _summarise(study, out_dir)


def _summarise(study: dict, out_dir: Path) -> None:
    long = {k: pd.read_csv(out_dir / "long" / f"{k}.csv") for k in LONG_TABLES}
    tables = build_tables(long, study)
    table_dir = out_dir / "tables"
    for directory, content in [(table_dir, tables), (table_dir / "legacy_replica_gate", build_tables(long, study, legacy=True))]:
        directory.mkdir(parents=True, exist_ok=True)
        for name, df in content.items():
            df.to_csv(directory / f"{name}.csv", index=False)
    modes = [m["label"] for m in study["modes"]]
    reference_group = None
    for p in study["scenarios"].values():
        sub = yaml.safe_load((study["_path"].parent / p).read_text(encoding="utf-8")).get("subgroup")
        if sub:
            reference_group = sub.get("reference_group")
    # Ungated modes reuse a gated mode's replicas, so the gate figure shows gated modes only;
    # every policy is drawn so all diagnostics appear (the benchmark policy is a superset).
    gated = [m["label"] for m in study["modes"] if m.get("gate")]
    make_figures(tables, modes, reference_group, out_dir / "figures", gated_modes=gated,
                 gate_policies=list(study["gate"]["policies"]))
    print(f"Tables: {table_dir}\nFigures: {out_dir / 'figures'}")


def cmd_summarise(args: argparse.Namespace) -> None:
    study = load_study(Path(args.study))
    _summarise(study, Path(args.out) if args.out else _resolve(study, "output_dir"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ofeval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run the study: observed tasks, replicas, gates, linkage evaluation, tables, figures")
    run.add_argument("--study", default="configs/study.yaml")
    run.add_argument("--scenario", action="append", help="Run only this scenario (repeatable)")
    run.add_argument("--seeds", type=int, help="Override the number of replicate seeds (e.g. 2 for a smoke test)")
    run.add_argument("--first-seed", type=int, help="First replicate seed (default 1); use to check verdict stability")
    run.add_argument("--workers", type=int, default=1, help="Parallel worker processes for replicas")
    run.add_argument("--out", help="Output directory (default: output_dir in the study config)")
    run.set_defaults(func=cmd_run)
    summ = sub.add_parser("summarise", help="Rebuild tables and figures from existing long outputs")
    summ.add_argument("--study", default="configs/study.yaml")
    summ.add_argument("--out")
    summ.set_defaults(func=cmd_summarise)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
